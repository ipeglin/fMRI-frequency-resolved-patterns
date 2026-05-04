use anyhow::Result;
use ndarray::{Array3, s};
use scirs2_signal::hilbert::hilbert;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands;
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_dataset_old};
use utils::roi_migration::{check_roi_fingerprint, propagate_roi_attrs};

/// Number of log-spaced frequency bins for marginal spectra and the 2-D Hilbert
/// spectrum H(omega, t). Matches the CWT scale-grid height so HHT spectrograms
/// and CWT scalograms share both frequency axis and DenseNet201 input height.
const TARGET_N_FREQ: usize = 224;

/// Result of a Hilbert-Huang Transform applied to a set of IMF modes.
///
/// Modes tensor shape: [n_modes, n_channels, n_timepoints]
/// Envelope/inst_freq shape same.
/// Marginal spectra shape: [n_modes, n_channels, n_freq_bins]
/// Full spectrum shape: [n_channels, n_freq_bins]
struct HHTResult {
    /// Instantaneous amplitude (envelope) per mode per channel [n_modes, n_channels, n_timepoints]
    envelope: Vec<f64>,
    envelope_shape: [usize; 3],
    /// Instantaneous frequency per mode per channel [n_modes, n_channels, n_timepoints]
    inst_freq: Vec<f64>,
    inst_freq_shape: [usize; 3],
    /// Frequency axis for marginal/full spectra (Hz)
    freq_axis: Vec<f64>,
    /// Marginal Hilbert Spectrum per mode per channel [n_modes, n_channels, n_freq_bins]
    marginal_spectra: Vec<f64>,
    marginal_spectra_shape: [usize; 3],
    /// Normalized full Hilbert Spectrum (sum over modes) per channel [n_channels, n_freq_bins]
    full_spectrum: Vec<f64>,
    full_spectrum_shape: [usize; 2],
    /// 2-D Hilbert Spectrum H(ω,t): energy summed over modes, per channel [n_channels, n_freq_bins, n_timepoints]
    hilbert_spectrum: Vec<f64>,
    hilbert_spectrum_shape: [usize; 3],
}

/// Compute HHT from a modes array with shape [n_modes, n_channels, n_timepoints].
///
/// Modes are read from HDF5 as flat row-major with that shape ordering.
fn compute_hht(cfg: &AppConfig, modes_flat: &[f32], shape: &[usize]) -> Result<HHTResult> {
    let n_modes = shape[0];
    let n_channels = shape[1];
    let n_timepoints = shape[2];
    let sampling_rate = cfg.task_sampling_rate;

    // Log-spaced frequency grid in Hz, matching the CWT scale grid in
    // `crates/03cwt`. Bounds come from `frequency_bands::SLOW_BANDS` so HHT
    // spectra share the analysed BOLD window with CWT scalograms and MVMD.
    //
    // Energy is binned (not interpolated): each (amp^2, ifreq) sample at time t
    // is dropped into the nearest log-grid bin. Samples whose instantaneous
    // frequency falls outside [f_min, f_max] are discarded — same semantics as
    // CWT, which is only defined on the chosen scale grid.
    let f_min = frequency_bands::f_min();
    let f_max = frequency_bands::f_max();
    let n_freq = TARGET_N_FREQ;
    let log_ratio_max = (f_max / f_min).ln();
    let freq_axis: Vec<f64> = (0..n_freq)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n_freq - 1) as f64))
        .collect();

    let modes = Array3::from_shape_vec(
        (n_modes, n_channels, n_timepoints),
        modes_flat.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    )?;

    let env_total = n_modes * n_channels * n_timepoints;
    let marg_total = n_modes * n_channels * n_freq;
    let full_total = n_channels * n_freq;
    let hs_total = n_channels * n_freq * n_timepoints;

    let mut envelope_buf = vec![0f64; env_total];
    let mut inst_freq_buf = vec![0f64; env_total];
    let mut marginal_buf = vec![0f64; marg_total];
    let mut full_buf = vec![0f64; full_total];
    let mut hilbert_spectrum_buf = vec![0f64; hs_total];

    for m in 0..n_modes {
        for c in 0..n_channels {
            let channel_signal: Vec<f64> = modes.slice(s![m, c, ..]).to_vec();

            // Analytic signal via Hilbert transform
            let analytic = hilbert(&channel_signal)
                .map_err(|e| anyhow::anyhow!("hilbert failed mode={} ch={}: {}", m, c, e))?;

            // Envelope (instantaneous amplitude)
            let amp: Vec<f64> = analytic.iter().map(|z| z.norm()).collect();

            // Instantaneous frequency via unwrapped phase derivative
            let phase: Vec<f64> = analytic.iter().map(|z| z.im.atan2(z.re)).collect();
            let ifreq = unwrapped_phase_to_ifreq(&phase, sampling_rate);

            // Write envelope and inst_freq into flat buffers
            let base = m * n_channels * n_timepoints + c * n_timepoints;
            envelope_buf[base..base + n_timepoints].copy_from_slice(&amp);
            inst_freq_buf[base..base + n_timepoints].copy_from_slice(&ifreq);

            // Marginal Hilbert Spectrum: bin amplitude^2 by instantaneous frequency
            // 2-D Hilbert Spectrum H(ω,t): scatter energy into [freq_bin, t] cell
            //
            // Bin index found by inverting the log-spaced grid construction:
            //   f_i = f_min * (f_max/f_min)^(i/(n_freq-1))
            //   => i = round( log(f/f_min) / log(f_max/f_min) * (n_freq-1) )
            // Pure histogram assignment — no interpolation, no resampling.
            let marg_base = m * n_channels * n_freq + c * n_freq;
            let hs_base = c * n_freq * n_timepoints;
            for t in 0..n_timepoints {
                let f = ifreq[t];
                if f < f_min || f > f_max {
                    continue;
                }
                let log_ratio = (f / f_min).ln() / log_ratio_max;
                let bin = ((log_ratio * (n_freq - 1) as f64).round() as usize).min(n_freq - 1);
                let energy = amp[t] * amp[t];
                marginal_buf[marg_base + bin] += energy;
                // H(ω,t): sum over modes, row-major [freq_bin, timepoint]
                hilbert_spectrum_buf[hs_base + bin * n_timepoints + t] += energy;
            }

            // Accumulate into full spectrum (before normalization)
            let full_base = c * n_freq;
            for b in 0..n_freq {
                full_buf[full_base + b] += marginal_buf[marg_base + b];
            }
        }
    }

    // Normalize full spectrum per channel so it sums to 1 (avoid div-by-zero)
    for c in 0..n_channels {
        let base = c * n_freq;
        let sum: f64 = full_buf[base..base + n_freq].iter().sum();
        if sum > 0.0 {
            for b in 0..n_freq {
                full_buf[base + b] /= sum;
            }
        }
    }

    Ok(HHTResult {
        envelope: envelope_buf,
        envelope_shape: [n_modes, n_channels, n_timepoints],
        inst_freq: inst_freq_buf,
        inst_freq_shape: [n_modes, n_channels, n_timepoints],
        freq_axis,
        marginal_spectra: marginal_buf,
        marginal_spectra_shape: [n_modes, n_channels, n_freq],
        full_spectrum: full_buf,
        full_spectrum_shape: [n_channels, n_freq],
        hilbert_spectrum: hilbert_spectrum_buf,
        hilbert_spectrum_shape: [n_channels, n_freq, n_timepoints],
    })
}

/// Unwrap phase and compute instantaneous frequency via central differences.
fn unwrapped_phase_to_ifreq(phase: &[f64], fs: f64) -> Vec<f64> {
    use std::f64::consts::PI;
    let n = phase.len();
    if n == 0 {
        return vec![];
    }

    let mut unwrapped = vec![phase[0]];
    let mut prev = phase[0];
    for &p in &phase[1..] {
        let mut diff = p - prev;
        while diff > PI {
            diff -= 2.0 * PI;
        }
        while diff < -PI {
            diff += 2.0 * PI;
        }
        unwrapped.push(*unwrapped.last().unwrap() + diff);
        prev = p;
    }

    let mut ifreq = Vec::with_capacity(n);
    // Forward difference at first point
    ifreq.push(fs * (unwrapped[1] - unwrapped[0]) / (2.0 * PI));
    // Central differences
    for i in 1..n - 1 {
        ifreq.push(fs * (unwrapped[i + 1] - unwrapped[i - 1]) / (4.0 * PI));
    }
    // Backward difference at last point
    ifreq.push(fs * (unwrapped[n - 1] - unwrapped[n - 2]) / (2.0 * PI));
    ifreq
}

/// Write all HHT outputs to an HDF5 group.
fn write_hht(
    cfg: &AppConfig,
    hht_group: &hdf5::Group,
    result: &HHTResult,
    force: bool,
) -> Result<()> {
    let repetition_time: f64 = 1.0 / cfg.task_sampling_rate;
    let f_min = frequency_bands::f_min();
    let f_max = frequency_bands::f_max();
    write_dataset_old(
        hht_group,
        "envelope",
        &result.envelope,
        &result.envelope_shape,
        None,
    )?;
    write_dataset_old(
        hht_group,
        "instantaneous_frequency",
        &result.inst_freq,
        &result.inst_freq_shape,
        None,
    )?;
    write_dataset_old(
        hht_group,
        "frequency_axis",
        &result.freq_axis,
        &[result.freq_axis.len()],
        Some(&[
            H5Attr::f64("fs_hz", cfg.task_sampling_rate),
            H5Attr::f64("tr_s", repetition_time),
            H5Attr::string("spacing", "log"),
            H5Attr::f64("f_min", f_min),
            H5Attr::f64("f_max", f_max),
        ]),
    )?;

    let marg_group = open_or_create_group(hht_group, "marginal_spectra", force)?;
    write_dataset_old(
        &marg_group,
        "spectra",
        &result.marginal_spectra,
        &result.marginal_spectra_shape,
        None,
    )?;

    write_dataset_old(
        hht_group,
        "full_spectrum",
        &result.full_spectrum,
        &result.full_spectrum_shape,
        Some(&[H5Attr::string(
            "description",
            "normalized sum of marginal spectra across all modes per channel",
        )]),
    )?;

    write_dataset_old(
        hht_group,
        "hilbert_spectrum",
        &result.hilbert_spectrum,
        &result.hilbert_spectrum_shape,
        Some(&[H5Attr::string(
            "description",
            "2D Hilbert spectrum H(omega,t): energy summed over modes per channel [n_channels, n_freq, n_timepoints]",
        )]),
    )?;

    Ok(())
}

/// Copy `roi_indices` dataset from MVMD source group to HHT destination group, if present.
fn propagate_roi_indices(src: &hdf5::Group, dest: &hdf5::Group) -> Result<()> {
    let Ok(ds) = src.dataset("roi_indices") else {
        return Ok(());
    };
    if dest.dataset("roi_indices").is_ok() {
        return Ok(());
    }
    let data: Vec<u32> = ds.read_raw()?;
    write_dataset_old(dest, "roi_indices", &data, &[data.len()], None)?;
    Ok(())
}

/// Compute HHT for a single MVMD subgroup containing a `modes` dataset and write outputs
/// to a mirror group under `hht_parent` named `name`.
///
/// Skips work if destination already contains `hilbert_spectrum`.
/// Propagates `roi_indices` if present in source.
fn process_mvmd_modes_group(
    cfg: &AppConfig,
    mvmd_parent: &hdf5::Group,
    hht_parent: &hdf5::Group,
    name: &str,
    task_name: &str,
    is_roi: bool,
) -> Result<()> {
    let mvmd_sub = match mvmd_parent.group(name) {
        Ok(g) => g,
        Err(_) => {
            debug!(
                task_name = task_name,
                group = name,
                "mvmd subgroup missing, skipping"
            );
            return Ok(());
        }
    };

    if is_roi {
        let expected = cfg.roi_selection.fingerprint();
        check_roi_fingerprint(&mvmd_sub, &expected, &format!("/04mvmd/.../{name}"))?;
    }

    let hht_done = !cfg.force
        && hht_parent
            .group(name)
            .map(|g| g.dataset("hilbert_spectrum").is_ok())
            .unwrap_or(false);

    if hht_done {
        if is_roi {
            let existing = hht_parent.group(name)?;
            check_roi_fingerprint(
                &existing,
                &cfg.roi_selection.fingerprint(),
                &format!("/05hht/.../{name}"),
            )?;
        }
    }

    let modes_ds = mvmd_sub.dataset("modes")?;
    let modes_shape = modes_ds.shape();
    let modes_flat: Vec<f32> = modes_ds.read_raw()?;

    let [n_modes, n_channels, n_timepoints] = match modes_shape.as_slice() {
        &[a, b, c] => [a, b, c],
        _ => anyhow::bail!(
            "unexpected modes shape {:?} for /mvmd/{}",
            modes_shape,
            name
        ),
    };

    info!(
        task_name = task_name,
        group = name,
        n_modes = n_modes,
        n_channels = n_channels,
        n_timepoints = n_timepoints,
        "computing HHT"
    );

    let hht_start = Instant::now();
    let result = compute_hht(cfg, &modes_flat, &modes_shape)?;
    let hht_duration_ms = hht_start.elapsed().as_millis();

    let write_start = Instant::now();
    let dest = open_or_create_group(hht_parent, name, cfg.force)?;
    write_hht(cfg, &dest, &result, cfg.force)?;
    propagate_roi_indices(&mvmd_sub, &dest)?;
    if is_roi {
        propagate_roi_attrs(&mvmd_sub, &dest)?;
    }
    let write_duration_ms = write_start.elapsed().as_millis();

    info!(
        task_name = task_name,
        group = name,
        hht_duration_ms = hht_duration_ms,
        write_duration_ms = write_duration_ms,
        "HHT complete"
    );

    Ok(())
}

/// Iterate `block_*` subgroups under `mvmd_parent/name` and compute HHT for each,
/// mirroring outputs under `hht_parent/name`.
fn process_blocks_parent(
    cfg: &AppConfig,
    mvmd_parent: &hdf5::Group,
    hht_parent: &hdf5::Group,
    name: &str,
    task_name: &str,
    error_count: &mut usize,
    is_roi: bool,
) -> Result<()> {
    let mvmd_blocks = match mvmd_parent.group(name) {
        Ok(g) => g,
        Err(_) => {
            debug!(
                task_name = task_name,
                group = name,
                "mvmd blocks parent missing, skipping"
            );
            return Ok(());
        }
    };

    let block_names: Vec<String> = mvmd_blocks
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("block_"))
        .collect();

    if block_names.is_empty() {
        debug!(
            task_name = task_name,
            group = name,
            "no blocks found, skipping"
        );
        return Ok(());
    }

    let hht_blocks = open_or_create_group(hht_parent, name, false)?;

    for block_name in &block_names {
        if let Err(e) = process_mvmd_modes_group(
            cfg,
            &mvmd_blocks,
            &hht_blocks,
            block_name,
            task_name,
            is_roi,
        ) {
            *error_count += 1;
            warn!(
                task_name = task_name,
                group = name,
                block = block_name,
                error = %e,
                "skipping block HHT due to error"
            );
        }
    }

    info!(
        task_name = task_name,
        group = name,
        total_blocks = total_blocks,
        "finished all block HHT decompositions"
    );

    Ok(())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        "starting HHT pipeline (MVMD-based Hilbert-Huang Transform)"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.consolidated_data_dir)?
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            let formatted = BidsSubjectId::parse(id).to_dir_name();
            Some((formatted, path))
        })
        .collect();

    let total_subjects = subjects.len();
    info!(num_subjects = total_subjects, "found subject directories");

    let mut subject_idx = 0;
    let mut error_count: usize = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;

        let _subject_span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();

        let available_timeseries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter_map(|path| {
                let p = path.file_name()?.to_str()?.to_string();
                if p.contains(".h5") { Some(path) } else { None }
            })
            .collect();

        info!(num_files = available_timeseries.len(), "processing subject");

        for file_path in &available_timeseries {
            let file_result: anyhow::Result<()> = (|| {
                let bids =
                    BidsFilename::parse(match file_path.file_name().and_then(|n| n.to_str()) {
                        Some(name) => name,
                        None => return Ok(()),
                    });
                let task_name = bids.get("task").unwrap_or("unknown");

                let h5_file = open_or_create(file_path)?;

                // MVMD group must exist — this step depends on step 04
                let mvmd_group = match h5_file.group("04mvmd") {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            "no mvmd group found, skipping (run tcp-mvmd first)"
                        );
                        return Ok(());
                    }
                };

                let hht_group = open_or_create_group(&h5_file, "05hht", false)?;

                match task_name {
                    "restAP" => {
                        process_mvmd_modes_group(
                            cfg,
                            &mvmd_group,
                            &hht_group,
                            "full_run_std",
                            task_name,
                            false,
                        )?;
                        if !cfg.roi_selection.is_empty() {
                            process_mvmd_modes_group(
                                cfg,
                                &mvmd_group,
                                &hht_group,
                                "full_run_std_roi",
                                task_name,
                                true,
                            )?;
                        }
                    }
                    "hammerAP" => {
                        process_blocks_parent(
                            cfg,
                            &mvmd_group,
                            &hht_group,
                            "blocks_std",
                            task_name,
                            &mut error_count,
                            false,
                        )?;
                        if !cfg.roi_selection.is_empty() {
                            process_blocks_parent(
                                cfg,
                                &mvmd_group,
                                &hht_group,
                                "blocks_std_roi",
                                task_name,
                                &mut error_count,
                                true,
                            )?;
                        }
                    }
                    other => {
                        debug!(task_name = other, "unrecognized task type, skipping HHT");
                    }
                }

                Ok(())
            })();

            if let Err(e) = file_result {
                error_count += 1;
                warn!(
                    file = %file_path.display(),
                    error = %e,
                    "skipping file due to error"
                );
            }
        }
    }

    if error_count > 0 {
        warn!(
            error_count = error_count,
            "some subjects/blocks were skipped due to errors"
        );
    }

    let total_duration_ms = run_start.elapsed().as_millis();
    info!(
        error_count = error_count,
        total_duration_ms = total_duration_ms,
        "HHT pipeline complete"
    );

    Ok(())
}
