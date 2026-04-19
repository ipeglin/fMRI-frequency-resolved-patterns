use anyhow::Result;
use config::{HilbertHuangConfig, bids_filename::BidsFilename, bids_subject_id::BidsSubjectId};
use hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_dataset};
use ndarray::{Array3, s};
use scirs2_signal::hilbert::hilbert;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};

/// Sampling period (TR) in seconds — matches fMRI acquisition.
const TR: f64 = 0.8;
/// Sampling frequency in Hz.
const FS: f64 = 1.0 / TR;

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
    /// Z-score standardized envelope per mode per channel [n_modes, n_channels, n_timepoints]
    std_envelope: Vec<f64>,
    /// Z-score standardized marginal spectra per mode per channel [n_modes, n_channels, n_freq_bins]
    std_marginal_spectra: Vec<f64>,
    /// Z-score standardized full spectrum per channel [n_channels, n_freq_bins]
    std_full_spectrum: Vec<f64>,
    /// Z-score standardized 2-D Hilbert Spectrum per channel [n_channels, n_freq_bins, n_timepoints]
    std_hilbert_spectrum: Vec<f64>,
}

/// Compute HHT from a modes array with shape [n_modes, n_channels, n_timepoints].
///
/// Modes are read from HDF5 as flat row-major with that shape ordering.
fn compute_hht(modes_flat: &[f32], shape: &[usize]) -> Result<HHTResult> {
    let n_modes = shape[0];
    let n_channels = shape[1];
    let n_timepoints = shape[2];

    // Frequency bins for marginal spectrum: linspace(0, FS/2, n_timepoints/2 + 1)
    let n_freq = n_timepoints / 2 + 1;
    let df = FS / n_timepoints as f64;
    let freq_axis: Vec<f64> = (0..n_freq).map(|i| i as f64 * df).collect();

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
            let ifreq = unwrapped_phase_to_ifreq(&phase, FS);

            // Write envelope and inst_freq into flat buffers
            let base = m * n_channels * n_timepoints + c * n_timepoints;
            envelope_buf[base..base + n_timepoints].copy_from_slice(&amp);
            inst_freq_buf[base..base + n_timepoints].copy_from_slice(&ifreq);

            // Marginal Hilbert Spectrum: bin amplitude^2 by instantaneous frequency
            // 2-D Hilbert Spectrum H(ω,t): scatter energy into [freq_bin, t] cell
            let marg_base = m * n_channels * n_freq + c * n_freq;
            let hs_base = c * n_freq * n_timepoints;
            for t in 0..n_timepoints {
                let f = ifreq[t];
                if f < 0.0 || f > FS / 2.0 {
                    continue;
                }
                let bin = ((f / df).round() as usize).min(n_freq - 1);
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

    // Z-score standardize envelope per (mode, channel) slice
    let mut std_envelope_buf = vec![0f64; env_total];
    for m in 0..n_modes {
        for c in 0..n_channels {
            let base = m * n_channels * n_timepoints + c * n_timepoints;
            let standardized = zscore(&envelope_buf[base..base + n_timepoints]);
            std_envelope_buf[base..base + n_timepoints].copy_from_slice(&standardized);
        }
    }

    // Z-score standardize marginal spectra per (mode, channel) slice
    let mut std_marginal_buf = vec![0f64; marg_total];
    for m in 0..n_modes {
        for c in 0..n_channels {
            let base = m * n_channels * n_freq + c * n_freq;
            let standardized = zscore(&marginal_buf[base..base + n_freq]);
            std_marginal_buf[base..base + n_freq].copy_from_slice(&standardized);
        }
    }

    // Z-score standardize full spectrum per channel
    let mut std_full_buf = vec![0f64; full_total];
    for c in 0..n_channels {
        let base = c * n_freq;
        let standardized = zscore(&full_buf[base..base + n_freq]);
        std_full_buf[base..base + n_freq].copy_from_slice(&standardized);
    }

    // Z-score standardize 2-D Hilbert Spectrum per channel (flatten freq×time slice)
    let mut std_hilbert_spectrum_buf = vec![0f64; hs_total];
    for c in 0..n_channels {
        let base = c * n_freq * n_timepoints;
        let standardized = zscore(&hilbert_spectrum_buf[base..base + n_freq * n_timepoints]);
        std_hilbert_spectrum_buf[base..base + n_freq * n_timepoints]
            .copy_from_slice(&standardized);
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
        std_envelope: std_envelope_buf,
        std_hilbert_spectrum: std_hilbert_spectrum_buf,
        std_marginal_spectra: std_marginal_buf,
        std_full_spectrum: std_full_buf,
    })
}

/// Z-score standardize a slice in-place: (x - mean) / std per provided window.
/// Returns a new Vec with standardized values; if std == 0, output is zero.
fn zscore(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 {
        return vec![];
    }
    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
    let std = variance.sqrt();
    if std == 0.0 {
        return vec![0.0; n];
    }
    data.iter().map(|&x| (x - mean) / std).collect()
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
fn write_hht(hht_group: &hdf5::Group, result: &HHTResult, force: bool) -> Result<()> {
    write_dataset(
        hht_group,
        "envelope",
        &result.envelope,
        &result.envelope_shape,
        None,
    )?;
    write_dataset(
        hht_group,
        "instantaneous_frequency",
        &result.inst_freq,
        &result.inst_freq_shape,
        None,
    )?;
    write_dataset(
        hht_group,
        "frequency_axis",
        &result.freq_axis,
        &[result.freq_axis.len()],
        Some(&[H5Attr::f64("fs_hz", FS), H5Attr::f64("tr_s", TR)]),
    )?;

    let marg_group = open_or_create_group(hht_group, "marginal_spectra", force)?;
    write_dataset(
        &marg_group,
        "spectra",
        &result.marginal_spectra,
        &result.marginal_spectra_shape,
        None,
    )?;

    write_dataset(
        hht_group,
        "full_spectrum",
        &result.full_spectrum,
        &result.full_spectrum_shape,
        Some(&[H5Attr::string(
            "description",
            "normalized sum of marginal spectra across all modes per channel",
        )]),
    )?;

    write_dataset(
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

/// Write z-score standardized HHT outputs to an HDF5 group.
///
/// Mirrors the structure of `write_hht` but uses standardized spectra. Envelope is
/// z-scored per (mode, channel); marginal spectra and full spectrum are z-scored per
/// channel across the frequency axis.
fn write_hht_standardized(hht_std_group: &hdf5::Group, result: &HHTResult, force: bool) -> Result<()> {
    write_dataset(
        hht_std_group,
        "envelope",
        &result.std_envelope,
        &result.envelope_shape,
        Some(&[H5Attr::string("standardization", "zscore_per_mode_channel")]),
    )?;
    write_dataset(
        hht_std_group,
        "instantaneous_frequency",
        &result.inst_freq,
        &result.inst_freq_shape,
        None,
    )?;
    write_dataset(
        hht_std_group,
        "frequency_axis",
        &result.freq_axis,
        &[result.freq_axis.len()],
        Some(&[H5Attr::f64("fs_hz", FS), H5Attr::f64("tr_s", TR)]),
    )?;

    let marg_group = open_or_create_group(hht_std_group, "marginal_spectra", force)?;
    write_dataset(
        &marg_group,
        "spectra",
        &result.std_marginal_spectra,
        &result.marginal_spectra_shape,
        Some(&[H5Attr::string("standardization", "zscore_per_mode_channel")]),
    )?;

    write_dataset(
        hht_std_group,
        "full_spectrum",
        &result.std_full_spectrum,
        &result.full_spectrum_shape,
        Some(&[H5Attr::string(
            "standardization",
            "zscore_per_channel_across_frequency",
        )]),
    )?;

    write_dataset(
        hht_std_group,
        "hilbert_spectrum",
        &result.std_hilbert_spectrum,
        &result.hilbert_spectrum_shape,
        Some(&[H5Attr::string(
            "standardization",
            "zscore_per_channel_across_freq_time",
        )]),
    )?;

    Ok(())
}

pub fn run(cfg: &HilbertHuangConfig) -> Result<()> {
    let run_start = Instant::now();

    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        bold_ts_dir = %cfg.bold_ts_dir.display(),
        force = cfg.force,
        "starting HHT pipeline (MVMD-based Hilbert-Huang Transform)"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.bold_ts_dir)?
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
    let mut error_count = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;

        let available_timeseries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter_map(|path| {
                let p = path.file_name()?.to_str()?.to_string();
                if p.contains(".h5") { Some(path) } else { None }
            })
            .collect();

        info!(
            subject = formatted_id,
            subject_idx = subject_idx,
            total_subjects = total_subjects,
            num_files = available_timeseries.len(),
            "processing subject"
        );

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
                let mvmd_group = match h5_file.group("mvmd") {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            subject = formatted_id,
                            task_name = task_name,
                            "no mvmd group found, skipping (run tcp-mvmd first)"
                        );
                        return Ok(());
                    }
                };

                let hht_group = open_or_create_group(&h5_file, "hht", false)?;
                let hht_std_group = open_or_create_group(&h5_file, "hht_standardized", false)?;

                //////////////////////////
                // Whole-band HHT       //
                //////////////////////////

                let wb_hht_done = !cfg.force
                    && hht_group.group("whole-band")
                        .map(|g| g.dataset("hilbert_spectrum").is_ok())
                        .unwrap_or(false);
                let wb_std_done = !cfg.force
                    && hht_std_group.group("whole-band")
                        .map(|g| g.dataset("hilbert_spectrum").is_ok())
                        .unwrap_or(false);

                if wb_hht_done && wb_std_done {
                    debug!(
                        subject = formatted_id,
                        task_name = task_name,
                        "whole-band HHT already computed, skipping (use --force to recompute)"
                    );
                } else {
                    match mvmd_group.group("whole-band") {
                        Err(_) => {
                            debug!(
                                subject = formatted_id,
                                task_name = task_name,
                                "no mvmd/whole-band group found, skipping whole-band HHT"
                            );
                        }
                        Ok(wb_mvmd) => {
                            let modes_ds = wb_mvmd.dataset("modes")?;
                            let modes_shape = modes_ds.shape();
                            let modes_flat: Vec<f32> = modes_ds.read_raw()?;

                            let [n_modes, n_channels, n_timepoints] = match modes_shape.as_slice() {
                                &[a, b, c] => [a, b, c],
                                _ => anyhow::bail!("unexpected modes shape {:?}", modes_shape),
                            };

                            info!(
                                subject = formatted_id,
                                task_name = task_name,
                                n_modes = n_modes,
                                n_channels = n_channels,
                                n_timepoints = n_timepoints,
                                "computing whole-band HHT"
                            );

                            let hht_start = Instant::now();
                            let result = compute_hht(&modes_flat, &modes_shape)?;
                            let hht_duration_ms = hht_start.elapsed().as_millis();

                            let write_start = Instant::now();
                            if !wb_hht_done {
                                let wb_hht_group =
                                    open_or_create_group(&hht_group, "whole-band", cfg.force)?;
                                if !cfg.force && wb_hht_group.dataset("full_spectrum").is_ok() {
                                    write_dataset(&wb_hht_group, "hilbert_spectrum", &result.hilbert_spectrum, &result.hilbert_spectrum_shape, Some(&[H5Attr::string("description", "2D Hilbert spectrum H(omega,t): energy summed over modes per channel [n_channels, n_freq, n_timepoints]")]))?;
                                } else {
                                    write_hht(&wb_hht_group, &result, cfg.force)?;
                                }
                            }
                            if !wb_std_done {
                                let wb_hht_std_group =
                                    open_or_create_group(&hht_std_group, "whole-band", cfg.force)?;
                                if !cfg.force && wb_hht_std_group.dataset("full_spectrum").is_ok() {
                                    write_dataset(&wb_hht_std_group, "hilbert_spectrum", &result.std_hilbert_spectrum, &result.hilbert_spectrum_shape, Some(&[H5Attr::string("standardization", "zscore_per_channel_across_freq_time")]))?;
                                } else {
                                    write_hht_standardized(&wb_hht_std_group, &result, cfg.force)?;
                                }
                            }
                            let write_duration_ms = write_start.elapsed().as_millis();

                            info!(
                                subject = formatted_id,
                                task_name = task_name,
                                n_modes = n_modes,
                                n_channels = n_channels,
                                n_timepoints = n_timepoints,
                                hht_duration_ms = hht_duration_ms,
                                write_duration_ms = write_duration_ms,
                                output_file = %file_path.display(),
                                "whole-band HHT complete"
                            );
                        }
                    }
                }

                //////////////////////////
                // Block-level HHT      //
                //////////////////////////

                let mvmd_blocks_group = match mvmd_group.group("blocks") {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            subject = formatted_id,
                            task_name = task_name,
                            "no mvmd/blocks group found, skipping block HHT"
                        );
                        return Ok(());
                    }
                };

                let block_names: Vec<String> = mvmd_blocks_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    return Ok(());
                }

                let hht_blocks_group = open_or_create_group(&hht_group, "blocks", false)?;
                let hht_std_blocks_group = open_or_create_group(&hht_std_group, "blocks", false)?;

                for block_name in &block_names {
                    let block_hht_done = !cfg.force
                        && hht_blocks_group.group(block_name)
                            .map(|g| g.dataset("hilbert_spectrum").is_ok())
                            .unwrap_or(false);
                    let block_std_done = !cfg.force
                        && hht_std_blocks_group.group(block_name)
                            .map(|g| g.dataset("hilbert_spectrum").is_ok())
                            .unwrap_or(false);

                    if block_hht_done && block_std_done {
                        debug!(
                            subject = formatted_id,
                            task_name = task_name,
                            block = block_name,
                            "block HHT already computed, skipping (use --force to recompute)"
                        );
                        continue;
                    }

                    let block_mvmd = mvmd_blocks_group.group(block_name)?;
                    let modes_ds = block_mvmd.dataset("modes")?;
                    let modes_shape = modes_ds.shape();
                    let modes_flat: Vec<f32> = modes_ds.read_raw()?;

                    let [n_modes, n_channels, n_timepoints] = match modes_shape.as_slice() {
                        &[a, b, c] => [a, b, c],
                        _ => {
                            error_count += 1;
                            warn!(
                                subject = formatted_id,
                                task_name = task_name,
                                block = block_name,
                                shape = ?modes_shape,
                                reason = "unexpected_modes_shape",
                                "skipping block HHT"
                            );
                            continue;
                        }
                    };

                    info!(
                        subject = formatted_id,
                        task_name = task_name,
                        block = block_name,
                        n_modes = n_modes,
                        n_channels = n_channels,
                        n_timepoints = n_timepoints,
                        "computing block HHT"
                    );

                    let hht_start = Instant::now();
                    let result = match compute_hht(&modes_flat, &modes_shape) {
                        Ok(r) => r,
                        Err(e) => {
                            error_count += 1;
                            warn!(
                                subject = formatted_id,
                                task_name = task_name,
                                block = block_name,
                                error = %e,
                                reason = "hht_failed",
                                "skipping block HHT due to error"
                            );
                            continue;
                        }
                    };
                    let hht_duration_ms = hht_start.elapsed().as_millis();

                    let write_start = Instant::now();
                    if !block_hht_done {
                        let block_hht_group =
                            open_or_create_group(&hht_blocks_group, block_name, cfg.force)?;
                        if !cfg.force && block_hht_group.dataset("full_spectrum").is_ok() {
                            write_dataset(&block_hht_group, "hilbert_spectrum", &result.hilbert_spectrum, &result.hilbert_spectrum_shape, Some(&[H5Attr::string("description", "2D Hilbert spectrum H(omega,t): energy summed over modes per channel [n_channels, n_freq, n_timepoints]")]))?;
                        } else {
                            write_hht(&block_hht_group, &result, cfg.force)?;
                        }
                    }
                    if !block_std_done {
                        let block_hht_std_group =
                            open_or_create_group(&hht_std_blocks_group, block_name, cfg.force)?;
                        if !cfg.force && block_hht_std_group.dataset("full_spectrum").is_ok() {
                            write_dataset(&block_hht_std_group, "hilbert_spectrum", &result.std_hilbert_spectrum, &result.hilbert_spectrum_shape, Some(&[H5Attr::string("standardization", "zscore_per_channel_across_freq_time")]))?;
                        } else {
                            write_hht_standardized(&block_hht_std_group, &result, cfg.force)?;
                        }
                    }
                    let write_duration_ms = write_start.elapsed().as_millis();

                    debug!(
                        subject = formatted_id,
                        task_name = task_name,
                        block = block_name,
                        n_modes = n_modes,
                        hht_duration_ms = hht_duration_ms,
                        write_duration_ms = write_duration_ms,
                        "block HHT complete"
                    );
                }

                info!(
                    subject = formatted_id,
                    task_name = task_name,
                    num_blocks = block_names.len(),
                    "finished block HHT decompositions"
                );

                Ok(())
            })();

            if let Err(e) = file_result {
                error_count += 1;
                warn!(
                    subject = formatted_id,
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
        total_subjects = total_subjects,
        error_count = error_count,
        total_duration_ms = total_duration_ms,
        "HHT pipeline complete"
    );

    Ok(())
}
