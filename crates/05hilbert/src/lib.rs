use anyhow::{Context, Result};
use ndarray::{Array3, s};
use scirs2_signal::hilbert::hilbert;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};
use utils::bids_filename::{BidsFilename, filter_directory_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands;
use utils::hdf5_io::{H5Attr, ensure_path, open_or_create_group, path_exists, write_dataset_old};
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
    /// Instantaneous amplitude (envelope) [n_modes, n_channels, n_timepoints]
    envelope: Vec<f64>,
    envelope_shape: [usize; 3],
    /// Instantaneous frequency (Hz) [n_modes, n_channels, n_timepoints]
    inst_freq: Vec<f64>,
    inst_freq_shape: [usize; 3],
    /// Frequency axis (Hz) [Size: 224], mapped from f_max to f_min
    freq_axis: Vec<f64>,
    /// Marginal Hilbert Spectrum [n_modes, n_channels, n_freq_bins]
    marginal_spectra: Vec<f64>,
    marginal_spectra_shape: [usize; 3],
    /// Normalized full spectrum [n_channels, n_freq_bins]
    full_spectrum: Vec<f64>,
    full_spectrum_shape: [usize; 2],
    /// 2-D Hilbert Spectrum H(f,t) [n_channels, n_freq_bins, n_timepoints]
    hilbert_spectrum: Vec<f64>,
    hilbert_spectrum_shape: [usize; 3],
}

use num_complex::Complex64;

fn compute_instantaneous_angular_freq(analytic: &[Complex64], sampling_rate: f64) -> Vec<f64> {
    let n = analytic.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut omega = vec![0.0; n];
    let dt = 1.0 / sampling_rate;

    for i in 1..n {
        // The phase difference Δθ can be found by taking the argument of
        // the product of the current sample and the conjugate of the previous one.
        // Formula: Δθ = arg(z[t] * conj(z[t-1]))
        let phase_diff = (analytic[i] * analytic[i - 1].conj()).arg();

        // ω = Δθ / Δt gives rad/s
        omega[i] = phase_diff / dt;
    }

    // Handle the first element by back-filling (constant extension)
    // so the vector length matches the input signal.
    omega[0] = omega[1];

    omega
}

/// Compute HHT from a modes array with shape [n_modes, n_channels, n_timepoints].
///
/// Modes are read from HDF5 as flat row-major with that shape ordering.
fn compute_hht(cfg: &AppConfig, modes_flat: &[f32], shape: &[usize]) -> Result<HHTResult> {
    let n_modes = shape[0];
    let n_channels = shape[1];
    let n_timepoints = shape[2];
    let sampling_rate = cfg.task_sampling_rate;

    // 1. Angular boundaries for instantaneous-frequency binning (rad/s)
    let f_min_hz = frequency_bands::f_min();
    let f_max_hz = frequency_bands::f_max();
    let w_min = 2.0 * std::f64::consts::PI * f_min_hz;
    let w_max = 2.0 * std::f64::consts::PI * f_max_hz;
    debug!(
        f_min_hz = f_min_hz,
        f_max_hz = f_max_hz,
        w_min = w_min,
        w_max = w_max,
        "Using angular frequencies"
    );

    // Generate axis in angular space first
    let angular_freq_axis: Vec<f64> = (0..TARGET_N_FREQ)
        .map(|i| w_max * (w_min / w_max).powf(i as f64 / (TARGET_N_FREQ - 1) as f64))
        .collect();

    let modes = Array3::from_shape_vec(
        (n_modes, n_channels, n_timepoints),
        modes_flat.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    )?;

    let mut envelope_buf = vec![0f64; n_modes * n_channels * n_timepoints];
    let mut inst_freq_buf = vec![0f64; n_modes * n_channels * n_timepoints]; // Will hold rad/s, then converted to Hz
    let mut marginal_buf = vec![0f64; n_modes * n_channels * TARGET_N_FREQ];
    let mut full_buf = vec![0f64; n_channels * TARGET_N_FREQ];
    let mut hilbert_spectrum_buf = vec![0f64; n_channels * TARGET_N_FREQ * n_timepoints];

    let log_w_max = w_max.ln();
    let log_total_w_ratio = (w_min / w_max).ln();

    for m in 0..n_modes {
        for c in 0..n_channels {
            let channel_signal: Vec<f64> = modes.slice(s![m, c, ..]).to_vec();
            let analytic = hilbert(&channel_signal)
                .map_err(|e| anyhow::anyhow!("hilbert failed mode={} ch={}: {}", m, c, e))?;

            let amp: Vec<f64> = analytic.iter().map(|z| z.norm()).collect();
            let i_omega = compute_instantaneous_angular_freq(&analytic, sampling_rate);

            let base = m * n_channels * n_timepoints + c * n_timepoints;
            envelope_buf[base..base + n_timepoints].copy_from_slice(&amp);
            inst_freq_buf[base..base + n_timepoints].copy_from_slice(&i_omega);

            let marg_base = m * n_channels * TARGET_N_FREQ + c * TARGET_N_FREQ;
            let hs_base = c * TARGET_N_FREQ * n_timepoints;

            for t in 0..n_timepoints {
                let w = i_omega[t];
                if w < w_min || w > w_max {
                    warn!(
                        mode_idx = m,
                        angular_freq = w,
                        f_min_hz = f_min_hz,
                        f_max_hz = f_max_hz,
                        "Skipping mode. Instantaneous frequency out of range"
                    );
                    continue;
                }

                let log_ratio = (w.ln() - log_w_max) / log_total_w_ratio;
                let bin = ((log_ratio * (TARGET_N_FREQ - 1) as f64).round() as usize)
                    .min(TARGET_N_FREQ - 1);

                let energy = amp[t] * amp[t];
                marginal_buf[marg_base + bin] += energy;
                hilbert_spectrum_buf[hs_base + bin * n_timepoints + t] += energy;
            }

            let full_base = c * TARGET_N_FREQ;
            for b in 0..TARGET_N_FREQ {
                full_buf[full_base + b] += marginal_buf[marg_base + b];
            }
        }
    }

    // --- Post-Processing: Conversion to Hz ---

    // Convert Axis: rad/s -> Hz
    let freq_axis_hz: Vec<f64> = angular_freq_axis
        .iter()
        .map(|w| w / (2.0 * std::f64::consts::PI))
        .collect();

    // Convert Inst Freq Buffer: rad/s -> Hz
    for val in inst_freq_buf.iter_mut() {
        *val /= 2.0 * std::f64::consts::PI;
    }

    // Normalization (keeping as previously designed)
    for c in 0..n_channels {
        let base = c * TARGET_N_FREQ;
        let sum: f64 = full_buf[base..base + TARGET_N_FREQ].iter().sum();
        if sum > 0.0 {
            for b in 0..TARGET_N_FREQ {
                full_buf[base + b] /= sum;
            }
        }
    }

    Ok(HHTResult {
        envelope: envelope_buf,
        envelope_shape: [n_modes, n_channels, n_timepoints],
        inst_freq: inst_freq_buf,
        inst_freq_shape: [n_modes, n_channels, n_timepoints],
        freq_axis: freq_axis_hz,
        marginal_spectra: marginal_buf,
        marginal_spectra_shape: [n_modes, n_channels, TARGET_N_FREQ],
        full_spectrum: full_buf,
        full_spectrum_shape: [n_channels, TARGET_N_FREQ],
        hilbert_spectrum: hilbert_spectrum_buf,
        hilbert_spectrum_shape: [n_channels, TARGET_N_FREQ, n_timepoints],
    })
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
            "2D Hilbert spectrum H(f,t): energy summed over modes per channel [n_channels, n_freq, n_timepoints]",
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
        check_roi_fingerprint(&mvmd_sub, &expected)?;
    }

    let hht_done = !cfg.force
        && hht_parent
            .group(name)
            .map(|g| g.dataset("hilbert_spectrum").is_ok())
            .unwrap_or(false);

    if hht_done {
        if is_roi {
            let existing = hht_parent.group(name)?;
            // We only return early if the fingerprint is unchanged (is_ok).
            // If it fails (is_err), we skip this block and continue execution.
            if check_roi_fingerprint(&existing, &cfg.roi_selection.fingerprint()).is_ok() {
                debug!(
                    task_name = task_name,
                    group = name,
                    "HHT already computed and ROI matches, skipping"
                );
                return Ok(());
            }
        } else {
            // If it's not a ROI task but hht_done is true, return early as before
            debug!(
                task_name = task_name,
                group = name,
                "HHT already computed, skipping"
            );
            return Ok(());
        }
    }

    let modes_ds = mvmd_sub.dataset("modes")?;
    let modes_shape = modes_ds.shape();
    let modes_flat: Vec<f32> = modes_ds.read_raw()?;
    let center_freqs: Vec<f32> = modes_ds.attr("center_frequencies")?.read_raw()?;

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
        n_center_frequencies = center_freqs.len(),
        "computing HHT"
    );

    let hht_start = Instant::now();
    let result = compute_hht(cfg, &modes_flat, &modes_shape)?;
    let hht_duration_ms = hht_start.elapsed().as_millis();

    let write_start = Instant::now();
    let dest = ensure_path(hht_parent, name, cfg.force)?;
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

/// Iterate trial-type subgroups (e.g. "face") under `mvmd_parent/name`, then `block_*`
/// subgroups within each, and compute HHT for each block, mirroring outputs under
/// `hht_parent/name/{trial_type}/{block_name}`.
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

    // MVMD writes blocks_std/{trial_type}/{block_name}, not blocks_std/{block_name}.
    // Collect trial-type subgroups (anything that is not itself a block_* entry).
    let trial_types: Vec<String> = mvmd_blocks
        .member_names()?
        .into_iter()
        .filter(|n| !n.starts_with("block_"))
        .collect();

    if trial_types.is_empty() {
        debug!(
            task_name = task_name,
            group = name,
            "no trial-type subgroups found under blocks parent, skipping"
        );
        return Ok(());
    }

    let hht_blocks = open_or_create_group(hht_parent, name, false)?;
    let mut total_blocks = 0usize;

    for trial_type in &trial_types {
        let mvmd_trial = match mvmd_blocks.group(trial_type) {
            Ok(g) => g,
            Err(_) => {
                debug!(
                    task_name = task_name,
                    group = name,
                    trial_type = trial_type,
                    "trial-type subgroup missing, skipping"
                );
                continue;
            }
        };

        let block_names: Vec<String> = mvmd_trial
            .member_names()?
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect();

        if block_names.is_empty() {
            debug!(
                task_name = task_name,
                group = name,
                trial_type = trial_type,
                "no blocks found under trial type, skipping"
            );
            continue;
        }

        let hht_trial = open_or_create_group(&hht_blocks, trial_type, false)?;

        for block_name in &block_names {
            if let Err(e) = process_mvmd_modes_group(
                cfg,
                &mvmd_trial,
                &hht_trial,
                block_name,
                task_name,
                is_roi,
            ) {
                *error_count += 1;
                warn!(
                    task_name = task_name,
                    group = name,
                    trial_type = trial_type,
                    block = block_name,
                    error = %e,
                    "skipping block HHT due to error"
                );
            }
        }

        total_blocks += block_names.len();

        info!(
            task_name = task_name,
            group = name,
            trial_type = trial_type,
            num_blocks = block_names.len(),
            "finished block HHT decompositions for trial type"
        );
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

        let available_timeseries: Vec<BidsFilename> = filter_directory_bids_files(dir, |bids| {
            bids.get("task") == Some("hammerAP") || bids.get("task") == Some("restAP")
        })
        .expect("Failed to read the directory");

        info!(num_files = available_timeseries.len(), "processing subject");

        for file in &available_timeseries {
            let task_name = file.get("task").unwrap_or("unknown");

            // Ensure output file exists
            let path = file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;

            let h5_file = hdf5::File::open_rw(&path)?;

            // MVMD group must exist — this step depends on step 04
            let has_mvmd_results = path_exists(&h5_file, "04mvmd");
            if !has_mvmd_results {
                anyhow::bail!(
                    "Missing 04mvmd results in file {file}. Bailing HHT for subject {subject}",
                    subject = formatted_id,
                    file = file,
                )
            }

            let mvmd_crate_results = h5_file.group("04mvmd")?;
            let hht_group = ensure_path(&h5_file, "05hht", cfg.force)?;

            match task_name {
                "restAP" => {
                    process_mvmd_modes_group(
                        cfg,
                        &mvmd_crate_results,
                        &hht_group,
                        "full_run_std",
                        task_name,
                        false,
                    )?;
                    if !cfg.roi_selection.is_empty() {
                        process_mvmd_modes_group(
                            cfg,
                            &mvmd_crate_results,
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
                        &mvmd_crate_results,
                        &hht_group,
                        "blocks_std",
                        task_name,
                        &mut error_count,
                        false,
                    )?;
                    if !cfg.roi_selection.is_empty() {
                        process_blocks_parent(
                            cfg,
                            &mvmd_crate_results,
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
