mod algorithms;
mod hilbert;

use anyhow::{Context, Result};
use ndarray::{Array2, Axis};
use polars::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex};
use utils::atlas::BrainAtlas;
use utils::bids_filename::{BidsFilename, filter_directory_bids_files, sort_bids_vec};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::{AppConfig, FrequencyInitConfig};
use utils::hdf5_io::{
    H5Attr, ensure_path, open_or_create_group, path_exists, prepare_dataset, write_attrs,
    write_dataset_old,
};
use utils::roi_migration::{check_roi_fingerprint, propagate_roi_attrs};

use crate::algorithms::admm::ADMMConfig;
use crate::algorithms::mvmd::{MVMD, MvmdVariant};
use crate::hilbert::{HHTResult, compute_hht};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

const HHT_CRATE_GROUP: &str = "04hht";
const FULL_RUN_GROUP: &str = "full_run_std";
const FULL_RUN_GROUP_ROI: &str = "full_run_std_roi";
const ALL_BLOCKS_GROUP: &str = "blocks_std";
const ALL_BLOCKS_GROUP_ROI: &str = "blocks_std_roi";

// NA-MVMD variant writes to separate groups so classic and NA results coexist on disk.
const FULL_RUN_GROUP_NA: &str = "full_run_std_na";
const FULL_RUN_GROUP_ROI_NA: &str = "full_run_std_roi_na";
const ALL_BLOCKS_GROUP_NA: &str = "blocks_std_na";
const ALL_BLOCKS_GROUP_ROI_NA: &str = "blocks_std_roi_na";

/// One-sided power spectrum of a single channel (normalised by N). Returns `n/2+1` values.
fn compute_psd_channel(signal: &[f32]) -> Vec<f64> {
    let n = signal.len();
    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n);
    let mut buf: Vec<Complex<f32>> = signal.iter().map(|&x| Complex::new(x, 0.0)).collect();
    fft.process(&mut buf);
    buf[..n / 2 + 1]
        .iter()
        .map(|c| (c.norm_sqr() as f64) / (n as f64))
        .collect()
}

/// Frequency axis for a one-sided FFT of length `n` at sample rate `fs_hz`.
fn freq_axis(n: usize, fs: f64) -> Vec<f64> {
    (0..n / 2 + 1).map(|k| k as f64 * fs / n as f64).collect()
}

/// Mean and std of PSD across channels (columns of `signal` are channels).
/// Returns `(freqs_hz, mean_power_db, std_power_db)`.
fn psd_mean_std(signal: &Array2<f32>, fs: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let n_timepoints = signal.shape()[1];
    let freqs = freq_axis(n_timepoints, fs);
    let n_freq = freqs.len();

    let mut acc = vec![0f64; n_freq];
    let mut acc2 = vec![0f64; n_freq];
    let n_ch = signal.shape()[0] as f64;

    for ch in signal.outer_iter() {
        let psd = compute_psd_channel(ch.as_slice().expect("contiguous"));
        for (i, p) in psd.iter().enumerate() {
            acc[i] += p;
            acc2[i] += p * p;
        }
    }

    let mean: Vec<f64> = acc.iter().map(|s| 10.0 * (s / n_ch).log10()).collect();
    let std: Vec<f64> = acc2
        .iter()
        .zip(acc.iter())
        .map(|(s2, s)| {
            let var = s2 / n_ch - (s / n_ch).powi(2);
            let std_linear = var.max(0.0).sqrt();
            // approximate dB std via delta-method: std_db ≈ 10/ln(10) * std_linear/mean_linear
            let mean_linear = s / n_ch;
            if mean_linear > 0.0 {
                10.0 / std::f64::consts::LN_10 * std_linear / mean_linear
            } else {
                0.0
            }
        })
        .collect();

    (freqs, mean, std)
}

fn write_mvmd_algorithm_attrs_if_missing(
    loc: &hdf5::Location,
    alpha: f64,
    sampling_rate: f64,
    num_modes: usize,
    admm_config: &ADMMConfig,
) -> Result<()> {
    if loc.attr("algorithm").is_err() {
        write_attrs(loc, &[H5Attr::string("algorithm", "mvmd")])?;
    }
    if loc.attr("alpha").is_err() {
        write_attrs(loc, &[H5Attr::f64("alpha", alpha)])?;
    }
    if loc.attr("sampling_rate").is_err() {
        write_attrs(loc, &[H5Attr::f64("sampling_rate", sampling_rate)])?;
    }
    if loc.attr("num_modes").is_err() {
        write_attrs(loc, &[H5Attr::u32("num_modes", num_modes as u32)])?;
    }
    if loc.attr("admm_tolerance").is_err() {
        write_attrs(loc, &[H5Attr::f64("admm_tolerance", admm_config.tolerance)])?;
    }
    if loc.attr("admm_tau").is_err() {
        write_attrs(loc, &[H5Attr::f64("admm_tau", admm_config.tau)])?;
    }
    if loc.attr("admm_max_iterations").is_err() {
        write_attrs(
            loc,
            &[H5Attr::u32(
                "admm_max_iterations",
                admm_config.max_iterations,
            )],
        )?;
    }
    Ok(())
}

fn write_center_frequencies_attr(
    dataset: &hdf5::Dataset,
    center_frequencies: &[f64],
) -> Result<()> {
    if let Ok(attr) = dataset.attr("center_frequencies") {
        attr.write_raw(center_frequencies)?;
    } else {
        dataset
            .new_attr::<f64>()
            .shape([center_frequencies.len()])
            .create("center_frequencies")?
            .write_raw(center_frequencies)?;
    }
    Ok(())
}

fn sync_center_frequencies_attr_from_group(group: &hdf5::Group) -> Result<()> {
    let center_frequencies = group
        .dataset("center_frequencies")
        .context("failed to open dataset center_frequencies for attribute sync")?
        .read_raw::<f64>()?;
    let modes_dataset = group
        .dataset("modes")
        .context("failed to open dataset modes for attribute sync")?;
    write_center_frequencies_attr(&modes_dataset, &center_frequencies)
}

/// Check whether envelope normalization attrs match current config.
fn envelope_norm_unchanged(group: &hdf5::Group, cfg: &AppConfig) -> bool {
    let expected_norm = cfg.hht.hht_envelope_normalize;
    let expected_log = cfg.hht.hht_log_amp;
    group
        .dataset("envelope")
        .ok()
        .map(|ds| {
            let stored_norm = ds
                .attr("normalized")
                .ok()
                .and_then(|a| a.read_scalar::<u32>().ok())
                .map(|v| v != 0)
                .unwrap_or(false);
            let stored_log = ds
                .attr("log_amp_applied")
                .ok()
                .and_then(|a| a.read_scalar::<u32>().ok())
                .map(|v| v != 0)
                .unwrap_or(false);
            stored_norm == expected_norm && stored_log == expected_log
        })
        .unwrap_or(true)
}

/// Check whether the required always-write datasets exist in a group.
fn required_datasets_exist(group: &hdf5::Group) -> bool {
    group.dataset("modes").is_ok()
        && group.dataset("envelope").is_ok()
        && group.dataset("instantaneous_frequency").is_ok()
}

/// Write MVMD modes + HHT results to an HDF5 group under `/04hht/`.
///
/// Writes: modes, center_frequencies attr, envelope, instantaneous_frequency.
#[allow(clippy::too_many_arguments)]
fn write_hht_group(
    cfg: &AppConfig,
    dest: &hdf5::Group,
    modes: &ndarray::Array3<f64>,
    center_frequencies: &[f64],
    num_iterations: u32,
    converged: bool,
    hht: &HHTResult,
    alpha: f64,
    admm_config: &ADMMConfig,
) -> Result<()> {
    write_attrs(dest, &[
        H5Attr::u32("num_iterations", num_iterations),
        H5Attr::u32("converged", converged as u32),
    ])?;

    // Always-write: modes + center_frequencies attr
    let m_shape = modes.shape();
    let modes_ds = prepare_dataset::<f64>(dest, "modes", &[m_shape[0], m_shape[1], m_shape[2]])?;
    modes_ds.write_raw(modes.as_slice().unwrap())?;
    write_center_frequencies_attr(&modes_ds, center_frequencies)?;

    // Always-write: envelope
    write_dataset_old(
        dest,
        "envelope",
        &hht.envelope,
        &hht.envelope_shape,
        Some(&[
            H5Attr::u32("normalized", cfg.hht.hht_envelope_normalize as u32),
            H5Attr::string(
                "normalize_method",
                if cfg.hht.hht_envelope_normalize {
                    "per_channel_max_divide"
                } else {
                    "none"
                },
            ),
            H5Attr::u32("log_amp_applied", cfg.hht.hht_log_amp as u32),
        ]),
    )?;

    // Always-write: instantaneous_frequency
    write_dataset_old(
        dest,
        "instantaneous_frequency",
        &hht.inst_freq,
        &hht.inst_freq_shape,
        None,
    )?;

    // Always-write: center_frequencies standalone dataset (for sync_center_frequencies_attr_from_group)
    let cf_shape = &[center_frequencies.len()];
    let cf_ds = prepare_dataset::<f64>(dest, "center_frequencies", cf_shape)?;
    cf_ds.write_raw(center_frequencies)?;

    write_mvmd_algorithm_attrs_if_missing(
        dest,
        alpha,
        cfg.task_sampling_rate,
        m_shape[0],
        admm_config,
    )?;

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_signal_group(
    cfg: &AppConfig,
    h5_file: &hdf5::File,
    hht_parent: &hdf5::Group,
    name: &str,
    task_name: &str,
    signal: Array2<f32>,
    channel_labels: Vec<String>,
    is_roi: bool,
    roi_indices: &[u32],
    roi_labels: &[String],
    roi_matched_regions: &[String],
    roi_selection_name: &str,
    roi_selection_fingerprint: &str,
    alpha: f64,
    num_modes: usize,
    admm_config: &ADMMConfig,
    mvmd_variant: &MvmdVariant,
    // BIDS-ish filename stem for diagnostic TSVs (no extension). Pass None to skip.
    diag_stem: Option<&str>,
) -> Result<()> {
    let existing = hht_parent.group(name).ok();

    let roi_fingerprint_ok = if is_roi {
        existing
            .as_ref()
            .map(|g| check_roi_fingerprint(g, roi_selection_fingerprint).is_ok())
            .unwrap_or(true)
    } else {
        true
    };

    let norm_ok = existing
        .as_ref()
        .map(|g| envelope_norm_unchanged(g, cfg))
        .unwrap_or(true);

    let data_ok = existing
        .as_ref()
        .map(required_datasets_exist)
        .unwrap_or(false);

    if !cfg.force && data_ok && roi_fingerprint_ok && norm_ok {
        // Still sync center_frequencies attr in case it was missing on a legacy file
        if let Some(g) = &existing {
            let _ = sync_center_frequencies_attr_from_group(g);
        }
        debug!(task_name, group = name, "HHT already computed, skipping");
        return Ok(());
    }

    let sampling_rate = cfg.task_sampling_rate;

    let columns: Vec<Column> = signal
        .outer_iter()
        .zip(channel_labels.iter())
        .map(|(row, label)| {
            let slice = row.as_slice().expect("ndarray must be contiguous");
            Series::new(label.as_str().into(), slice).into()
        })
        .collect();

    let df = DataFrame::new(columns)?;
    let freq_init = match cfg.hht.frequency_init {
        FrequencyInitConfig::Zero => crate::algorithms::mvmd::FrequencyInit::Zero,
        FrequencyInitConfig::Linear => crate::algorithms::mvmd::FrequencyInit::Linear,
        FrequencyInitConfig::Exponential => crate::algorithms::mvmd::FrequencyInit::Exponential,
    };
    let mvmd = MVMD::from_dataframe(&df, alpha, sampling_rate)?
        .with_admm_config(admm_config.clone())
        .with_variant(mvmd_variant.clone())
        .with_init(freq_init);

    let mvmd_start = Instant::now();
    let decomp = mvmd.decompose(num_modes);
    let mvmd_ms = mvmd_start.elapsed().as_millis();
    let converged = decomp.num_iterations < admm_config.max_iterations;

    info!(
        task_name,
        group = name,
        n_modes = decomp.modes.shape()[0],
        n_channels = decomp.modes.shape()[1],
        n_timepoints = decomp.modes.shape()[2],
        mvmd_ms,
        converged,
        "MVMD decomposition complete, computing HHT"
    );

    let hht_start = Instant::now();
    let hht_result = compute_hht(cfg, &decomp.modes)?;
    let hht_ms = hht_start.elapsed().as_millis();

    let force_write = cfg.force || !roi_fingerprint_ok || !norm_ok;
    let dest = ensure_path(hht_parent, name, force_write)?;

    write_hht_group(
        cfg,
        &dest,
        &decomp.modes,
        decomp.center_frequencies.as_slice().unwrap(),
        decomp.num_iterations,
        converged,
        &hht_result,
        alpha,
        admm_config,
    )?;

    if is_roi {
        write_attrs(
            &dest,
            &[
                H5Attr::u32("n_rois", roi_indices.len() as u32),
                H5Attr::string("roi_labels", roi_labels.join(",")),
                H5Attr::string("roi_matched_regions", roi_matched_regions.join(",")),
                H5Attr::string("roi_selection_name", roi_selection_name),
                H5Attr::string("roi_selection_fingerprint", roi_selection_fingerprint),
                H5Attr::string("channels", channel_labels.join(",")),
            ],
        )?;
        let roi_indices_u32: Vec<u32> = roi_indices.to_vec();
        let bi_ds = prepare_dataset::<u32>(&dest, "roi_indices", &[roi_indices_u32.len()])?;
        bi_ds.write_raw(&roi_indices_u32)?;
        propagate_roi_attrs(h5_file, &dest)?;
    } else {
        write_attrs(
            &dest,
            &[H5Attr::string("channels", channel_labels.join(","))],
        )?;
    }

    // Diagnostic TSVs: mode summary + PSD (original signal and per-mode).
    if cfg.dump_intermediates
        && let Some(stem) = diag_stem
    {
        // --- mode summary ---
        let n_modes = decomp.center_frequencies.len();
        let mode_indices: Vec<i64> = (0..n_modes as i64).collect();
        let center_freqs: Vec<f64> = decomp.center_frequencies.to_vec();
        let iterations: Vec<u32> = vec![decomp.num_iterations; n_modes];
        let alphas: Vec<f64> = vec![alpha; n_modes];
        let taus: Vec<f64> = vec![admm_config.tau; n_modes];

        let modes_df = polars::prelude::DataFrame::new(vec![
            polars::prelude::Column::new("mode_idx".into(), mode_indices),
            polars::prelude::Column::new("center_frequency_hz".into(), center_freqs),
            polars::prelude::Column::new("admm_iterations".into(), iterations),
            polars::prelude::Column::new("alpha".into(), alphas),
            polars::prelude::Column::new("admm_tau".into(), taus),
        ])?;
        let filename = format!("{stem}_desc-modes_summary.tsv");
        if let Err(e) = utils::intermediates::dump_tsv(cfg, "04hht", &filename, &modes_df) {
            warn!(stem, error = %e, "failed to write mode summary TSV");
        }

        // --- PSD of original signal ---
        let (freqs, mean_db, std_db) = psd_mean_std(&signal, sampling_rate);
        let n_freq = freqs.len();
        let orig_psd_df = polars::prelude::DataFrame::new(vec![
            polars::prelude::Column::new("freq_hz".into(), freqs.clone()),
            polars::prelude::Column::new("mean_power_db".into(), mean_db),
            polars::prelude::Column::new("std_power_db".into(), std_db),
        ])?;
        let filename = format!("{stem}_desc-psd_orig.tsv");
        if let Err(e) = utils::intermediates::dump_tsv(cfg, "04hht", &filename, &orig_psd_df) {
            warn!(stem, error = %e, "failed to write orig PSD TSV");
        }

        // --- PSD per MVMD mode (long format) ---
        let mut all_freqs: Vec<f64> = Vec::with_capacity(n_modes * n_freq);
        let mut all_mode_idx: Vec<i64> = Vec::with_capacity(n_modes * n_freq);
        let mut all_mean_db: Vec<f64> = Vec::with_capacity(n_modes * n_freq);
        let mut all_std_db: Vec<f64> = Vec::with_capacity(n_modes * n_freq);

        for m in 0..n_modes {
            let mode_view = decomp.modes.index_axis(Axis(0), m);
            let mode_f32: ndarray::Array2<f32> = mode_view.mapv(|x| x as f32);
            let (mf, mm, ms) = psd_mean_std(&mode_f32, sampling_rate);
            all_freqs.extend_from_slice(&mf);
            all_mode_idx.extend(std::iter::repeat_n(m as i64, n_freq));
            all_mean_db.extend_from_slice(&mm);
            all_std_db.extend_from_slice(&ms);
        }

        let modes_psd_df = polars::prelude::DataFrame::new(vec![
            polars::prelude::Column::new("freq_hz".into(), all_freqs),
            polars::prelude::Column::new("mode_idx".into(), all_mode_idx),
            polars::prelude::Column::new("mean_power_db".into(), all_mean_db),
            polars::prelude::Column::new("std_power_db".into(), all_std_db),
        ])?;
        let filename = format!("{stem}_desc-psd_modes.tsv");
        if let Err(e) = utils::intermediates::dump_tsv(cfg, "04hht", &filename, &modes_psd_df) {
            warn!(stem, error = %e, "failed to write modes PSD TSV");
        }
    }

    info!(
        task_name,
        group = name,
        mvmd_ms,
        hht_ms,
        "HHT group complete"
    );

    Ok(())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    let alpha = cfg.hht.alpha;
    let num_modes = cfg.hht.num_modes;
    let sampling_rate = cfg.task_sampling_rate;
    let admm_config = ADMMConfig {
        tolerance: cfg.hht.admm_tolerance,
        tau: cfg.hht.admm_tau,
        max_iterations: cfg.hht.admm_max_iterations as u32,
    };
    let mvmd_variant = if cfg.hht.na_mvmd {
        MvmdVariant::NoiseAssisted {
            noise_channels: cfg.hht.noise_channels,
            noise_std_ratio: cfg.hht.noise_std_ratio,
            seed: cfg.hht.noise_seed,
        }
    } else {
        MvmdVariant::Classic
    };
    // Resolve HDF5 group names based on algorithm variant so both results coexist.
    let full_run_group = if cfg.hht.na_mvmd {
        FULL_RUN_GROUP_NA
    } else {
        FULL_RUN_GROUP
    };
    let full_run_group_roi = if cfg.hht.na_mvmd {
        FULL_RUN_GROUP_ROI_NA
    } else {
        FULL_RUN_GROUP_ROI
    };
    let all_blocks_group = if cfg.hht.na_mvmd {
        ALL_BLOCKS_GROUP_NA
    } else {
        ALL_BLOCKS_GROUP
    };
    let all_blocks_group_roi = if cfg.hht.na_mvmd {
        ALL_BLOCKS_GROUP_ROI_NA
    } else {
        ALL_BLOCKS_GROUP_ROI
    };

    let brain_atlas =
        BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let spec = &cfg.roi_selection;
    let roi_subset_enabled = spec.stratified_decomposition && !spec.is_empty();
    let selected = brain_atlas.selected_rois(spec);
    let roi_row_indices: Vec<usize> = selected.iter().map(|r| r.row_index).collect();
    let roi_indices_u32: Vec<u32> = roi_row_indices.iter().map(|i| *i as u32).collect();
    let roi_labels: Vec<String> = selected.iter().map(|r| r.label.clone()).collect();
    let roi_matched_regions: Vec<String> =
        selected.iter().map(|r| r.matched_region.clone()).collect();
    let roi_selection_name = spec.name.clone();
    let roi_selection_fingerprint = spec.fingerprint();

    if roi_subset_enabled && roi_row_indices.is_empty() {
        anyhow::bail!(
            "ROI selection '{}' matched no atlas rows — check LUTs ({}, {}) and config [roi_selection]",
            spec.name,
            cfg.cortical_atlas_lut.display(),
            cfg.subcortical_atlas_lut.display()
        );
    }

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        num_modes,
        dump_intermediates = cfg.dump_intermediates,
        "starting HHT pipeline (MVMD + Hilbert-Huang Transform)"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.consolidated_data_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
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
    let mut error_count = 0usize;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;

        let _span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();

        let mut resting_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |b| b.get("task") == Some("restAP"))
                .expect("Failed to read directory");
        sort_bids_vec(&mut resting_ts, &["run"], |key, a, b| match key {
            "run" => match (a.parse::<u32>(), b.parse::<u32>()) {
                (Ok(na), Ok(nb)) => na.cmp(&nb),
                _ => a.cmp(b),
            },
            _ => a.cmp(b),
        });

        let mut task_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |b| b.get("task") == Some("hammerAP"))
                .expect("Failed to read directory");
        sort_bids_vec(&mut task_ts, &["run"], |key, a, b| match key {
            "run" => match (a.parse::<u32>(), b.parse::<u32>()) {
                (Ok(na), Ok(nb)) => na.cmp(&nb),
                _ => a.cmp(b),
            },
            _ => a.cmp(b),
        });

        // --- Resting-state (full-run) ---
        for rs_file in &resting_ts {
            let task_name = rs_file.get("task").unwrap_or("unknown");
            let path = rs_file
                .try_to_path_buf()
                .context("BidsFilename has no path")?;
            let h5_file = hdf5::File::open_rw(&path)?;
            let hht_group = ensure_path(&h5_file, HHT_CRATE_GROUP, cfg.force)?;
            write_mvmd_algorithm_attrs_if_missing(
                &hht_group,
                alpha,
                sampling_rate,
                num_modes,
                &admm_config,
            )?;

            let parc_group = h5_file
                .group("01fmri_parcellation")
                .context("missing /01fmri_parcellation")?;
            let data: Array2<f32> = parc_group
                .dataset("full_run_std")
                .context("missing /01fmri_parcellation/full_run_std")?
                .read_2d()?;

            let n_channels = data.shape()[0];
            let ch_labels: Vec<String> = (0..n_channels).map(|c| format!("ch_{}", c)).collect();

            if let Err(e) = process_signal_group(
                cfg,
                &h5_file,
                &hht_group,
                full_run_group,
                task_name,
                data.clone(),
                ch_labels,
                false,
                &[],
                &[],
                &[],
                "",
                "",
                alpha,
                num_modes,
                &admm_config,
                &mvmd_variant,
                Some(&format!(
                    "{}_task-{}_run-{}_group-full",
                    formatted_id,
                    rs_file.get("task").unwrap_or("unknown"),
                    rs_file.get("run").unwrap_or("01")
                )),
            ) {
                error_count += 1;
                warn!(task_name, error = %e, "full-run HHT failed");
            }

            if roi_subset_enabled {
                let roi_data = data.select(Axis(0), &roi_row_indices);
                if let Err(e) = process_signal_group(
                    cfg,
                    &h5_file,
                    &hht_group,
                    full_run_group_roi,
                    task_name,
                    roi_data,
                    roi_labels.clone(),
                    true,
                    &roi_indices_u32,
                    &roi_labels,
                    &roi_matched_regions,
                    &roi_selection_name,
                    &roi_selection_fingerprint,
                    alpha,
                    num_modes,
                    &admm_config,
                    &mvmd_variant,
                    Some(&format!(
                        "{}_task-{}_run-{}_group-fullroi",
                        formatted_id,
                        rs_file.get("task").unwrap_or("unknown"),
                        rs_file.get("run").unwrap_or("01")
                    )),
                ) {
                    error_count += 1;
                    warn!(task_name, error = %e, "full-run ROI HHT failed");
                }
            }
        }

        // --- Task (block-wise) ---
        for task_file in &task_ts {
            let task_name = task_file.get("task").unwrap_or("unknown");
            let path = task_file
                .try_to_path_buf()
                .context("BidsFilename has no path")?;
            let h5_file = hdf5::File::open_rw(&path)?;
            let hht_group = ensure_path(&h5_file, HHT_CRATE_GROUP, cfg.force)?;
            write_mvmd_algorithm_attrs_if_missing(
                &hht_group,
                alpha,
                sampling_rate,
                num_modes,
                &admm_config,
            )?;

            let src_blocks = h5_file
                .group("02fmri_segment_trials")?
                .group("blocks_std")
                .context("missing /02fmri_segment_trials/blocks_std")?;

            let trial_types: Vec<String> = src_blocks
                .member_names()
                .unwrap_or_default()
                .into_iter()
                .filter(|n| !n.starts_with("block_"))
                .collect();
            info!(
                task_name,
                ?trial_types,
                "detected trial types in blocks_std"
            );
            for trial_type in &trial_types {
                if !path_exists(&src_blocks, trial_type.as_str()) {
                    warn!(
                        task_name,
                        trial_type, "trial type group not found, skipping"
                    );
                    continue;
                }

                let trial_group = src_blocks.group(trial_type)?;
                let block_names: Vec<String> = trial_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    continue;
                }

                let hht_blocks = open_or_create_group(&hht_group, all_blocks_group, false)?;
                let hht_trial = open_or_create_group(&hht_blocks, trial_type, false)?;

                for block_name in &block_names {
                    let input_ds = trial_group.dataset(block_name).context(format!(
                        "failed to open /02fmri_segment_trials/blocks_std/{trial_type}/{block_name}"
                    ))?;
                    let block_signal: Array2<f32> = input_ds.read_2d()?;
                    let n_ch = block_signal.shape()[0];
                    let ch_labels: Vec<String> = (0..n_ch).map(|c| format!("ch_{}", c)).collect();

                    if let Err(e) = process_signal_group(
                        cfg,
                        &h5_file,
                        &hht_trial,
                        block_name,
                        task_name,
                        block_signal.clone(),
                        ch_labels,
                        false,
                        &[],
                        &[],
                        &[],
                        "",
                        "",
                        alpha,
                        num_modes,
                        &admm_config,
                        &mvmd_variant,
                        Some(&format!(
                            "{}_task-{}_run-{}_{}_group-full",
                            formatted_id,
                            task_file.get("task").unwrap_or("unknown"),
                            task_file.get("run").unwrap_or("01"),
                            block_name
                        )),
                    ) {
                        error_count += 1;
                        warn!(task_name, block = block_name, error = %e, "block HHT failed");
                    }

                    if roi_subset_enabled {
                        let roi_block = block_signal.select(Axis(0), &roi_row_indices);
                        let hht_blocks_roi =
                            open_or_create_group(&hht_group, all_blocks_group_roi, false)?;
                        let hht_trial_roi =
                            open_or_create_group(&hht_blocks_roi, trial_type, false)?;
                        if let Err(e) = process_signal_group(
                            cfg,
                            &h5_file,
                            &hht_trial_roi,
                            block_name,
                            task_name,
                            roi_block,
                            roi_labels.clone(),
                            true,
                            &roi_indices_u32,
                            &roi_labels,
                            &roi_matched_regions,
                            &roi_selection_name,
                            &roi_selection_fingerprint,
                            alpha,
                            num_modes,
                            &admm_config,
                            &mvmd_variant,
                            Some(&format!(
                                "{}_task-{}_run-{}_{}_group-fullroi",
                                formatted_id,
                                task_file.get("task").unwrap_or("unknown"),
                                task_file.get("run").unwrap_or("01"),
                                block_name
                            )),
                        ) {
                            error_count += 1;
                            warn!(
                                task_name,
                                block = block_name,
                                error = %e,
                                "block ROI HHT failed"
                            );
                        }
                    }
                }

                info!(
                    task_name,
                    trial_type,
                    num_blocks = block_names.len(),
                    "finished block HHT for trial type"
                );
            }
        }
    }

    if error_count > 0 {
        warn!(
            error_count,
            "some subjects/blocks were skipped due to errors"
        );
    }

    let total_ms = run_start.elapsed().as_millis();
    info!(error_count, total_ms, "HHT pipeline complete");

    Ok(())
}
