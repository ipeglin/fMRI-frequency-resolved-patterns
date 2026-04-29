mod algorithms;

use anyhow::{Context, Result};
use ndarray::{Array2, Axis, concatenate};
use polars::prelude::*;
use utils::atlas::BrainAtlas;
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands;
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_attrs, write_dataset};
use utils::roi_migration::check_roi_fingerprint;

use crate::algorithms::admm::ADMMConfig;
use crate::algorithms::mvmd::{FrequencyInit, MVMD};
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, info, warn};

fn write_mvmd_algorithm_attrs_if_missing(
    loc: &hdf5::Location,
    alpha: f64,
    sampling_rate: f64,
    f_min: f64,
    f_max: f64,
    n_scales: usize,
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
    if loc.attr("f_min").is_err() {
        write_attrs(loc, &[H5Attr::f64("f_min", f_min)])?;
    }
    if loc.attr("f_max").is_err() {
        write_attrs(loc, &[H5Attr::f64("f_max", f_max)])?;
    }
    if loc.attr("n_scales").is_err() {
        write_attrs(loc, &[H5Attr::u32("n_scales", n_scales as u32)])?;
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

fn filter_directory_bids_files<F>(
    dir: &Path,
    predicate: F,
) -> Result<Vec<BidsFilename>, Box<dyn std::error::Error>>
where
    F: Fn(&BidsFilename) -> bool,
{
    let files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let bids = BidsFilename::from_path_buf(&path);
            if predicate(&bids) { Some(bids) } else { None }
        })
        .collect();

    Ok(files)
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let _run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    // Global MVMD config
    // Frequency bounds derived from project-wide SLOW_BANDS so MVMD modes share
    // the same analysed BOLD frequency window as CWT scalograms and HHT spectra.
    let f_min: f64 = frequency_bands::f_min();
    let f_max: f64 = frequency_bands::f_max();
    let alpha: f64 = 2000.0;
    let n_scales: usize = 224;
    let num_modes: usize = cfg.mvmd.num_modes;
    let sampling_rate: f64 = cfg.task_sampling_rate;
    let admm_config = ADMMConfig::default();
    let custom_init_freqs: Vec<f64> = (0..num_modes)
        .map(|i| {
            // Explicitly using f64 literals (0.0_f64) or casting fixes the powf ambiguity
            let ratio = f_max / f_min;
            let exponent = i as f64 / (num_modes - 1).max(1) as f64;
            f_min * ratio.powf(exponent)
        })
        .collect();

    let target_trial_types = vec!["face"];
    let group_name_blocks_raw = "blocks_raw";
    let group_name_full_run_roi = "full_run_raw_roi";
    let group_name_blocks_roi = "blocks_raw_roi";
    // let group_name_blocks_std = "blocks_std";

    let brain_atlas =
        BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let spec = &cfg.roi_selection;
    let roi_subset_enabled = !spec.is_empty();
    let selected = brain_atlas.selected_rois(spec);
    let roi_row_indices: Vec<usize> = selected.iter().map(|r| r.row_index).collect();
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
    let roi_indices_u32: Vec<u32> = roi_row_indices.iter().map(|i| *i as u32).collect();
    if roi_subset_enabled {
        info!(
            n_target_rois = roi_row_indices.len(),
            rois = ?roi_labels,
            roi_selection_name = %roi_selection_name,
            roi_selection_fingerprint = %roi_selection_fingerprint,
            "selected ROI subset for ROI-only MVMD"
        );
    } else {
        info!("no ROI selection configured — skipping ROI-only MVMD paths");
    }

    info!(
        tcp_repo_dir = %cfg.tcp_repo_dir.display(),
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        num_modes = %cfg.mvmd.num_modes,
        sampling_rate = %cfg.task_sampling_rate,
        admm_config = ?admm_config,
        alpha = alpha,
        "starting fMRI MVMD decomposition"
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
    let mut error_count = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;

        let _subject_span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();

        let available_resting_state_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("restAP"))
                .expect("Failed to read the directory");

        let available_hammer_task_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("hammerAP"))
                .expect("Failed to read the directory");
        let available_timeseries: Vec<BidsFilename> = filter_directory_bids_files(dir, |bids| {
            let task = bids.get("task");
            task == Some("restAP") || task == Some("hammerAP")
        })
        .expect("Failed to read the directory");

        debug!(
            resting_state_count = available_resting_state_ts.len(),
            hammer_task_count = available_hammer_task_ts.len(),
            total_available_timeseries = available_timeseries.len(),
            "extracted resting-state and task-based timeseries"
        );

        for rs_file in &available_resting_state_ts {
            // Ensure output file exists
            let path = rs_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;
            debug!(
                output_file = %path.display(),
                "opening subject HDF5 file for resting-state MVMD"
            );
            let h5_file = open_or_create(&path)?;
            debug!(
                group_path = "/04mvmd",
                force = cfg.force,
                "opening output group"
            );
            // Top-level /04mvmd never wiped: per-subgroup `open_or_create_group`
            // calls below honour `cfg.force`. Wiping the parent on heavy files
            // can leave HDF5's symbol table in a stale state where the next
            // H5Gcreate2 fails with "name already exists".
            let mvmd_group = open_or_create_group(&h5_file, "04mvmd", false)
                .context("failed to open/create group /mvmd")?;
            write_mvmd_algorithm_attrs_if_missing(
                &mvmd_group,
                alpha,
                sampling_rate,
                f_min,
                f_max,
                n_scales,
                num_modes,
                &admm_config,
            )?;

            let task_name = rs_file.get("task").unwrap_or("unknown");
            let run_idx = rs_file.get("run").unwrap_or("unknown");

            // Full-run Decomposition //
            let fr_done = !cfg.force && mvmd_group.group("full_run_raw").is_ok();
            if fr_done {
                let fr_group = mvmd_group.group("full_run_raw").context(
                    "failed to open group /mvmd/full_run_raw for center_frequencies sync",
                )?;
                sync_center_frequencies_attr_from_group(&fr_group).context(
                    "failed to sync center_frequencies attribute to /mvmd/full_run_raw/modes",
                )?;
                debug!(
                    task_name = task_name,
                    run = run_idx,
                    "whole-signal resting-state MVMD already computed, skipping (use --force to recompute)"
                );
            } else {
                info!(
                    task_name = task_name,
                    run = run_idx,
                    signal_type = "full_run",
                    input_dataset = "/01fmri_parcellation/full_run_raw",
                    output_group = "/04mvmd/full_run_raw",
                    "starting MVMD decomposition"
                );
                let mvmd_start = Instant::now();

                debug!(
                    dataset_path = "/01fmri_parcellation/full_run_raw",
                    "reading input dataset"
                );
                let parc_group = h5_file
                    .group("01fmri_parcellation")
                    .context("failed to open group /01fmri_parcellation")?;
                let dataset = parc_group
                    .dataset("full_run_raw")
                    .context("failed to open dataset /01fmri_parcellation/full_run_raw")?;
                let data: Array2<f32> = dataset.read_2d()?;

                let columns: Vec<Column> = data
                    .outer_iter()
                    .enumerate()
                    .map(|(c, row_view)| {
                        let slice = row_view
                            .as_slice()
                            .expect("Data in ndarray must be contiguous for Polars conversion");
                        Series::new(format!("ch_{}", c).into(), slice).into()
                    })
                    .collect();

                let full_df = DataFrame::new(columns)?;
                let fr_mvmd = MVMD::from_dataframe(&full_df, alpha, sampling_rate)?
                    .with_admm_config(admm_config.clone());
                // .with_init(FrequencyInit::Custom(custom_init_freqs.clone())); // Initialize with log-spaced Hz

                let signal_decomposition = fr_mvmd.decompose(num_modes);
                let grid_aligned_modes = signal_decomposition.remap_to_grid(f_min, f_max, n_scales);
                let modes_shape = grid_aligned_modes.shape();
                let mvmd_wb_duration_ms = mvmd_start.elapsed().as_millis();

                debug!(
                    group_path = "/mvmd/full_run_raw",
                    force = cfg.force,
                    "opening output group"
                );
                let fr_group = open_or_create_group(&mvmd_group, "full_run_raw", cfg.force)
                    .context("failed to open/create group /mvmd/full_run_raw")?;
                let write_start = Instant::now();

                debug!(
                    dataset_path = "/mvmd/full_run_raw/modes_gridded",
                    "writing output dataset"
                );
                write_dataset(
                    &fr_group,
                    "modes_gridded",
                    grid_aligned_modes.as_slice().unwrap(),
                    &[modes_shape[0], modes_shape[1], modes_shape[2]],
                    None,
                )?;

                let modes_shape = signal_decomposition.modes.shape();
                debug!(
                    dataset_path = "/mvmd/full_run_raw/modes",
                    "writing output dataset"
                );
                write_dataset(
                    &fr_group,
                    "modes",
                    signal_decomposition.modes.as_slice().unwrap(),
                    &[modes_shape[0], modes_shape[1], modes_shape[2]],
                    None,
                )?;
                let fr_modes_dataset = fr_group
                    .dataset("modes")
                    .context("failed to open dataset /mvmd/full_run_raw/modes")?;
                write_center_frequencies_attr(
                    &fr_modes_dataset,
                    signal_decomposition.center_frequencies.as_slice().unwrap(),
                )?;

                let cf_shape = signal_decomposition.frequency_traces.shape();
                debug!(
                    dataset_path = "/mvmd/full_run_raw/frequency_traces",
                    "writing output dataset"
                );
                write_dataset(
                    &fr_group,
                    "frequency_traces",
                    signal_decomposition.frequency_traces.as_slice().unwrap(),
                    &[cf_shape[0], cf_shape[1]],
                    None,
                )?;

                debug!(
                    dataset_path = "/mvmd/full_run_raw/center_frequencies",
                    "writing output dataset"
                );
                write_dataset(
                    &fr_group,
                    "center_frequencies",
                    signal_decomposition.center_frequencies.as_slice().unwrap(),
                    &[signal_decomposition.center_frequencies.len()],
                    None,
                )?;

                write_attrs(
                    &fr_group,
                    &[H5Attr::u32(
                        "num_iterations",
                        signal_decomposition.num_iterations as u32,
                    )],
                )?;
                write_mvmd_algorithm_attrs_if_missing(
                    &fr_group,
                    alpha,
                    sampling_rate,
                    f_min,
                    f_max,
                    n_scales,
                    num_modes,
                    &admm_config,
                )?;

                if mvmd_group.attr("channels").is_err() {
                    write_attrs(
                        &mvmd_group,
                        &[H5Attr::string(
                            "channels",
                            signal_decomposition.channels.join(","),
                        )],
                    )?;
                }

                let write_duration_ms = write_start.elapsed().as_millis();

                debug!(
                    task_name = task_name,
                    num_modes = num_modes,
                    mvmd_iterations = signal_decomposition.num_iterations,
                    mvmd_duration_ms = mvmd_wb_duration_ms,
                    write_duration_ms = write_duration_ms,
                    output_file = %path.display(),
                    "computed whole-signal MVMD decomposition"
                );
            }

            // ROI-specific Full-run Decomposition //
            // Skipped entirely when no ROI selection is configured; the
            // whole-signal `full_run_raw` block above still runs.
            if !roi_subset_enabled {
                debug!(
                    task_name = task_name,
                    run = run_idx,
                    "no ROI selection configured, skipping ROI-only full-run MVMD"
                );
            } else {
            let fr_roi_done = !cfg.force && mvmd_group.group(group_name_full_run_roi).is_ok();
            if fr_roi_done {
                let fr_roi_group = mvmd_group.group(group_name_full_run_roi).with_context(|| {
                    format!(
                        "failed to open group /mvmd/{group_name_full_run_roi} for center_frequencies sync"
                    )
                })?;
                check_roi_fingerprint(
                    &fr_roi_group,
                    &roi_selection_fingerprint,
                    &format!("/04mvmd/{group_name_full_run_roi}"),
                )?;
                sync_center_frequencies_attr_from_group(&fr_roi_group).with_context(|| {
                    format!(
                        "failed to sync center_frequencies attribute to /mvmd/{group_name_full_run_roi}/modes"
                    )
                })?;
                debug!(
                    task_name = task_name,
                    run = run_idx,
                    "ROI-only full-run MVMD already computed, skipping (use --force to recompute)"
                );
            } else {
                info!(
                    task_name = task_name,
                    run = run_idx,
                    signal_type = "full_run_roi",
                    input_dataset = "/01fmri_parcellation/full_run_raw",
                    output_group = %format!("/04mvmd/{group_name_full_run_roi}"),
                    n_target_rois = roi_row_indices.len(),
                    "starting ROI-only MVMD decomposition"
                );
                let mvmd_roi_start = Instant::now();

                let parc_group = h5_file
                    .group("01fmri_parcellation")
                    .context("failed to open group /01fmri_parcellation for ROI subset")?;
                let dataset = parc_group.dataset("full_run_raw").context(
                    "failed to open dataset /01fmri_parcellation/full_run_raw for ROI subset",
                )?;
                let data: Array2<f32> = dataset.read_2d()?;
                let roi_data = data.select(Axis(0), &roi_row_indices);

                let roi_columns: Vec<Column> = roi_data
                    .outer_iter()
                    .zip(roi_labels.iter())
                    .map(|(row_view, label)| {
                        let slice = row_view
                            .as_slice()
                            .expect("Data in ndarray must be contiguous for Polars conversion");
                        Series::new(label.as_str().into(), slice).into()
                    })
                    .collect();

                let roi_df = DataFrame::new(roi_columns)?;
                let fr_roi_mvmd = MVMD::from_dataframe(&roi_df, alpha, sampling_rate)?
                    .with_admm_config(admm_config.clone());

                let roi_decomposition = fr_roi_mvmd.decompose(num_modes);
                let roi_grid_aligned_modes =
                    roi_decomposition.remap_to_grid(f_min, f_max, n_scales);
                let roi_modes_shape = roi_grid_aligned_modes.shape();
                let mvmd_roi_duration_ms = mvmd_roi_start.elapsed().as_millis();

                let fr_roi_group =
                    open_or_create_group(&mvmd_group, group_name_full_run_roi, cfg.force)
                        .with_context(|| {
                            format!("failed to open/create group /mvmd/{group_name_full_run_roi}")
                        })?;
                let roi_write_start = Instant::now();

                write_dataset(
                    &fr_roi_group,
                    "modes_gridded",
                    roi_grid_aligned_modes.as_slice().unwrap(),
                    &[roi_modes_shape[0], roi_modes_shape[1], roi_modes_shape[2]],
                    None,
                )?;

                let roi_modes_shape = roi_decomposition.modes.shape();
                write_dataset(
                    &fr_roi_group,
                    "modes",
                    roi_decomposition.modes.as_slice().unwrap(),
                    &[roi_modes_shape[0], roi_modes_shape[1], roi_modes_shape[2]],
                    None,
                )?;
                let fr_roi_modes_dataset = fr_roi_group.dataset("modes").with_context(|| {
                    format!("failed to open dataset /mvmd/{group_name_full_run_roi}/modes")
                })?;
                write_center_frequencies_attr(
                    &fr_roi_modes_dataset,
                    roi_decomposition.center_frequencies.as_slice().unwrap(),
                )?;

                let roi_cf_shape = roi_decomposition.frequency_traces.shape();
                write_dataset(
                    &fr_roi_group,
                    "frequency_traces",
                    roi_decomposition.frequency_traces.as_slice().unwrap(),
                    &[roi_cf_shape[0], roi_cf_shape[1]],
                    None,
                )?;

                write_dataset(
                    &fr_roi_group,
                    "center_frequencies",
                    roi_decomposition.center_frequencies.as_slice().unwrap(),
                    &[roi_decomposition.center_frequencies.len()],
                    None,
                )?;

                write_dataset(
                    &fr_roi_group,
                    "roi_indices",
                    &roi_indices_u32,
                    &[roi_indices_u32.len()],
                    None,
                )?;

                write_attrs(
                    &fr_roi_group,
                    &[
                        H5Attr::u32("num_iterations", roi_decomposition.num_iterations as u32),
                        H5Attr::u32("n_rois", roi_indices_u32.len() as u32),
                        H5Attr::string("roi_labels", roi_labels.join(",")),
                        H5Attr::string("roi_matched_regions", roi_matched_regions.join(",")),
                        H5Attr::string("roi_selection_name", &roi_selection_name),
                        H5Attr::string(
                            "roi_selection_fingerprint",
                            &roi_selection_fingerprint,
                        ),
                        H5Attr::string("channels", roi_decomposition.channels.join(",")),
                    ],
                )?;
                write_mvmd_algorithm_attrs_if_missing(
                    &fr_roi_group,
                    alpha,
                    sampling_rate,
                    f_min,
                    f_max,
                    n_scales,
                    num_modes,
                    &admm_config,
                )?;

                let roi_write_duration_ms = roi_write_start.elapsed().as_millis();

                debug!(
                    task_name = task_name,
                    n_target_rois = roi_row_indices.len(),
                    num_modes = num_modes,
                    mvmd_iterations = roi_decomposition.num_iterations,
                    mvmd_duration_ms = mvmd_roi_duration_ms,
                    write_duration_ms = roi_write_duration_ms,
                    output_file = %path.display(),
                    "computed ROI-only full-run MVMD decomposition"
                );
            }
            }
        }

        for task_file in &available_hammer_task_ts {
            // Ensure output file exists
            let path = task_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;
            debug!(
                output_file = %path.display(),
                "opening subject HDF5 file for task block MVMD"
            );
            let h5_file = open_or_create(&path)?;
            debug!(
                group_path = "/04mvmd",
                force = cfg.force,
                "opening output group"
            );
            let mvmd_group = open_or_create_group(&h5_file, "04mvmd", cfg.force)
                .context("failed to open/create group /04mvmd")?;
            write_mvmd_algorithm_attrs_if_missing(
                &mvmd_group,
                alpha,
                sampling_rate,
                f_min,
                f_max,
                n_scales,
                num_modes,
                &admm_config,
            )?;

            let task_name = task_file.get("task").unwrap_or("unknown");
            info!(
                task_name = task_name,
                signal_type = "blocks",
                input_group = "/02fmri_segment_trials/blocks_raw",
                output_group = "/04mvmd/blocks_raw",
                "starting MVMD decomposition"
            );

            // Block-wise Decomposition //
            debug!(
                group_path = "/02fmri_segment_trials/blocks_raw",
                "opening input group"
            );
            let segment_root = h5_file
                .group("02fmri_segment_trials")
                .context("failed to open input group /02fmri_segment_trials")?;
            let ts_blocks_group = segment_root
                .group(group_name_blocks_raw)
                .context("failed to open input group /02fmri_segment_trials/blocks_raw")?;
            let mvmd_blocks_group =
                open_or_create_group(&mvmd_group, group_name_blocks_raw, cfg.force)
                    .context("failed to open/create output group /04mvmd/blocks_raw")?;

            for trial_type in &target_trial_types {
                let trial_group_path = format!("/02fmri_segment_trials/blocks_raw/{trial_type}");
                let trial_group = match ts_blocks_group.group(trial_type) {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            trial_type = trial_type,
                            group_path = %trial_group_path,
                            "trial type group not found, skipping"
                        );
                        continue;
                    }
                };

                let block_names: Vec<String> = trial_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    debug!(
                        task_name = task_name,
                        trial_type = trial_type,
                        "trial type group is empty, skipping"
                    );
                    continue;
                }

                info!(
                    task_name = task_name,
                    trial_type = trial_type,
                    num_blocks = block_names.len(),
                    "starting block MVMD decomposition"
                );

                for (block_idx, block_name) in block_names.iter().enumerate() {
                    let input_block_dataset_path = format!("{trial_group_path}/{block_name}");
                    let output_block_group_path = format!("/04mvmd/blocks_raw/{block_name}");

                    info!(
                        task_name = task_name,
                        trial_type = trial_type,
                        block = block_name,
                        block_idx = block_idx,
                        num_blocks = block_names.len(),
                        signal_type = "block",
                        input_dataset = %input_block_dataset_path,
                        output_group = %output_block_group_path,
                        "starting MVMD decomposition"
                    );

                    // Check if output group already exists (/04mvmd/blocks_raw/block_X/)
                    if !cfg.force && mvmd_blocks_group.group(block_name).is_ok() {
                        let existing_block_group =
                            mvmd_blocks_group.group(block_name).with_context(|| {
                                format!(
                                    "failed to open existing output group {output_block_group_path}"
                                )
                            })?;
                        sync_center_frequencies_attr_from_group(&existing_block_group).with_context(|| {
                            format!(
                                "failed to sync center_frequencies attribute to {output_block_group_path}/modes"
                            )
                        })?;
                        debug!(
                            task_name = task_name,
                            trial_type = trial_type,
                            block = block_name,
                            block_idx = block_idx,
                            num_blocks = block_names.len(),
                            "MVMD block modes already computed, skipping (use --force to recompute)"
                        );
                        continue;
                    }

                    debug!(dataset_path = %input_block_dataset_path, "reading input dataset");
                    let input_ds = trial_group.dataset(block_name).with_context(|| {
                        format!("failed to open input dataset {input_block_dataset_path}")
                    })?;
                    let block_signal_f32: Array2<f32> = input_ds.read_2d()?;

                    // Read metadata attributes from the input dataset
                    let onset_s: f64 = input_ds.attr("onset_s")?.read_scalar()?;
                    let block_end_s: f64 = input_ds.attr("block_end_s")?.read_scalar()?;

                    let [block_channels, block_timepoints] = match block_signal_f32.shape() {
                        &[r, c] => [r, c],
                        _ => {
                            error_count += 1;
                            warn!(
                                task_name = task_name,
                                trial_type = trial_type,
                                block = block_name,
                                block_idx = block_idx,
                                num_blocks = block_names.len(),
                                reason = "unexpected_block_shape",
                                shape = ?block_signal_f32.shape(),
                                "skipping mvmd block decompositin due to unexpected signal shape"
                            );
                            continue;
                        }
                    };
                    let block_signal_f64 = block_signal_f32.mapv(|val| val as f64);

                    let block_columns: Vec<Column> = block_signal_f32
                        .outer_iter()
                        .enumerate()
                        .map(|(c, row_view)| {
                            let slice = row_view
                                .as_slice()
                                .expect("Data in ndarray must be contiguous for Polars conversion");
                            Series::new(format!("ch_{}", c).into(), slice).into()
                        })
                        .collect();

                    let block_df = DataFrame::new(block_columns)?;
                    let block_mvmd = MVMD::from_dataframe(&block_df, alpha, sampling_rate)?
                        .with_admm_config(admm_config.clone());
                    // .with_init(FrequencyInit::Custom(custom_init_freqs.clone())); // Initialize with log-spaced Hz

                    // Run actual decomposition
                    let block_start = Instant::now();
                    let block_decomposition = block_mvmd.decompose(num_modes);
                    let block_mvmd_duration_ms = block_start.elapsed().as_millis();
                    debug!(
                        group_path = %output_block_group_path,
                        force = cfg.force,
                        "opening output block group"
                    );
                    let mvmd_block_group =
                        open_or_create_group(&mvmd_blocks_group, block_name, cfg.force)
                            .with_context(|| {
                                format!(
                                    "failed to open/create output group {output_block_group_path}"
                                )
                            })?; // /04mvmd/blocks_raw/block_X/
                    let block_write_start = Instant::now();

                    debug!(dataset_path = %format!("{output_block_group_path}/modes"), "writing output dataset");

                    let bm_shape = block_decomposition.modes.shape();
                    write_dataset(
                        &mvmd_block_group,
                        "modes",
                        block_decomposition.modes.as_slice().unwrap(),
                        &[bm_shape[0], bm_shape[1], bm_shape[2]],
                        None,
                    )?;
                    let block_modes_dataset =
                        mvmd_block_group.dataset("modes").with_context(|| {
                            format!("failed to open dataset {output_block_group_path}/modes")
                        })?;
                    write_center_frequencies_attr(
                        &block_modes_dataset,
                        block_decomposition.center_frequencies.as_slice().unwrap(),
                    )?;

                    let bcf_shape = block_decomposition.frequency_traces.shape();
                    debug!(dataset_path = %format!("{output_block_group_path}/frequency_traces"), "writing output dataset");
                    write_dataset(
                        &mvmd_block_group,
                        "frequency_traces",
                        block_decomposition.frequency_traces.as_slice().unwrap(),
                        &[bcf_shape[0], bcf_shape[1]],
                        None,
                    )?;

                    debug!(dataset_path = %format!("{output_block_group_path}/center_frequencies"), "writing output dataset");
                    write_dataset(
                        &mvmd_block_group,
                        "center_frequencies",
                        block_decomposition.center_frequencies.as_slice().unwrap(),
                        &[block_decomposition.center_frequencies.len()],
                        None,
                    )?;

                    write_attrs(
                        &mvmd_block_group,
                        &[H5Attr::u32(
                            "num_iterations",
                            block_decomposition.num_iterations as u32,
                        )],
                    )?;
                    write_mvmd_algorithm_attrs_if_missing(
                        &mvmd_block_group,
                        alpha,
                        sampling_rate,
                        f_min,
                        f_max,
                        n_scales,
                        num_modes,
                        &admm_config,
                    )?;

                    let block_write_duration_ms = block_write_start.elapsed().as_millis();

                    debug!(
                        task_name = task_name,
                        block = block_name,
                        num_modes = num_modes,
                        mvmd_iterations = block_decomposition.num_iterations,
                        mvmd_duration_ms = block_mvmd_duration_ms,
                        write_duration_ms = block_write_duration_ms,
                        "computed block MVMD decomposition"
                    );
                }

                debug!(
                    task_name = task_name,
                    num_blocks = block_names.len(),
                    "finished block MVMD decompositions"
                );
            }

            // ROI-specific Block-wise Decomposition //
            // Skipped entirely when no ROI selection is configured.
            if !roi_subset_enabled {
                debug!(
                    task_name = task_name,
                    "no ROI selection configured, skipping ROI-only block MVMD"
                );
                continue;
            }
            let mvmd_blocks_roi_group = open_or_create_group(
                &mvmd_group,
                group_name_blocks_roi,
                cfg.force,
            )
            .with_context(|| {
                format!("failed to open/create output group /04mvmd/{group_name_blocks_roi}")
            })?;

            for trial_type in &target_trial_types {
                let trial_group_path = format!("/02fmri_segment_trials/blocks_raw/{trial_type}");
                let trial_group = match ts_blocks_group.group(trial_type) {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            trial_type = trial_type,
                            group_path = %trial_group_path,
                            "trial type group not found, skipping ROI block-wise"
                        );
                        continue;
                    }
                };

                let block_names: Vec<String> = trial_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    continue;
                }

                info!(
                    task_name = task_name,
                    trial_type = trial_type,
                    num_blocks = block_names.len(),
                    n_target_rois = roi_row_indices.len(),
                    "starting ROI-only block MVMD decomposition"
                );

                for (block_idx, block_name) in block_names.iter().enumerate() {
                    let input_block_dataset_path = format!("{trial_group_path}/{block_name}");
                    let output_block_group_path =
                        format!("/04mvmd/{group_name_blocks_roi}/{block_name}");

                    if !cfg.force && mvmd_blocks_roi_group.group(block_name).is_ok() {
                        let existing_block_group =
                            mvmd_blocks_roi_group.group(block_name).with_context(|| {
                                format!(
                                    "failed to open existing output group {output_block_group_path}"
                                )
                            })?;
                        check_roi_fingerprint(
                            &existing_block_group,
                            &roi_selection_fingerprint,
                            &output_block_group_path,
                        )?;
                        sync_center_frequencies_attr_from_group(&existing_block_group)
                            .with_context(|| {
                                format!(
                                    "failed to sync center_frequencies attribute to {output_block_group_path}/modes"
                                )
                            })?;
                        debug!(
                            task_name = task_name,
                            trial_type = trial_type,
                            block = block_name,
                            block_idx = block_idx,
                            num_blocks = block_names.len(),
                            "ROI MVMD block modes already computed, skipping (use --force to recompute)"
                        );
                        continue;
                    }

                    let input_ds = trial_group.dataset(block_name).with_context(|| {
                        format!("failed to open input dataset {input_block_dataset_path}")
                    })?;
                    let block_signal_f32: Array2<f32> = input_ds.read_2d()?;

                    let [_block_channels, _block_timepoints] = match block_signal_f32.shape() {
                        &[r, c] => [r, c],
                        _ => {
                            error_count += 1;
                            warn!(
                                task_name = task_name,
                                trial_type = trial_type,
                                block = block_name,
                                block_idx = block_idx,
                                num_blocks = block_names.len(),
                                reason = "unexpected_block_shape",
                                shape = ?block_signal_f32.shape(),
                                "skipping ROI mvmd block decomposition due to unexpected signal shape"
                            );
                            continue;
                        }
                    };

                    let roi_block_signal = block_signal_f32.select(Axis(0), &roi_row_indices);

                    let roi_block_columns: Vec<Column> = roi_block_signal
                        .outer_iter()
                        .zip(roi_labels.iter())
                        .map(|(row_view, label)| {
                            let slice = row_view
                                .as_slice()
                                .expect("Data in ndarray must be contiguous for Polars conversion");
                            Series::new(label.as_str().into(), slice).into()
                        })
                        .collect();

                    let roi_block_df = DataFrame::new(roi_block_columns)?;
                    let roi_block_mvmd = MVMD::from_dataframe(&roi_block_df, alpha, sampling_rate)?
                        .with_admm_config(admm_config.clone());

                    let block_start = Instant::now();
                    let roi_block_decomposition = roi_block_mvmd.decompose(num_modes);
                    let block_mvmd_duration_ms = block_start.elapsed().as_millis();

                    let mvmd_block_group =
                        open_or_create_group(&mvmd_blocks_roi_group, block_name, cfg.force)
                            .with_context(|| {
                                format!(
                                    "failed to open/create output group {output_block_group_path}"
                                )
                            })?;
                    let block_write_start = Instant::now();

                    let bm_shape = roi_block_decomposition.modes.shape();
                    write_dataset(
                        &mvmd_block_group,
                        "modes",
                        roi_block_decomposition.modes.as_slice().unwrap(),
                        &[bm_shape[0], bm_shape[1], bm_shape[2]],
                        None,
                    )?;
                    let block_modes_dataset =
                        mvmd_block_group.dataset("modes").with_context(|| {
                            format!("failed to open dataset {output_block_group_path}/modes")
                        })?;
                    write_center_frequencies_attr(
                        &block_modes_dataset,
                        roi_block_decomposition
                            .center_frequencies
                            .as_slice()
                            .unwrap(),
                    )?;

                    let bcf_shape = roi_block_decomposition.frequency_traces.shape();
                    write_dataset(
                        &mvmd_block_group,
                        "frequency_traces",
                        roi_block_decomposition.frequency_traces.as_slice().unwrap(),
                        &[bcf_shape[0], bcf_shape[1]],
                        None,
                    )?;

                    write_dataset(
                        &mvmd_block_group,
                        "center_frequencies",
                        roi_block_decomposition
                            .center_frequencies
                            .as_slice()
                            .unwrap(),
                        &[roi_block_decomposition.center_frequencies.len()],
                        None,
                    )?;

                    write_dataset(
                        &mvmd_block_group,
                        "roi_indices",
                        &roi_indices_u32,
                        &[roi_indices_u32.len()],
                        None,
                    )?;

                    write_attrs(
                        &mvmd_block_group,
                        &[
                            H5Attr::u32(
                                "num_iterations",
                                roi_block_decomposition.num_iterations as u32,
                            ),
                            H5Attr::u32("n_rois", roi_indices_u32.len() as u32),
                            H5Attr::string("roi_labels", roi_labels.join(",")),
                            H5Attr::string("channels", roi_block_decomposition.channels.join(",")),
                            H5Attr::string(
                                "roi_matched_regions",
                                roi_matched_regions.join(","),
                            ),
                            H5Attr::string("roi_selection_name", &roi_selection_name),
                            H5Attr::string(
                                "roi_selection_fingerprint",
                                &roi_selection_fingerprint,
                            ),
                        ],
                    )?;
                    write_mvmd_algorithm_attrs_if_missing(
                        &mvmd_block_group,
                        alpha,
                        sampling_rate,
                        f_min,
                        f_max,
                        n_scales,
                        num_modes,
                        &admm_config,
                    )?;

                    let block_write_duration_ms = block_write_start.elapsed().as_millis();

                    debug!(
                        task_name = task_name,
                        block = block_name,
                        n_target_rois = roi_row_indices.len(),
                        num_modes = num_modes,
                        mvmd_iterations = roi_block_decomposition.num_iterations,
                        mvmd_duration_ms = block_mvmd_duration_ms,
                        write_duration_ms = block_write_duration_ms,
                        "computed ROI-only block MVMD decomposition"
                    );
                }
            }
        }
    }

    if error_count > 0 {
        warn!(
            error_count = error_count,
            "some subjects were skipped due to errors"
        );
    }

    Ok(())
}
