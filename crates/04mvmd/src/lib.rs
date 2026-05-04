mod algorithms;

use anyhow::{Context, Result};
use ndarray::{Array2, Axis};
use polars::prelude::*;
use utils::atlas::BrainAtlas;
use utils::bids_filename::{BidsFilename, filter_directory_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::hdf5_io::{H5Attr, H5AttrValue, ensure_path, path_exists, prepare_dataset, write_attrs};
use utils::roi_migration::check_roi_fingerprint;

use crate::algorithms::admm::ADMMConfig;
use crate::algorithms::mvmd::MVMD;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

// HDF5 groups and datasets
const MVMD_CRATE_GROUP: &str = "04mvmd";

const FULL_RUN_GROUP: &str = "full_run_std";
const FULL_RUN_GROUP_ROI_STRATIFIED: &str = "full_run_std_roi";

const ALL_BLOCKS_GROUP: &str = "blocks_std";
const ALL_BLOCKS_GROUP_ROI_STRATIFIED: &str = "blocks_std_roi";

const TARGET_TRIAL_TYPES: &[&str] = &["face"];

// MVMD
const MVMD_ALPHA: f64 = 2000.0;

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

pub fn run(cfg: &AppConfig) -> Result<()> {
    let _run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    // Global MVMD config
    // Frequency bounds derived from project-wide SLOW_BANDS so MVMD modes share
    // the same analysed BOLD frequency window as CWT scalograms and HHT spectra.
    let num_modes: usize = cfg.mvmd.num_modes;
    let sampling_rate: f64 = cfg.task_sampling_rate;
    let admm_config = ADMMConfig::default();

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
            "Using ROI-stratified configuration in additional to all-channel decomposition."
        );
    } else {
        info!("no ROI selection configured — skipping ROI-only MVMD paths");
    }


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

        let mut available_resting_state_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("restAP"))
                .expect("Failed to read the directory");
        sort_bids_vec(
            &mut available_resting_state_ts,
            &["run"],
            |key, a, b| match key {
                "run" => {
                    // Attempt to parse both strings as integers
                    let num_a = a.parse::<u32>();
                    let num_b = b.parse::<u32>();

                    match (num_a, num_b) {
                        (Ok(n_a), Ok(n_b)) => n_a.cmp(&n_b), // Numeric sort
                        _ => a.cmp(b), // Fallback to string sort if parsing fails
                    }
                }
                _ => a.cmp(b), // Default string comparison for other keys
            },
        );

        let mut available_hammer_task_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("hammerAP"))
                .expect("Failed to read the directory");
        sort_bids_vec(
            &mut available_hammer_task_ts,
            &["run"],
            |key, a, b| match key {
                "run" => {
                    // Attempt to parse both strings as integers
                    let num_a = a.parse::<u32>();
                    let num_b = b.parse::<u32>();

                    match (num_a, num_b) {
                        (Ok(n_a), Ok(n_b)) => n_a.cmp(&n_b), // Numeric sort
                        _ => a.cmp(b), // Fallback to string sort if parsing fails
                    }
                }
                _ => a.cmp(b), // Default string comparison for other keys
            },
        );

        debug!(
            resting_state_count = available_resting_state_ts.len(),
            hammer_task_count = available_hammer_task_ts.len(),
            "extracted resting-state and task-based timeseries"
        );

        for rs_file in &available_resting_state_ts {
            let task_name = rs_file.get("task").unwrap_or("unknown");
            let run_idx = rs_file.get("run").unwrap_or("unknown");

            info!(
                file = %rs_file,
                "Processeing resting state fMRI scan"
            );

            // Ensure output file exists
            let path = rs_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;

            let h5_file = hdf5::File::open_rw(&path)?;
            debug!(
                group_path = format!("/{MVMD_CRATE_GROUP}"),
                force = cfg.force,
                "opening output group"
            );

            // Top-level /04mvmd never wiped: per-subgroup `open_or_create_group`
            // calls below honour `cfg.force`. Wiping the parent on heavy files
            // can leave HDF5's symbol table in a stale state where the next
            // H5Gcreate2 fails with "name already exists".
            let mvmd_group = ensure_path(&h5_file, MVMD_CRATE_GROUP, cfg.force)
                .context(format!("failed to open/create group /{MVMD_CRATE_GROUP}"))?;
            write_mvmd_algorithm_attrs_if_missing(
                &mvmd_group,
                MVMD_ALPHA,
                sampling_rate,
                num_modes,
                &admm_config,
            )?;

            // Full-run Decomposition //
            let fr_already_done = !cfg.force
                && path_exists(
                    &mvmd_group,
                    format!("{FULL_RUN_GROUP}/center_frequencies").as_str(),
                );
            if fr_already_done {
                let fr_group = mvmd_group.group(FULL_RUN_GROUP).context(
                    format!("failed to open group /{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP} for center_frequencies sync"),
                )?;
                sync_center_frequencies_attr_from_group(&fr_group).context(format!(
                    "failed to sync center_frequencies attribute to /{}/{}/modes",
                    MVMD_CRATE_GROUP, FULL_RUN_GROUP
                ))?;
                warn!(
                    task_name = task_name,
                    run = run_idx,
                    "full-run resting-state MVMD already computed, skipping (use --force to recompute)"
                );
            } else {
                info!(
                    file = %rs_file,
                    task_name = task_name,
                    run = run_idx,
                    signal_type = "full_run",
                    input_dataset = "/01fmri_parcellation/full_run_std",
                    output_group = format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}"),
                    "Commencing analysis. Full-run decomposition missing or overwritten"
                );
                let mvmd_start = Instant::now();

                let parc_group = h5_file
                    .group("01fmri_parcellation")
                    .context("failed to open group /01fmri_parcellation")?;
                let dataset = parc_group
                    .dataset("full_run_std")
                    .context("failed to open dataset /01fmri_parcellation/full_run_std")?;
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
                let fr_mvmd = MVMD::from_dataframe(&full_df, MVMD_ALPHA, sampling_rate)?
                    .with_admm_config(admm_config.clone());

                let fr_decomposition = fr_mvmd.decompose(num_modes);
                let mvmd_fr_duration_ms = mvmd_start.elapsed().as_millis();

                debug!(
                    group_path = format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}"),
                    force = cfg.force,
                    "opening output group"
                );

                let mvmd_metadata = vec![H5Attr {
                    name: "channels".to_string(),
                    value: H5AttrValue::String(fr_decomposition.channels.join(",")),
                }];
                write_attrs(&mvmd_group, &mvmd_metadata)?;

                // Force-wipe: cache check determined the group is incomplete (or
                // cfg.force is set), so unlink any stale partial group rather
                // than opening it — `write_dataset` would otherwise fail with
                // "unknown library error" trying to create a dataset that
                // already exists from the prior aborted run.
                let fr_group = ensure_path(&mvmd_group, FULL_RUN_GROUP, cfg.force).context(
                    format!("failed to open/create group /{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}"),
                )?;
                let fr_group_metadata = vec![H5Attr {
                    name: "num_iterations".to_string(),
                    value: H5AttrValue::U32(fr_decomposition.num_iterations),
                }];

                write_attrs(&fr_group, &fr_group_metadata)?;

                let write_start = Instant::now();

                debug!(
                    dataset_path = format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}/modes_gridded"),
                    "writing output dataset"
                );

                debug!(
                    dataset_path = format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}/modes"),
                    "writing output dataset"
                );

                let modes_shape = fr_decomposition.modes.shape();
                let modes_ds = prepare_dataset::<f64>(
                    &fr_group,
                    "modes",
                    &[modes_shape[0], modes_shape[1], modes_shape[2]],
                )?;
                let modes_metadata = vec![H5Attr {
                    name: "center_frequencies".to_string(),
                    value: H5AttrValue::F64Slice(
                        fr_decomposition
                            .center_frequencies
                            .as_slice()
                            .unwrap()
                            .to_vec(),
                    ),
                }];
                modes_ds.write_raw(fr_decomposition.modes.as_slice().unwrap())?;
                write_attrs(&modes_ds, &modes_metadata)?;

                debug!(
                    dataset_path = format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}/frequency_traces"),
                    "writing output dataset"
                );

                let cf_shape = fr_decomposition.frequency_traces.shape();
                let centre_freq_ds = prepare_dataset::<f64>(
                    &fr_group,
                    "frequency_traces",
                    &[cf_shape[0], cf_shape[1]],
                )?;
                centre_freq_ds.write_raw(fr_decomposition.frequency_traces.as_slice().unwrap())?;

                debug!(
                    dataset_path =
                        format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP}/center_frequencies"),
                    "writing output dataset"
                );

                let freq_shape = &[fr_decomposition.center_frequencies.len()];
                let freq_ds = prepare_dataset::<f64>(&fr_group, "center_frequencies", freq_shape)?;
                freq_ds.write_raw(fr_decomposition.center_frequencies.as_slice().unwrap())?;

                write_mvmd_algorithm_attrs_if_missing(
                    &fr_group,
                    MVMD_ALPHA,
                    sampling_rate,
                    num_modes,
                    &admm_config,
                )?;

                let write_duration_ms = write_start.elapsed().as_millis();

                debug!(
                    task_name = task_name,
                    num_modes = num_modes,
                    mvmd_iterations = fr_decomposition.num_iterations,
                    mvmd_duration_ms = mvmd_fr_duration_ms,
                    write_duration_ms = write_duration_ms,
                    output_file = %path.display(),
                    "computed full-run MVMD decomposition"
                );
            }

            // ROI-specific Full-run Decomposition //
            // Skipped entirely when no ROI selection is configured; the
            // full-run `full_run_std` block above still runs.
            if !roi_subset_enabled {
                warn!(
                    task_name = task_name,
                    run = run_idx,
                    "no ROI selection configured, skipping ROI-only full-run MVMD"
                );
            } else {
                let fr_roi_group = mvmd_group.group(FULL_RUN_GROUP_ROI_STRATIFIED).with_context(|| {
                    format!(
                        "failed to open group /{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP_ROI_STRATIFIED} for center_frequencies sync"
                    )
                })?;

                // roi fingerprint config same as HDF5 attribute
                let roi_fingerprint_unchanged = check_roi_fingerprint(
                    &fr_roi_group,
                    &roi_selection_fingerprint,
                    &format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP_ROI_STRATIFIED}"),
                )
                .is_ok();
                let fr_roi_done = !cfg.force
                    // already has MVMD results
                    && path_exists(
                        &mvmd_group,
                        format!("{FULL_RUN_GROUP_ROI_STRATIFIED}/center_frequencies").as_str(),
                    )
                    && roi_fingerprint_unchanged;
                if fr_roi_done {
                    sync_center_frequencies_attr_from_group(&fr_roi_group).with_context(|| {
                        format!(
                            "failed to sync center_frequencies attribute to /{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP_ROI_STRATIFIED}/modes"
                        )
                    })?;

                warn!(
                    file = %rs_file,
                    task_name = task_name,
                    run = run_idx,
                    "ROI-only full-run MVMD already computed, skipping (use --force to recompute)"
                );
            } else {
                info!(
                    file = %rs_file,
                    task_name = task_name,
                    run = run_idx,
                    signal_type = "full_run_roi",
                    input_dataset = "/01fmri_parcellation/full_run_std",
                    output_group = %format!("/{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP_ROI_STRATIFIED}"),
                    n_target_rois = roi_row_indices.len(),
                    "Commencing analysis. ROI-stratified decomposition missing or overwritten"
                );

                    let mvmd_roi_start = Instant::now();

                    let parc_group = h5_file
                        .group("01fmri_parcellation")
                        .context("failed to open group /01fmri_parcellation for ROI subset")?;
                    let dataset = parc_group.dataset("full_run_std").context(
                        "failed to open dataset /01fmri_parcellation/full_run_std for ROI subset",
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
                    let fr_roi_mvmd = MVMD::from_dataframe(&roi_df, MVMD_ALPHA, sampling_rate)?
                        .with_admm_config(admm_config.clone());

                    let roi_decomposition = fr_roi_mvmd.decompose(num_modes);

                    let mvmd_roi_duration_ms = mvmd_roi_start.elapsed().as_millis();

                    let fr_roi_group =
                        ensure_path(&mvmd_group, FULL_RUN_GROUP_ROI_STRATIFIED, cfg.force)
                            .with_context(|| {
                                format!(
                                    "failed to open/create group /{MVMD_CRATE_GROUP}/{FULL_RUN_GROUP_ROI_STRATIFIED}"
                                )
                            })?;

                    let roi_write_start = Instant::now();

                    let roi_modes_shape = roi_decomposition.modes.shape();
                    let fr_roi_modes_ds = prepare_dataset::<f64>(
                        &fr_roi_group,
                        "modes",
                        &[roi_modes_shape[0], roi_modes_shape[1], roi_modes_shape[2]],
                    )?;
                    let fr_roi_modes_metadata = vec![H5Attr {
                        name: "center_frequencies".to_string(),
                        value: H5AttrValue::F64Slice(
                            roi_decomposition
                                .center_frequencies
                                .as_slice()
                                .unwrap()
                                .to_vec(),
                        ),
                    }];
                    fr_roi_modes_ds.write_raw(roi_decomposition.modes.as_slice().unwrap())?;
                    write_attrs(&fr_roi_modes_ds, &fr_roi_modes_metadata)?;

                    let roi_cf_shape = roi_decomposition.frequency_traces.shape();
                    let freq_trace_ds = prepare_dataset::<f64>(
                        &fr_roi_group,
                        "frequency_traces",
                        &[roi_cf_shape[0], roi_cf_shape[1]],
                    )?;
                    freq_trace_ds
                        .write_raw(roi_decomposition.frequency_traces.as_slice().unwrap())?;

                    let center_freqs_shape = &[roi_decomposition.center_frequencies.len()];
                    let center_freqs_ds = prepare_dataset::<f64>(
                        &fr_roi_group,
                        "center_frequencies",
                        center_freqs_shape,
                    )?;
                    center_freqs_ds
                        .write_raw(roi_decomposition.center_frequencies.as_slice().unwrap())?;

                    let roi_indices_shape = &[roi_indices_u32.len()];
                    let roi_indice_ds =
                        prepare_dataset::<f64>(&fr_roi_group, "roi_indices", roi_indices_shape)?;
                    roi_indice_ds.write_raw(&roi_indices_u32)?;

                    write_attrs(
                        &fr_roi_group,
                        &[
                            H5Attr::u32("num_iterations", roi_decomposition.num_iterations as u32),
                            H5Attr::u32("n_rois", roi_indices_u32.len() as u32),
                            H5Attr::string("roi_labels", roi_labels.join(",")),
                            H5Attr::string("roi_matched_regions", roi_matched_regions.join(",")),
                            H5Attr::string("roi_selection_name", &roi_selection_name),
                            H5Attr::string("roi_selection_fingerprint", &roi_selection_fingerprint),
                            H5Attr::string("channels", roi_decomposition.channels.join(",")),
                        ],
                    )?;
                    write_mvmd_algorithm_attrs_if_missing(
                        &fr_roi_group,
                        MVMD_ALPHA,
                        sampling_rate,
                        num_modes,
                        &admm_config,
                    )?;

                    let roi_write_duration_ms = roi_write_start.elapsed().as_millis();

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

        for task_file in &available_hammer_task_ts {
            let task_name = task_file.get("task").unwrap_or("unknown");

            info!(
                file = %task_file,
                "Processeing task-based fMRI scan"
            );

            let path = task_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;

            let h5_file = hdf5::File::open_rw(&path)?;
            debug!(
                group_path = format!("/{MVMD_CRATE_GROUP}"),
                force = cfg.force,
                "opening output group"
            );

            let mvmd_group = ensure_path(&h5_file, MVMD_CRATE_GROUP, cfg.force)
                .context(format!("failed to open/create group /{MVMD_CRATE_GROUP}"))?;
            write_mvmd_algorithm_attrs_if_missing(
                &mvmd_group,
                MVMD_ALPHA,
                sampling_rate,
                num_modes,
                &admm_config,
            )?;

            // Block-wise Decomposition //
            let src_blocks_group = h5_file
                .group("02fmri_segment_trials")?
                .group("blocks_std")
                .context("failed to open input group /02fmri_segment_trials")?;

            let blocks_parent_group = format!("{}/{}", MVMD_CRATE_GROUP, ALL_BLOCKS_GROUP);
            let blocks_parent_group = ensure_path(&h5_file, &blocks_parent_group, cfg.force)
                .context("Failed to prepare deep HDF5 path")?;

            for trial_type in TARGET_TRIAL_TYPES {
                let trial_group_path = format!("/02fmri_segment_trials/blocks_std/{trial_type}");
                if !path_exists(&src_blocks_group, trial_type) {
                    warn!(
                        task_name = task_name,
                        trial_type = trial_type,
                        group_path = %trial_group_path,
                        "trial type group not found in blocks_std, skipping"
                    );
                    continue;
                }

                let trial_group = src_blocks_group.group(trial_type)?;
                let blocks_available: Vec<String> = trial_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if blocks_available.is_empty() {
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
                    num_blocks = blocks_available.len(),
                    "starting block MVMD decomposition"
                );

                for (block_idx, block_name) in blocks_available.iter().enumerate() {
                    let input_block_dataset_path = format!("{trial_group_path}/{block_name}");
                    let block_name_path =
                        format!("/{MVMD_CRATE_GROUP}/{ALL_BLOCKS_GROUP}/{trial_type}/{block_name}");


                    // Check if output group already exists (/04mvmd/blocks_std/block_X/)
                    let block_already_done = !cfg.force
                        && path_exists(
                            &h5_file,
                            format!("{}/center_frequencies", block_name_path).as_str(),
                        );
                    if block_already_done {
                        let existing_block_group =
                            h5_file.group(&block_name_path).with_context(|| {
                                format!("failed to open existing output group {block_name_path}")
                            })?;

                        sync_center_frequencies_attr_from_group(&existing_block_group).with_context(|| {
                            format!(
                                "failed to sync center_frequencies attribute to {block_name_path}/modes"
                            )
                        })?;

                        debug!(
                            task_name = task_name,
                            trial_type = trial_type,
                            block = block_name,
                            block_idx = block_idx,
                            num_blocks = blocks_available.len(),
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
                                num_blocks = blocks_available.len(),
                                reason = "unexpected_block_shape",
                                shape = ?block_signal_f32.shape(),
                                "skipping mvmd block decompositin due to unexpected signal shape"
                            );
                            continue;
                        }
                    };
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
                    let block_mvmd = MVMD::from_dataframe(&block_df, MVMD_ALPHA, sampling_rate)?
                        .with_admm_config(admm_config.clone());

                    // Run actual decomposition
                    let block_start = Instant::now();
                    let block_decomposition = block_mvmd.decompose(num_modes);
                    let block_mvmd_duration_ms = block_start.elapsed().as_millis();

                    debug!(
                        group_path = %block_name_path,
                        force = cfg.force,
                        "opening output block group"
                    );

                    let block_group = ensure_path(&h5_file, &block_name_path, cfg.force)
                        .with_context(|| {
                            format!("failed to open/create output group {block_name_path}")
                        })?; // /04mvmd/blocks_std/block_X/

                    let block_write_start = Instant::now();

                    write_attrs(
                        &block_group,
                        &[H5Attr::u32(
                            "num_iterations",
                            block_decomposition.num_iterations as u32,
                        )],
                    )?;

                    debug!(dataset_path = %format!("{block_name_path}/modes"), "writing output dataset");

                    let bm_shape = block_decomposition.modes.shape();
                    let modes_ds = prepare_dataset::<f64>(
                        &block_group,
                        "modes",
                        &[bm_shape[0], bm_shape[1], bm_shape[2]],
                    )?;
                    let modes_metadata = vec![H5Attr {
                        name: "center_frequencies".to_string(),
                        value: H5AttrValue::F64Slice(
                            block_decomposition
                                .center_frequencies
                                .as_slice()
                                .unwrap()
                                .to_vec(),
                        ),
                    }];
                    modes_ds.write_raw(block_decomposition.modes.as_slice().unwrap())?;
                    write_attrs(&modes_ds, &modes_metadata)?;

                    let bcf_shape = block_decomposition.frequency_traces.shape();
                    debug!(dataset_path = %format!("{block_name_path}/frequency_traces"), "writing output dataset");
                    let freq_traces_ds = prepare_dataset::<f64>(
                        &block_group,
                        "frequency_traces",
                        &[bcf_shape[0], bcf_shape[1]],
                    )?;
                    freq_traces_ds
                        .write_raw(block_decomposition.frequency_traces.as_slice().unwrap())?;

                    let center_freqs_shape = &[block_decomposition.center_frequencies.len()];
                    let center_freqs_ds = prepare_dataset::<f64>(
                        &block_group,
                        "center_frequencies",
                        center_freqs_shape,
                    )?;
                    debug!(dataset_path = %format!("{block_name_path}/center_frequencies"), "writing output dataset");

                    write_mvmd_algorithm_attrs_if_missing(
                        &block_group,
                        MVMD_ALPHA,
                        sampling_rate,
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
                    num_blocks = blocks_available.len(),
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
            let mvmd_blocks_roi_group = ensure_path(
                &mvmd_group,
                ALL_BLOCKS_GROUP_ROI_STRATIFIED,
                cfg.force,
            )
            .with_context(|| {
                format!(
                    "failed to open/create output group /{MVMD_CRATE_GROUP}/{ALL_BLOCKS_GROUP_ROI_STRATIFIED}"
                )
            })?;

            for trial_type in TARGET_TRIAL_TYPES {
                let trial_group_path = format!("/02fmri_segment_trials/blocks_std/{trial_type}");
                if !path_exists(&src_blocks_group, trial_type) {
                    warn!(
                        task_name = task_name,
                        trial_type = trial_type,
                        group_path = %trial_group_path,
                        "trial type group not found in blocks_std, skipping"
                    );
                    continue;
                }

                let trial_group = src_blocks_group.group(trial_type)?;
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
                    n_target_rois = roi_row_indices.len(),
                    "starting ROI-only block MVMD decomposition"
                );

                for (block_idx, block_name) in block_names.iter().enumerate() {
                    let input_block_dataset_path = format!("{trial_group_path}/{block_name}");
                    let output_block_name_path = format!(
                        "/{MVMD_CRATE_GROUP}/{ALL_BLOCKS_GROUP_ROI_STRATIFIED}/{trial_type}/{block_name}"
                    );

                    let existing_block_group =
                        h5_file.group(&output_block_name_path).with_context(|| {
                            format!("failed to open existing output group {output_block_name_path}")
                        })?;

                    // roi fingerprint config same as HDF5 attribute
                    let roi_fingerprint_unchanged = check_roi_fingerprint(
                        &existing_block_group,
                        &roi_selection_fingerprint,
                        &output_block_name_path,
                    )
                    .is_ok();
                    let roi_block_already_done = !cfg.force
                        && path_exists(
                            &h5_file,
                            format!("{}/center_frequencies", output_block_name_path).as_str(),
                        )
                        && roi_fingerprint_unchanged;

                    if roi_block_already_done {
                        sync_center_frequencies_attr_from_group(&existing_block_group)
                            .with_context(|| {
                                format!(
                                    "failed to sync center_frequencies attribute to {output_block_name_path}/modes"
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

                    debug!(dataset_path = %input_block_dataset_path, "reading input dataset");
                    let input_ds = trial_group.dataset(block_name).with_context(|| {
                        format!("failed to open input dataset {input_block_dataset_path}")
                    })?;
                    let block_signal_f32: Array2<f32> = input_ds.read_2d()?;

                    // Read metadata attributes from the input dataset
                    let onset_s: f64 = input_ds.attr("onset_s")?.read_scalar()?;
                    let block_end_s: f64 = input_ds.attr("block_end_s")?.read_scalar()?;

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
                    let roi_block_mvmd =
                        MVMD::from_dataframe(&roi_block_df, MVMD_ALPHA, sampling_rate)?
                            .with_admm_config(admm_config.clone());

                    let block_start = Instant::now();
                    let roi_block_decomposition = roi_block_mvmd.decompose(num_modes);
                    let block_mvmd_duration_ms = block_start.elapsed().as_millis();

                    let block_name_group = ensure_path(
                        &h5_file,
                        &output_block_name_path,
                        cfg.force,
                    )
                    .with_context(|| {
                        format!("failed to open/create output group {output_block_name_path}")
                    })?;

                    let block_write_start = Instant::now();

                    write_attrs(
                        &block_name_group,
                        &[
                            H5Attr::u32(
                                "num_iterations",
                                roi_block_decomposition.num_iterations as u32,
                            ),
                            H5Attr::u32("n_rois", roi_indices_u32.len() as u32),
                            H5Attr::string("roi_labels", roi_labels.join(",")),
                            H5Attr::string("channels", roi_block_decomposition.channels.join(",")),
                            H5Attr::string("roi_matched_regions", roi_matched_regions.join(",")),
                            H5Attr::string("roi_selection_name", &roi_selection_name),
                            H5Attr::string("roi_selection_fingerprint", &roi_selection_fingerprint),
                        ],
                    )?;

                    let bm_shape = roi_block_decomposition.modes.shape();
                    let bm_ds = prepare_dataset::<f64>(
                        &block_name_group,
                        "modes",
                        &[bm_shape[0], bm_shape[1], bm_shape[2]],
                    )?;
                    let bm_metadata = vec![H5Attr {
                        name: "center_frequencies".to_string(),
                        value: H5AttrValue::F64Slice(
                            roi_block_decomposition
                                .center_frequencies
                                .as_slice()
                                .unwrap()
                                .to_vec(),
                        ),
                    }];
                    bm_ds.write_raw(roi_block_decomposition.modes.as_slice().unwrap())?;
                    write_attrs(&bm_ds, &bm_metadata)?;

                    let bft_shape = roi_block_decomposition.frequency_traces.shape();
                    let bft_ds = prepare_dataset::<f64>(
                        &block_name_group,
                        "frequency_traces",
                        &[bft_shape[0], bft_shape[1]],
                    )?;
                    bft_ds
                        .write_raw(roi_block_decomposition.frequency_traces.as_slice().unwrap())?;

                    let bcf_shape = &[roi_block_decomposition.center_frequencies.len()];
                    let bcf_ds =
                        prepare_dataset::<f64>(&block_name_group, "center_frequencies", bcf_shape)?;
                    bcf_ds.write_raw(
                        roi_block_decomposition
                            .center_frequencies
                            .as_slice()
                            .unwrap(),
                    )?;

                    let bi_shape = &[roi_indices_u32.len()];
                    let bi_ds = prepare_dataset::<u32>(&block_name_group, "roi_indices", bi_shape)?;
                    bi_ds.write_raw(&roi_indices_u32)?;

                    write_mvmd_algorithm_attrs_if_missing(
                        &block_name_group,
                        MVMD_ALPHA,
                        sampling_rate,
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
