use anyhow::{Context, Result};
use hdf5::types::VarLenUnicode;
use ndarray::Array2;
use rayon::prelude::*;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands;
use utils::hdf5_io::{open_or_create, open_or_create_group, write_dataset};

use scirs2_signal::wavelets::{complex_morlet, scalogram};

/// Compute the CWT scalogram (squared magnitude) for each channel using the complex Morlet wavelet.
///
/// Returns a flat row-major buffer and the 3D shape `[n_channels, n_scales, n_timepoints]`.
/// The buffer layout is: for each channel, for each scale, the power values over time.
fn cwt_scalogram(signal: &Array2<f64>) -> (Vec<f64>, [usize; 3]) {
    let n_channels = signal.nrows();
    let n_timepoints = signal.ncols();

    // Scale grid derived from target frequencies via scale = w0 / (f * dt),
    // where w0=6.0 is the Morlet center frequency and dt=0.8 s is the TR.
    //
    // Frequency bounds come from `utils::frequency_bands::SLOW_BANDS` so the CWT
    // scalogram, MVMD grid, and HHT spectrum bins all share one project-wide
    // analysed BOLD frequency range.
    //
    // Log-spacing is used because BOLD dynamics span two decades. Linear spacing
    // would oversample the high-frequency end and undersample the low end.
    let tr: f64 = 0.8;
    let w0: f64 = 6.0;
    let f_min: f64 = frequency_bands::f_min();
    let f_max: f64 = frequency_bands::f_max();
    let n_scales: usize = 224; // input height of DenseNet201
    let scales: Vec<f64> = (0..n_scales)
        .map(|i| {
            // Linearly interpolate in log-space
            let f = f_min * (f_max / f_min).powf(i as f64 / (n_scales - 1) as f64);
            // Convert frequency to scale
            w0 / (f * tr)
        })
        .collect();

    // Parallelize over channels: each channel's scalogram is independent. Contiguous
    // copy avoids requiring row-major views to be Sync across threads.
    let channels_rows: Vec<Vec<f64>> = (0..n_channels).map(|ch| signal.row(ch).to_vec()).collect();

    let per_channel: Vec<Vec<f64>> = channels_rows
        .par_iter()
        .enumerate()
        .map(|(ch, channel_slice)| {
            if ch % 50 == 0 {
                debug!(
                    channel = ch,
                    n_channels = n_channels,
                    n_timepoints = n_timepoints,
                    n_scales = n_scales,
                    "computing scalogram for channel"
                );
            }

            // symmetry=0.0 → standard symmetric Morlet (non-zero cancels the Gaussian envelope)
            let scalo = scalogram(
                channel_slice,
                |points, scale| complex_morlet(points, 6.0, 1.0, 0.0, scale),
                &scales,
                Some(false),
            )
            .expect("scalogram computation should succeed");

            let mut out: Vec<f64> = Vec::with_capacity(n_scales * n_timepoints);
            for scale_row in &scalo {
                out.extend_from_slice(scale_row);
            }
            out
        })
        .collect();

    let mut flat: Vec<f64> = Vec::with_capacity(n_channels * n_scales * n_timepoints);
    for ch_buf in per_channel {
        flat.extend(ch_buf);
    }

    (flat, [n_channels, n_scales, n_timepoints])
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    let target_trial_types = vec!["face"];
    // let group_name_blocks_raw = "blocks_raw";
    let group_name_blocks_std = "blocks_std";

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        "starting fMRI CWT pipeline"
    );

    /////////////////////
    // Get Time Series //
    /////////////////////

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

                debug!(
                    output_file = %file_path.display(),
                    "opening subject HDF5 file for CWT"
                );
                let h5_file = open_or_create(&file_path)?;
                debug!(group_path = "/03cwt", force = false, "opening output group");
                let cwt_group = open_or_create_group(&h5_file, "03cwt", false)
                    .context("failed to open/create group /03cwt")?;

                /////////////////////////////////////
                // Full run — Z-score standardized //
                /////////////////////////////////////

                let wb_std_done = !cfg.force && cwt_group.dataset("full_run_std").is_ok();

                if wb_std_done {
                    info!(
                        task_name = task_name,
                        "whole-signal standardized scalogram already computed, skipping (use --force to recompute)"
                    );
                } else if task_name == "hammerAP" {
                    warn!(
                        task_name = task_name,
                        "We only compute block-wise CWT for task-based fMRI - Should yield the same results as doing full CWT and chunking later."
                    );
                } else {
                    info!(
                        task_name = task_name,
                        signal_type = "full_run",
                        input_dataset = "/01fmri_parcellation/full_run_std",
                        output_dataset = "/03cwt/full_run_std",
                        "starting CWT decomposition"
                    );
                    debug!(
                        dataset_path = "/01fmri_parcellation/full_run_std",
                        "reading input dataset"
                    );
                    let parc_group = h5_file
                        .group("01fmri_parcellation")
                        .context("failed to open group /01fmri_parcellation")?;
                    let dataset = parc_group
                        .dataset("full_run_std")
                        .context("failed to open dataset /01fmri_parcellation/full_run_std")?;
                    let data_f32: Array2<f32> = dataset.read_2d()?;
                    let [n_channels, n_timepoints] = match data_f32.shape() {
                        &[r, c] => [r, c],
                        _ => anyhow::bail!(
                            "expected 2D standardized timeseries, got shape {:?}",
                            data_f32.shape()
                        ),
                    };
                    let data_f64 = data_f32.mapv(|val| val as f64);

                    info!(
                        task_name = task_name,
                        n_channels = n_channels,
                        n_timepoints = n_timepoints,
                        "starting whole-signal standardized scalogram"
                    );

                    let cwt_start = Instant::now();
                    let (scalogram_data, shape) = cwt_scalogram(&data_f64);
                    let cwt_duration_ms = cwt_start.elapsed().as_millis();

                    let write_start = Instant::now();
                    debug!(
                        dataset_path = "/03cwt/full_run_std",
                        "writing output dataset"
                    );
                    write_dataset(&cwt_group, "full_run_std", &scalogram_data, &shape, None)?;
                    let write_duration_ms = write_start.elapsed().as_millis();

                    info!(
                        task_name = task_name,
                        n_channels = n_channels,
                        n_timepoints = n_timepoints,
                        output_shape = ?shape,
                        cwt_duration_ms = cwt_duration_ms,
                        write_duration_ms = write_duration_ms,
                        output_file = %file_path.display(),
                        "whole-signal standardized scalogram complete"
                    );
                }

                ///////////////////////////////////////////////////////////
                // Block-level — standardized (blocks_standardized group) //
                ///////////////////////////////////////////////////////////

                info!(
                    task_name = task_name,
                    signal_type = "blocks",
                    input_group = "/02fmri_segment_trials/blocks_std",
                    output_group = "/03cwt/blocks_std",
                    "starting CWT decomposition"
                );
                let segment_root = h5_file.group("02fmri_segment_trials").ok();
                let blocks_std_group_opt = segment_root
                    .as_ref()
                    .and_then(|g| g.group(group_name_blocks_std).ok());
                match blocks_std_group_opt {
                    None => {
                        debug!(
                            task_name = task_name,
                            "no /02fmri_segment_trials/blocks_std group found, skipping standardized block scalograms"
                        );
                    }
                    Some(blocks_std_group) => {
                        debug!(
                            group_path = "/03cwt/blocks_std",
                            force = cfg.force,
                            "opening output blocks group"
                        );
                        let cwt_blocks_group =
                            open_or_create_group(&cwt_group, group_name_blocks_std, cfg.force)
                                .context("failed to open/create group /03cwt/blocks_std")?;

                        for trial_type in &target_trial_types {
                            let trial_group_path =
                                format!("/02fmri_segment_trials/blocks_std/{trial_type}");
                            let trial_group = match blocks_std_group.group(trial_type) {
                                Ok(g) => g,
                                Err(_) => {
                                    debug!(
                                        task_name = task_name,
                                        trial_type = trial_type,
                                        group_path = %trial_group_path,
                                        "trial type group not found in blocks_std, skipping"
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
                                "starting standardized block scalograms"
                            );

                            for (block_idx, block_name) in block_names.iter().enumerate() {
                                let input_block_dataset_path =
                                    format!("{trial_group_path}/{block_name}");
                                let output_block_dataset_path =
                                    format!("/03cwt/blocks_std/{block_name}");

                                info!(
                                    task_name = task_name,
                                    trial_type = trial_type,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    signal_type = "block",
                                    input_dataset = %input_block_dataset_path,
                                    output_dataset = %output_block_dataset_path,
                                    "starting CWT decomposition"
                                );

                                // Check if output dataset already exists
                                if !cfg.force && cwt_blocks_group.dataset(block_name).is_ok() {
                                    debug!(
                                        task_name = task_name,
                                        trial_type = trial_type,
                                        block = block_name,
                                        block_idx = block_idx,
                                        num_blocks = block_names.len(),
                                        "std block scalogram already computed, skipping (use --force to recompute)"
                                    );
                                    continue;
                                }

                                // Read Input Dataset & Attributes
                                debug!(
                                    dataset_path = %input_block_dataset_path,
                                    "reading input dataset"
                                );
                                let input_ds =
                                    trial_group.dataset(block_name).with_context(|| {
                                        format!("failed to open dataset {input_block_dataset_path}")
                                    })?;

                                // Read the single dataset instead of concatenating cortical/subcortical
                                let block_signal_f32: Array2<f32> = input_ds.read_2d()?;

                                // Read metadata attributes from the input dataset
                                let onset_s: f64 = input_ds.attr("onset_s")?.read_scalar()?;
                                let block_end_s: f64 =
                                    input_ds.attr("block_end_s")?.read_scalar()?;

                                let [block_channels, block_timepoints] = match block_signal_f32
                                    .shape()
                                {
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
                                            "skipping std block scalogram due to unexpected signal shape"
                                        );
                                        continue;
                                    }
                                };
                                let block_signal_f64 = block_signal_f32.mapv(|val| val as f64);

                                info!(
                                    task_name = task_name,
                                    trial_type = trial_type,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    "starting std block scalogram"
                                );

                                // Compute CWT
                                let block_cwt_start = Instant::now();
                                let (scalogram_data, shape) = cwt_scalogram(&block_signal_f64);
                                let block_cwt_duration_ms = block_cwt_start.elapsed().as_millis();

                                // Write Output Dataset & Attach Attributes
                                let block_write_start = Instant::now();

                                if cfg.force {
                                    debug!(
                                        dataset_path = %output_block_dataset_path,
                                        "unlinking existing output dataset"
                                    );
                                    let _ = cwt_blocks_group.unlink(block_name);
                                }

                                // Since we are writing the output directly as a dataset (not a group),
                                // we bypass your custom `write_dataset` helper so we can attach attributes directly.
                                let output_ds = cwt_blocks_group
                                    .new_dataset::<f32>() // Assuming scalogram_data is f32; change to f64 if needed
                                    .shape(shape.clone())
                                    .create(block_name.as_str())
                                    .with_context(|| {
                                        format!(
                                            "failed to create output dataset {output_block_dataset_path}"
                                        )
                                    })?;

                                // Write the raw array data (assumes scalogram_data is C-contiguous standard layout)
                                let data_view = ndarray::ArrayView::from_shape(
                                    shape.clone(),
                                    &scalogram_data,
                                )
                                .expect("Failed to reshape flat scalogram data into ND-array view");
                                debug!(
                                    dataset_path = %output_block_dataset_path,
                                    "writing output dataset"
                                );
                                output_ds.write(&data_view)?;

                                // Write metadata attributes to the new CWT dataset
                                let trial_type_val: VarLenUnicode = trial_type.parse()?;
                                output_ds
                                    .new_attr::<VarLenUnicode>()
                                    .shape(())
                                    .create("trial_type")?
                                    .as_writer()
                                    .write_scalar(&trial_type_val)?;
                                output_ds
                                    .new_attr::<f64>()
                                    .shape(())
                                    .create("onset_s")?
                                    .as_writer()
                                    .write_scalar(&onset_s)?;
                                output_ds
                                    .new_attr::<f64>()
                                    .shape(())
                                    .create("block_end_s")?
                                    .as_writer()
                                    .write_scalar(&block_end_s)?;

                                let block_write_duration_ms =
                                    block_write_start.elapsed().as_millis();

                                info!(
                                    task_name = task_name,
                                    trial_type = trial_type,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    output_shape = ?shape,
                                    cwt_duration_ms = block_cwt_duration_ms,
                                    write_duration_ms = block_write_duration_ms,
                                    "std block scalogram complete"
                                );
                            }

                            info!(
                                task_name = task_name,
                                trial_type = trial_type,
                                num_blocks = block_names.len(),
                                "finished all standardized block scalograms for trial type"
                            );
                        }
                    }
                }
                Ok(())
            })();

            if let Err(e) = file_result {
                error_count += 1;
                warn!(
                    file = %file_path.display(),
                    error = %e,
                    "skipping file due to HDF5 error"
                );
            }
        }
    }

    if error_count > 0 {
        warn!(
            error_count = error_count,
            "some blocks were skipped due to errors"
        );
    }

    let total_duration_ms = run_start.elapsed().as_millis();
    info!(
        error_count = error_count,
        total_duration_ms = total_duration_ms,
        "CWT scalogram pipeline complete"
    );

    Ok(())
}
