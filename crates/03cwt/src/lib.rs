use anyhow::{Context, Result};
use ndarray::Array2;
use rayon::prelude::*;
use std::f64::consts::PI;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};
use utils::bids_filename::{BidsFilename, filter_directory_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands;
use utils::hdf5_io::{
    H5Attr, H5AttrValue, ensure_path, open_or_create_group, path_exists, prepare_dataset,
    write_attrs,
};

use scirs2_signal::wavelets::{complex_morlet, scalogram};

// HDF5 groups and datasets
const CWT_CRATE_GROUP: &str = "03cwt";
const FULL_RUN_DATASET: &str = "full_run_std";
const BLOCKS_GROUP: &str = "blocks_std";

const TARGET_TRIAL_TYPES: &[&str] = &["face"];

// Other params
const ANGULAR_FREQ: f64 = 6.0; // Angular center frequency
const NUM_SCALES: usize = 224; // DenseNet201 height

/// Compute the CWT scalogram (squared magnitude) for each channel using the complex Morlet wavelet.
///
/// Returns a flat row-major buffer and the 3D shape `[n_channels, n_scales, n_timepoints]`.
/// The buffer layout is: for each channel, for each scale, the power values over time.
fn cwt_scalogram(cfg: &AppConfig, signal: &Array2<f64>) -> (Vec<f64>, [usize; 3]) {
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
    let tr: f64 = 1.0 / cfg.task_sampling_rate; // Your sampling period (1/fs)
    let f_min: f64 = frequency_bands::f_min(); // Target min frequency in Hz
    let f_max: f64 = frequency_bands::f_max(); // Target max frequency in Hz

    let scales: Vec<f64> = (0..NUM_SCALES)
        .map(|i| {
            // Calculate the target frequency for this step in log-space
            // We go from f_max to f_min so that the resulting scales
            // are in ascending order (standard for many CNN inputs).
            let f = f_max * (f_min / f_max).powf(i as f64 / (NUM_SCALES - 1) as f64);

            // Convert physical frequency (Hz) to CWT scale
            // Formula: s = w0 / (2 * PI * f * tr)
            ANGULAR_FREQ / (2.0 * PI * f * tr)
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
                    n_scales = NUM_SCALES,
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

            let mut out: Vec<f64> = Vec::with_capacity(NUM_SCALES * n_timepoints);
            for scale_row in &scalo {
                out.extend_from_slice(scale_row);
            }
            out
        })
        .collect();

    let mut flat: Vec<f64> = Vec::with_capacity(n_channels * NUM_SCALES * n_timepoints);
    for ch_buf in per_channel {
        flat.extend(ch_buf);
    }

    (flat, [n_channels, NUM_SCALES, n_timepoints])
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

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

        let available_resting_state_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("restAP"))
                .expect("Failed to read the directory");

        let available_hammer_task_ts: Vec<BidsFilename> =
            filter_directory_bids_files(dir, |bids| bids.get("task") == Some("hammerAP"))
                .expect("Failed to read the directory");

        debug!(
            resting_state_count = available_resting_state_ts.len(),
            hammer_task_count = available_hammer_task_ts.len(),
            "extracted resting-state and task-based timeseries"
        );

        for rs_file in &available_resting_state_ts {
            let task_name = rs_file.get("task").unwrap_or("unknown");
            let path = rs_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;

            let h5_file = hdf5::File::open_rw(&path)?;

            let fr_dataset = format!("{CWT_CRATE_GROUP}/{FULL_RUN_DATASET}");
            let already_done = !cfg.force && path_exists(&h5_file, &fr_dataset);
            if already_done {
                info!(
                    task_name = task_name,
                    "full-run standardized scalogram already computed, skipping (use --force to recompute)"
                );
                continue;
            }

            let cwt_group = ensure_path(&h5_file, CWT_CRATE_GROUP, cfg.force)
                .context("Failed to prepare deep HDF5 path")?;

            info!(
                subject = formatted_id,
                task_name = task_name,
                signal_type = "full_run",
                input_dataset = format!("/01fmri_parcellation/{}", FULL_RUN_DATASET),
                output_dataset = format!("/{}/{}", CWT_CRATE_GROUP, FULL_RUN_DATASET),
                "starting CWT decomposition"
            );
            let parc_group = h5_file
                .group("01fmri_parcellation")
                .context("failed to open group /01fmri_parcellation")?;
            let dataset = parc_group.dataset("full_run_std").context(format!(
                "failed to open dataset /01fmri_parcellation/{}",
                FULL_RUN_DATASET
            ))?;
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
                "starting full-run standardized scalogram"
            );

            let cwt_start = Instant::now();
            let (scalogram_data, shape) = cwt_scalogram(cfg, &data_f64);
            let cwt_duration_ms = cwt_start.elapsed().as_millis();

            let write_start = Instant::now();
            debug!(
                dataset_path = format!("/{}/{}", CWT_CRATE_GROUP, FULL_RUN_DATASET),
                "writing output dataset"
            );

            let fr_ds = prepare_dataset::<f32>(&cwt_group, FULL_RUN_DATASET, &shape)?;
            fr_ds.write_raw(&scalogram_data)?;

            let write_duration_ms = write_start.elapsed().as_millis();

            info!(
                task_name = task_name,
                n_channels = n_channels,
                n_timepoints = n_timepoints,
                output_shape = ?shape,
                cwt_duration_ms = cwt_duration_ms,
                write_duration_ms = write_duration_ms,
                output_file = %path.display(),
                "full-run standardized scalogram complete"
            );
        }

        for task_file in &available_hammer_task_ts {
            let task_name = task_file.get("task").unwrap_or("unknown");

            info!(
                task_name = task_name,
                signal_type = "blocks",
                input_group = format!("/02fmri_segment_trials/{}", BLOCKS_GROUP),
                output_group = format!("/{}/{}", CWT_CRATE_GROUP, BLOCKS_GROUP),
                "starting CWT decomposition"
            );

            let path = task_file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;

            let h5_file = hdf5::File::open_rw(&path)?;
            debug!(
                group_path = format!("/{}", CWT_CRATE_GROUP),
                force = cfg.force,
                "opening output group"
            );

            let trial_block_dataset = format!("02fmri_segment_trials/blocks_std");
            let missing_trial_blocks = !path_exists(&h5_file, &trial_block_dataset);
            if missing_trial_blocks {
                warn!(
                    task_name = task_name,
                    "no /02fmri_segment_trials/blocks_std group found, skipping standardized block scalograms",
                );
                continue;
            }

            let blocks_std_group = h5_file
                .group("02fmri_segment_trials")?
                .group("blocks_std")
                .expect("failed to unwrap blocks group");

            let cwt_blocks_group = format!("{}/{}", CWT_CRATE_GROUP, BLOCKS_GROUP);
            let cwt_blocks_group = ensure_path(&h5_file, &cwt_blocks_group, cfg.force)
                .context("Failed to prepare deep HDF5 path")?;

            for trial_type in TARGET_TRIAL_TYPES {
                let trial_group_path = format!("/02fmri_segment_trials/blocks_std/{trial_type}");
                if !path_exists(&blocks_std_group, trial_type) {
                    warn!(
                        task_name = task_name,
                        trial_type = trial_type,
                        group_path = %trial_group_path,
                        "trial type group not found in blocks_std, skipping"
                    );
                    continue;
                }

                let trial_group = blocks_std_group.group(trial_type)?;
                let block_names: Vec<String> = trial_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    warn!(
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
                    let input_block_dataset_path = format!("{trial_group_path}/{block_name}");
                    let output_block_dataset_path =
                        format!("/{CWT_CRATE_GROUP}/{BLOCKS_GROUP}/{trial_type}/{block_name}");

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
                    let block_already_done = !cfg.force
                        && path_exists(&cwt_blocks_group, &format!("{trial_type}/{block_name}"));
                    if block_already_done {
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
                    let input_ds = trial_group.dataset(block_name).with_context(|| {
                        format!("failed to open dataset {input_block_dataset_path}")
                    })?;

                    // Read the single dataset instead of concatenating cortical/subcortical
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
                    let (scalogram_data, shape) = cwt_scalogram(cfg, &block_signal_f64);
                    let block_cwt_duration_ms = block_cwt_start.elapsed().as_millis();

                    // Write Output Dataset & Attach Attributes
                    let block_write_start = Instant::now();

                    let trial_group = open_or_create_group(&cwt_blocks_group, trial_type, false)
                        .with_context(|| {
                            format!("failed to open/create trial group {trial_type}")
                        })?;

                    if cfg.force {
                        debug!(
                            dataset_path = %output_block_dataset_path,
                            "unlinking existing output dataset"
                        );
                        let _ = trial_group.unlink(block_name);
                    }

                    let output_ds = prepare_dataset::<f32>(&trial_group, &block_name, &shape)
                        .with_context(|| {
                            format!(
                                "failed to create output dataset {trial_group:?}/{block_name:?}"
                            )
                        })?;

                    // Write the raw array data (assumes scalogram_data is C-contiguous standard layout)
                    let data_view = ndarray::ArrayView::from_shape(shape.clone(), &scalogram_data)
                        .expect("Failed to reshape flat scalogram data into ND-array view");
                    debug!(
                        dataset_path = %output_block_dataset_path,
                        "writing output dataset"
                    );
                    output_ds.write(&data_view)?;

                    // Write metadata attributes to the new CWT dataset
                    // let trial_type_val: VarLenUnicode = trial_type.parse()?;
                    let metadata = vec![
                        H5Attr {
                            name: "trial_type".to_string(),
                            value: H5AttrValue::String(trial_type.to_string()),
                        },
                        H5Attr {
                            name: "onset_s".to_string(),
                            value: H5AttrValue::F64(onset_s),
                        },
                        H5Attr {
                            name: "block_end_s".to_string(),
                            value: H5AttrValue::F64(block_end_s),
                        },
                    ];

                    write_attrs(&output_ds, &metadata)?;

                    let block_write_duration_ms = block_write_start.elapsed().as_millis();

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
