use anyhow::Result;
use ndarray::{Array2, Axis, concatenate};
use rayon::prelude::*;
use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};
use tracing::{debug, info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
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
    // Frequency bounds:
    //   - f_min = 0.008 Hz: below the slowest resting-state low-frequency fluctuations (~0.01 Hz)
    //   - f_max = 0.5 Hz: near Nyquist (0.625 Hz at TR=0.8 s); captures physiological noise bands
    //     (respiratory ~0.3 Hz) that are relevant for task-based fMRI denoising
    //
    // Log-spacing is used because BOLD dynamics span two decades (0.01–1 Hz). Linear spacing
    // would oversample the high-frequency end and undersample the low end. Log-spacing gives
    // equal relative resolution per octave across the whole band.
    //
    // TODO: consider splitting into separate resting-state (0.008–0.1 Hz) and task
    // (0.008–0.5 Hz) scale grids once the two paradigms are processed separately.
    let tr: f64 = 0.8;
    let w0: f64 = 6.0;
    let f_min: f64 = 0.008; // Hz
    let f_max: f64 = 0.5; // Hz — below Nyquist (0.625 Hz at TR=0.8 s)
    let n_scales: usize = 64;
    let scales: Vec<f64> = (0..n_scales)
        .map(|i| {
            let f = f_min * (f_max / f_min).powf(i as f64 / (n_scales - 1) as f64);
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

    info!(
        parcellated_ts_dir = %cfg.parcellated_ts_dir.display(),
        force = cfg.force,
        "starting fMRI CWT pipeline"
    );

    /////////////////////
    // Get Time Series //
    /////////////////////

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.parcellated_ts_dir)?
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

                let h5_file = open_or_create(&file_path)?;
                let cwt_group = open_or_create_group(&h5_file, "cwt", false)?;

                //////////////////////////////////////////////
                // Whole-band — raw (tcp_timeseries_raw)   //
                //////////////////////////////////////////////

                let wb_raw_done = !cfg.force && cwt_group.group("whole-band").is_ok();
                if wb_raw_done {
                    info!(
                        task_name = task_name,
                        "whole-band raw scalogram already computed, skipping (use --force to recompute)"
                    );
                } else {
                    let dataset = h5_file.dataset("tcp_timeseries_raw")?;
                    let data_f32: Array2<f32> = dataset.read_2d()?;
                    let [n_channels, n_timepoints] = match data_f32.shape() {
                        &[r, c] => [r, c],
                        _ => anyhow::bail!(
                            "expected 2D timeseries, got shape {:?}",
                            data_f32.shape()
                        ),
                    };
                    let data_f64 = data_f32.mapv(|val| val as f64);

                    info!(
                        task_name = task_name,
                        n_channels = n_channels,
                        n_timepoints = n_timepoints,
                        "starting whole-band raw scalogram"
                    );

                    let cwt_start = Instant::now();
                    let (scalogram_data, shape) = cwt_scalogram(&data_f64);
                    let cwt_duration_ms = cwt_start.elapsed().as_millis();

                    let write_start = Instant::now();
                    let wb_group = open_or_create_group(&cwt_group, "whole-band", cfg.force)?;
                    write_dataset(&wb_group, "scalogram", &scalogram_data, &shape, None)?;
                    let write_duration_ms = write_start.elapsed().as_millis();

                    info!(
                        task_name = task_name,
                        n_channels = n_channels,
                        n_timepoints = n_timepoints,
                        output_shape = ?shape,
                        cwt_duration_ms = cwt_duration_ms,
                        write_duration_ms = write_duration_ms,
                        output_file = %file_path.display(),
                        "whole-band raw scalogram complete"
                    );
                }

                ////////////////////////////////////////////////////
                // Whole-band — standardized (tcp_timeseries_std) //
                ////////////////////////////////////////////////////

                match h5_file.dataset("tcp_timeseries_standardized") {
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            "no tcp_timeseries_standardized found, skipping standardized whole-band scalogram"
                        );
                    }
                    Ok(std_dataset) => {
                        let cwt_std_group =
                            open_or_create_group(&h5_file, "cwt_standardized", false)?;
                        let wb_std_done = !cfg.force && cwt_std_group.group("whole-band").is_ok();

                        if wb_std_done {
                            info!(
                                task_name = task_name,
                                "whole-band standardized scalogram already computed, skipping (use --force to recompute)"
                            );
                        } else {
                            let data_f32: Array2<f32> = std_dataset.read_2d()?;
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
                                "starting whole-band standardized scalogram"
                            );

                            let cwt_start = Instant::now();
                            let (scalogram_data, shape) = cwt_scalogram(&data_f64);
                            let cwt_duration_ms = cwt_start.elapsed().as_millis();

                            let write_start = Instant::now();
                            let wb_std_group =
                                open_or_create_group(&cwt_std_group, "whole-band", cfg.force)?;
                            write_dataset(
                                &wb_std_group,
                                "scalogram",
                                &scalogram_data,
                                &shape,
                                None,
                            )?;
                            let write_duration_ms = write_start.elapsed().as_millis();

                            info!(
                                task_name = task_name,
                                n_channels = n_channels,
                                n_timepoints = n_timepoints,
                                output_shape = ?shape,
                                cwt_duration_ms = cwt_duration_ms,
                                write_duration_ms = write_duration_ms,
                                output_file = %file_path.display(),
                                "whole-band standardized scalogram complete"
                            );
                        }
                    }
                }

                ///////////////////////////////////////////
                // Block-level — raw (blocks group)      //
                ///////////////////////////////////////////

                match h5_file.group("blocks") {
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            "no blocks group found, skipping raw block scalograms"
                        );
                    }
                    Ok(blocks_group) => {
                        let block_names: Vec<String> = blocks_group
                            .member_names()?
                            .into_iter()
                            .filter(|n| n.starts_with("block_"))
                            .collect();

                        if block_names.is_empty() {
                            debug!(
                                task_name = task_name,
                                "blocks group is empty, skipping raw block scalograms"
                            );
                        } else {
                            info!(
                                task_name = task_name,
                                num_blocks = block_names.len(),
                                "starting raw block scalograms"
                            );

                            let cwt_blocks_group =
                                open_or_create_group(&cwt_group, "blocks", cfg.force)?;

                            for (block_idx, block_name) in block_names.iter().enumerate() {
                                if !cfg.force && cwt_blocks_group.group(block_name).is_ok() {
                                    debug!(
                                        task_name = task_name,
                                        block = block_name,
                                        block_idx = block_idx,
                                        num_blocks = block_names.len(),
                                        "raw block scalogram already computed, skipping (use --force to recompute)"
                                    );
                                    continue;
                                }

                                let block_group = blocks_group.group(block_name)?;
                                let cortical: Array2<f32> =
                                    block_group.dataset("cortical_raw")?.read_2d()?;
                                let subcortical: Array2<f32> =
                                    block_group.dataset("subcortical_raw")?.read_2d()?;
                                let block_signal_f32 =
                                    concatenate(Axis(0), &[cortical.view(), subcortical.view()])?;
                                let [block_channels, block_timepoints] = match block_signal_f32
                                    .shape()
                                {
                                    &[r, c] => [r, c],
                                    _ => {
                                        error_count += 1;
                                        warn!(
                                            task_name = task_name,
                                            block = block_name,
                                            block_idx = block_idx,
                                            num_blocks = block_names.len(),
                                            reason = "unexpected_block_shape",
                                            shape = ?block_signal_f32.shape(),
                                            "skipping raw block scalogram due to unexpected signal shape"
                                        );
                                        continue;
                                    }
                                };
                                let block_signal_f64 = block_signal_f32.mapv(|val| val as f64);

                                info!(
                                    task_name = task_name,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    "starting raw block scalogram"
                                );

                                let block_cwt_start = Instant::now();
                                let (scalogram_data, shape) = cwt_scalogram(&block_signal_f64);
                                let block_cwt_duration_ms = block_cwt_start.elapsed().as_millis();

                                let block_write_start = Instant::now();
                                let cwt_block_group =
                                    open_or_create_group(&cwt_blocks_group, block_name, cfg.force)?;
                                write_dataset(
                                    &cwt_block_group,
                                    "scalogram",
                                    &scalogram_data,
                                    &shape,
                                    None,
                                )?;
                                let block_write_duration_ms =
                                    block_write_start.elapsed().as_millis();

                                info!(
                                    task_name = task_name,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    output_shape = ?shape,
                                    cwt_duration_ms = block_cwt_duration_ms,
                                    write_duration_ms = block_write_duration_ms,
                                    "raw block scalogram complete"
                                );
                            }

                            info!(
                                task_name = task_name,
                                num_blocks = block_names.len(),
                                "finished all raw block scalograms"
                            );
                        }
                    }
                }

                ///////////////////////////////////////////////////////////
                // Block-level — standardized (blocks_standardized group) //
                ///////////////////////////////////////////////////////////

                match h5_file.group("blocks_standardized") {
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            "no blocks_standardized group found, skipping standardized block scalograms"
                        );
                    }
                    Ok(blocks_std_group) => {
                        let block_names: Vec<String> = blocks_std_group
                            .member_names()?
                            .into_iter()
                            .filter(|n| n.starts_with("block_"))
                            .collect();

                        if block_names.is_empty() {
                            debug!(
                                task_name = task_name,
                                "blocks_standardized group is empty, skipping standardized block scalograms"
                            );
                        } else {
                            info!(
                                task_name = task_name,
                                num_blocks = block_names.len(),
                                "starting standardized block scalograms"
                            );

                            let cwt_std_group =
                                open_or_create_group(&h5_file, "cwt_standardized", false)?;
                            let cwt_std_blocks_group =
                                open_or_create_group(&cwt_std_group, "blocks", cfg.force)?;

                            for (block_idx, block_name) in block_names.iter().enumerate() {
                                if !cfg.force && cwt_std_blocks_group.group(block_name).is_ok() {
                                    debug!(
                                        task_name = task_name,
                                        block = block_name,
                                        block_idx = block_idx,
                                        num_blocks = block_names.len(),
                                        "standardized block scalogram already computed, skipping (use --force to recompute)"
                                    );
                                    continue;
                                }

                                let block_group = blocks_std_group.group(block_name)?;
                                let cortical: Array2<f32> =
                                    block_group.dataset("cortical_standardized")?.read_2d()?;
                                let subcortical: Array2<f32> =
                                    block_group.dataset("subcortical_standardized")?.read_2d()?;
                                let block_signal_f32 =
                                    concatenate(Axis(0), &[cortical.view(), subcortical.view()])?;
                                let [block_channels, block_timepoints] = match block_signal_f32
                                    .shape()
                                {
                                    &[r, c] => [r, c],
                                    _ => {
                                        error_count += 1;
                                        warn!(
                                            task_name = task_name,
                                            block = block_name,
                                            block_idx = block_idx,
                                            num_blocks = block_names.len(),
                                            reason = "unexpected_block_shape",
                                            shape = ?block_signal_f32.shape(),
                                            "skipping standardized block scalogram due to unexpected signal shape"
                                        );
                                        continue;
                                    }
                                };
                                let block_signal_f64 = block_signal_f32.mapv(|val| val as f64);

                                info!(
                                    task_name = task_name,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    "starting standardized block scalogram"
                                );

                                let block_cwt_start = Instant::now();
                                let (scalogram_data, shape) = cwt_scalogram(&block_signal_f64);
                                let block_cwt_duration_ms = block_cwt_start.elapsed().as_millis();

                                let block_write_start = Instant::now();
                                let cwt_std_block_group = open_or_create_group(
                                    &cwt_std_blocks_group,
                                    block_name,
                                    cfg.force,
                                )?;
                                write_dataset(
                                    &cwt_std_block_group,
                                    "scalogram",
                                    &scalogram_data,
                                    &shape,
                                    None,
                                )?;
                                let block_write_duration_ms =
                                    block_write_start.elapsed().as_millis();

                                info!(
                                    task_name = task_name,
                                    block = block_name,
                                    block_idx = block_idx,
                                    num_blocks = block_names.len(),
                                    n_channels = block_channels,
                                    n_timepoints = block_timepoints,
                                    output_shape = ?shape,
                                    cwt_duration_ms = block_cwt_duration_ms,
                                    write_duration_ms = block_write_duration_ms,
                                    "standardized block scalogram complete"
                                );
                            }

                            info!(
                                task_name = task_name,
                                num_blocks = block_names.len(),
                                "finished all standardized block scalograms"
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
