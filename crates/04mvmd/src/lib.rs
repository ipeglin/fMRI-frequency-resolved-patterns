mod algorithms;

use anyhow::Result;
use ndarray::{Array2, Axis, concatenate};
use polars::prelude::*;
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_attrs, write_dataset};

use crate::algorithms::admm::ADMMConfig;
use crate::algorithms::mvmd::MVMD;
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};

pub fn run(cfg: &AppConfig) -> Result<()> {
    let _run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        tcp_repo_dir = % cfg.tcp_repo_dir.display(),
        parcellated_ts_dir = %cfg.parcellated_ts_dir.display(),
        num_modes = %cfg.mvmd.num_modes,
        "starting fMRI MVMD decomposition"
    );

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

        for file_path in &available_timeseries {
            let file_result: anyhow::Result<()> = (|| {
                let bids =
                    BidsFilename::parse(match file_path.file_name().and_then(|n| n.to_str()) {
                        Some(name) => name,
                        None => return Ok(()),
                    });
                let task_name = bids.get("task").unwrap_or("unknown");

                ////////////////////////
                // MVMD Decomposition //
                ////////////////////////

                let num_modes = cfg.mvmd.num_modes;
                let admm_config = ADMMConfig::default();

                let h5_file = open_or_create(file_path)?;
                let mvmd_group = open_or_create_group(&h5_file, "mvmd", cfg.force)?;

                // Whole-band decomposition on tcp_timeseries_raw
                let wb_done = !cfg.force && mvmd_group.group("whole-band").is_ok();
                if wb_done {
                    debug!(
                        task_name = task_name,
                        "whole-band MVMD already computed, skipping (use --force to recompute)"
                    );
                } else {
                    let mvmd_start = Instant::now();

                    let dataset = h5_file.dataset("tcp_timeseries_raw")?;
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

                    let mvmd_wb = match MVMD::from_dataframe(&full_df, 1.0) {
                        Ok(m) => m.with_admm_config(admm_config.clone()),
                        Err(e) => {
                            error_count += 1;
                            warn!(
                                task_name = task_name,
                                error = %e,
                                reason = "mvmd_init_failed",
                                "skipping MVMD due to error"
                            );
                            return Ok(());
                        }
                    };

                    let signal_decomposition = mvmd_wb.decompose(num_modes);
                    let mvmd_wb_duration_ms = mvmd_start.elapsed().as_millis();

                    let wb_group = open_or_create_group(&mvmd_group, "whole-band", cfg.force)?;
                    let write_start = Instant::now();

                    let modes_shape = signal_decomposition.modes.shape();
                    write_dataset(
                        &wb_group,
                        "modes",
                        signal_decomposition.modes.as_slice().unwrap(),
                        &[modes_shape[0], modes_shape[1], modes_shape[2]],
                        None,
                    )?;

                    let cf_shape = signal_decomposition.frequency_traces.shape();
                    write_dataset(
                        &wb_group,
                        "frequency_traces",
                        signal_decomposition.frequency_traces.as_slice().unwrap(),
                        &[cf_shape[0], cf_shape[1]],
                        None,
                    )?;

                    write_dataset(
                        &wb_group,
                        "center_frequencies",
                        signal_decomposition.center_frequencies.as_slice().unwrap(),
                        &[signal_decomposition.center_frequencies.len()],
                        None,
                    )?;

                    write_attrs(
                        &wb_group,
                        &[H5Attr::u32(
                            "num_iterations",
                            signal_decomposition.num_iterations as u32,
                        )],
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
                        output_file = %file_path.display(),
                        "computed whole-band MVMD decomposition"
                    );
                }

                // Block-level decomposition — only for task files that have trial blocks
                let blocks_group = match h5_file.group("blocks") {
                    Ok(g) => g,
                    Err(_) => {
                        debug!(
                            task_name = task_name,
                            "no blocks group found, skipping block decomposition"
                        );
                        return Ok(());
                    }
                };

                let block_names: Vec<String> = blocks_group
                    .member_names()?
                    .into_iter()
                    .filter(|n| n.starts_with("block_"))
                    .collect();

                if block_names.is_empty() {
                    return Ok(());
                }

                let mvmd_blocks_group = open_or_create_group(&mvmd_group, "blocks", cfg.force)?;

                for block_name in &block_names {
                    if !cfg.force && mvmd_blocks_group.group(block_name).is_ok() {
                        debug!(
                            task_name = task_name,
                            block = block_name,
                            "block MVMD already computed, skipping (use --force to recompute)"
                        );
                        continue;
                    }

                    let block_group = blocks_group.group(block_name)?;

                    let cortical: Array2<f32> = block_group.dataset("cortical_raw")?.read_2d()?;
                    let subcortical: Array2<f32> =
                        block_group.dataset("subcortical_raw")?.read_2d()?;

                    let block_signal =
                        concatenate(Axis(0), &[cortical.view(), subcortical.view()])?;

                    let block_columns: Vec<Column> = block_signal
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

                    let block_mvmd = match MVMD::from_dataframe(&block_df, 1.0) {
                        Ok(m) => m.with_admm_config(admm_config.clone()),
                        Err(e) => {
                            error_count += 1;
                            warn!(
                                task_name = task_name,
                                block = block_name,
                                error = %e,
                                reason = "mvmd_init_failed",
                                "skipping block MVMD due to error"
                            );
                            continue;
                        }
                    };

                    let block_start = Instant::now();
                    let block_decomposition = block_mvmd.decompose(num_modes);
                    let block_mvmd_duration_ms = block_start.elapsed().as_millis();

                    let mvmd_block_group =
                        open_or_create_group(&mvmd_blocks_group, block_name, cfg.force)?;

                    let block_write_start = Instant::now();

                    let bm_shape = block_decomposition.modes.shape();
                    write_dataset(
                        &mvmd_block_group,
                        "modes",
                        block_decomposition.modes.as_slice().unwrap(),
                        &[bm_shape[0], bm_shape[1], bm_shape[2]],
                        None,
                    )?;

                    let bcf_shape = block_decomposition.frequency_traces.shape();
                    write_dataset(
                        &mvmd_block_group,
                        "frequency_traces",
                        block_decomposition.frequency_traces.as_slice().unwrap(),
                        &[bcf_shape[0], bcf_shape[1]],
                        None,
                    )?;

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
            "some subjects were skipped due to errors"
        );
    }

    Ok(())
}
