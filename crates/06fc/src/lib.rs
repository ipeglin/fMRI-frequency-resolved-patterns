use anyhow::Result;
use ndarray::{Array2, Array3, Axis, concatenate};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_attrs, write_dataset};

/// fMRI slow-band (Buzsáki) frequency ranges in Hz.
/// Intervals are [low, high) — a mode with center frequency `f` falls in a band iff low <= f < high.
const SLOW_BANDS: &[(&str, f64, f64)] = &[
    ("slow_5", 0.010, 0.027),
    ("slow_4", 0.027, 0.073),
    ("slow_3", 0.073, 0.198),
    ("slow_2", 0.198, 0.250),
];

/// Clip bound for Fisher Z transform to avoid ±inf at r = ±1.
const FISHER_CLIP: f64 = 0.9999;

/// Compute NxN Pearson correlation across rows (channels) of a [C, T] matrix.
/// Returns NaN rows/cols where a channel has zero variance.
fn pearson_matrix(signal: &Array2<f64>) -> Array2<f64> {
    let n_channels = signal.nrows();
    let n_timepoints = signal.ncols();

    let mut centered = signal.clone();
    let mut stds = vec![0.0f64; n_channels];

    for (i, mut row) in centered.axis_iter_mut(Axis(0)).enumerate() {
        let mean = row.sum() / n_timepoints as f64;
        row.mapv_inplace(|v| v - mean);
        let var = row.iter().map(|v| v * v).sum::<f64>() / n_timepoints as f64;
        stds[i] = var.sqrt();
    }

    let mut out = Array2::<f64>::zeros((n_channels, n_channels));
    for i in 0..n_channels {
        for j in i..n_channels {
            let r = if stds[i] == 0.0 || stds[j] == 0.0 {
                f64::NAN
            } else {
                let dot: f64 = centered
                    .row(i)
                    .iter()
                    .zip(centered.row(j).iter())
                    .map(|(a, b)| a * b)
                    .sum();
                (dot / n_timepoints as f64) / (stds[i] * stds[j])
            };
            out[[i, j]] = r;
            out[[j, i]] = r;
        }
    }
    out
}

/// Apply Fisher Z transform (arctanh) with clipping to avoid ±inf.
fn fisher_z(pearson: &Array2<f64>) -> Array2<f64> {
    pearson.mapv(|v| {
        if v.is_nan() {
            f64::NAN
        } else {
            v.clamp(-FISHER_CLIP, FISHER_CLIP).atanh()
        }
    })
}

/// Write both pearson and fisher_z datasets into `group`.
/// `source` is stored as a group attribute to trace provenance.
fn write_fc_pair(group: &hdf5::Group, pearson: &Array2<f64>, attrs: &[H5Attr]) -> Result<()> {
    let shape = [pearson.nrows(), pearson.ncols()];
    let fz = fisher_z(pearson);

    write_dataset(group, "pearson", pearson.as_slice().unwrap(), &shape, None)?;
    write_dataset(group, "fisher_z", fz.as_slice().unwrap(), &shape, None)?;

    if !attrs.is_empty() {
        write_attrs(group, attrs)?;
    }
    Ok(())
}

/// Read a [C, T] timeseries dataset and convert to f64 for FC computation.
fn read_timeseries_2d(ds: &hdf5::Dataset) -> Result<Array2<f64>> {
    let data_f32: Array2<f32> = ds.read_2d()?;
    Ok(data_f32.mapv(|v| v as f64))
}

/// Concatenate cortical + subcortical row-wise (matches CWT/MVMD pattern).
fn concat_cortical_subcortical(
    block_group: &hdf5::Group,
    cortical_name: &str,
    subcortical_name: &str,
) -> Result<Array2<f64>> {
    let cortical: Array2<f32> = block_group.dataset(cortical_name)?.read_2d()?;
    let subcortical: Array2<f32> = block_group.dataset(subcortical_name)?.read_2d()?;
    let stacked = concatenate(Axis(0), &[cortical.view(), subcortical.view()])?;
    Ok(stacked.mapv(|v| v as f64))
}

/// Read an MVMD modes dataset as [K, C, T] f64.
fn read_modes_3d(group: &hdf5::Group) -> Result<Array3<f64>> {
    let ds = group.dataset("modes")?;
    let raw: Vec<f64> = ds.read_raw::<f64>()?;
    let shape = ds.shape();
    if shape.len() != 3 {
        anyhow::bail!("expected 3D modes dataset, got shape {:?}", shape);
    }
    Ok(Array3::from_shape_vec((shape[0], shape[1], shape[2]), raw)?)
}

/// Read the MVMD center_frequencies dataset (K,) as f64.
fn read_center_frequencies(group: &hdf5::Group) -> Result<Vec<f64>> {
    let ds = group.dataset("center_frequencies")?;
    Ok(ds.read_raw::<f64>()?)
}

/// For each slow-band, average Fisher-Z FC matrices whose mode center frequency falls in the band.
/// Stores `fisher_z_mean` and `pearson` (tanh of mean) under `parent/{slow_band}`.
fn write_slow_band_aggregates(
    parent: &hdf5::Group,
    mode_fisher: &[Array2<f64>],
    center_frequencies: &[f64],
    force: bool,
) -> Result<()> {
    if mode_fisher.is_empty() {
        return Ok(());
    }
    let (n, m) = (mode_fisher[0].nrows(), mode_fisher[0].ncols());

    for (band_name, low, high) in SLOW_BANDS {
        let members: Vec<usize> = center_frequencies
            .iter()
            .enumerate()
            .filter_map(|(k, f)| {
                if *f >= *low && *f < *high {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();

        let band_group = open_or_create_group(parent, band_name, force)?;

        if members.is_empty() {
            write_attrs(
                &band_group,
                &[
                    H5Attr::f64("freq_low_hz", *low),
                    H5Attr::f64("freq_high_hz", *high),
                    H5Attr::u32("num_modes_aggregated", 0),
                ],
            )?;
            continue;
        }

        let mut mean_fz = Array2::<f64>::zeros((n, m));
        let mut counts = Array2::<f64>::zeros((n, m));

        for k in &members {
            let fz = &mode_fisher[*k];
            for i in 0..n {
                for j in 0..m {
                    let v = fz[[i, j]];
                    if !v.is_nan() {
                        mean_fz[[i, j]] += v;
                        counts[[i, j]] += 1.0;
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..m {
                mean_fz[[i, j]] = if counts[[i, j]] > 0.0 {
                    mean_fz[[i, j]] / counts[[i, j]]
                } else {
                    f64::NAN
                };
            }
        }

        let pearson_back = mean_fz.mapv(|v| if v.is_nan() { f64::NAN } else { v.tanh() });

        write_dataset(
            &band_group,
            "fisher_z_mean",
            mean_fz.as_slice().unwrap(),
            &[n, m],
            None,
        )?;
        write_dataset(
            &band_group,
            "pearson",
            pearson_back.as_slice().unwrap(),
            &[n, m],
            None,
        )?;
        write_attrs(
            &band_group,
            &[
                H5Attr::f64("freq_low_hz", *low),
                H5Attr::f64("freq_high_hz", *high),
                H5Attr::u32("num_modes_aggregated", members.len() as u32),
                H5Attr::string(
                    "mode_indices",
                    members
                        .iter()
                        .map(|k| k.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
        )?;
    }
    Ok(())
}

/// Compute FC (pearson + fisher_z) for each MVMD mode and write per-mode datasets under `parent`.
/// Returns the per-mode Fisher-Z matrices so the caller can aggregate into slow-band datasets.
fn process_mvmd_modes(
    parent: &hdf5::Group,
    modes: &Array3<f64>,
    center_frequencies: &[f64],
    force: bool,
) -> Result<Vec<Array2<f64>>> {
    let n_modes = modes.shape()[0];
    let mut fisher_per_mode = Vec::with_capacity(n_modes);

    for k in 0..n_modes {
        let mode_slice = modes.index_axis(Axis(0), k);
        let mode_2d = mode_slice.to_owned();

        let pearson = pearson_matrix(&mode_2d);
        let fz = fisher_z(&pearson);

        let mode_group = open_or_create_group(parent, &format!("mode_{}", k), force)?;
        let shape = [pearson.nrows(), pearson.ncols()];
        write_dataset(
            &mode_group,
            "pearson",
            pearson.as_slice().unwrap(),
            &shape,
            None,
        )?;
        write_dataset(
            &mode_group,
            "fisher_z",
            fz.as_slice().unwrap(),
            &shape,
            None,
        )?;
        write_attrs(
            &mode_group,
            &[
                H5Attr::u32("mode_index", k as u32),
                H5Attr::f64("center_frequency_hz", center_frequencies[k]),
            ],
        )?;

        fisher_per_mode.push(fz);
    }

    Ok(fisher_per_mode)
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        parcellated_ts_dir = %cfg.parcellated_ts_dir.display(),
        force = cfg.force,
        "starting fMRI FC pipeline"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.parcellated_ts_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            Some((BidsSubjectId::parse(id).to_dir_name(), path))
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
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.contains(".h5"))
                    .unwrap_or(false)
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

                let h5_file = open_or_create(file_path)?;
                let fc_group = open_or_create_group(&h5_file, "fc", false)?;

                /////////////////////////////////////////
                // Whole-band raw (tcp_timeseries_raw) //
                /////////////////////////////////////////
                {
                    let done = !cfg.force && fc_group.group("raw").is_ok();
                    if done {
                        debug!(task_name = task_name, "fc/raw already computed, skipping");
                    } else if let Ok(ds) = h5_file.dataset("tcp_timeseries_raw") {
                        let t0 = Instant::now();
                        let ts = read_timeseries_2d(&ds)?;
                        let pearson = pearson_matrix(&ts);
                        let raw_group = open_or_create_group(&fc_group, "raw", cfg.force)?;
                        write_fc_pair(
                            &raw_group,
                            &pearson,
                            &[
                                H5Attr::string("source", "tcp_timeseries_raw"),
                                H5Attr::u32("n_channels", ts.nrows() as u32),
                                H5Attr::u32("n_timepoints", ts.ncols() as u32),
                            ],
                        )?;
                        debug!(
                            task_name = task_name,
                            n = ts.nrows(),
                            duration_ms = t0.elapsed().as_millis(),
                            "computed fc/raw"
                        );
                    }
                }

                /////////////////////////////////////////////////////
                // Whole-band standardized (tcp_timeseries_standardized) //
                /////////////////////////////////////////////////////
                {
                    let done = !cfg.force && fc_group.group("standardized").is_ok();
                    if done {
                        debug!(
                            task_name = task_name,
                            "fc/standardized already computed, skipping"
                        );
                    } else if let Ok(ds) = h5_file.dataset("tcp_timeseries_standardized") {
                        let t0 = Instant::now();
                        let ts = read_timeseries_2d(&ds)?;
                        let pearson = pearson_matrix(&ts);
                        let std_group = open_or_create_group(&fc_group, "standardized", cfg.force)?;
                        write_fc_pair(
                            &std_group,
                            &pearson,
                            &[
                                H5Attr::string("source", "tcp_timeseries_standardized"),
                                H5Attr::u32("n_channels", ts.nrows() as u32),
                                H5Attr::u32("n_timepoints", ts.ncols() as u32),
                            ],
                        )?;
                        debug!(
                            task_name = task_name,
                            duration_ms = t0.elapsed().as_millis(),
                            "computed fc/standardized"
                        );
                    }
                }

                ////////////////////
                // Raw blocks     //
                ////////////////////
                if let Ok(blocks_group) = h5_file.group("blocks") {
                    let block_names: Vec<String> = blocks_group
                        .member_names()?
                        .into_iter()
                        .filter(|n| n.starts_with("block_"))
                        .collect();

                    if !block_names.is_empty() {
                        let fc_blocks_raw =
                            open_or_create_group(&fc_group, "blocks_raw", cfg.force)?;

                        for block_name in &block_names {
                            if !cfg.force && fc_blocks_raw.group(block_name).is_ok() {
                                continue;
                            }
                            let block_group = blocks_group.group(block_name)?;
                            let ts = concat_cortical_subcortical(
                                &block_group,
                                "cortical_raw",
                                "subcortical_raw",
                            )?;
                            let pearson = pearson_matrix(&ts);
                            let out = open_or_create_group(&fc_blocks_raw, block_name, cfg.force)?;
                            write_fc_pair(
                                &out,
                                &pearson,
                                &[
                                    H5Attr::string("source", "blocks/cortical_raw+subcortical_raw"),
                                    H5Attr::u32("n_channels", ts.nrows() as u32),
                                    H5Attr::u32("n_timepoints", ts.ncols() as u32),
                                ],
                            )?;
                        }
                        debug!(
                            task_name = task_name,
                            num_blocks = block_names.len(),
                            "computed fc/blocks_raw"
                        );
                    }
                }

                ////////////////////////////
                // Standardized blocks    //
                ////////////////////////////
                if let Ok(blocks_std_group) = h5_file.group("blocks_standardized") {
                    let block_names: Vec<String> = blocks_std_group
                        .member_names()?
                        .into_iter()
                        .filter(|n| n.starts_with("block_"))
                        .collect();

                    if !block_names.is_empty() {
                        let fc_blocks_std =
                            open_or_create_group(&fc_group, "blocks_standardized", cfg.force)?;

                        for block_name in &block_names {
                            if !cfg.force && fc_blocks_std.group(block_name).is_ok() {
                                continue;
                            }
                            let block_group = blocks_std_group.group(block_name)?;
                            let ts = concat_cortical_subcortical(
                                &block_group,
                                "cortical_standardized",
                                "subcortical_standardized",
                            )?;
                            let pearson = pearson_matrix(&ts);
                            let out = open_or_create_group(&fc_blocks_std, block_name, cfg.force)?;
                            write_fc_pair(
                                &out,
                                &pearson,
                                &[
                                    H5Attr::string(
                                        "source",
                                        "blocks_standardized/cortical_standardized+subcortical_standardized",
                                    ),
                                    H5Attr::u32("n_channels", ts.nrows() as u32),
                                    H5Attr::u32("n_timepoints", ts.ncols() as u32),
                                ],
                            )?;
                        }
                        debug!(
                            task_name = task_name,
                            num_blocks = block_names.len(),
                            "computed fc/blocks_standardized"
                        );
                    }
                }

                //////////////////////////////
                // MVMD whole-band modes    //
                //////////////////////////////
                if let Ok(mvmd_group) = h5_file.group("mvmd") {
                    let fc_mvmd_group = open_or_create_group(&fc_group, "mvmd", cfg.force)?;

                    if let Ok(wb_group) = mvmd_group.group("whole-band") {
                        let already = !cfg.force && fc_mvmd_group.group("whole-band").is_ok();
                        if already {
                            debug!(
                                task_name = task_name,
                                "fc/mvmd/whole-band already computed, skipping"
                            );
                        } else {
                            let t0 = Instant::now();
                            let modes = read_modes_3d(&wb_group)?;
                            let cfreqs = read_center_frequencies(&wb_group)?;
                            let fc_wb =
                                open_or_create_group(&fc_mvmd_group, "whole-band", cfg.force)?;
                            let per_mode_fz =
                                process_mvmd_modes(&fc_wb, &modes, &cfreqs, cfg.force)?;
                            write_slow_band_aggregates(&fc_wb, &per_mode_fz, &cfreqs, cfg.force)?;
                            write_attrs(
                                &fc_wb,
                                &[
                                    H5Attr::u32("n_modes", modes.shape()[0] as u32),
                                    H5Attr::u32("n_channels", modes.shape()[1] as u32),
                                    H5Attr::u32("n_timepoints", modes.shape()[2] as u32),
                                ],
                            )?;
                            debug!(
                                task_name = task_name,
                                n_modes = modes.shape()[0],
                                duration_ms = t0.elapsed().as_millis(),
                                "computed fc/mvmd/whole-band"
                            );
                        }
                    }

                    if let Ok(mvmd_blocks_group) = mvmd_group.group("blocks") {
                        let block_names: Vec<String> = mvmd_blocks_group
                            .member_names()?
                            .into_iter()
                            .filter(|n| n.starts_with("block_"))
                            .collect();

                        if !block_names.is_empty() {
                            let fc_mvmd_blocks =
                                open_or_create_group(&fc_mvmd_group, "blocks", cfg.force)?;

                            for block_name in &block_names {
                                if !cfg.force && fc_mvmd_blocks.group(block_name).is_ok() {
                                    continue;
                                }
                                let src_block = mvmd_blocks_group.group(block_name)?;
                                let modes = read_modes_3d(&src_block)?;
                                let cfreqs = read_center_frequencies(&src_block)?;
                                let out =
                                    open_or_create_group(&fc_mvmd_blocks, block_name, cfg.force)?;
                                let per_mode_fz =
                                    process_mvmd_modes(&out, &modes, &cfreqs, cfg.force)?;
                                write_slow_band_aggregates(&out, &per_mode_fz, &cfreqs, cfg.force)?;
                                write_attrs(
                                    &out,
                                    &[
                                        H5Attr::u32("n_modes", modes.shape()[0] as u32),
                                        H5Attr::u32("n_channels", modes.shape()[1] as u32),
                                        H5Attr::u32("n_timepoints", modes.shape()[2] as u32),
                                    ],
                                )?;
                            }
                            debug!(
                                task_name = task_name,
                                num_blocks = block_names.len(),
                                "computed fc/mvmd/blocks"
                            );
                        }
                    }
                }

                info!(
                    task_name = task_name,
                    file = %file_path.display(),
                    "fc file complete"
                );
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
            "some files were skipped due to errors"
        );
    }

    info!(
        error_count = error_count,
        total_duration_ms = run_start.elapsed().as_millis(),
        "FC pipeline complete"
    );

    // TODO: ROI-specific correlation coefficient and Fisher Z-score extraction is
    // handled in the feature extraction crate (07feature_extraction), using the
    // atlas LUTs to map channel indices to ROI labels.

    Ok(())
}
