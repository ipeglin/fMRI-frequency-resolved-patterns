use anyhow::Result;
use ndarray::{Array2, Array3, Axis};
use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{debug, info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands::{self, SLOW_BANDS};
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_attrs, write_dataset};

/// Clip bound for Fisher Z transform to avoid ±inf at r = ±1.
const FISHER_CLIP: f64 = 0.9999;

/// Number of CWT scales (mirrors `crates/03cwt`).
const CWT_N_SCALES: usize = 224;

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

/// Write both pearson and fisher_z datasets into `group` plus optional attrs.
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

/// Read a 3D HDF5 dataset as f64, handling either f32 or f64 on-disk storage.
fn read_3d_as_f64(ds: &hdf5::Dataset) -> Result<Array3<f64>> {
    let shape = ds.shape();
    if shape.len() != 3 {
        anyhow::bail!("expected 3D dataset, got shape {:?}", shape);
    }
    let dims = (shape[0], shape[1], shape[2]);
    if let Ok(raw) = ds.read_raw::<f32>() {
        let raw_f64: Vec<f64> = raw.into_iter().map(|v| v as f64).collect();
        return Ok(Array3::from_shape_vec(dims, raw_f64)?);
    }
    let raw: Vec<f64> = ds.read_raw()?;
    Ok(Array3::from_shape_vec(dims, raw)?)
}

/// Read an MVMD `modes` dataset as `[K, C, T]` f64.
fn read_modes_3d(group: &hdf5::Group) -> Result<Array3<f64>> {
    let ds = group.dataset("modes")?;
    read_3d_as_f64(&ds)
}

/// Read the MVMD `center_frequencies` dataset (K,) as f64.
fn read_center_frequencies(group: &hdf5::Group) -> Result<Vec<f64>> {
    let ds = group.dataset("center_frequencies")?;
    Ok(ds.read_raw::<f64>()?)
}

/// Copy `roi_indices` dataset from MVMD source group to FC destination group, if present.
fn propagate_roi_indices(src: &hdf5::Group, dest: &hdf5::Group) -> Result<()> {
    let Ok(ds) = src.dataset("roi_indices") else {
        return Ok(());
    };
    if dest.dataset("roi_indices").is_ok() {
        return Ok(());
    }
    let data: Vec<u32> = ds.read_raw()?;
    write_dataset(dest, "roi_indices", &data, &[data.len()], None)?;
    Ok(())
}

/// CWT scale → centre-frequency grid. Mirrors the log-spaced grid built in
/// `crates/03cwt`, so each scale index maps to a known Hz value here.
fn cwt_freq_grid() -> Vec<f64> {
    let f_min = frequency_bands::f_min();
    let f_max = frequency_bands::f_max();
    let n = CWT_N_SCALES;
    (0..n)
        .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n - 1) as f64))
        .collect()
}

/// For each slow-band, average Fisher-Z FC matrices across components whose
/// centre frequency falls in the band. Stores `fisher_z_mean` and `pearson`
/// (`tanh` of the mean) under `parent/{slow_band}`.
fn write_slow_band_aggregates(
    parent: &hdf5::Group,
    component_fisher: &[Array2<f64>],
    component_freqs: &[f64],
    force: bool,
) -> Result<()> {
    if component_fisher.is_empty() {
        return Ok(());
    }
    let (n, m) = (component_fisher[0].nrows(), component_fisher[0].ncols());

    for (band_name, low, high) in SLOW_BANDS {
        let members: Vec<usize> = component_freqs
            .iter()
            .enumerate()
            .filter_map(|(k, f)| (*f >= *low && *f < *high).then_some(k))
            .collect();

        let band_group = open_or_create_group(parent, band_name, force)?;

        if members.is_empty() {
            write_attrs(
                &band_group,
                &[
                    H5Attr::f64("freq_low_hz", *low),
                    H5Attr::f64("freq_high_hz", *high),
                    H5Attr::u32("num_components_aggregated", 0),
                ],
            )?;
            continue;
        }

        let mut mean_fz = Array2::<f64>::zeros((n, m));
        let mut counts = Array2::<f64>::zeros((n, m));

        for k in &members {
            let fz = &component_fisher[*k];
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
                H5Attr::u32("num_components_aggregated", members.len() as u32),
                H5Attr::string(
                    "component_indices",
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
        let mode_2d = modes.index_axis(Axis(0), k).to_owned();

        let pearson = pearson_matrix(&mode_2d);
        let fz = fisher_z(&pearson);

        let mode_group = open_or_create_group(parent, &format!("mode_{}", k), force)?;
        write_fc_pair(
            &mode_group,
            &pearson,
            &[
                H5Attr::u32("mode_index", k as u32),
                H5Attr::f64("center_frequency_hz", center_frequencies[k]),
            ],
        )?;

        fisher_per_mode.push(fz);
    }

    Ok(fisher_per_mode)
}

/// Compute slow-band FC for a CWT power scalogram `[C, S, T]`.
///
/// For each slow-band: average power across the scales whose centre frequency
/// falls in that band → `[C, T]` band-power signal → Pearson FC across channels.
/// Aggregating power before correlating (rather than averaging per-scale FC) keeps
/// storage bounded to one matrix per band rather than 224 per scenario.
fn process_cwt_scalogram(
    parent: &hdf5::Group,
    scalo: &Array3<f64>,
    scale_freqs: &[f64],
    force: bool,
) -> Result<()> {
    let [n_channels, n_scales, n_timepoints] = match scalo.shape() {
        &[c, s, t] => [c, s, t],
        s => anyhow::bail!("expected 3D scalogram, got shape {:?}", s),
    };

    if scale_freqs.len() != n_scales {
        anyhow::bail!(
            "scale_freqs length {} != n_scales {}",
            scale_freqs.len(),
            n_scales
        );
    }

    for (band_name, low, high) in SLOW_BANDS {
        let scale_indices: Vec<usize> = scale_freqs
            .iter()
            .enumerate()
            .filter_map(|(s, f)| (*f >= *low && *f < *high).then_some(s))
            .collect();

        let band_group = open_or_create_group(parent, band_name, force)?;

        if scale_indices.is_empty() {
            write_attrs(
                &band_group,
                &[
                    H5Attr::f64("freq_low_hz", *low),
                    H5Attr::f64("freq_high_hz", *high),
                    H5Attr::u32("num_scales_aggregated", 0),
                ],
            )?;
            continue;
        }

        let denom = scale_indices.len() as f64;
        let mut band_power = Array2::<f64>::zeros((n_channels, n_timepoints));
        for &s in &scale_indices {
            for c in 0..n_channels {
                for t in 0..n_timepoints {
                    band_power[[c, t]] += scalo[[c, s, t]];
                }
            }
        }
        band_power.mapv_inplace(|v| v / denom);

        let pearson = pearson_matrix(&band_power);
        write_fc_pair(
            &band_group,
            &pearson,
            &[
                H5Attr::f64("freq_low_hz", *low),
                H5Attr::f64("freq_high_hz", *high),
                H5Attr::u32("num_scales_aggregated", scale_indices.len() as u32),
                H5Attr::string(
                    "scale_indices",
                    scale_indices
                        .iter()
                        .map(|s| s.to_string())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
            ],
        )?;
    }
    Ok(())
}

/// True if `parent/name` exists and carries the named completion-sentinel attr.
/// Sentinels are written last in each `fc_for_*` so their presence proves the
/// prior run finished writing all FC + Fisher-Z matrices for that subgroup.
fn subgroup_complete(parent: &hdf5::Group, name: &str, sentinel_attr: &str) -> bool {
    parent
        .group(name)
        .map_or(false, |g| g.attr(sentinel_attr).is_ok())
}

/// FC for one MVMD subgroup (e.g. `/mvmd/full_run_raw` or `/mvmd/blocks_raw/block_X`).
/// Writes per-mode FC + slow-band aggregates under `fc_parent/name`.
fn fc_for_mvmd_subgroup(
    src_parent: &hdf5::Group,
    fc_parent: &hdf5::Group,
    name: &str,
    task_name: &str,
    force: bool,
) -> Result<()> {
    let src = match src_parent.group(name) {
        Ok(g) => g,
        Err(_) => {
            debug!(
                task_name = task_name,
                group = name,
                "mvmd subgroup missing, skipping FC"
            );
            return Ok(());
        }
    };

    if !force && fc_parent.group(name).is_ok() {
        if subgroup_complete(fc_parent, name, "n_modes") {
            debug!(
                task_name = task_name,
                group = name,
                "fc/mvmd subgroup already computed, skipping (use --force to recompute)"
            );
        } else {
            warn!(
                task_name = task_name,
                group = name,
                "fc/mvmd subgroup exists but is incomplete (no n_modes sentinel) — skipping; rerun with --force to recompute"
            );
        }
        return Ok(());
    }

    let modes = read_modes_3d(&src)?;
    let cfreqs = read_center_frequencies(&src)?;
    let dest = open_or_create_group(fc_parent, name, force)?;

    let t0 = Instant::now();
    let per_mode_fz = process_mvmd_modes(&dest, &modes, &cfreqs, force)?;
    write_slow_band_aggregates(&dest, &per_mode_fz, &cfreqs, force)?;
    propagate_roi_indices(&src, &dest)?;
    write_attrs(
        &dest,
        &[
            H5Attr::u32("n_modes", modes.shape()[0] as u32),
            H5Attr::u32("n_channels", modes.shape()[1] as u32),
            H5Attr::u32("n_timepoints", modes.shape()[2] as u32),
        ],
    )?;
    debug!(
        task_name = task_name,
        group = name,
        n_modes = modes.shape()[0],
        n_channels = modes.shape()[1],
        duration_ms = t0.elapsed().as_millis(),
        "computed fc/mvmd subgroup"
    );
    Ok(())
}

/// Iterate `block_*` MVMD subgroups under `mvmd_root/name` and write per-block FC.
fn fc_for_mvmd_blocks(
    mvmd_root: &hdf5::Group,
    fc_mvmd: &hdf5::Group,
    name: &str,
    task_name: &str,
    force: bool,
) -> Result<()> {
    let blocks_src = match mvmd_root.group(name) {
        Ok(g) => g,
        Err(_) => {
            debug!(
                task_name = task_name,
                group = name,
                "mvmd blocks parent missing, skipping FC"
            );
            return Ok(());
        }
    };

    let block_names: Vec<String> = blocks_src
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("block_"))
        .collect();

    if block_names.is_empty() {
        debug!(
            task_name = task_name,
            group = name,
            "no mvmd blocks found, skipping FC"
        );
        return Ok(());
    }

    let dest_parent = open_or_create_group(fc_mvmd, name, false)?;
    for block_name in &block_names {
        fc_for_mvmd_subgroup(&blocks_src, &dest_parent, block_name, task_name, force)?;
    }
    debug!(
        task_name = task_name,
        group = name,
        num_blocks = block_names.len(),
        "computed fc/mvmd blocks"
    );
    Ok(())
}

/// FC for one CWT scalogram dataset. Writes slow-band FC under `fc_parent/name`.
fn fc_for_cwt_dataset(
    ds: &hdf5::Dataset,
    fc_parent: &hdf5::Group,
    name: &str,
    task_name: &str,
    force: bool,
) -> Result<()> {
    if !force && fc_parent.group(name).is_ok() {
        if subgroup_complete(fc_parent, name, "n_channels") {
            debug!(
                task_name = task_name,
                group = name,
                "fc/cwt subgroup already computed, skipping (use --force to recompute)"
            );
        } else {
            warn!(
                task_name = task_name,
                group = name,
                "fc/cwt subgroup exists but is incomplete (no n_channels sentinel) — skipping; rerun with --force to recompute"
            );
        }
        return Ok(());
    }

    let scalo = read_3d_as_f64(ds)?;
    let dest = open_or_create_group(fc_parent, name, force)?;
    let freqs = cwt_freq_grid();

    let t0 = Instant::now();
    process_cwt_scalogram(&dest, &scalo, &freqs, force)?;
    write_attrs(
        &dest,
        &[
            H5Attr::u32("n_channels", scalo.shape()[0] as u32),
            H5Attr::u32("n_scales", scalo.shape()[1] as u32),
            H5Attr::u32("n_timepoints", scalo.shape()[2] as u32),
        ],
    )?;
    debug!(
        task_name = task_name,
        group = name,
        n_channels = scalo.shape()[0],
        n_scales = scalo.shape()[1],
        duration_ms = t0.elapsed().as_millis(),
        "computed fc/cwt subgroup"
    );
    Ok(())
}

/// Iterate `block_*` CWT scalogram datasets under `blocks_src` and write per-block FC.
fn fc_for_cwt_blocks(
    blocks_src: &hdf5::Group,
    fc_cwt: &hdf5::Group,
    name: &str,
    task_name: &str,
    force: bool,
) -> Result<()> {
    let block_names: Vec<String> = blocks_src
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("block_"))
        .collect();

    if block_names.is_empty() {
        debug!(
            task_name = task_name,
            group = name,
            "no cwt blocks found, skipping FC"
        );
        return Ok(());
    }

    let dest_parent = open_or_create_group(fc_cwt, name, false)?;
    for block_name in &block_names {
        match blocks_src.dataset(block_name) {
            Ok(ds) => fc_for_cwt_dataset(&ds, &dest_parent, block_name, task_name, force)?,
            Err(_) => {
                debug!(
                    task_name = task_name,
                    group = name,
                    block = block_name,
                    "cwt block dataset missing, skipping"
                );
            }
        }
    }
    debug!(
        task_name = task_name,
        group = name,
        num_blocks = block_names.len(),
        "computed fc/cwt blocks"
    );
    Ok(())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        "starting fMRI FC pipeline"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.consolidated_data_dir)?
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
                let fc_group = open_or_create_group(&h5_file, "06fc", false)?;

                match task_name {
                    "restAP" => {
                        // CWT whole-signal scalogram on all channels
                        if let Ok(cwt_root) = h5_file.group("03cwt") {
                            if let Ok(ds) = cwt_root.dataset("full_run_std") {
                                let fc_cwt = open_or_create_group(&fc_group, "cwt", false)?;
                                fc_for_cwt_dataset(
                                    &ds,
                                    &fc_cwt,
                                    "full_run_std",
                                    task_name,
                                    cfg.force,
                                )?;
                            }
                        }
                        // MVMD whole-signal modes (all channels + ROI subset)
                        if let Ok(mvmd_root) = h5_file.group("04mvmd") {
                            let fc_mvmd = open_or_create_group(&fc_group, "mvmd", false)?;
                            fc_for_mvmd_subgroup(
                                &mvmd_root,
                                &fc_mvmd,
                                "full_run_raw",
                                task_name,
                                cfg.force,
                            )?;
                            fc_for_mvmd_subgroup(
                                &mvmd_root,
                                &fc_mvmd,
                                "full_run_raw_roi",
                                task_name,
                                cfg.force,
                            )?;
                        }
                    }
                    "hammerAP" => {
                        // CWT block scalograms on all channels
                        if let Ok(cwt_root) = h5_file.group("03cwt") {
                            if let Ok(blocks_std) = cwt_root.group("blocks_std") {
                                let fc_cwt = open_or_create_group(&fc_group, "cwt", false)?;
                                fc_for_cwt_blocks(
                                    &blocks_std,
                                    &fc_cwt,
                                    "blocks_std",
                                    task_name,
                                    cfg.force,
                                )?;
                            }
                        }
                        // MVMD block modes (all channels + ROI subset)
                        if let Ok(mvmd_root) = h5_file.group("04mvmd") {
                            let fc_mvmd = open_or_create_group(&fc_group, "mvmd", false)?;
                            fc_for_mvmd_blocks(
                                &mvmd_root,
                                &fc_mvmd,
                                "blocks_raw",
                                task_name,
                                cfg.force,
                            )?;
                            fc_for_mvmd_blocks(
                                &mvmd_root,
                                &fc_mvmd,
                                "blocks_raw_roi",
                                task_name,
                                cfg.force,
                            )?;
                        }
                    }
                    other => {
                        debug!(task_name = other, "unrecognized task type, skipping FC");
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

    Ok(())
}
