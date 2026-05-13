use anyhow::Result;
use ndarray::{Array3, Axis};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::frequency_bands::SLOW_BANDS;

mod aggregation;
mod io;
pub mod stats;

use aggregation::{aggregate_face_blocks, read_fc_matrix};
use stats::permutation::run_permutation;

/// Enumerate (subject_dir_name, first_h5_path) pairs under `dir`.
fn enumerate_subjects(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(out);
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let subject_dir = entry.path();
        if !subject_dir.is_dir() {
            continue;
        }
        let subject = BidsSubjectId::parse(
            subject_dir.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        )
        .to_dir_name();
        for h5_entry in std::fs::read_dir(&subject_dir)?.filter_map(|e| e.ok()) {
            let p = h5_entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("h5") {
                out.push((subject, p));
                break;
            }
        }
    }
    Ok(out)
}

/// Load subject_dir_name -> is_anhedonic from `/metadata/cohort` attr in each subject's H5.
fn load_cohort_labels(dir: &Path) -> Result<HashMap<String, bool>> {
    let mut map = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(map);
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let subject_dir = entry.path();
        if !subject_dir.is_dir() {
            continue;
        }
        let subject = BidsSubjectId::parse(
            subject_dir.file_name().and_then(|n| n.to_str()).unwrap_or(""),
        )
        .to_dir_name();
        // Find first H5 file.
        let Some(h5_path) = std::fs::read_dir(&subject_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .find(|p| p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("h5"))
        else {
            continue;
        };
        let Ok(file) = hdf5::File::open(&h5_path) else {
            continue;
        };
        let Ok(meta) = file.group("metadata") else {
            continue;
        };
        let Ok(attr) = meta.attr("cohort") else {
            continue;
        };
        let Ok(cohort) = attr.read_scalar::<hdf5::types::VarLenUnicode>() else {
            continue;
        };
        let is_anhedonic = match cohort.as_str() {
            "anhedonic" => true,
            "control" => false,
            _ => continue,
        };
        map.insert(subject, is_anhedonic);
    }
    Ok(map)
}

/// Read ROI indices (as usize) from a subject's H5 file.
/// Tries MVMD full-run ROI group first, then per-block as fallback.
fn read_roi_indices(file: &hdf5::File) -> Option<Vec<usize>> {
    let candidates = [
        "/06fc/mvmd/full_run_std_roi/roi_indices",
        "/04hht/full_run_std_roi/roi_indices",
    ];
    for path in &candidates {
        // path is like "/group/subgroup/dataset" — split off dataset name.
        let (group_path, ds_name) = path.rsplit_once('/')?;
        let Ok(grp) = file.group(group_path) else {
            continue;
        };
        let Ok(ds) = grp.dataset(ds_name) else {
            continue;
        };
        if let Ok(raw) = ds.read_raw::<u32>() {
            return Some(raw.into_iter().map(|v| v as usize).collect());
        }
    }
    None
}

/// Extract a symmetric ROI submatrix.
fn extract_roi_submatrix(full: &ndarray::Array2<f64>, idx: &[usize]) -> ndarray::Array2<f64> {
    full.select(Axis(0), idx).select(Axis(1), idx)
}

/// Collect FC matrices from all subjects, stack into [N, C, C] and aligned bool labels.
/// Subjects missing data or lacking cohort labels are skipped.
fn collect_stack(
    subjects: &[(String, PathBuf)],
    labels: &HashMap<String, bool>,
    reader: &impl Fn(&hdf5::File) -> Result<Option<ndarray::Array2<f64>>>,
) -> Result<Option<(Array3<f64>, Vec<bool>)>> {
    let mut mats: Vec<ndarray::Array2<f64>> = Vec::new();
    let mut lab: Vec<bool> = Vec::new();

    for (subject, h5_path) in subjects {
        let Some(&is_anhedonic) = labels.get(subject) else {
            debug!(subject, "no cohort label, skipping");
            continue;
        };
        let Ok(file) = hdf5::File::open(h5_path) else {
            debug!(subject, "cannot open H5, skipping");
            continue;
        };
        match reader(&file) {
            Ok(Some(mat)) => {
                mats.push(mat);
                lab.push(is_anhedonic);
            }
            Ok(None) => {
                debug!(subject, "FC data absent, skipping");
            }
            Err(e) => {
                warn!(subject, error = %e, "error reading FC, skipping");
            }
        }
    }

    if mats.is_empty() {
        return Ok(None);
    }
    let (n_rows, n_cols) = mats[0].dim();
    let n = mats.len();
    let mut stack = Array3::<f64>::zeros((n, n_rows, n_cols));
    for (i, mat) in mats.into_iter().enumerate() {
        stack.slice_mut(ndarray::s![i, .., ..]).assign(&mat);
    }
    Ok(Some((stack, lab)))
}

/// Run one analysis: collect → permutation → write. Skips if too few subjects.
#[allow(clippy::too_many_arguments)]
fn run_one(
    subjects: &[(String, PathBuf)],
    labels: &HashMap<String, bool>,
    results_dir: &Path,
    task: &str,
    source: &str,
    level: &str,
    roi_suffix: &str,
    n_perm: u32,
    primary_t: f64,
    seed: u64,
    reader: impl Fn(&hdf5::File) -> Result<Option<ndarray::Array2<f64>>>,
) -> Result<()> {
    let Some((stack, lab)) = collect_stack(subjects, labels, &reader)? else {
        warn!(task, source, level, roi_suffix, "no data collected, skipping");
        return Ok(());
    };
    let n_a = lab.iter().filter(|&&l| l).count();
    let n_c = lab.len() - n_a;
    if n_a < 2 || n_c < 2 {
        warn!(task, source, level, n_anhedonic = n_a, n_control = n_c, "insufficient group size, skipping");
        return Ok(());
    }
    info!(task, source, level, roi_suffix, n_subjects = lab.len(), n_anhedonic = n_a, n_control = n_c, "running permutation test");
    let result = run_permutation(stack.view(), &lab, n_perm, seed, primary_t);
    io::write_analysis_result(results_dir, task, source, level, roi_suffix, &result, n_perm, primary_t, seed)?;
    info!(task, source, level, roi_suffix, "wrote results");
    Ok(())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let results_dir = cfg
        .consolidated_data_dir
        .parent()
        .unwrap_or(&cfg.consolidated_data_dir)
        .join("09fc_analysis");
    std::fs::create_dir_all(&results_dir)?;

    let n_perm = cfg.fc_analysis.n_permutations;
    let primary_t = cfg.fc_analysis.nbs_primary_t;
    let seed = cfg.fc_analysis.permutation_seed;
    let num_modes = cfg.hht.num_modes;
    let run_roi = !cfg.roi_selection.name.is_empty();

    info!("Enumerating subjects...");
    let subjects = enumerate_subjects(&cfg.consolidated_data_dir)?;
    if subjects.is_empty() {
        warn!("no subjects found in {}", cfg.consolidated_data_dir.display());
        return Ok(());
    }
    info!(n_subjects = subjects.len(), "loaded subject list");

    let labels = load_cohort_labels(&cfg.consolidated_data_dir)?;
    info!(n_labeled = labels.len(), "loaded cohort labels");

    // ------------------------------------------------------------------
    // restAP — TS full-run
    // ------------------------------------------------------------------
    run_one(&subjects, &labels, &results_dir, "restAP", "ts", "full_run", "", n_perm, primary_t, seed, |f| {
        read_fc_matrix(f, "/06fc/ts/full_run_std", None, "fisher_z")
    })?;

    // restAP — CWT per slow band
    for &(band, _, _) in SLOW_BANDS {
        let group = format!("/06fc/cwt/full_run_std/{band}");
        run_one(&subjects, &labels, &results_dir, "restAP", "cwt", band, "", n_perm, primary_t, seed, move |f| {
            read_fc_matrix(f, &group, None, "fisher_z")
        })?;
    }

    // restAP — MVMD per mode
    for k in 0..num_modes {
        let group = format!("/06fc/mvmd/full_run_std/mode_{k}");
        let level = format!("mode_{k}");
        run_one(&subjects, &labels, &results_dir, "restAP", "mvmd", &level, "", n_perm, primary_t, seed, move |f| {
            read_fc_matrix(f, &group, None, "fisher_z")
        })?;
    }

    // restAP — MVMD per slow band
    for &(band, _, _) in SLOW_BANDS {
        let group = format!("/06fc/mvmd/full_run_std/{band}");
        run_one(&subjects, &labels, &results_dir, "restAP", "mvmd", band, "", n_perm, primary_t, seed, move |f| {
            read_fc_matrix(f, &group, None, "fisher_z_mean")
        })?;
    }

    // ------------------------------------------------------------------
    // hammerAP — TS face block average
    // ------------------------------------------------------------------
    run_one(&subjects, &labels, &results_dir, "hammerAP", "ts", "face_block_avg", "", n_perm, primary_t, seed, |f| {
        aggregate_face_blocks(f, "/06fc/ts/blocks_std", None, "fisher_z")
    })?;

    // hammerAP — CWT per slow band, face block average
    for &(band, _, _) in SLOW_BANDS {
        let level = format!("face_block_avg_{band}");
        run_one(&subjects, &labels, &results_dir, "hammerAP", "cwt", &level, "", n_perm, primary_t, seed, move |f| {
            aggregate_face_blocks(f, "/06fc/cwt/blocks_std", Some(band), "fisher_z")
        })?;
    }

    // hammerAP — MVMD per mode, face block average
    for k in 0..num_modes {
        let sub = format!("mode_{k}");
        let level = format!("face_block_avg_mode_{k}");
        run_one(&subjects, &labels, &results_dir, "hammerAP", "mvmd", &level, "", n_perm, primary_t, seed, move |f| {
            aggregate_face_blocks(f, "/06fc/mvmd/blocks_std", Some(&sub), "fisher_z")
        })?;
    }

    // hammerAP — MVMD per slow band, face block average
    for &(band, _, _) in SLOW_BANDS {
        let level = format!("face_block_avg_{band}");
        run_one(&subjects, &labels, &results_dir, "hammerAP", "mvmd_band", &level, "", n_perm, primary_t, seed, move |f| {
            aggregate_face_blocks(f, "/06fc/mvmd/blocks_std", Some(band), "fisher_z_mean")
        })?;
    }

    // ------------------------------------------------------------------
    // ROI variants (if configured)
    // ------------------------------------------------------------------
    if run_roi {
        // Resolve ROI indices from first available subject.
        let roi_indices: Option<Vec<usize>> = subjects.iter().find_map(|(_, h5_path)| {
            hdf5::File::open(h5_path).ok().and_then(|f| read_roi_indices(&f))
        });
        let Some(roi_idx) = roi_indices else {
            warn!("roi_selection configured but no roi_indices found in any subject H5; skipping ROI analyses");
            return Ok(());
        };
        let roi_idx = std::sync::Arc::new(roi_idx);

        // restAP TS ROI — submatrix extraction (no _roi group for TS).
        {
            let idx = std::sync::Arc::clone(&roi_idx);
            run_one(&subjects, &labels, &results_dir, "restAP", "ts", "full_run", "_roi", n_perm, primary_t, seed, move |f| {
                let full = read_fc_matrix(f, "/06fc/ts/full_run_std", None, "fisher_z")?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            })?;
        }

        // restAP CWT ROI
        for &(band, _, _) in SLOW_BANDS {
            let group = format!("/06fc/cwt/full_run_std_roi/{band}");
            run_one(&subjects, &labels, &results_dir, "restAP", "cwt", band, "_roi", n_perm, primary_t, seed, move |f| {
                read_fc_matrix(f, &group, None, "fisher_z")
            })?;
        }

        // restAP MVMD mode ROI
        for k in 0..num_modes {
            let group = format!("/06fc/mvmd/full_run_std_roi/mode_{k}");
            let level = format!("mode_{k}");
            run_one(&subjects, &labels, &results_dir, "restAP", "mvmd", &level, "_roi", n_perm, primary_t, seed, move |f| {
                read_fc_matrix(f, &group, None, "fisher_z")
            })?;
        }

        // restAP MVMD band ROI
        for &(band, _, _) in SLOW_BANDS {
            let group = format!("/06fc/mvmd/full_run_std_roi/{band}");
            run_one(&subjects, &labels, &results_dir, "restAP", "mvmd", band, "_roi", n_perm, primary_t, seed, move |f| {
                read_fc_matrix(f, &group, None, "fisher_z_mean")
            })?;
        }

        // hammerAP TS ROI — submatrix extraction.
        {
            let idx = std::sync::Arc::clone(&roi_idx);
            run_one(&subjects, &labels, &results_dir, "hammerAP", "ts", "face_block_avg", "_roi", n_perm, primary_t, seed, move |f| {
                let full = aggregate_face_blocks(f, "/06fc/ts/blocks_std", None, "fisher_z")?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            })?;
        }

        // hammerAP CWT band ROI
        for &(band, _, _) in SLOW_BANDS {
            let level = format!("face_block_avg_{band}");
            run_one(&subjects, &labels, &results_dir, "hammerAP", "cwt", &level, "_roi", n_perm, primary_t, seed, move |f| {
                aggregate_face_blocks(f, "/06fc/cwt/blocks_std_roi", Some(band), "fisher_z")
            })?;
        }

        // hammerAP MVMD mode ROI
        for k in 0..num_modes {
            let sub = format!("mode_{k}");
            let level = format!("face_block_avg_mode_{k}");
            run_one(&subjects, &labels, &results_dir, "hammerAP", "mvmd", &level, "_roi", n_perm, primary_t, seed, move |f| {
                aggregate_face_blocks(f, "/06fc/mvmd/blocks_std_roi", Some(&sub), "fisher_z")
            })?;
        }

        // hammerAP MVMD band ROI
        for &(band, _, _) in SLOW_BANDS {
            let level = format!("face_block_avg_{band}");
            run_one(&subjects, &labels, &results_dir, "hammerAP", "mvmd_band", &level, "_roi", n_perm, primary_t, seed, move |f| {
                aggregate_face_blocks(f, "/06fc/mvmd/blocks_std_roi", Some(band), "fisher_z_mean")
            })?;
        }
    }

    info!("FC analysis complete. Results in {}", results_dir.display());
    Ok(())
}
