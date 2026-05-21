use anyhow::Result;
use ndarray::{Array2, Array3, Axis};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use tracing::{debug, info, warn};
use utils::config::AppConfig;

type SubjectList = Vec<(String, PathBuf)>;
type CohortMap = HashMap<String, bool>;

use crate::stats::permutation::{PermResult, run_permutation};

/// Enumerate `(subject_dir_name, first_h5_path)` pairs under `dir`.
pub fn enumerate_subjects(dir: &Path) -> Result<Vec<(String, PathBuf)>> {
    let mut out = Vec::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(out);
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let subject_dir = entry.path();
        if !subject_dir.is_dir() {
            continue;
        }
        let subject = utils::bids_subject_id::BidsSubjectId::parse(
            subject_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(""),
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

/// Load `subject_dir_name -> is_anhedonic` from `/metadata/cohort` in each subject's H5.
pub fn load_cohort_labels(dir: &Path) -> Result<HashMap<String, bool>> {
    let mut map = HashMap::new();
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Ok(map);
    };
    for entry in entries.filter_map(|e| e.ok()) {
        let subject_dir = entry.path();
        if !subject_dir.is_dir() {
            continue;
        }
        let subject = utils::bids_subject_id::BidsSubjectId::parse(
            subject_dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(""),
        )
        .to_dir_name();
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

/// Enumerate subjects and load cohort labels in one call.
pub fn enumerate_and_labels(cfg: &AppConfig) -> Result<(SubjectList, CohortMap)> {
    let subjects = enumerate_subjects(&cfg.consolidated_data_dir)?;
    if subjects.is_empty() {
        warn!(
            "no subjects found in {}",
            cfg.consolidated_data_dir.display()
        );
    }
    info!(n_subjects = subjects.len(), "loaded subject list");
    let labels = load_cohort_labels(&cfg.consolidated_data_dir)?;
    info!(n_labeled = labels.len(), "loaded cohort labels");
    Ok((subjects, labels))
}

/// Collect FC matrices from all subjects, stack into `[N, C, C]` with aligned cohort labels.
/// Subjects missing data or cohort labels are skipped.
pub fn collect_stack(
    subjects: &[(String, PathBuf)],
    labels: &HashMap<String, bool>,
    reader: &impl Fn(&hdf5::File) -> Result<Option<Array2<f64>>>,
) -> Result<Option<(Array3<f64>, Vec<bool>)>> {
    let mut mats: Vec<Array2<f64>> = Vec::new();
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

/// Extract a symmetric ROI submatrix from a full FC matrix.
pub fn extract_roi_submatrix(full: &Array2<f64>, idx: &[usize]) -> Array2<f64> {
    full.select(Axis(0), idx).select(Axis(1), idx)
}

pub struct RunOneParams<'a> {
    pub subjects: &'a [(String, PathBuf)],
    pub labels: &'a HashMap<String, bool>,
    pub results_dir: &'a Path,
    pub task: &'a str,
    pub source: &'a str,
    pub level: &'a str,
    pub roi_suffix: &'a str,
    pub n_perm: u32,
    pub primary_t: f64,
    pub seed: u64,
}

/// Collect → permute → write for one analysis cell. Skips if too few subjects.
pub fn run_one(
    p: &RunOneParams<'_>,
    reader: impl Fn(&hdf5::File) -> Result<Option<Array2<f64>>>,
) -> Result<()> {
    let Some((stack, lab)) = collect_stack(p.subjects, p.labels, &reader)? else {
        warn!(
            task = p.task,
            source = p.source,
            level = p.level,
            "no data collected, skipping"
        );
        return Ok(());
    };
    let n_a = lab.iter().filter(|&&l| l).count();
    let n_c = lab.len() - n_a;
    if n_a < 2 || n_c < 2 {
        warn!(
            task = p.task,
            source = p.source,
            level = p.level,
            n_anhedonic = n_a,
            n_control = n_c,
            "insufficient group size, skipping"
        );
        return Ok(());
    }
    if stack.shape()[1] < 4 {
        warn!(
            task = p.task,
            source = p.source,
            level = p.level,
            n_rois = stack.shape()[1],
            "ROI count < 4; FDR/NBS may be unreliable"
        );
    }
    info!(
        task = p.task,
        source = p.source,
        level = p.level,
        roi_suffix = p.roi_suffix,
        n_subjects = lab.len(),
        n_anhedonic = n_a,
        n_control = n_c,
        "running permutation test"
    );
    let result: PermResult = run_permutation(stack.view(), &lab, p.n_perm, p.seed, p.primary_t);
    crate::io::write_analysis_result(
        p.results_dir,
        p.task,
        p.source,
        p.level,
        p.roi_suffix,
        &result,
        p.n_perm,
        p.primary_t,
        p.seed,
    )?;
    info!(
        task = p.task,
        source = p.source,
        level = p.level,
        roi_suffix = p.roi_suffix,
        "wrote results"
    );
    Ok(())
}

/// Read ROI indices (as usize) from a subject's H5 file.
#[allow(dead_code)]
pub fn read_roi_indices(file: &hdf5::File) -> Option<Vec<usize>> {
    let candidates = [
        "/06fc/mvmd/full_run_std_roi/roi_indices",
        "/04hht/full_run_std_roi/roi_indices",
    ];
    for path in &candidates {
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

/// Compute `results_dir` for stage 09 outputs.
pub fn results_dir(cfg: &AppConfig) -> PathBuf {
    cfg.consolidated_data_dir
        .parent()
        .unwrap_or(&cfg.consolidated_data_dir)
        .join("09fc_analysis")
}
