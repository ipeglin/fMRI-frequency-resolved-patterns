use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tracing::{debug, warn};
use utils::bids_subject_id::BidsSubjectId;
use utils::hdf5_io;
use utils::polars_csv;

/// Binary class labels for the classification task.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    HealthyControl = 0,
    Anhedonic = 1,
}

impl Label {
    pub fn as_i32(self) -> i32 {
        self as i32
    }

    pub fn from_i32(v: i32) -> Option<Self> {
        match v {
            0 => Some(Self::HealthyControl),
            1 => Some(Self::Anhedonic),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum FeatureSource {
    Cwt,
    Hht,
}

impl FeatureSource {
    fn dir(&self) -> &'static str {
        match self {
            Self::Cwt => "cwt",
            Self::Hht => "hht",
        }
    }
}

#[derive(Debug, Clone)]
pub enum FeatureSubgroup {
    WholeBand,
    Block(usize),
}

impl FeatureSubgroup {
    fn path(&self) -> String {
        match self {
            Self::WholeBand => "whole-band".to_string(),
            Self::Block(n) => format!("blocks/block_{}", n),
        }
    }
}

/// How to collapse a per-file feature tensor into a single sample vector.
#[derive(Debug, Clone, Copy)]
pub enum FeatureAggregation {
    /// Use the precomputed mean across ROIs: `[feat_dim]` (default 1920).
    Mean,
    /// Use one ROI row from `per_roi[n_rois, feat_dim]`.
    PerRoi(usize),
    /// Flatten the full `per_roi` matrix: `[n_rois * feat_dim]`.
    Concat,
}

#[derive(Debug, Clone)]
pub struct FeatureSpec {
    pub source: FeatureSource,
    pub subgroup: FeatureSubgroup,
    pub aggregation: FeatureAggregation,
}

impl FeatureSpec {
    fn group_path(&self) -> String {
        format!("features/{}/{}", self.source.dir(), self.subgroup.path())
    }
}

/// Load `subjectkey` column as BIDS directory names (e.g. `sub-NDARINV...`).
pub fn load_subject_ids(csv: &Path) -> Result<Vec<String>> {
    let df = polars_csv::read_dataframe(csv)
        .with_context(|| format!("failed to read {}", csv.display()))?;

    let col = df
        .column("subjectkey")
        .with_context(|| format!("missing 'subjectkey' column in {}", csv.display()))?;

    Ok(col
        .str()?
        .into_no_null_iter()
        .map(|s| BidsSubjectId::parse(s).to_dir_name())
        .collect())
}

/// Build a `sub-<id> -> Label` map from `healthy_controls.csv` + `anhedonic.csv`.
pub fn load_labels(filter_dir: &Path) -> Result<HashMap<String, Label>> {
    let mut map = HashMap::new();

    for id in load_subject_ids(&filter_dir.join("healthy_controls.csv"))? {
        map.insert(id, Label::HealthyControl);
    }
    for id in load_subject_ids(&filter_dir.join("anhedonic.csv"))? {
        map.insert(id, Label::Anhedonic);
    }

    Ok(map)
}

/// Read a single feature vector from one `.h5` file per the given spec.
fn read_feature_vector(h5_path: &Path, spec: &FeatureSpec) -> Result<Vec<f32>> {
    let group_path = spec.group_path();

    match spec.aggregation {
        FeatureAggregation::Mean => {
            let (data, shape, _): (Vec<f32>, _, _) = hdf5_io::read(h5_path, &group_path, "mean")
                .with_context(|| {
                    format!(
                        "failed to read {}/mean from {}",
                        group_path,
                        h5_path.display()
                    )
                })?;
            if shape.len() != 1 {
                bail!(
                    "mean dataset in {} has unexpected shape {:?}",
                    h5_path.display(),
                    shape
                );
            }
            Ok(data)
        }
        FeatureAggregation::PerRoi(roi_idx) => {
            let (data, shape, _): (Vec<f32>, _, _) = hdf5_io::read(h5_path, &group_path, "per_roi")
                .with_context(|| {
                    format!(
                        "failed to read {}/per_roi from {}",
                        group_path,
                        h5_path.display()
                    )
                })?;
            let [n_rois, feat_dim] = match shape.as_slice() {
                &[a, b] => [a, b],
                _ => bail!(
                    "per_roi dataset in {} has unexpected shape {:?}",
                    h5_path.display(),
                    shape
                ),
            };
            if roi_idx >= n_rois {
                bail!(
                    "roi index {} out of range for per_roi with {} rows",
                    roi_idx,
                    n_rois
                );
            }
            let start = roi_idx * feat_dim;
            Ok(data[start..start + feat_dim].to_vec())
        }
        FeatureAggregation::Concat => {
            let (data, shape, _): (Vec<f32>, _, _) = hdf5_io::read(h5_path, &group_path, "per_roi")
                .with_context(|| {
                    format!(
                        "failed to read {}/per_roi from {}",
                        group_path,
                        h5_path.display()
                    )
                })?;
            if shape.len() != 2 {
                bail!(
                    "per_roi dataset in {} has unexpected shape {:?}",
                    h5_path.display(),
                    shape
                );
            }
            Ok(data)
        }
    }
}

/// Enumerate `.h5` files directly under a subject directory.
fn list_subject_h5(subject_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(subject_dir)
        .with_context(|| format!("failed to read {}", subject_dir.display()))?
    {
        let p = entry?.path();
        if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some("h5") {
            out.push(p);
        }
    }
    out.sort();
    Ok(out)
}

/// For a list of BIDS directory names (`sub-...`), build `(X, y)` by reading one
/// feature vector per `.h5` file. Each `.h5` becomes one labeled sample.
/// Subjects without matching labels, without `.h5` files, or with failed reads
/// are logged and skipped.
pub fn build_dataset(
    parcellated_ts_dir: &Path,
    subjects: &[String],
    labels: &HashMap<String, Label>,
    spec: &FeatureSpec,
) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut ys: Vec<i32> = Vec::new();

    for subject in subjects {
        let Some(label) = labels.get(subject) else {
            warn!(subject = %subject, "no label found, skipping");
            continue;
        };

        let dir = parcellated_ts_dir.join(subject);
        if !dir.is_dir() {
            warn!(subject = %subject, path = %dir.display(), "subject directory missing");
            continue;
        }

        let files = list_subject_h5(&dir)?;
        if files.is_empty() {
            warn!(subject = %subject, "no .h5 files found");
            continue;
        }

        for file in files {
            match read_feature_vector(&file, spec) {
                Ok(v) => {
                    debug!(
                        subject = %subject,
                        file = %file.display(),
                        feat_dim = v.len(),
                        label = ?label,
                        "loaded feature vector"
                    );
                    xs.push(v);
                    ys.push(label.as_i32());
                }
                Err(e) => {
                    warn!(
                        subject = %subject,
                        file = %file.display(),
                        error = %e,
                        "failed to read features, skipping"
                    );
                }
            }
        }
    }

    Ok((xs, ys))
}

/// Number of ROIs recorded in the first `per_roi` dataset found for `spec`.
/// Useful when iterating per-ROI classifiers without hardcoding the atlas size.
pub fn detect_n_rois(
    parcellated_ts_dir: &Path,
    subjects: &[String],
    source: FeatureSource,
    subgroup: &FeatureSubgroup,
) -> Result<usize> {
    let group_path = format!("features/{}/{}", source.dir(), subgroup.path());
    for subject in subjects {
        let dir = parcellated_ts_dir.join(subject);
        let Ok(files) = list_subject_h5(&dir) else {
            continue;
        };
        for file in files {
            let Ok(f) = hdf5::File::open(&file) else {
                continue;
            };
            let Ok(g) = f.group(&group_path) else {
                continue;
            };
            if let Ok(ds) = g.dataset("per_roi") {
                let shape = ds.shape();
                if let &[n_rois, _] = shape.as_slice() {
                    return Ok(n_rois);
                }
            }
        }
    }
    bail!("could not detect n_rois: no readable per_roi dataset in any subject")
}

/// Read `(labels, roi_indices)` from the first available features subgroup.
///
/// Labels come from the group-level `labels` attribute (comma-separated). ROI
/// indices come from the `roi_indices` dataset when present; otherwise the
/// returned `roi_indices` is empty, signalling older h5 files written before
/// that dataset was added.
pub fn read_roi_metadata(
    parcellated_ts_dir: &Path,
    subjects: &[String],
    source: FeatureSource,
    subgroup: &FeatureSubgroup,
) -> Result<(Vec<String>, Vec<usize>)> {
    let group_path = format!("features/{}/{}", source.dir(), subgroup.path());
    for subject in subjects {
        let dir = parcellated_ts_dir.join(subject);
        let Ok(files) = list_subject_h5(&dir) else {
            continue;
        };
        for file in files {
            let Ok(f) = hdf5::File::open(&file) else {
                continue;
            };
            let Ok(g) = f.group(&group_path) else {
                continue;
            };

            let attrs = hdf5_io::read_attrs(&g).unwrap_or_default();
            let labels_csv = attrs.iter().find_map(|a| {
                if a.name == "labels" {
                    if let utils::hdf5_io::H5AttrValue::String(s) = &a.value {
                        return Some(s.clone());
                    }
                }
                None
            });
            let labels: Vec<String> = labels_csv
                .map(|s| s.split(',').map(|t| t.to_string()).collect())
                .unwrap_or_default();

            let roi_indices: Vec<usize> = g
                .dataset("roi_indices")
                .and_then(|ds| ds.read_raw::<u32>())
                .map(|v| v.into_iter().map(|x| x as usize).collect())
                .unwrap_or_default();

            return Ok((labels, roi_indices));
        }
    }
    bail!("could not read ROI metadata: no readable features group in any subject")
}

// -----------------------------------------------------------------------------
// FC (functional connectivity) features
// -----------------------------------------------------------------------------

/// How to subset an `[n, n]` FC matrix before flattening.
#[derive(Debug, Clone)]
pub enum FcSelection {
    /// Keep every row and column (full `[n, n]`).
    All,
    /// Keep only rows and columns at these atlas indices.
    /// The list is deduped and sorted internally.
    SubsetRois(Vec<usize>),
}

/// Strategy for turning a 2D FC matrix (possibly subset) into a 1D feature.
#[derive(Debug, Clone, Copy)]
pub enum FcFlatten {
    /// Strict upper triangle (i < j). Requires a symmetric square selection.
    /// Length = k*(k-1)/2 for a k x k matrix.
    UpperTriangle,
    /// Flatten selected rows across all columns (seed-to-whole-brain).
    /// Length = k_rows * n_cols.
    RowsFlat,
}

/// Specification for a single FC feature vector per `.h5` file.
#[derive(Debug, Clone)]
pub struct FcSpec {
    /// HDF5 group path, e.g. `"fc/standardized"` or
    /// `"fc/mvmd/whole-band/slow_4"` or `"fc/blocks_standardized/block_0"`.
    pub group_path: String,
    /// Dataset within the group, e.g. `"pearson"`, `"fisher_z"`, or
    /// `"fisher_z_mean"` (for slow-band aggregates).
    pub dataset: String,
    pub selection: FcSelection,
    pub flatten: FcFlatten,
}

fn read_fc_matrix(h5_path: &Path, group_path: &str, dataset: &str) -> Result<(Vec<f64>, usize)> {
    let (data, shape, _): (Vec<f64>, _, _) = hdf5_io::read(h5_path, group_path, dataset)
        .with_context(|| {
            format!(
                "failed to read {}/{} from {}",
                group_path,
                dataset,
                h5_path.display()
            )
        })?;
    let n = match shape.as_slice() {
        &[a, b] if a == b => a,
        _ => bail!(
            "FC dataset {}/{} in {} has non-square shape {:?}",
            group_path,
            dataset,
            h5_path.display(),
            shape
        ),
    };
    Ok((data, n))
}

fn read_fc_vector(h5_path: &Path, spec: &FcSpec) -> Result<Vec<f32>> {
    let (mat, n) = read_fc_matrix(h5_path, &spec.group_path, &spec.dataset)?;

    let rows: Vec<usize> = match &spec.selection {
        FcSelection::All => (0..n).collect(),
        FcSelection::SubsetRois(idxs) => {
            let mut v = idxs.clone();
            v.sort_unstable();
            v.dedup();
            for &r in &v {
                if r >= n {
                    bail!("roi index {} out of range for FC dim {}", r, n);
                }
            }
            v
        }
    };

    let get = |i: usize, j: usize| mat[i * n + j];
    let clean = |v: f64| if v.is_finite() { v as f32 } else { 0.0 };

    match spec.flatten {
        FcFlatten::UpperTriangle => {
            // Symmetric selection (rows == cols) only.
            let k = rows.len();
            if k < 2 {
                bail!("UpperTriangle needs at least 2 rows, got {}", k);
            }
            let mut out = Vec::with_capacity(k * (k - 1) / 2);
            for i in 0..k {
                for j in (i + 1)..k {
                    out.push(clean(get(rows[i], rows[j])));
                }
            }
            Ok(out)
        }
        FcFlatten::RowsFlat => {
            let mut out = Vec::with_capacity(rows.len() * n);
            for &i in &rows {
                for j in 0..n {
                    out.push(clean(get(i, j)));
                }
            }
            Ok(out)
        }
    }
}

/// Build `(X, y)` over subjects by loading one FC feature vector per `.h5` file.
/// Mirrors `build_dataset` but for FC-derived features.
pub fn build_fc_dataset(
    parcellated_ts_dir: &Path,
    subjects: &[String],
    labels: &HashMap<String, Label>,
    spec: &FcSpec,
) -> Result<(Vec<Vec<f32>>, Vec<i32>)> {
    let mut xs: Vec<Vec<f32>> = Vec::new();
    let mut ys: Vec<i32> = Vec::new();

    for subject in subjects {
        let Some(label) = labels.get(subject) else {
            warn!(subject = %subject, "no label found, skipping");
            continue;
        };

        let dir = parcellated_ts_dir.join(subject);
        if !dir.is_dir() {
            warn!(subject = %subject, path = %dir.display(), "subject directory missing");
            continue;
        }

        let files = list_subject_h5(&dir)?;
        if files.is_empty() {
            warn!(subject = %subject, "no .h5 files found");
            continue;
        }

        for file in files {
            match read_fc_vector(&file, spec) {
                Ok(v) => {
                    debug!(
                        subject = %subject,
                        file = %file.display(),
                        feat_dim = v.len(),
                        label = ?label,
                        "loaded FC feature vector"
                    );
                    xs.push(v);
                    ys.push(label.as_i32());
                }
                Err(e) => {
                    warn!(
                        subject = %subject,
                        file = %file.display(),
                        error = %e,
                        "failed to read FC features, skipping"
                    );
                }
            }
        }
    }

    Ok((xs, ys))
}
