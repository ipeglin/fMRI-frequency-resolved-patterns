use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use tracing::debug;

use utils::bids_filename::BidsFilename;
pub use utils::bids_subject_id::BidsSubjectId;
use utils::hdf5_io;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FeatureSource {
    Cwt,
    Hht,
}

impl FeatureSource {
    pub const fn dir(&self) -> &'static str {
        match self {
            Self::Cwt => "cwt",
            Self::Hht => "hht",
        }
    }
}

impl std::str::FromStr for FeatureSource {
    type Err = String;
    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s {
            "cwt" => Ok(Self::Cwt),
            "hht" => Ok(Self::Hht),
            _ => Err(format!("unknown FeatureSource: {}", s)),
        }
    }
}

/// One of the five DenseNet feature extraction strategies emitted by
/// `crates/07feature_extraction/src/strategies.rs`.
///
/// The `*_chunked` and `*_per_block` variants produce multiple leaf groups per
/// HDF5 file (one per chunk/block); the others produce a single leaf at the
/// analysis group itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisKind {
    BaselineChunked,
    BaselineAveraged,
    BaselineResized,
    TaskConcat,
    TaskPerBlock,
    TaskPerBlockResized,
    TaskAveraged,
    TaskAveragedResized,
}

impl AnalysisKind {
    pub const fn dir(self) -> &'static str {
        match self {
            Self::BaselineChunked => "baseline_chunked",
            Self::BaselineAveraged => "baseline_averaged",
            Self::BaselineResized => "baseline_resized",
            Self::TaskConcat => "task_concat",
            Self::TaskPerBlock => "task_per_block",
            Self::TaskPerBlockResized => "task_per_block_resized",
            Self::TaskAveraged => "task_averaged",
            Self::TaskAveragedResized => "task_averaged_resized",
        }
    }

    pub const fn task(self) -> &'static str {
        match self {
            Self::BaselineChunked | Self::BaselineAveraged | Self::BaselineResized => "restAP",
            Self::TaskConcat
            | Self::TaskPerBlock
            | Self::TaskPerBlockResized
            | Self::TaskAveraged
            | Self::TaskAveragedResized => "hammerAP",
        }
    }

    pub const fn is_multi_leaf(self) -> bool {
        matches!(self, Self::BaselineChunked | Self::TaskPerBlock | Self::TaskPerBlockResized)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Label {
    Control = 0,
    Anhedonic = 1,
}

impl Label {
    pub fn as_i32(self) -> i32 {
        self as i32
    }
}

pub fn load_subject_ids(csv: &Path) -> Result<Vec<String>> {
    let df = utils::polars_csv::read_dataframe(csv)
        .with_context(|| format!("failed to read {}", csv.display()))?;
    let col = df
        .column("subjectkey")
        .with_context(|| format!("missing 'subjectkey' column in {}", csv.display()))?;
    Ok(col
        .str()
        .with_context(|| "subjectkey column is not a string type")?
        .into_no_null_iter()
        .map(|s| BidsSubjectId::parse(s).to_dir_name())
        .collect())
}

/// Load `subjectkey -> Label` from `controls.csv` and `anhedonic.csv`.
pub fn load_labels(dir: &Path) -> Result<HashMap<String, Label>> {
    let mut map = HashMap::new();
    for (filename, label) in [
        ("controls.csv", Label::Control),
        ("anhedonic.csv", Label::Anhedonic),
    ] {
        let path = dir.join(filename);
        if !path.exists() {
            eprintln!("Warning: expected file not found: {:?}", path);
            continue;
        }
        let df = utils::polars_csv::read_dataframe(&path)?;
        let col = df.column("subjectkey")?.str()?;
        for s in col.into_no_null_iter() {
            map.insert(BidsSubjectId::parse(s).to_dir_name(), label);
        }
    }
    Ok(map)
}

pub fn list_subject_h5(subject_dir: &Path) -> Result<Vec<PathBuf>> {
    let mut out = Vec::new();
    if !subject_dir.exists() {
        return Ok(out);
    }
    for entry in std::fs::read_dir(subject_dir)? {
        let path = entry?.path();
        if path.is_file() && path.extension().and_then(|e| e.to_str()) == Some("h5") {
            out.push(path);
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Feature group access
// ---------------------------------------------------------------------------

fn analysis_group_path(source: FeatureSource, kind: AnalysisKind) -> String {
    format!("07feature_extraction/{}/{}", source.dir(), kind.dir())
}

fn leaf_group_path(source: FeatureSource, kind: AnalysisKind, leaf: &str) -> String {
    if leaf.is_empty() {
        analysis_group_path(source, kind)
    } else {
        format!("{}/{}", analysis_group_path(source, kind), leaf)
    }
}

/// Return the names of leaves under an analysis group, or `[""]` for
/// single-leaf analyses (where the analysis group is itself the leaf).
/// Returns empty if the analysis group does not exist for this file.
pub fn list_analysis_leaves(
    h5_path: &Path,
    source: FeatureSource,
    kind: AnalysisKind,
) -> Vec<String> {
    let path = analysis_group_path(source, kind);
    let Ok(file) = hdf5::File::open(h5_path) else {
        return Vec::new();
    };
    let Ok(group) = file.group(&path) else {
        return Vec::new();
    };
    if !kind.is_multi_leaf() {
        return vec![String::new()];
    }
    let prefix = match kind {
        AnalysisKind::BaselineChunked => "chunk_",
        AnalysisKind::TaskPerBlock | AnalysisKind::TaskPerBlockResized => "block_",
        _ => unreachable!(),
    };
    let mut names: Vec<String> = group
        .member_names()
        .unwrap_or_default()
        .into_iter()
        .filter(|n| n.starts_with(prefix))
        .collect();
    names.sort();
    names
}

/// Read the `per_roi [n_rois, feat_dim]` matrix from a leaf and return one
/// `Vec<f32>` per ROI row. `leaf` is `""` for single-leaf analyses.
pub fn read_per_roi(
    h5_path: &Path,
    source: FeatureSource,
    kind: AnalysisKind,
    leaf: &str,
) -> Result<Vec<Vec<f32>>> {
    let group_path = leaf_group_path(source, kind, leaf);
    let file = hdf5::File::open(h5_path)
        .with_context(|| format!("failed to open {}", h5_path.display()))?;
    let group = file
        .group(&group_path)
        .with_context(|| format!("missing group {} in {}", group_path, h5_path.display()))?;
    let (data, shape, _): (Vec<f32>, _, _) = hdf5_io::read_dataset(&group, "per_roi")?;
    let (n_rois, feat_dim) = match shape.as_slice() {
        &[r, d] => (r, d),
        _ => bail!(
            "unexpected per_roi shape {:?} in {}/{}",
            shape,
            h5_path.display(),
            group_path
        ),
    };
    let mut rows = Vec::with_capacity(n_rois);
    for r in 0..n_rois {
        rows.push(data[r * feat_dim..(r + 1) * feat_dim].to_vec());
    }
    Ok(rows)
}

/// Read the `mean [feat_dim]` vector from a leaf.
pub fn read_mean(
    h5_path: &Path,
    source: FeatureSource,
    kind: AnalysisKind,
    leaf: &str,
) -> Result<Vec<f32>> {
    let group_path = leaf_group_path(source, kind, leaf);
    let file = hdf5::File::open(h5_path)
        .with_context(|| format!("failed to open {}", h5_path.display()))?;
    let group = file
        .group(&group_path)
        .with_context(|| format!("missing group {} in {}", group_path, h5_path.display()))?;
    let (data, shape, _): (Vec<f32>, _, _) = hdf5_io::read_dataset(&group, "mean")?;
    if shape.len() != 1 {
        bail!(
            "unexpected mean shape {:?} in {}/{}",
            shape,
            h5_path.display(),
            group_path
        );
    }
    Ok(data)
}

// ---------------------------------------------------------------------------
// Dataset builders
// ---------------------------------------------------------------------------

/// Build a flat per-ROI dataset for a given (source, analysis) pair.
///
/// One row per (file × leaf × ROI). Returns `(xs, ys, groups)` where `groups`
/// is the subject id for each row — useful for subject-stratified or
/// group-aware splits. Files whose BIDS task does not match `kind.task()` are
/// skipped automatically.
pub fn build_per_roi_dataset<I, S>(
    consolidated_data_dir: &Path,
    subject_ids: I,
    labels: &HashMap<String, Label>,
    source: FeatureSource,
    kind: AnalysisKind,
) -> Result<(Vec<Vec<f32>>, Vec<Label>, Vec<String>)>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    let mut groups = Vec::new();
    let task = kind.task();

    for subject in subject_ids {
        let subject = subject.as_ref().to_string();
        let Some(&label) = labels.get(&subject) else {
            debug!(subject, "missing label, skipping");
            continue;
        };
        let dir = consolidated_data_dir.join(&subject);
        if !dir.is_dir() {
            continue;
        }
        let files = list_subject_h5(&dir)?;
        for file in files {
            let bids = BidsFilename::from_path_buf(&file);
            if bids.get("task") != Some(task) {
                continue;
            }
            for leaf in list_analysis_leaves(&file, source, kind) {
                match read_per_roi(&file, source, kind, &leaf) {
                    Ok(rows) => {
                        for (i, row) in rows.into_iter().enumerate() {
                            xs.push(row);
                            ys.push(label);
                            let mut g = subject.clone();
                            if !leaf.is_empty() {
                                g.push_str(&format!("_{}", leaf));
                            }
                            g.push_str(&format!("_roi{:03}", i));
                            groups.push(g);
                        }
                    }
                    Err(e) => {
                        debug!(
                            file = %file.display(),
                            leaf,
                            error = %e,
                            "failed to read per_roi"
                        );
                    }
                }
            }
        }
    }
    Ok((xs, ys, groups))
}

/// Build a per-leaf grouping for analyses that emit multiple leaves per file
/// (`BaselineChunked`, `TaskPerBlock`).
///
/// Returns `leaf_name -> (xs, ys, subject_ids)`. Each row is one ROI of one
/// (subject, file, leaf) combination. Used by ensemble classifiers that fit
/// one model per chunk/block and combine predictions across leaves.
pub fn build_per_leaf_per_roi_dataset<I, S>(
    consolidated_data_dir: &Path,
    subject_ids: I,
    labels: &HashMap<String, Label>,
    source: FeatureSource,
    kind: AnalysisKind,
) -> Result<BTreeMap<String, (Vec<Vec<f32>>, Vec<Label>, Vec<String>)>>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut by_leaf: BTreeMap<String, (Vec<Vec<f32>>, Vec<Label>, Vec<String>)> = BTreeMap::new();
    let task = kind.task();

    for subject in subject_ids {
        let subject = subject.as_ref().to_string();
        let Some(&label) = labels.get(&subject) else {
            continue;
        };
        let dir = consolidated_data_dir.join(&subject);
        if !dir.is_dir() {
            continue;
        }
        let files = list_subject_h5(&dir)?;
        for file in files {
            let bids = BidsFilename::from_path_buf(&file);
            if bids.get("task") != Some(task) {
                continue;
            }
            for leaf in list_analysis_leaves(&file, source, kind) {
                let entry = by_leaf
                    .entry(leaf.clone())
                    .or_insert_with(|| (Vec::new(), Vec::new(), Vec::new()));
                if let Ok(rows) = read_per_roi(&file, source, kind, &leaf) {
                    for (i, row) in rows.into_iter().enumerate() {
                        entry.0.push(row);
                        entry.1.push(label);
                        let mut g = subject.clone();
                        g.push_str(&format!("_roi{:03}", i));
                        entry.2.push(g);
                    }
                }
            }
        }
    }
    Ok(by_leaf)
}
