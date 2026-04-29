use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::atlas::RoiSelectionSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    // Shared IO paths (used by multiple stages)
    #[serde(default)]
    pub task_sampling_rate: f64,
    pub tcp_annex_remote: String,
    pub csv_output_dir: PathBuf,
    pub tcp_repo_dir: PathBuf,
    pub fmriprep_output_dir: PathBuf,
    pub consolidated_data_dir: PathBuf,
    pub subject_filter_dir: PathBuf,
    pub task_regressors_output_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    pub data_splitting_output_dir: PathBuf,
    /// Directory where classification runners write per-analysis JSON results
    /// (one file per `<analysis>__<source>.json`) for downstream plotting.
    pub classification_results_dir: PathBuf,

    // Global behavior flags
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub dry_run: bool,

    // Stage-local params
    #[serde(default)]
    pub parcellation: ParcellationParams,
    #[serde(default)]
    pub mvmd: MvmdParams,
    #[serde(default)]
    pub feature_extraction: FeatureExtractionParams,
    #[serde(default)]
    pub classification: ClassificationParams,

    /// Single source of truth for which atlas rows the spec-dependent stages
    /// (04mvmd `_roi`, 05hilbert `_roi`, 06fc `_roi`, 07feature_extraction)
    /// operate on. Empty selection is currently rejected by 07; reserved for
    /// future "all ROIs" mode.
    #[serde(default)]
    pub roi_selection: RoiSelectionSpec,
}

impl AppConfig {
    /// Resolved output directory for classification result JSON files. The
    /// configured `classification_results_dir` is suffixed with the active
    /// `roi_selection.name` so different ROI selections (e.g. `vpfc_mpfc_amy`
    /// vs `dmn`) write to disjoint subdirectories. Falls back to the unsuffixed
    /// directory when `roi_selection.name` is empty.
    pub fn resolved_classification_results_dir(&self) -> PathBuf {
        if self.roi_selection.name.is_empty() {
            self.classification_results_dir.clone()
        } else {
            self.classification_results_dir.join(&self.roi_selection.name)
        }
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            task_sampling_rate: 1.25, // TR = 800ms
            tcp_repo_dir: PathBuf::from("/path/to/tcp"),
            csv_output_dir: PathBuf::from("/path/to/csv"),
            fmriprep_output_dir: PathBuf::from("/path/to/raw_fmri_data"),
            consolidated_data_dir: PathBuf::from("/path/to/fmri_timeseries"),
            subject_filter_dir: PathBuf::from("/path/to/subject_filters"),
            task_regressors_output_dir: PathBuf::from("/path/to/glm_conditions"),
            cortical_atlas: PathBuf::from("/path/to/cortical_atlas"),
            subcortical_atlas: PathBuf::from("/path/to/subcortical_atlas"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            data_splitting_output_dir: PathBuf::from("/path/to/data_split"),
            classification_results_dir: PathBuf::from("/path/to/classification_results"),
            tcp_annex_remote: String::new(),
            force: false,
            dry_run: false,
            parcellation: ParcellationParams::default(),
            mvmd: MvmdParams::default(),
            feature_extraction: FeatureExtractionParams::default(),
            classification: ClassificationParams::default(),
            roi_selection: RoiSelectionSpec::default(),
        }
    }
}

impl fmt::Display for AppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AppConfig:")?;
        writeln!(f, "  task_sampling_rate: {} Hz", self.task_sampling_rate)?;
        writeln!(f, "  tcp_repo_dir: {}", self.tcp_repo_dir.display())?;
        writeln!(f, "  cvs_output_dir: {}", self.csv_output_dir.display())?;
        writeln!(
            f,
            "  fmriprep_output_dir: {}",
            self.fmriprep_output_dir.display()
        )?;
        writeln!(
            f,
            "  consolidated_data_dir: {}",
            self.consolidated_data_dir.display()
        )?;
        writeln!(
            f,
            "  subject_filter_dir: {}",
            self.subject_filter_dir.display()
        )?;
        writeln!(
            f,
            "  task_regressors_output_dir: {}",
            self.task_regressors_output_dir.display()
        )?;
        writeln!(f, "  cortical_atlas: {}", self.cortical_atlas.display())?;
        writeln!(
            f,
            "  subcortical_atlas: {}",
            self.subcortical_atlas.display()
        )?;
        writeln!(
            f,
            "  cortical_atlas_lut: {}",
            self.cortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  subcortical_atlas_lut: {}",
            self.subcortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  data_splitting_output_dir: {}",
            self.data_splitting_output_dir.display()
        )?;
        writeln!(
            f,
            "  classification_results_dir: {}",
            self.classification_results_dir.display()
        )?;
        writeln!(f, "  force: {}", self.force)?;
        writeln!(f, "  dry_run: {}", self.dry_run)?;
        writeln!(
            f,
            "  parcellation.voxelwise_zscore: {}",
            self.parcellation.voxelwise_zscore
        )?;
        writeln!(f, "  mvmd.num_modes: {}", self.mvmd.num_modes)?;
        match &self.feature_extraction.cnn_weights_path {
            Some(p) => writeln!(f, "  feature_extraction.cnn_weights_path: {}", p.display())?,
            None => writeln!(f, "  feature_extraction.cnn_weights_path: <random init>")?,
        }
        writeln!(
            f,
            "  roi_selection: name={} cortical={:?} subcortical={:?}",
            self.roi_selection.name,
            self.roi_selection.cortical_regions,
            self.roi_selection.subcortical_regions
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcellationParams {
    /// Apply voxel-wise z-score normalization before parcellation. Produces
    /// additional HDF5 datasets (`tcp_cortical_voxelzscore`, etc.) alongside
    /// the raw outputs.
    #[serde(default)]
    pub voxelwise_zscore: bool,
}

impl Default for ParcellationParams {
    fn default() -> Self {
        Self {
            voxelwise_zscore: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvmdParams {
    pub num_modes: usize,
}

impl Default for MvmdParams {
    fn default() -> Self {
        Self { num_modes: 10 }
    }
}

/// How to coerce a spectrogram (height = 224 frequency bins, width = T time
/// samples) to the DenseNet-201 expected `224×224` input.
///
/// `Pad`: zero-pad the time axis on the right to width 224. Preserves the
/// original signal granularity exactly — no interpolation.
///
/// `Resize`: bicubic upsample/downsample to `224×224`. Compromises granularity
/// but exposes the full receptive field. Kept as an opt-in for ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ImageFitMode {
    Pad,
    Resize,
}

impl Default for ImageFitMode {
    fn default() -> Self {
        Self::Pad
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionParams {
    #[serde(default)]
    pub cnn_weights_path: Option<PathBuf>,
    /// Apply log1p amplitude compression to HHT spectrograms before min-max
    /// normalization. Recommended: heavy-tailed Hilbert spectra benefit from
    /// log compression to preserve granularity in low-amplitude bins.
    #[serde(default = "default_hht_log_amp")]
    pub hht_log_amp: bool,
    #[serde(default)]
    pub image_fit: ImageFitMode,
}

fn default_hht_log_amp() -> bool {
    true
}

impl Default for FeatureExtractionParams {
    fn default() -> Self {
        Self {
            cnn_weights_path: Some(PathBuf::from("cnn_model_weights/densenet201_imagenet.pt")),
            hht_log_amp: default_hht_log_amp(),
            image_fit: ImageFitMode::default(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationParams {
    pub knn_num_neighbors: usize,
    #[serde(default = "default_knn_metric")]
    pub knn_metric: String,
}

fn default_knn_metric() -> String {
    "cosine".to_string()
}

impl Default for ClassificationParams {
    fn default() -> Self {
        Self {
            knn_num_neighbors: 3,
            knn_metric: default_knn_metric(),
        }
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    toml::from_str(&s).with_context(|| format!("Failed to parse config: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_shared_fields() {
        let toml = r#"
            task_sampling_rate = 1.25
            csv_output_dir = "/c"
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            consolidated_data_dir = "/b"
            subject_filter_dir = "/sf"
            task_regressors_output_dir = "/glm"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            data_splitting_output_dir = "/ds"
            classification_results_dir = "/cr"
            tcp_annex_remote = "uuid"
        "#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.consolidated_data_dir.to_str().unwrap(), "/b");
        assert_eq!(cfg.tcp_repo_dir.to_str().unwrap(), "/t");
        assert!(!cfg.force);
        assert_eq!(cfg.mvmd.num_modes, 10);
    }

    #[test]
    fn parses_stage_params() {
        let toml = r#"
            task_sampling_rate = 1.25
            csv_output_dir = "/c"
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            consolidated_data_dir = "/b"
            subject_filter_dir = "/sf"
            task_regressors_output_dir = "/glm"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            data_splitting_output_dir = "/ds"
            classification_results_dir = "/cr"
            tcp_annex_remote = "uuid"

            [mvmd]
            num_modes = 20

            [parcellation]
            voxelwise_zscore = true
        "#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.mvmd.num_modes, 20);
        assert!(cfg.parcellation.voxelwise_zscore);
    }

    #[test]
    fn resolved_classification_results_dir_suffixes_with_roi_name() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = "vpfc_mpfc_amy".to_string();
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results/vpfc_mpfc_amy")
        );
    }

    #[test]
    fn resolved_classification_results_dir_unsuffixed_when_name_empty() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = String::new();
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results")
        );
    }
}
