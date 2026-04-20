use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    // Shared IO paths (used by multiple stages)
    pub tcp_annex_remote: String,
    pub tcp_repo_dir: PathBuf,
    pub fmriprep_output_dir: PathBuf,
    pub parcellated_ts_dir: PathBuf,
    pub subject_filter_dir: PathBuf,
    pub task_regressors_output_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    pub data_splitting_output_dir: PathBuf,

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
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            tcp_repo_dir: PathBuf::from("/path/to/tcp"),
            fmriprep_output_dir: PathBuf::from("/path/to/raw_fmri_data"),
            parcellated_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            subject_filter_dir: PathBuf::from("/path/to/subject_filters"),
            task_regressors_output_dir: PathBuf::from("/path/to/glm_conditions"),
            cortical_atlas: PathBuf::from("/path/to/cortical_atlas"),
            subcortical_atlas: PathBuf::from("/path/to/subcortical_atlas"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            data_splitting_output_dir: PathBuf::from("/path/to/data_split"),
            tcp_annex_remote: String::new(),
            force: false,
            dry_run: false,
            parcellation: ParcellationParams::default(),
            mvmd: MvmdParams::default(),
            feature_extraction: FeatureExtractionParams::default(),
        }
    }
}

impl fmt::Display for AppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AppConfig:")?;
        writeln!(f, "  tcp_repo_dir: {}", self.tcp_repo_dir.display())?;
        writeln!(
            f,
            "  fmriprep_output_dir: {}",
            self.fmriprep_output_dir.display()
        )?;
        writeln!(
            f,
            "  parcellated_ts_dir: {}",
            self.parcellated_ts_dir.display()
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionParams {
    #[serde(default)]
    pub cnn_weights_path: Option<PathBuf>,
}

impl Default for FeatureExtractionParams {
    fn default() -> Self {
        Self {
            cnn_weights_path: Some(PathBuf::from("cnn_model_weights/densenet201_imagenet.pt")),
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
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            parcellated_ts_dir = "/b"
            subject_filter_dir = "/sf"
            task_regressors_output_dir = "/glm"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            data_splitting_output_dir = "/ds"
            tcp_annex_remote = "uuid"
        "#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.parcellated_ts_dir.to_str().unwrap(), "/b");
        assert_eq!(cfg.tcp_repo_dir.to_str().unwrap(), "/t");
        assert!(!cfg.force);
        assert_eq!(cfg.mvmd.num_modes, 10);
    }

    #[test]
    fn parses_stage_params() {
        let toml = r#"
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            parcellated_ts_dir = "/b"
            subject_filter_dir = "/sf"
            task_regressors_output_dir = "/glm"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            data_splitting_output_dir = "/ds"
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
}
