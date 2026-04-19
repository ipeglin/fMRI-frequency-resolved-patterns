use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionConfig {
    pub bold_ts_dir: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    #[serde(default)]
    pub cnn_weights_path: Option<PathBuf>,
    #[serde(default)]
    pub force: bool,
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            cnn_weights_path: Some(PathBuf::from("cnn_model_weights/densenet201_imagenet.pt")),
            force: false,
        }
    }
}

impl fmt::Display for FeatureExtractionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CNN Feature Extraction")?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        writeln!(f, "  Cortical Atlas LUT:   {}", self.cortical_atlas_lut.display())?;
        writeln!(f, "  Subcortical Atlas LUT:{}", self.subcortical_atlas_lut.display())?;
        match &self.cnn_weights_path {
            Some(p) => writeln!(f, "  CNN Weights: {}", p.display())?,
            None => writeln!(f, "  CNN Weights: <random init>")?,
        }
        Ok(())
    }
}
