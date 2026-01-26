pub mod annex;
pub mod polars_csv;
pub mod tcp_config;

pub use tcp_config::{TCPSubjectSelectionConfig, TCPfMRIPreprocessConfig, TCPfMRIProcessConfig};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_subject_selection: TCPSubjectSelectionConfig,
    #[serde(default)]
    pub tcp_fmri_preprocess: TCPfMRIPreprocessConfig,
    pub tcp_fmri_process: TCPfMRIProcessConfig,
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let cfg: AppConfig =
        toml::from_str(&s).with_context(|| format!("Failed to parse TOML: {}", path.display()))?;

    Ok(cfg)
}
