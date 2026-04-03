pub mod annex;
pub mod bids_filename;
pub mod bids_subject_id;
pub mod polars_csv;
pub mod tcp_config;

pub use tcp_config::{
    CwtConfig, MvmdConfig, TcpFmriParcellationConfig, TcpFmriProcessConfig,
    TcpSubjectSelectionConfig, TcpTrialSegmentationConfig,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_subject_selection: TcpSubjectSelectionConfig,
    #[serde(default)]
    pub tcp_fmri_parcellation: TcpFmriParcellationConfig,
    #[serde(default)]
    pub tcp_fmri_segment_trials: TcpTrialSegmentationConfig,
    #[serde(default)]
    pub tcp_mvmd: MvmdConfig,
    #[serde(default)]
    pub tcp_cwt: CwtConfig,
    #[serde(default)]
    pub tcp_fmri_process: TcpFmriProcessConfig,
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let cfg: AppConfig =
        toml::from_str(&s).with_context(|| format!("Failed to parse TOML: {}", path.display()))?;

    Ok(cfg)
}
