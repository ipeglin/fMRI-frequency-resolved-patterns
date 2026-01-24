pub mod annex;
pub mod polars_csv;

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_subject_selection: TCPSubjectSelectionConfig,
    #[serde(default)]
    pub tcp_fmri_preprocess: TCPfMRIPreprocessConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPSubjectSelectionConfig {
    pub tcp_dir: PathBuf,
    pub tcp_annex_remote: String,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub filters: Option<Vec<String>>,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TCPSubjectSelectionConfig {
    fn default() -> Self {
        Self {
            tcp_dir: PathBuf::from("/path/to/tcp"),
            tcp_annex_remote: String::from(""),
            output_dir: PathBuf::from("/path/to/output"),
            filters: None,
            dry_run: false,
        }
    }
}

impl fmt::Display for TCPSubjectSelectionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC Subject Selection:")?;
        writeln!(f, "  TCP Dir: {}", self.tcp_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;

        // Handling the Option for cleaner output
        match &self.filters {
            Some(flts) => writeln!(f, "  Filters: {:?}", flts)?,
            None => writeln!(f, "  Filters: None")?,
        }

        write!(f, "  Dry run: {}", self.dry_run)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TCPfMRIPreprocessConfig {
    pub fmri_dir: PathBuf,
    pub filter_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TCPfMRIPreprocessConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/raw_fmri_data"),
            filter_dir: PathBuf::from("/path/to/output"),
            cortical_atlas: PathBuf::from("/path/to/atlas"),
            subcortical_atlas: PathBuf::from("/path/to/atlas"),
            dry_run: false,
        }
    }
}

impl fmt::Display for TCPfMRIPreprocessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.fmri_dir.display())?;
        writeln!(f, "  Filter Dir: {}", self.filter_dir.display())?;

        write!(f, "  Dry run: {}", self.dry_run)
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;

    let cfg: AppConfig =
        toml::from_str(&s).with_context(|| format!("Failed to parse TOML: {}", path.display()))?;

    Ok(cfg)
}
