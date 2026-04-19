use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MvmdConfig {
    pub tcp_dir: PathBuf,
    pub bold_ts_dir: PathBuf,
    pub num_modes: usize,
    #[serde(default)]
    pub force: bool,
}

impl Default for MvmdConfig {
    fn default() -> Self {
        Self {
            tcp_dir: PathBuf::from("/path/to/tcp"),
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            num_modes: 10 as usize,
            force: false,
        }
    }
}

impl fmt::Display for MvmdConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "MVMD Decomposition:")?;
        writeln!(f, "  TCP Dir: {}", self.tcp_dir.display())?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        Ok(())
    }
}