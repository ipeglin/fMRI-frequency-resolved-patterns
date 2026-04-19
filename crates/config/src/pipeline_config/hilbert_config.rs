use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HilbertHuangConfig {
    pub bold_ts_dir: PathBuf,
    #[serde(default)]
    pub force: bool,
}

impl Default for HilbertHuangConfig {
    fn default() -> Self {
        Self {
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            force: false,
        }
    }
}

impl fmt::Display for HilbertHuangConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Hilbert-Huang Transform")?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        Ok(())
    }
}
