use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FcConfig {
    pub bold_ts_dir: PathBuf,
    #[serde(default)]
    pub force: bool,
}

impl Default for FcConfig {
    fn default() -> Self {
        Self {
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            force: false,
        }
    }
}

impl fmt::Display for FcConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Functional Connectivity")?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        Ok(())
    }
}
