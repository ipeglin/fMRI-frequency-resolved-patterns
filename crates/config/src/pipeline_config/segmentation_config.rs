use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialSegmentationConfig {
    pub tcp_dir: PathBuf,
    pub bold_ts_dir: PathBuf,
    /// Directory where per-condition GLM onset/duration TSV files are written.
    pub glm_output_dir: PathBuf,
    /// Force reprocessing of blocks that already exist in output files.
    #[serde(default)]
    pub force: bool,
}

impl Default for TrialSegmentationConfig {
    fn default() -> Self {
        Self {
            tcp_dir: PathBuf::from("/path/to/tcp"),
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            glm_output_dir: PathBuf::from("/path/to/glm_conditions"),
            force: false,
        }
    }
}

impl fmt::Display for TrialSegmentationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Trail Segmentation:")?;
        writeln!(f, "  TCP Dir: {}", self.tcp_dir.display())?;
        writeln!(f, "  fMRI Timeseries Dir: {}", self.bold_ts_dir.display())?;
        write!(f, "  GLM Output Dir: {}", self.glm_output_dir.display())?;
        Ok(())
    }
}
