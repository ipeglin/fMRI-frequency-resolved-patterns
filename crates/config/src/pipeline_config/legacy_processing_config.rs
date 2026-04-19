use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmriProcessConfig {
    pub bold_ts_dir: PathBuf,
    pub output_dir: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub subject_file: Option<PathBuf>,
    /// Force reprocessing of subjects that already exist in output files
    #[serde(default)]
    pub force: bool,
}

impl Default for FmriProcessConfig {
    fn default() -> Self {
        Self {
            bold_ts_dir: PathBuf::from("/path/to/raw_fmri_data"),
            output_dir: PathBuf::from("/path/to/output"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            subject_file: None,
            force: false,
        }
    }
}

impl fmt::Display for FmriProcessConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.bold_ts_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;
        writeln!(
            f,
            "  Cortical Atlast LUT: {}",
            self.cortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  Subcortical Atlast LUT: {}",
            self.subcortical_atlas_lut.display()
        )?;
        write!(f, "  Subjects: {:?}", self.subject_file)
    }
}
