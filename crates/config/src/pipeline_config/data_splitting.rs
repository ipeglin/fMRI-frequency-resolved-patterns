use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataSplitConfig {
    pub subject_filter_dir: PathBuf,
    pub bold_ts_dir: PathBuf,
    pub training_subjects_path: PathBuf,
    pub test_subjects_path: PathBuf,
    pub validation_subjects_path: PathBuf,
    #[serde(default)]
    pub force: bool,
}

impl Default for DataSplitConfig {
    fn default() -> Self {
        Self {
            subject_filter_dir: PathBuf::from("/path/to/subject_filter_output"),
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            training_subjects_path: PathBuf::from("/path/to/training_subjects.csv"),
            test_subjects_path: PathBuf::from("/path/to/test_subjects.csv"),
            validation_subjects_path: PathBuf::from("/path/to/validation_subjects.csv"),
            force: false,
        }
    }
}

impl fmt::Display for DataSplitConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Data Splitting for Classifier Training")?;
        writeln!(
            f,
            "  Subject Group Filter Dir: {}",
            self.subject_filter_dir.display()
        )?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        writeln!(
            f,
            "  Model Training Subjects File: {}",
            self.training_subjects_path.display()
        )?;
        writeln!(
            f,
            "  Model Test Subjects File: {}",
            self.test_subjects_path.display()
        )?;
        writeln!(
            f,
            "  Model Validation Subjects File: {}",
            self.validation_subjects_path.display()
        )?;
        Ok(())
    }
}
