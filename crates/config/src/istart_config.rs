use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ISTARTSubjectSelectionConfig {
    pub istart_dir: PathBuf,
    pub istart_annex_remote: String,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for ISTARTSubjectSelectionConfig {
    fn default() -> Self {
        Self {
            istart_dir: PathBuf::from("/path/to/tcp"),
            istart_annex_remote: String::from(""),
            output_dir: PathBuf::from("/path/to/output"),
            dry_run: false,
        }
    }
}

impl fmt::Display for ISTARTSubjectSelectionConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "ISTART Subject Selection:")?;
        writeln!(f, "  ISTART Dir: {}", self.istart_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;

        write!(f, "  Dry run: {}", self.dry_run)
    }
}
