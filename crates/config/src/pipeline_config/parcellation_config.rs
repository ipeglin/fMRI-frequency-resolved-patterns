use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FmriParcellationConfig {
    pub fmri_dir: PathBuf,
    pub filter_dir: PathBuf,
    pub output_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    /// Force reprocessing of subjects that already have preprocessed output
    #[serde(default)]
    pub force: bool,
    /// Apply voxel-wise z-score normalization before parcellation.
    ///
    /// Each voxel's timeseries is normalized by its own temporal mean and
    /// standard deviation prior to ROI averaging. Produces additional HDF5
    /// datasets (`tcp_cortical_voxelzscore`, etc.) alongside the raw outputs.
    #[serde(default)]
    pub voxelwise_zscore: bool,
}

impl Default for FmriParcellationConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/raw_fmri_data"),
            filter_dir: PathBuf::from("/path/to/output"),
            output_dir: PathBuf::from("/path/to/output"),
            cortical_atlas: PathBuf::from("/path/to/atlas"),
            subcortical_atlas: PathBuf::from("/path/to/atlas"),
            force: false,
            voxelwise_zscore: false,
        }
    }
}

impl fmt::Display for FmriParcellationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.fmri_dir.display())?;
        writeln!(f, "  Filter Dir: {}", self.filter_dir.display())?;
        write!(f, "  Output Dir: {}", self.output_dir.display())
    }
}
