use std::{fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpSubjectSelectionConfig {
    pub tcp_dir: PathBuf,
    pub tcp_annex_remote: String,
    pub output_dir: PathBuf,
    #[serde(default)]
    pub filters: Option<Vec<String>>,
    #[serde(default)]
    pub dry_run: bool,
}

impl Default for TcpSubjectSelectionConfig {
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

impl fmt::Display for TcpSubjectSelectionConfig {
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
pub struct TcpFmriParcellationConfig {
    pub fmri_dir: PathBuf,
    pub filter_dir: PathBuf,
    pub output_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    #[serde(default)]
    pub dry_run: bool,
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

impl Default for TcpFmriParcellationConfig {
    fn default() -> Self {
        Self {
            fmri_dir: PathBuf::from("/path/to/raw_fmri_data"),
            filter_dir: PathBuf::from("/path/to/output"),
            output_dir: PathBuf::from("/path/to/output"),
            cortical_atlas: PathBuf::from("/path/to/atlas"),
            subcortical_atlas: PathBuf::from("/path/to/atlas"),
            dry_run: false,
            force: false,
            voxelwise_zscore: false,
        }
    }
}

impl fmt::Display for TcpFmriParcellationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Preprocessing:")?;
        writeln!(f, "  fMRI Dir: {}", self.fmri_dir.display())?;
        writeln!(f, "  Filter Dir: {}", self.filter_dir.display())?;
        writeln!(f, "  Output Dir: {}", self.output_dir.display())?;

        write!(f, "  Dry run: {}", self.dry_run)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpTrialSegmentationConfig {
    pub tcp_dir: PathBuf,
    pub bold_ts_dir: PathBuf,
    /// Directory where per-condition GLM onset/duration TSV files are written.
    pub glm_output_dir: PathBuf,
    /// Force reprocessing of blocks that already exist in output files.
    #[serde(default)]
    pub force: bool,
}

impl Default for TcpTrialSegmentationConfig {
    fn default() -> Self {
        Self {
            tcp_dir: PathBuf::from("/path/to/tcp"),
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            glm_output_dir: PathBuf::from("/path/to/glm_conditions"),
            force: false,
        }
    }
}

impl fmt::Display for TcpTrialSegmentationConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "TPC fMRI Trail Segmentation:")?;
        writeln!(f, "  TCP Dir: {}", self.tcp_dir.display())?;
        writeln!(f, "  fMRI Timeseries Dir: {}", self.bold_ts_dir.display())?;
        write!(f, "  GLM Output Dir: {}", self.glm_output_dir.display())?;
        Ok(())
    }
}

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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CwtConfig {
    pub bold_ts_dir: PathBuf,
    #[serde(default)]
    pub force: bool,
}

impl Default for CwtConfig {
    fn default() -> Self {
        Self {
            bold_ts_dir: PathBuf::from("/path/to/fmri_timeseries"),
            force: false,
        }
    }
}

impl fmt::Display for CwtConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Continuous Wavelet Transform")?;
        writeln!(f, "  fMRI Time Series Dir: {}", self.bold_ts_dir.display())?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TcpFmriProcessConfig {
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

impl Default for TcpFmriProcessConfig {
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

impl fmt::Display for TcpFmriProcessConfig {
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
