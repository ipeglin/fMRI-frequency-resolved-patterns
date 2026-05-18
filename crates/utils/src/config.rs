use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::{
    fmt,
    path::{Path, PathBuf},
};

use crate::atlas::RoiSelectionSpec;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    // Shared IO paths (used by multiple stages)
    #[serde(default)]
    pub task_sampling_rate: f64,
    pub tcp_annex_remote: String,
    pub tcp_repo_dir: PathBuf,
    pub fmriprep_output_dir: PathBuf,
    pub consolidated_data_dir: PathBuf,
    pub subject_filter_dir: PathBuf,
    pub cortical_atlas: PathBuf,
    pub subcortical_atlas: PathBuf,
    pub cortical_atlas_lut: PathBuf,
    pub subcortical_atlas_lut: PathBuf,
    /// Directory where classification runners write per-analysis JSON results
    /// (one file per `<analysis>__<source>.json`) for downstream plotting.
    pub classification_results_dir: PathBuf,
    /// Root directory for diagnostic TSV dumps (gated by `dump_intermediates`).
    /// Sub-directories are created per pipeline stage (e.g. `01fmri_parcellation/`,
    /// `04hht/`). Files are named with BIDS-ish `desc-` entities.
    #[serde(default = "default_intermediates_output_dir")]
    pub intermediates_output_dir: PathBuf,

    // Global behavior flags
    #[serde(default)]
    pub force: bool,
    #[serde(default)]
    pub dry_run: bool,
    /// When true, subjects must have a `task-restAP_run-01` file to qualify for
    /// inclusion. When false (default), any `task-restAP` run is sufficient.
    #[serde(default)]
    pub restap_run01_only: bool,
    /// Write diagnostic TSV files for each pipeline stage. Files land in
    /// `intermediates_output_dir/<stage>/` with BIDS-ish naming. Default false.
    #[serde(default)]
    pub dump_intermediates: bool,

    // Stage-local params
    #[serde(default)]
    pub parcellation: ParcellationParams,
    #[serde(default)]
    pub hht: HhtParams,
    #[serde(default)]
    pub feature_extraction: FeatureExtractionParams,
    #[serde(default)]
    pub classification: ClassificationParams,
    #[serde(default)]
    pub fc_analysis: FcAnalysisParams,

    /// Single source of truth for which atlas rows the spec-dependent stages
    /// (04mvmd `_roi`, 05hilbert `_roi`, 06fc `_roi`, 07feature_extraction)
    /// operate on. Empty selection is currently rejected by 07; reserved for
    /// future "all ROIs" mode.
    #[serde(default)]
    pub roi_selection: RoiSelectionSpec,
}

impl AppConfig {
    /// Resolved output directory for classification result JSON files. The
    /// configured `classification_results_dir` is suffixed with the active
    /// `roi_selection.name` so different ROI selections (e.g. `vpfc_mpfc_amy`
    /// vs `dmn`) write to disjoint subdirectories. When
    /// `roi_selection.cortical_networks` is non-empty the leaf is further
    /// suffixed with `__net-{sorted_networks.join('_')}` so swapping the
    /// network filter under the same `name` does not overwrite prior results.
    /// Falls back to the unsuffixed directory when `roi_selection.name` is
    /// empty.
    pub fn resolved_classification_results_dir(&self) -> PathBuf {
        if self.roi_selection.name.is_empty() {
            return self.classification_results_dir.clone();
        }
        let mut leaf = self.roi_selection.name.clone();
        if !self.roi_selection.cortical_networks.is_empty() {
            let mut nets: Vec<String> = self
                .roi_selection
                .cortical_networks
                .iter()
                .map(|r| r.name().to_string())
                .collect();
            nets.sort();
            leaf.push_str("__net-");
            leaf.push_str(&nets.join("_"));
        }
        self.classification_results_dir.join(leaf)
    }
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            task_sampling_rate: 1.25, // TR = 800ms
            tcp_repo_dir: PathBuf::from("/path/to/tcp"),
            fmriprep_output_dir: PathBuf::from("/path/to/raw_fmri_data"),
            consolidated_data_dir: PathBuf::from("/path/to/fmri_timeseries"),
            subject_filter_dir: PathBuf::from("/path/to/subject_filters"),
            cortical_atlas: PathBuf::from("/path/to/cortical_atlas"),
            subcortical_atlas: PathBuf::from("/path/to/subcortical_atlas"),
            cortical_atlas_lut: PathBuf::from("/path/to/cortical_atlas_lut"),
            subcortical_atlas_lut: PathBuf::from("/path/to/subcortical_atlas_lut"),
            classification_results_dir: PathBuf::from("/path/to/classification_results"),
            intermediates_output_dir: default_intermediates_output_dir(),
            tcp_annex_remote: String::new(),
            force: false,
            dry_run: false,
            restap_run01_only: false,
            dump_intermediates: false,
            parcellation: ParcellationParams::default(),
            hht: HhtParams::default(),
            feature_extraction: FeatureExtractionParams::default(),
            classification: ClassificationParams::default(),
            fc_analysis: FcAnalysisParams::default(),
            roi_selection: RoiSelectionSpec::default(),
        }
    }
}

fn default_intermediates_output_dir() -> PathBuf {
    PathBuf::from("out/intermediates")
}

impl fmt::Display for AppConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "AppConfig:")?;
        writeln!(f, "  task_sampling_rate: {} Hz", self.task_sampling_rate)?;
        writeln!(f, "  tcp_repo_dir: {}", self.tcp_repo_dir.display())?;
        writeln!(
            f,
            "  fmriprep_output_dir: {}",
            self.fmriprep_output_dir.display()
        )?;
        writeln!(
            f,
            "  consolidated_data_dir: {}",
            self.consolidated_data_dir.display()
        )?;
        writeln!(
            f,
            "  subject_filter_dir: {}",
            self.subject_filter_dir.display()
        )?;
        writeln!(f, "  cortical_atlas: {}", self.cortical_atlas.display())?;
        writeln!(
            f,
            "  subcortical_atlas: {}",
            self.subcortical_atlas.display()
        )?;
        writeln!(
            f,
            "  cortical_atlas_lut: {}",
            self.cortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  subcortical_atlas_lut: {}",
            self.subcortical_atlas_lut.display()
        )?;
        writeln!(
            f,
            "  classification_results_dir: {}",
            self.classification_results_dir.display()
        )?;
        writeln!(
            f,
            "  intermediates_output_dir: {}",
            self.intermediates_output_dir.display()
        )?;
        writeln!(f, "  force: {}", self.force)?;
        writeln!(f, "  dry_run: {}", self.dry_run)?;
        writeln!(f, "  restap_run01_only: {}", self.restap_run01_only)?;
        writeln!(f, "  dump_intermediates: {}", self.dump_intermediates)?;
        writeln!(
            f,
            "  parcellation.standardize: {}",
            self.parcellation.standardize
        )?;
        writeln!(
            f,
            "  parcellation.voxelwise_zscore: {}",
            self.parcellation.voxelwise_zscore
        )?;
        writeln!(f, "  hht.num_modes: {}", self.hht.num_modes)?;
        match &self.feature_extraction.cnn_weights_path {
            Some(p) => writeln!(f, "  feature_extraction.cnn_weights_path: {}", p.display())?,
            None => writeln!(f, "  feature_extraction.cnn_weights_path: <random init>")?,
        }
        writeln!(
            f,
            "  roi_selection: name={} cortical={:?} subcortical={:?}",
            self.roi_selection.name,
            self.roi_selection.cortical_regions,
            self.roi_selection.subcortical_regions
        )?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParcellationParams {
    /// Apply per-ROI z-score standardization (sample std, ddof=1) to the
    /// parcellated timeseries after ROI averaging. Matches nilearn
    /// `NiftiLabelsMasker(standardize="zscore_sample")`. Default true.
    #[serde(default = "default_parcellation_standardize")]
    pub standardize: bool,
    /// Apply voxel-wise z-score normalization **before** parcellation. Each
    /// voxel's timeseries is independently normalized by its own temporal mean
    /// and std prior to ROI averaging. Expensive — adds significant wall time.
    /// Opt-in only (default false).
    #[serde(default)]
    pub voxelwise_zscore: bool,
}

fn default_parcellation_standardize() -> bool {
    true
}

impl Default for ParcellationParams {
    fn default() -> Self {
        Self {
            standardize: default_parcellation_standardize(),
            voxelwise_zscore: false,
        }
    }
}

/// Single-source default values for ADMM and MVMD parameters.
pub mod defaults {
    pub const ADMM_TOLERANCE: f64 = 1e-8;
    pub const ADMM_TAU: f64 = 1e-3;
    pub const ADMM_MAX_ITERATIONS: usize = 500;
    pub const MVMD_ALPHA: f64 = 1000.0;
}

fn default_hht_num_modes() -> usize {
    10
}
fn default_hht_log_amp() -> bool {
    true
}
fn default_hht_envelope_normalize() -> bool {
    true
}
fn default_mvmd_alpha() -> f64 {
    defaults::MVMD_ALPHA
}
fn default_admm_tolerance() -> f64 {
    defaults::ADMM_TOLERANCE
}
fn default_admm_tau() -> f64 {
    defaults::ADMM_TAU
}
fn default_admm_max_iterations() -> usize {
    defaults::ADMM_MAX_ITERATIONS
}
fn default_na_mvmd() -> bool {
    false
}
fn default_noise_channels() -> usize {
    1
}
fn default_noise_std_ratio() -> f64 {
    0.8
}
fn default_noise_seed() -> u64 {
    0x00C0_FFEE
}
fn default_frequency_init() -> FrequencyInitConfig {
    FrequencyInitConfig::Exponential
}

/// Initialization method for MVMD center frequencies.
///
/// Matches the `FrequencyInit` enum in the `04hht` crate; defined here so
/// config deserialization does not depend on that crate.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FrequencyInitConfig {
    /// All omegas start at 0 (classic VMD convention).
    Zero,
    /// Omegas linearly spaced in [0, 0.5].
    Linear,
    /// Omegas log-spaced in [0, 0.5] — reference NA-MVMD init.
    #[default]
    Exponential,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HhtParams {
    /// Number of MVMD modes (K).
    #[serde(default = "default_hht_num_modes")]
    pub num_modes: usize,
    /// MVMD penalty term (alpha).
    #[serde(default = "default_mvmd_alpha")]
    pub alpha: f64,
    /// ADMM tolerance.
    #[serde(default = "default_admm_tolerance")]
    pub admm_tolerance: f64,
    /// ADMM step-size (tau).
    #[serde(default = "default_admm_tau")]
    pub admm_tau: f64,
    /// ADMM max iterations.
    #[serde(default = "default_admm_max_iterations")]
    pub admm_max_iterations: usize,
    /// Apply log1p amplitude compression to the HHT envelope before normalization.
    /// Compresses raw dynamic range before per-channel max-divide.
    #[serde(default = "default_hht_log_amp")]
    pub hht_log_amp: bool,
    /// Normalize HHT envelope per-channel (divide each channel across all modes
    /// and timepoints by its maximum). Bounds the per-channel spectrogram in [0, 1].
    #[serde(default = "default_hht_envelope_normalize")]
    pub hht_envelope_normalize: bool,
    /// Enable Noise-Assisted MVMD (NA-MVMD). Normalizes each input channel to
    /// unit variance, appends WGN channels, applies the Generalized Cross-Spectrum
    /// (GCS) single-snapshot centroid for the omega update, then rescales output
    /// modes by the original per-channel std.
    #[serde(default = "default_na_mvmd")]
    pub na_mvmd: bool,
    /// Number of White Gaussian Noise channels to append (N). Only used when na_mvmd = true.
    #[serde(default = "default_noise_channels")]
    pub noise_channels: usize,
    /// Noise standard deviation relative to per-channel unit-variance-normalized signal.
    /// Matches the reference `na_mvmd.m` default of 0.8.
    #[serde(default = "default_noise_std_ratio")]
    pub noise_std_ratio: f64,
    /// RNG seed for deterministic noise generation.
    #[serde(default = "default_noise_seed")]
    pub noise_seed: u64,
    /// Center-frequency initialization method for MVMD.
    /// `exponential` (log-spaced) matches the reference NA-MVMD initialization.
    #[serde(default = "default_frequency_init")]
    pub frequency_init: FrequencyInitConfig,
}

impl Default for HhtParams {
    fn default() -> Self {
        Self {
            num_modes: default_hht_num_modes(),
            alpha: default_mvmd_alpha(),
            admm_tolerance: default_admm_tolerance(),
            admm_tau: default_admm_tau(),
            admm_max_iterations: default_admm_max_iterations(),
            hht_log_amp: default_hht_log_amp(),
            hht_envelope_normalize: default_hht_envelope_normalize(),
            na_mvmd: default_na_mvmd(),
            noise_channels: default_noise_channels(),
            noise_std_ratio: default_noise_std_ratio(),
            noise_seed: default_noise_seed(),
            frequency_init: default_frequency_init(),
        }
    }
}

/// How to coerce a spectrogram (height = 224 frequency bins, width = T time
/// samples) to the DenseNet-201 expected `224×224` input.
///
/// `Pad`: zero-pad the time axis on the right to width 224. Preserves the
/// original signal granularity exactly — no interpolation.
///
/// `Resize`: bicubic upsample/downsample to `224×224`. Compromises granularity
/// but exposes the full receptive field. Kept as an opt-in for ablation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum ImageFitMode {
    #[default]
    Pad,
    Resize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureExtractionParams {
    #[serde(default)]
    pub cnn_weights_path: Option<PathBuf>,
    #[serde(default)]
    pub image_fit: ImageFitMode,
    /// Also produce a frequency-smoothed Hilbert spectrum (`hht_smoothed` /
    /// `hht_roi_smoothed`) alongside the skeleton. Smoothing uses a per-segment
    /// moving average along the frequency axis only (time axis untouched), with
    /// kernel width derived from Huang's optimal cell count `N = round(n_t / 5)`.
    /// Default true.
    #[serde(default = "default_hht_smoothed")]
    pub hht_smoothed: bool,
    /// Also produce instantaneous energy analyses (`hht_ie` / `hht_roi_ie`).
    /// IE(t) = Σ_ω H²(ω, t) collapses frequency → a per-ROI power time-series,
    /// encoded as a 224-level one-hot amplitude image for DenseNet input.
    /// Computed from the skeleton spectrum (H = amplitude after log1p/normalize).
    /// Default true.
    #[serde(default = "default_hht_ie")]
    pub hht_ie: bool,
}

fn default_hht_smoothed() -> bool {
    true
}

fn default_hht_ie() -> bool {
    true
}

impl Default for FeatureExtractionParams {
    fn default() -> Self {
        Self {
            cnn_weights_path: Some(PathBuf::from("cnn_model_weights/densenet201_imagenet.pt")),
            image_fit: ImageFitMode::default(),
            hht_smoothed: default_hht_smoothed(),
            hht_ie: default_hht_ie(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationParams {
    pub knn_num_neighbors: usize,
    #[serde(default = "default_knn_metric")]
    pub knn_metric: String,
}

fn default_knn_metric() -> String {
    "cosine".to_string()
}

impl Default for ClassificationParams {
    fn default() -> Self {
        Self {
            knn_num_neighbors: 3,
            knn_metric: default_knn_metric(),
        }
    }
}

/// Parameters for the supplementary FC group-difference analysis (stage 09).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FcAnalysisParams {
    /// Number of label permutations for NBS / FWER / FDR null distributions.
    #[serde(default = "default_fc_n_permutations")]
    pub n_permutations: u32,
    /// Primary t-statistic threshold for NBS suprathreshold graph (≈ p<0.001 one-tail).
    #[serde(default = "default_fc_nbs_primary_t")]
    pub nbs_primary_t: f64,
    /// Seed for ChaCha20 RNG — ensures reproducible permutation results.
    #[serde(default = "default_fc_permutation_seed")]
    pub permutation_seed: u64,
}

fn default_fc_n_permutations() -> u32 {
    10_000
}
fn default_fc_nbs_primary_t() -> f64 {
    3.1
}
fn default_fc_permutation_seed() -> u64 {
    0x00C0_FFEE_C0FF_EE00
}

impl Default for FcAnalysisParams {
    fn default() -> Self {
        Self {
            n_permutations: default_fc_n_permutations(),
            nbs_primary_t: default_fc_nbs_primary_t(),
            permutation_seed: default_fc_permutation_seed(),
        }
    }
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    toml::from_str(&s).with_context(|| format!("Failed to parse config: {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_flat_shared_fields() {
        let toml = r#"
            task_sampling_rate = 1.25
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            consolidated_data_dir = "/b"
            subject_filter_dir = "/sf"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            classification_results_dir = "/cr"
            tcp_annex_remote = "uuid"
        "#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.consolidated_data_dir.to_str().unwrap(), "/b");
        assert_eq!(cfg.tcp_repo_dir.to_str().unwrap(), "/t");
        assert!(!cfg.force);
        assert_eq!(cfg.hht.num_modes, 10);
    }

    #[test]
    fn parses_stage_params() {
        let toml = r#"
            task_sampling_rate = 1.25
            tcp_repo_dir = "/t"
            fmriprep_output_dir = "/f"
            consolidated_data_dir = "/b"
            subject_filter_dir = "/sf"
            cortical_atlas = "/ca"
            subcortical_atlas = "/sca"
            cortical_atlas_lut = "/cal"
            subcortical_atlas_lut = "/scal"
            classification_results_dir = "/cr"
            tcp_annex_remote = "uuid"

            [hht]
            num_modes = 20

            [parcellation]
            voxelwise_zscore = true
        "#;
        let cfg: AppConfig = toml::from_str(toml).unwrap();
        assert_eq!(cfg.hht.num_modes, 20);
        assert!(cfg.parcellation.voxelwise_zscore);
    }

    #[test]
    fn resolved_classification_results_dir_suffixes_with_roi_name() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = "vpfc_mpfc_amy".to_string();
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results/vpfc_mpfc_amy")
        );
    }

    #[test]
    fn resolved_classification_results_dir_unsuffixed_when_name_empty() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = String::new();
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results")
        );
    }

    #[test]
    fn resolved_classification_results_dir_appends_sorted_networks() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = "vpfc_mpfc_amy".to_string();
        cfg.roi_selection.cortical_networks = vec!["LimbicB".into(), "LimbicA".into()];
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results/vpfc_mpfc_amy__net-LimbicA_LimbicB")
        );
    }

    #[test]
    fn resolved_classification_results_dir_no_network_suffix_when_empty() {
        let mut cfg = AppConfig::default();
        cfg.classification_results_dir = PathBuf::from("/results");
        cfg.roi_selection.name = "vpfc_mpfc_amy".to_string();
        cfg.roi_selection.cortical_networks = vec![];
        assert_eq!(
            cfg.resolved_classification_results_dir(),
            PathBuf::from("/results/vpfc_mpfc_amy")
        );
    }
}
