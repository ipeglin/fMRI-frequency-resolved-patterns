use anyhow::{bail, Result};
use nalgebra::Matrix4;
use ndarray::{Array2, Array3, Array4, ShapeBuilder};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, NiftiVolume, ReaderOptions};
use std::{collections::HashSet, path::PathBuf};
use tracing::{debug, trace};

/// Standardization strategy for masked signal preprocessing.
///
/// Based on nilearn's standardization options for NiftiLabelsMasker.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub enum Standardize {
    /// No standardization
    #[default]
    None,
    /// Z-score using sample standard deviation (ddof=1)
    /// This is the recommended method.
    ZscoreSample,
    /// Percent signal change: (signal - mean) / |mean| * 100
    Psc,
}

/// Configuration for signal preprocessing in maskers.
///
/// Based on nilearn's NiftiLabelsMasker preprocessing options.
/// This controls how extracted time series are cleaned before being returned.
#[derive(Debug, Clone)]
pub struct MaskerSignalConfig {
    /// Whether to perform linear detrending
    pub detrend: bool,
    /// Standardization strategy applied to ROI timeseries after parcellation
    pub standardize: Standardize,
    /// Whether to apply voxel-wise z-score normalization before parcellation.
    ///
    /// Each voxel's timeseries is independently normalized by its own temporal
    /// mean and standard deviation before ROI averaging. This corresponds to
    /// "subject-level Z-score maps" as described in the neuroimaging literature:
    /// each voxel's amplitude is standardized across the full scan duration,
    /// making signal amplitude comparable across voxels and subjects prior to
    /// any spatial averaging.
    pub voxelwise_zscore: bool,
}

impl Default for MaskerSignalConfig {
    fn default() -> Self {
        Self {
            detrend: false,
            standardize: Standardize::None,
            voxelwise_zscore: false,
        }
    }
}

impl MaskerSignalConfig {
    /// Create a preprocessing config with detrending and z-score standardization.
    /// This matches nilearn's default behavior.
    pub fn with_defaults() -> Self {
        Self {
            detrend: true,
            standardize: Standardize::ZscoreSample,
            voxelwise_zscore: false,
        }
    }

    /// Builder method to set detrending
    pub fn detrend(mut self, detrend: bool) -> Self {
        self.detrend = detrend;
        self
    }

    /// Builder method to set standardization
    pub fn standardize(mut self, standardize: Standardize) -> Self {
        self.standardize = standardize;
        self
    }

    /// Builder method to enable voxel-wise z-score normalization before parcellation
    pub fn voxelwise_zscore(mut self, enabled: bool) -> Self {
        self.voxelwise_zscore = enabled;
        self
    }

    /// Check if any preprocessing is enabled
    pub fn is_enabled(&self) -> bool {
        self.detrend || self.standardize != Standardize::None || self.voxelwise_zscore
    }
}

/// Linear detrending of a 2D signal array.
///
/// Removes the mean and linear trend from each row (each ROI's time series).
/// Based on nilearn's `_detrend` function.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
///
/// # Returns
/// Detrended signal with the same shape
fn detrend_signal(data: &Array2<f32>) -> Array2<f32> {
    let (n_rois, n_timepoints) = data.dim();

    if n_timepoints <= 1 {
        return data.clone();
    }

    let mut result = data.clone();

    // Create time regressor: [0, 1, 2, ..., n-1], then center and normalize
    let regressor_mean = (n_timepoints - 1) as f32 / 2.0;
    let mut regressor: Vec<f32> = (0..n_timepoints)
        .map(|i| i as f32 - regressor_mean)
        .collect();

    // Normalize regressor
    let regressor_std: f32 = regressor.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if regressor_std > f32::EPSILON {
        for r in &mut regressor {
            *r /= regressor_std;
        }
    }

    for roi_idx in 0..n_rois {
        // Step 1: Remove mean
        let mean: f32 = result.row(roi_idx).sum() / n_timepoints as f32;
        for t in 0..n_timepoints {
            result[[roi_idx, t]] -= mean;
        }

        // Step 2: Remove linear trend
        // Compute dot product of regressor with signal (already mean-centered)
        let dot: f32 = result
            .row(roi_idx)
            .iter()
            .zip(regressor.iter())
            .map(|(&val, &r)| val * r)
            .sum();

        // Subtract the linear component
        for (t, &r) in regressor.iter().enumerate() {
            result[[roi_idx, t]] -= dot * r;
        }
    }

    result
}

/// Standardize signal using the specified strategy.
///
/// Based on nilearn's `standardize_signal` function.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
/// * `standardize` - Standardization strategy
/// * `already_detrended` - Whether the signal has already been detrended (mean removed)
///
/// # Returns
/// Standardized signal with the same shape
fn standardize_signal(
    data: &Array2<f32>,
    standardize: Standardize,
    already_detrended: bool,
) -> Array2<f32> {
    let (n_rois, n_timepoints) = data.dim();

    if n_timepoints <= 1 {
        return data.clone();
    }

    match standardize {
        Standardize::None => data.clone(),

        Standardize::ZscoreSample => {
            let mut result = data.clone();

            for roi_idx in 0..n_rois {
                // Remove mean if not already detrended
                if !already_detrended {
                    let mean: f32 = result.row(roi_idx).sum() / n_timepoints as f32;
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] -= mean;
                    }
                }

                // Compute sample standard deviation (ddof=1)
                let variance: f32 = result.row(roi_idx).iter().map(|&v| v * v).sum::<f32>()
                    / (n_timepoints - 1) as f32;
                let std = variance.sqrt();

                // Avoid division by zero
                let std = if std < f32::EPSILON { 1.0 } else { std };

                for t in 0..n_timepoints {
                    result[[roi_idx, t]] /= std;
                }
            }

            result
        }

        Standardize::Psc => {
            let mut result = data.clone();

            for roi_idx in 0..n_rois {
                let mean: f32 = data.row(roi_idx).sum() / n_timepoints as f32;
                let abs_mean = mean.abs();

                if abs_mean < f32::EPSILON {
                    // Mean is zero, PSC is meaningless - set to zero
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] = 0.0;
                    }
                } else {
                    for t in 0..n_timepoints {
                        result[[roi_idx, t]] = (data[[roi_idx, t]] - mean) / abs_mean * 100.0;
                    }
                }
            }

            result
        }
    }
}

/// Apply voxel-wise z-score normalization to a 4D BOLD array in-place.
///
/// Each voxel's timeseries `[x, y, z, :]` is independently normalized by its
/// own temporal mean and sample standard deviation (ddof=1). Voxels with near-
/// zero variance (e.g. outside the brain mask) are left unchanged.
pub(super) fn voxelwise_zscore_bold(data: &mut Array4<f32>) {
    let shape = data.shape().to_vec();
    let (nx, ny, nz, nt) = (shape[0], shape[1], shape[2], shape[3]);

    if nt <= 1 {
        return;
    }

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let mean: f32 = (0..nt).map(|t| data[[x, y, z, t]]).sum::<f32>() / nt as f32;
                let variance: f32 = (0..nt)
                    .map(|t| {
                        let d = data[[x, y, z, t]] - mean;
                        d * d
                    })
                    .sum::<f32>()
                    / (nt - 1) as f32;
                let std = variance.sqrt();

                if std < f32::EPSILON {
                    continue;
                }

                for t in 0..nt {
                    data[[x, y, z, t]] = (data[[x, y, z, t]] - mean) / std;
                }
            }
        }
    }
}

/// Apply preprocessing (detrend and/or standardize) to extracted time series.
///
/// Based on nilearn's signal cleaning pipeline.
///
/// # Arguments
/// * `data` - Array2 of shape (n_rois, n_timepoints)
/// * `config` - Preprocessing configuration
///
/// # Returns
/// Preprocessed signal with the same shape
pub fn preprocess_signals(data: &Array2<f32>, config: &MaskerSignalConfig) -> Array2<f32> {
    if !config.is_enabled() {
        return data.clone();
    }

    // Apply detrending first (order matters: detrend before standardize)
    let data = if config.detrend {
        detrend_signal(data)
    } else {
        data.clone()
    };

    // Apply standardization
    standardize_signal(&data, config.standardize, config.detrend)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_standardize_zscore() {
        // Create a signal with known mean and std (1 ROI, 5 timepoints)
        let data = array![[1.0_f32, 2.0, 3.0, 4.0, 5.0]];

        let standardized = standardize_signal(&data, Standardize::ZscoreSample, false);

        // Check mean is approximately 0
        let mean: f32 = standardized.row(0).sum() / 5.0;
        assert!(mean.abs() < 1e-5, "Mean {} should be near zero", mean);

        // Check std is approximately 1 (using ddof=1)
        let variance: f32 = standardized.row(0).iter().map(|&v| v * v).sum::<f32>() / 4.0;
        let std = variance.sqrt();
        assert!((std - 1.0).abs() < 1e-5, "Std {} should be near 1.0", std);
    }

    #[test]
    fn test_standardize_psc() {
        // Create a signal with known mean (1 ROI, 5 timepoints)
        // Mean = 100
        let data = array![[100.0_f32, 110.0, 90.0, 105.0, 95.0]];

        let standardized = standardize_signal(&data, Standardize::Psc, false);

        // PSC[0] = (100 - 100) / 100 * 100 = 0
        // PSC[1] = (110 - 100) / 100 * 100 = 10
        // PSC[2] = (90 - 100) / 100 * 100 = -10
        assert!((standardized[[0, 0]] - 0.0).abs() < 1e-5);
        assert!((standardized[[0, 1]] - 10.0).abs() < 1e-5);
        assert!((standardized[[0, 2]] - (-10.0)).abs() < 1e-5);
    }

    #[test]
    fn test_voxelwise_zscore_bold() {
        // 2x1x1 spatial, 5 timepoints
        // voxel (0,0,0): [1, 2, 3, 4, 5]  mean=3, std=sqrt(2.5)
        // voxel (1,0,0): [10, 10, 10, 10, 10]  constant -> should be left unchanged
        let mut data = Array4::<f32>::zeros((2, 1, 1, 5));
        for t in 0..5usize {
            data[[0, 0, 0, t]] = (t + 1) as f32;
            data[[1, 0, 0, t]] = 10.0;
        }

        voxelwise_zscore_bold(&mut data);

        // Voxel 0: mean should be ~0, std ~1
        let mean0: f32 = (0..5).map(|t| data[[0, 0, 0, t]]).sum::<f32>() / 5.0;
        assert!(mean0.abs() < 1e-5, "mean {} should be near zero", mean0);
        let var0: f32 = (0..5).map(|t| data[[0, 0, 0, t]].powi(2)).sum::<f32>() / 4.0;
        assert!(
            (var0.sqrt() - 1.0).abs() < 1e-5,
            "std {} should be near 1",
            var0.sqrt()
        );

        // Voxel 1: constant, left unchanged
        for t in 0..5 {
            assert_eq!(data[[1, 0, 0, t]], 10.0);
        }
    }

    #[test]
    fn test_masker_signal_config_builder() {
        let config = MaskerSignalConfig::default()
            .detrend(true)
            .standardize(Standardize::ZscoreSample);

        assert!(config.detrend);
        assert_eq!(config.standardize, Standardize::ZscoreSample);
        assert!(config.is_enabled());
    }

    #[test]
    fn test_masker_signal_config_disabled() {
        let config = MaskerSignalConfig::default();

        assert!(!config.detrend);
        assert_eq!(config.standardize, Standardize::None);
        assert!(!config.is_enabled());
    }
}
