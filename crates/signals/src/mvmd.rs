use crate::admm::{ADMMConfig, ADMMOptimizer};
use ndarray::{Array1, Array2, Array3};
use polars::prelude::*;
use rustfft::{FftPlanner, num_complex::Complex64};
use tracing::{debug, info, trace};

/// Initialization method for center frequencies in MVMD/VMD algorithms.
#[derive(Debug, Clone, Copy, Default)]
pub enum FrequencyInit {
    /// All omegas start at 0
    #[default]
    Zero,
    /// Omegas are initialized linearly distributed in [0, 0.5]
    Linear,
    /// Omegas are initialized exponentially distributed
    Exponential,
}

/// A single MVMD mode with its time series data and center frequency.
///
/// The DataFrame has channels as columns and time points as rows,
/// compatible with `ConnectivityMatrix::new()` for computing functional connectivity.
pub struct ModeData {
    /// Mode index (0-indexed, ordered by frequency)
    pub mode_index: usize,
    /// Time series data: columns are channels, rows are time points
    pub timeseries: DataFrame,
    /// Final center frequency for this mode
    pub center_frequency: f64,
}

/// Result of MVMD decomposition.
pub struct MVMDResult {
    /// Labels across all channels
    pub channels: Vec<String>,
    /// Decomposed modes with shape (K modes x C channels x T time-points)
    pub modes: Array3<f64>,
    /// Estimated mode center-frequencies over iterations (iter x K)
    pub frequency_traces: Array2<f64>,
    /// Final center frequencies for each mode (K,)
    pub center_frequencies: Array1<f64>,
    /// Number of iterations until convergence
    pub num_iterations: u32,
}

impl MVMDResult {
    /// Return each mode as a `ModeData` containing the time series DataFrame and center frequency.
    ///
    /// Each returned `ModeData` contains:
    /// - `mode_index`: The mode number (0 = lowest frequency)
    /// - `timeseries`: DataFrame with channels as columns and time points as rows,
    ///   directly compatible with `ConnectivityMatrix::new()`
    /// - `center_frequency`: The final center frequency for this mode
    ///
    /// # Returns
    /// A vector of `ModeData`, one per mode, ordered by frequency (lowest first).
    pub fn to_mode_dataframes(&self) -> PolarsResult<Vec<ModeData>> {
        let shape = self.modes.shape();
        let num_modes = shape[0];
        let num_channels = shape[1];
        let num_tpoints = shape[2];

        let mut result = Vec::with_capacity(num_modes);

        for k in 0..num_modes {
            // Build columns for this mode's DataFrame
            // Each column is a channel, each row is a time point
            let mut columns: Vec<Column> = Vec::with_capacity(num_channels);

            for c in 0..num_channels {
                let channel_name = &self.channels[c];
                let values: Vec<f64> = (0..num_tpoints).map(|t| self.modes[[k, c, t]]).collect();
                let series = Series::new(channel_name.as_str().into(), values);
                columns.push(series.into());
            }

            let df = DataFrame::new(columns)?;
            let center_freq = self.center_frequencies[k];

            result.push(ModeData {
                mode_index: k,
                timeseries: df,
                center_frequency: center_freq,
            });
        }

        Ok(result)
    }
}

/// Multivariate Variational Mode Decomposition (MVMD)
///
/// Implementation based on:
/// N. Rehman and H. Aftab (2019) "Multivariate Variational Mode Decomposition",
/// IEEE Transactions on Signal Processing
pub struct MVMD {
    /// Input signal data (channels x time-points)
    data: Vec<Vec<f64>>,
    /// Channel labels
    channels: Vec<String>,
    /// Number of channels
    num_channels: usize,
    /// Number of time points
    num_tpoints: usize,
    /// Bandwidth constraint parameter
    alpha: f64,
    /// Initialization method for center frequencies
    init: FrequencyInit,
    /// Sampling rate of the signal
    sampling_rate: f64,
    /// ADMM configuration for dual ascent
    admm_config: ADMMConfig,
}

impl ADMMOptimizer for MVMD {
    fn admm_config(&self) -> &ADMMConfig {
        &self.admm_config
    }

    fn admm_config_mut(&mut self) -> &mut ADMMConfig {
        &mut self.admm_config
    }
}

impl MVMD {
    /// Create a new MVMD instance from a DataFrame.
    ///
    /// The DataFrame should have columns representing channels and rows representing time-points.
    pub fn from_dataframe(df: &DataFrame, alpha: f64) -> PolarsResult<Self> {
        let channel_labels: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        let num_channels = channel_labels.len();
        let num_tpoints = df.height();

        // Extract data as Vec<Vec<f64>> (channels x time-points)
        // Supports both f32 and f64 columns
        let mut data = Vec::with_capacity(num_channels);
        for col_name in &channel_labels {
            let series = df.column(col_name.as_str())?;
            let values: Vec<f64> = if let Ok(ca) = series.f64() {
                // Column is f64
                ca.into_iter().map(|opt| opt.unwrap_or(0.0)).collect()
            } else if let Ok(ca) = series.f32() {
                // Column is f32, convert to f64
                ca.into_iter()
                    .map(|opt| opt.unwrap_or(0.0) as f64)
                    .collect()
            } else {
                // Try casting to f64
                let casted = series.cast(&DataType::Float64)?;
                casted
                    .f64()?
                    .into_iter()
                    .map(|opt| opt.unwrap_or(0.0))
                    .collect()
            };
            data.push(values);
        }

        Ok(Self {
            data,
            channels: channel_labels,
            num_channels,
            num_tpoints,
            alpha,
            init: FrequencyInit::default(),
            sampling_rate: 1.0,
            admm_config: ADMMConfig::default(),
        })
    }

    /// Create a new MVMD instance from raw data.
    ///
    /// Data should be provided as channels x time-points.
    pub fn new(data: Vec<Vec<f64>>, alpha: f64) -> Self {
        let num_channels = data.len();
        let num_tpoints = data.first().map(|v| v.len()).unwrap_or(0);
        let channels: Vec<String> = (0..num_channels).map(|i| format!("ch_{}", i)).collect();

        Self {
            data,
            channels,
            num_channels,
            num_tpoints,
            alpha,
            init: FrequencyInit::default(),
            sampling_rate: 1.0,
            admm_config: ADMMConfig::default(),
        }
    }

    /// Set the frequency initialization method
    pub fn with_init(mut self, init: FrequencyInit) -> Self {
        self.init = init;
        self
    }

    /// Set the sampling rate
    pub fn with_sampling_rate(mut self, sampling_rate: f64) -> Self {
        self.sampling_rate = sampling_rate;
        self
    }

    /// Set the ADMM configuration
    pub fn with_admm_config(mut self, config: ADMMConfig) -> Self {
        self.admm_config = config;
        self
    }

    /// Get channel labels
    pub fn channels(&self) -> &[String] {
        &self.channels
    }

    /// Get number of channels
    pub fn num_channels(&self) -> usize {
        self.num_channels
    }

    /// Get number of time points
    pub fn num_tpoints(&self) -> usize {
        self.num_tpoints
    }

    /// Decompose the signal into K modes.
    ///
    /// # Arguments
    /// * `num_modes` - Number of modes to decompose into (K)
    ///
    /// # Returns
    /// * `MVMDResult` containing the decomposed modes, center frequencies, and iteration count
    pub fn decompose(&self, num_modes: usize) -> MVMDResult {
        let num_fpoints = self.num_tpoints + 1;

        info!(
            num_modes = num_modes,
            num_channels = self.num_channels,
            num_tpoints = self.num_tpoints,
            num_fpoints = num_fpoints,
            alpha = self.alpha,
            max_iterations = self.admm_config.max_iterations,
            tolerance = self.admm_config.tolerance,
            tau = self.admm_config.tau,
            init = ?self.init,
            "starting MVMD decomposition"
        );

        // Frequency points in normalized frequency [0, 0.5]
        let f_points: Vec<f64> = (0..num_fpoints)
            .map(|i| 0.5 * i as f64 / (num_fpoints - 1) as f64)
            .collect();

        // Initialize center frequencies (omega) - only keep current and next iteration
        let mut omega_current: Vec<f64> = vec![0.0; num_modes];
        let mut omega_next: Vec<f64> = vec![0.0; num_modes];
        self.initialize_omegas(&mut omega_current, num_modes);

        // Store omega history for output
        let mut omega_history: Vec<Vec<f64>> =
            Vec::with_capacity(self.admm_config.max_iterations as usize);
        omega_history.push(omega_current.clone());

        debug!(
            initial_omega = ?omega_current,
            "initialized center frequencies"
        );

        // Transform signal to frequency domain
        debug!("transforming signal to frequency domain");
        let signal_hat = self.to_freq_domain();
        debug!("FFT completed for all channels");

        // Initialize modes in frequency domain: (K modes x C channels x F freq-points)
        let mut modes_hat: Vec<Vec<Vec<Complex64>>> =
            vec![vec![vec![Complex64::new(0.0, 0.0); num_fpoints]; self.num_channels]; num_modes];

        // Dual variables (lambda): only keep current and next iteration (memory optimization)
        let mut lambda_current: Vec<Vec<Complex64>> =
            vec![vec![Complex64::new(0.0, 0.0); num_fpoints]; self.num_channels];
        let mut lambda_next: Vec<Vec<Complex64>> =
            vec![vec![Complex64::new(0.0, 0.0); num_fpoints]; self.num_channels];

        let mut residual_diff = self.admm_config.tolerance + f64::EPSILON;
        let mut n: usize = 0;

        info!(
            estimated_ops_per_iter = num_modes * self.num_channels * num_fpoints * num_modes,
            "starting MVMD iteration loop"
        );

        // Pre-compute modes_sum once, then update incrementally
        let mut modes_sum: Vec<Vec<Complex64>> =
            vec![vec![Complex64::new(0.0, 0.0); num_fpoints]; self.num_channels];

        // Main MVMD iteration loop
        while n < self.admm_config.max_iterations as usize
            && residual_diff > self.admm_config.tolerance
        {
            residual_diff = 0.0;

            // Loop over modes
            for k in 0..num_modes {
                // Store previous mode values for residual calculation
                let omega_k = omega_current[k];

                // Update mode: modes_hat[k] = (signal - sum(other_modes) - 0.5*lambda) / (1 + alpha*(f - omega)^2)
                // sum(other_modes) = modes_sum - modes_hat[k]
                for c in 0..self.num_channels {
                    for f in 0..num_fpoints {
                        let old_val = modes_hat[k][c][f];

                        // sum of other modes = total sum - this mode
                        let sum_other = modes_sum[c][f] - old_val;

                        let numerator =
                            signal_hat[c][f] - sum_other - lambda_current[c][f].scale(0.5);

                        let freq_diff = f_points[f] - omega_k;
                        let denominator = 1.0 + self.alpha * freq_diff * freq_diff;

                        let new_val = numerator.scale(1.0 / denominator);
                        modes_hat[k][c][f] = new_val;

                        // Update modes_sum incrementally
                        modes_sum[c][f] = modes_sum[c][f] - old_val + new_val;

                        // Update residual
                        let diff = new_val - old_val;
                        residual_diff += diff.norm_sqr();
                    }
                }

                // Update center frequency (spectral centroid)
                let mut weighted_sum = 0.0;
                let mut total_power = 0.0;

                for c in 0..self.num_channels {
                    for f in 0..num_fpoints {
                        let power = modes_hat[k][c][f].norm_sqr();
                        weighted_sum += power * f_points[f];
                        total_power += power;
                    }
                }

                omega_next[k] = if total_power > 0.0 {
                    weighted_sum / total_power
                } else {
                    omega_k
                };
            }

            // Dual ascent: lambda = lambda + tau * (sum(modes) - signal)
            // modes_sum is already computed and maintained incrementally
            for c in 0..self.num_channels {
                for f in 0..num_fpoints {
                    let residual = modes_sum[c][f] - signal_hat[c][f];
                    lambda_next[c][f] = lambda_current[c][f] + residual.scale(self.admm_config.tau);
                }
            }

            // Swap current and next
            std::mem::swap(&mut omega_current, &mut omega_next);
            std::mem::swap(&mut lambda_current, &mut lambda_next);

            // Store omega history
            omega_history.push(omega_current.clone());

            n += 1;
            residual_diff /= self.num_tpoints as f64;

            // Log progress at INFO level so it's visible, every 10 iterations
            if n % 10 == 0 || n == 1 {
                info!(
                    iteration = n,
                    max_iterations = self.admm_config.max_iterations,
                    residual_diff = format!("{:.6e}", residual_diff),
                    tolerance = format!("{:.6e}", self.admm_config.tolerance),
                    "MVMD iteration"
                );
            }
        }

        let converged = residual_diff <= self.admm_config.tolerance;
        info!(
            iterations = n,
            converged = converged,
            final_residual = residual_diff,
            tolerance = self.admm_config.tolerance,
            "MVMD iteration loop completed"
        );

        // Post-processing: extract and order results
        debug!("post-processing: ordering results by frequency");
        let mut omega_result: Vec<Vec<f64>> = omega_history
            .iter()
            .map(|row| row.iter().map(|&w| w / self.sampling_rate).collect())
            .collect();

        // Get sorting indices based on final frequencies
        let mut indices: Vec<usize> = (0..num_modes).collect();
        if let Some(last_omega) = omega_result.last() {
            indices.sort_by(|&a, &b| {
                last_omega[a]
                    .partial_cmp(&last_omega[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        // Reorder omega columns
        for row in &mut omega_result {
            let original = row.clone();
            for (new_idx, &old_idx) in indices.iter().enumerate() {
                row[new_idx] = original[old_idx];
            }
        }

        // Reconstruct time-domain modes and reorder
        debug!("reconstructing time-domain modes via IFFT");
        let mut modes_vec: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_modes);
        for (i, &idx) in indices.iter().enumerate() {
            trace!(mode_idx = i, original_idx = idx, "reconstructing mode");
            modes_vec.push(self.to_time_domain(&modes_hat[idx]));
        }

        // Convert to ndarray types
        debug!("converting results to ndarray format");
        let n_timepoints = if !modes_vec.is_empty() && !modes_vec[0].is_empty() {
            modes_vec[0][0].len()
        } else {
            0
        };

        let mut modes = Array3::<f64>::zeros((num_modes, self.num_channels, n_timepoints));
        for (k, mode) in modes_vec.iter().enumerate() {
            for (c, channel) in mode.iter().enumerate() {
                for (t, &val) in channel.iter().enumerate() {
                    modes[[k, c, t]] = val;
                }
            }
        }

        // frequency_traces: (iter x K)
        let n_iters = omega_result.len();
        let mut frequency_traces = Array2::<f64>::zeros((n_iters, num_modes));
        for (i, row) in omega_result.iter().enumerate() {
            for (k, &val) in row.iter().enumerate() {
                frequency_traces[[i, k]] = val;
            }
        }

        // center_frequencies: (K,)
        let center_frequencies = if n_iters > 0 {
            Array1::from_vec(omega_result[n_iters - 1].clone())
        } else {
            Array1::zeros(num_modes)
        };

        info!(
            num_iterations = n as u32,
            modes_shape = ?[num_modes, self.num_channels, n_timepoints],
            center_frequencies = ?center_frequencies.as_slice(),
            "MVMD decomposition completed"
        );

        MVMDResult {
            channels: self.channels.clone(),
            modes,
            frequency_traces,
            center_frequencies,
            num_iterations: n as u32,
        }
    }

    /// Initialize center frequencies based on the chosen method
    fn initialize_omegas(&self, omega: &mut [f64], num_modes: usize) {
        match self.init {
            FrequencyInit::Zero => {
                // Already zero-initialized
            }
            FrequencyInit::Linear => {
                for (i, w) in omega.iter_mut().enumerate() {
                    *w = 0.5 * i as f64 / (num_modes - 1).max(1) as f64;
                }
            }
            FrequencyInit::Exponential => {
                for (i, w) in omega.iter_mut().enumerate() {
                    // 0.5 * 10^(-3 + 3*i/(K-1)) for K modes
                    let exponent = -3.0 + 3.0 * i as f64 / (num_modes - 1).max(1) as f64;
                    *w = 0.5 * 10_f64.powf(exponent);
                }
            }
        }
    }

    /// Transform signal to frequency domain with symmetric padding
    fn to_freq_domain(&self) -> Vec<Vec<Complex64>> {
        let tpoints = self.num_tpoints;
        let pad_left = tpoints / 2;
        let pad_right = tpoints - pad_left;
        let padded_len = tpoints + pad_left + pad_right;

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(padded_len);

        let mut result = Vec::with_capacity(self.num_channels);

        for channel_data in &self.data {
            // Symmetric padding
            let mut padded: Vec<Complex64> = Vec::with_capacity(padded_len);

            // Left padding (mirror)
            for i in (0..pad_left).rev() {
                let idx = i.min(tpoints - 1);
                padded.push(Complex64::new(channel_data[idx], 0.0));
            }

            // Original signal
            for &val in channel_data {
                padded.push(Complex64::new(val, 0.0));
            }

            // Right padding (mirror)
            for i in 0..pad_right {
                let idx = (tpoints - 1 - i).max(0);
                padded.push(Complex64::new(channel_data[idx], 0.0));
            }

            // Perform FFT
            fft.process(&mut padded);

            // Take first (tpoints + 1) frequency bins
            result.push(padded[..tpoints + 1].to_vec());
        }

        result
    }

    /// Transform frequency-domain signal back to time domain
    fn to_time_domain(&self, signal_hat: &[Vec<Complex64>]) -> Vec<Vec<f64>> {
        let fpoints = signal_hat[0].len();
        let red_ft = fpoints - 1;
        let full_len = 2 * red_ft;

        let mut planner = FftPlanner::<f64>::new();
        let ifft = planner.plan_fft_inverse(full_len);

        let mut result = Vec::with_capacity(self.num_channels);

        for channel_hat in signal_hat {
            // Construct Hermitian-symmetric spectrum for real signal reconstruction
            let mut full_hat: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); full_len];

            // positive frequencies: full_hat[red_ft:] = channel_hat[:red_ft]
            for i in 0..red_ft {
                full_hat[red_ft + i] = channel_hat[i];
            }

            // negative frequencies (conjugate mirror): full_hat[:red_ft] = conj(channel_hat[red_ft:0:-1])
            for i in 1..=red_ft {
                full_hat[red_ft - i] = channel_hat[i].conj();
            }

            // ifftshift equivalent
            let mut shifted: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); full_len];
            let mid = full_len / 2;
            for i in 0..full_len {
                let new_idx = (i + mid) % full_len;
                shifted[new_idx] = full_hat[i];
            }

            // Perform IFFT
            ifft.process(&mut shifted);

            // Extract real part and remove padding
            let start = red_ft / 2;
            let end = start + red_ft;
            let scale = 1.0 / full_len as f64;

            let time_signal: Vec<f64> = shifted[start..end].iter().map(|c| c.re * scale).collect();

            result.push(time_signal);
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mvmd_basic() {
        // Create a simple test signal: sum of two sinusoids
        let num_samples = 256;
        let t: Vec<f64> = (0..num_samples)
            .map(|i| i as f64 / num_samples as f64)
            .collect();

        // Two channels with different frequency combinations
        let channel1: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * PI * 5.0 * ti).sin() + 0.5 * (2.0 * PI * 20.0 * ti).sin())
            .collect();

        let channel2: Vec<f64> = t
            .iter()
            .map(|&ti| 0.8 * (2.0 * PI * 5.0 * ti).sin() + 0.3 * (2.0 * PI * 20.0 * ti).sin())
            .collect();

        let data = vec![channel1, channel2];
        let mvmd = MVMD::new(data, 2000.0)
            .with_init(FrequencyInit::Linear)
            .with_admm_config(ADMMConfig::new(1e-7, 0.0, 500));

        let result = mvmd.decompose(2);

        // Check modes shape: (K=2, C=2, T=256)
        assert_eq!(result.modes.shape(), &[2, 2, 256]);
        assert!(result.num_iterations > 0);
        assert!(result.num_iterations <= 500);

        // Check frequency_traces shape: (n_iters, K=2)
        assert_eq!(result.frequency_traces.shape()[1], 2);

        // Check center_frequencies shape: (K=2,)
        assert_eq!(result.center_frequencies.len(), 2);
    }
}
