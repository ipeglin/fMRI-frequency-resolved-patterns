use super::admm::{ADMMConfig, ADMMOptimizer};
use ndarray::{Array1, Array2, Array3, parallel::prelude::*, s};
use polars::prelude::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use rustfft::{FftPlanner, num_complex::Complex64};
use tracing::{debug, info, trace, warn};

/// Algorithm variant: classic MVMD or Noise-Assisted MVMD (NA-MVMD).
///
/// NA-MVMD appends WGN channels to the input and replaces the spectral-centroid
/// frequency update with a Generalized Cross-Spectrum (GCS) centroid computed via
/// power iteration on the frequency-smoothed coherence matrix.
#[derive(Debug, Clone, Default)]
pub enum MvmdVariant {
    /// Standard MVMD — no noise injection, spectral-centroid frequency update.
    #[default]
    Classic,
    /// NA-MVMD — normalizes channels to unit variance, appends WGN, applies the
    /// GCS single-snapshot centroid for the omega update, then rescales modes.
    NoiseAssisted {
        /// Number of WGN channels to append (N). 1 is usually sufficient.
        noise_channels: usize,
        /// Noise std relative to the unit-variance-normalized channels.
        /// Matches the reference `na_mvmd.m` default of 0.8.
        noise_std_ratio: f64,
        /// Seed for ChaCha8Rng — ensures deterministic noise.
        seed: u64,
    },
}

/// Sample standard deviation of a signal channel (unbiased, ddof=1). Returns 1.0 for empty/single-sample.
fn channel_std(ch: &[f64]) -> f64 {
    let n = ch.len();
    if n < 2 {
        return 1.0;
    }
    let mean = ch.iter().sum::<f64>() / n as f64;
    let var = ch.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    var.sqrt()
}

/// Initialization method for center frequencies in MVMD/VMD algorithms.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub enum FrequencyInit {
    /// All omegas start at 0
    #[default]
    Zero,
    /// Omegas are initialized linearly distributed in [0, 0.5]
    Linear,
    /// Omegas are initialized exponentially distributed
    Exponential,
    /// Custom initialization directly in normalized frequency space [0, 0.5]
    Custom(Vec<f64>),
}

/// A single MVMD mode with its time series data and center frequency.
///
/// The DataFrame has channels as columns and time points as rows,
/// compatible with `ConnectivityMatrix::new()` for computing functional connectivity.
#[allow(dead_code)]
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
    #[allow(dead_code)]
    /// Labels across all channels
    pub channels: Vec<String>,
    /// Decomposed modes with shape (K modes x C channels x T time-points)
    pub modes: Array3<f64>,
    /// Final center frequencies for each mode (K,)
    pub center_frequencies: Array1<f64>,
    /// Number of iterations until convergence
    pub num_iterations: u32,
}

#[allow(dead_code)]
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

    /// Maps the converged modes to the closest predefined logarithmic frequency bins.
    /// Returns a vector of tuples: (Mode Index, Bin Index, Distance to Bin)
    pub fn map_to_log_bins(
        &self,
        f_min: f64,
        f_max: f64,
        n_scales: usize,
    ) -> Vec<(usize, usize, f64)> {
        // 1. Generate the exact same log-frequencies you used for the CWT
        let log_freqs: Vec<f64> = (0..n_scales)
            .map(|i| f_min * (f_max / f_min).powf(i as f64 / (n_scales - 1) as f64))
            .collect();

        let mut mapped_modes = Vec::new();

        // 2. Iterate through the found center frequencies
        for (k, &cf) in self.center_frequencies.iter().enumerate() {
            // Filter out modes outside the frequency bounds
            if cf < f_min || cf > f_max {
                continue;
            }

            // 3. Find the closest log-bin index
            let mut closest_bin = 0;
            let mut min_diff = f64::MAX;

            for (bin_idx, &bin_freq) in log_freqs.iter().enumerate() {
                let diff = (cf - bin_freq).abs();
                if diff < min_diff {
                    min_diff = diff;
                    closest_bin = bin_idx;
                }
            }

            mapped_modes.push((k, closest_bin, min_diff));
        }

        mapped_modes
    }

    /// Snaps the converged sparse modes into a dense grid of specified size.
    /// If multiple modes map to the same bin, they are summed.
    pub fn remap_to_grid(&self, f_min: f64, f_max: f64, n_bins: usize) -> Array3<f64> {
        let shape = self.modes.shape();
        let _num_modes = shape[0];
        let num_channels = shape[1];
        let num_tpoints = shape[2];

        let mut grid_modes = Array3::<f64>::zeros((n_bins, num_channels, num_tpoints));
        let mappings = self.map_to_log_bins(f_min, f_max, n_bins);

        // Track which modes fall into which bins to detect redundancy
        let mut bin_assignments: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        for (k_idx, bin_idx, _) in mappings {
            bin_assignments.entry(bin_idx).or_default().push(k_idx);

            for c in 0..num_channels {
                for t in 0..num_tpoints {
                    grid_modes[[bin_idx, c, t]] += self.modes[[k_idx, c, t]];
                }
            }
        }

        // Explicitly log redundant modes needing merging
        for (bin_idx, mode_indices) in bin_assignments {
            if mode_indices.len() > 1 {
                let mode_list = mode_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");

                // Log as warning for researcher visibility
                warn!(
                    target_bin = bin_idx,
                    merged_modes = %mode_list,
                    reason = "redundant_modes_detected",
                    "MVMD collision: multiple modes merged into a single frequency bin"
                );
            }
        }

        grid_modes
    }
}

/// Multivariate Variational Mode Decomposition (MVMD)
///
/// Implementation based on:
/// N. Rehman and H. Aftab (2019) "Multivariate Variational Mode Decomposition",
/// IEEE Transactions on Signal Processing
#[allow(clippy::upper_case_acronyms)]
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
    /// Algorithm variant (Classic or NoiseAssisted)
    variant: MvmdVariant,
}

impl ADMMOptimizer for MVMD {
    fn admm_config(&self) -> &ADMMConfig {
        &self.admm_config
    }

    fn admm_config_mut(&mut self) -> &mut ADMMConfig {
        &mut self.admm_config
    }
}

#[allow(dead_code)]
impl MVMD {
    /// Create a new MVMD instance from a DataFrame.
    ///
    /// The DataFrame should have columns representing channels and rows representing time-points.
    pub fn from_dataframe(df: &DataFrame, alpha: f64, sampling_rate: f64) -> PolarsResult<Self> {
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
            sampling_rate,
            admm_config: ADMMConfig::default(),
            variant: MvmdVariant::Classic,
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
            sampling_rate: 1.25,
            admm_config: ADMMConfig::default(),
            variant: MvmdVariant::Classic,
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

    /// Set the algorithm variant (Classic or NoiseAssisted).
    pub fn with_variant(mut self, variant: MvmdVariant) -> Self {
        self.variant = variant;
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

        // Build augmented data (original channels + WGN) for NA-MVMD, or clone as-is for Classic.
        let (working_data, c_aug, na_scales) = self.build_augmented_data();

        info!(
            num_modes = num_modes,
            num_channels = self.num_channels,
            c_aug,
            num_tpoints = self.num_tpoints,
            num_fpoints = num_fpoints,
            alpha = self.alpha,
            max_iterations = self.admm_config.max_iterations,
            tolerance = self.admm_config.tolerance,
            tau = self.admm_config.tau,
            na_mvmd = na_scales.is_some(),
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

        // Transform (augmented) signal to frequency domain
        debug!("transforming signal to frequency domain");
        let signal_hat = Self::to_freq_domain_from_data(&working_data, self.num_tpoints);
        debug!("FFT completed for {} channels", c_aug);

        // Initialize modes in frequency domain: Vec of K Array2<[c_aug, F]>
        let mut modes_hat: Vec<Array2<Complex64>> = (0..num_modes)
            .map(|_| Array2::zeros((c_aug, num_fpoints)))
            .collect();

        // Dual variables (lambda): only keep current and next iteration (memory optimization)
        let mut lambda_current: Array2<Complex64> = Array2::zeros((c_aug, num_fpoints));
        let mut lambda_next: Array2<Complex64> = Array2::zeros((c_aug, num_fpoints));

        // Pre-compute scalars for the GCS single-snapshot centroid (NA-MVMD only).
        // γ_k(ω) = (Σ_c|û_{k,c}(ω)|² − 1) / (C+N−1)
        // ω_k = Σ_ω γ·f_points[ω] / Σ_ω γ
        //      = (Σ_c Σ_ω |û|²·f_points[ω]  −  Σ_ω f_points[ω])
        //        / (Σ_c Σ_ω |û|²  −  num_fpoints)
        // The (C+N-1) denominator cancels in the ratio and is not needed.
        let n_bins_sum: f64 = f_points.iter().sum();
        let n_bins_total: f64 = num_fpoints as f64;

        let mut residual_diff = self.admm_config.tolerance + f64::EPSILON;
        let mut n: usize = 0;

        // Pre-compute modes_sum once, then update incrementally
        let mut modes_sum: Array2<Complex64> = Array2::zeros((c_aug, num_fpoints));

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
                // Parallel over channels via ndarray Zip; residual contributions collected into Array1.
                let alpha = self.alpha;
                let mut residual_arr = ndarray::Array1::<f64>::zeros(c_aug);
                ndarray::Zip::from(modes_hat[k].rows_mut())
                    .and(modes_sum.rows_mut())
                    .and(signal_hat.rows())
                    .and(lambda_current.rows())
                    .and(residual_arr.view_mut())
                    .par_for_each(|mut mh_c, mut ms_c, sh_c, lam_c, res| {
                        let mut local: f64 = 0.0;
                        for f in 0..num_fpoints {
                            let old_val = mh_c[f];
                            let sum_other = ms_c[f] - old_val;
                            let numerator = sh_c[f] - sum_other - lam_c[f].scale(0.5);
                            let freq_diff = f_points[f] - omega_k;
                            let denominator = 1.0 + alpha * freq_diff * freq_diff;
                            let new_val = numerator.scale(1.0 / denominator);
                            mh_c[f] = new_val;
                            ms_c[f] = ms_c[f] - old_val + new_val;
                            local += (new_val - old_val).norm_sqr();
                        }
                        *res = local;
                    });
                residual_diff += residual_arr.sum();

                // Update center frequency.
                // Classic: average spectral centroid across all (augmented) channels.
                // NA-MVMD: GCS single-snapshot centroid — Λ_max(Σ_z(ω)) = Σ_c|û_c(ω)|²
                //   for the rank-1 raw cross-spectral matrix, giving:
                //   ω_k = (Σ_c Σ_f |û|²·f_points[f] − Σ_f f_points[f])
                //         / (Σ_c Σ_f |û|²          − num_fpoints)
                let (weighted_sum, total_power): (f64, f64) = modes_hat[k]
                    .outer_iter()
                    .into_par_iter()
                    .map(|mh_c| {
                        let mut ws: f64 = 0.0;
                        let mut tp: f64 = 0.0;
                        for f in 0..num_fpoints {
                            let power = mh_c[f].norm_sqr();
                            ws += power * f_points[f];
                            tp += power;
                        }
                        (ws, tp)
                    })
                    .reduce(|| (0.0, 0.0), |a, b| (a.0 + b.0, a.1 + b.1));

                omega_next[k] = if na_scales.is_some() {
                    // GCS: subtract the "-1" noise-floor per frequency bin.
                    let weighted = weighted_sum - n_bins_sum;
                    let total = total_power - n_bins_total;
                    if total > 0.0 {
                        weighted / total
                    } else {
                        omega_k
                    }
                } else if total_power > 0.0 {
                    weighted_sum / total_power
                } else {
                    omega_k
                };
            }

            // Dual ascent: lambda = lambda + tau * (sum(modes) - signal). Independent per c.
            let tau = self.admm_config.tau;
            ndarray::Zip::from(lambda_next.rows_mut())
                .and(lambda_current.rows())
                .and(modes_sum.rows())
                .and(signal_hat.rows())
                .par_for_each(|mut ln_c, lc_c, ms_c, sh_c| {
                    for f in 0..num_fpoints {
                        let residual = ms_c[f] - sh_c[f];
                        ln_c[f] = lc_c[f] + residual.scale(tau);
                    }
                });

            // Swap current and next
            std::mem::swap(&mut omega_current, &mut omega_next);
            std::mem::swap(&mut lambda_current, &mut lambda_next);

            // Store omega history
            omega_history.push(omega_current.clone());

            n += 1;
            residual_diff /= self.num_tpoints as f64;

            // Log progress at INFO level so it's visible, every 10 iterations
            if n.is_multiple_of(100) || n == 1 {
                debug!(
                    iteration = n,
                    max_iterations = self.admm_config.max_iterations,
                    residual_diff = format!("{:.6e}", residual_diff),
                    tolerance = format!("{:.6e}", self.admm_config.tolerance),
                    "MVMD iteration"
                );
            }
        }

        // Post-processing: extract and order results
        debug!("post-processing: ordering results by frequency");
        let mut omega_result: Vec<Vec<f64>> = omega_history
            .iter()
            .map(|row| row.iter().map(|&w| w * self.sampling_rate).collect())
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

        // Reconstruct time-domain modes and reorder.
        // Trim noise channels (indices ≥ self.num_channels) before IFFT — they are
        // never written to the output array.
        debug!("reconstructing time-domain modes via IFFT");
        let mut modes_vec: Vec<Vec<Vec<f64>>> = Vec::with_capacity(num_modes);
        for (i, &idx) in indices.iter().enumerate() {
            trace!(mode_idx = i, original_idx = idx, "reconstructing mode");
            modes_vec.push(self.to_time_domain(modes_hat[idx].slice(s![..self.num_channels, ..])));
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

        // Rescale modes by original per-channel std — mirrors `u(:,:,k) .* s'`
        // in the reference na_mvmd.m (undoes the unit-variance normalization).
        if let Some(ref stds) = na_scales {
            for k in 0..num_modes {
                for (c, &std_c) in stds.iter().enumerate() {
                    for t in 0..n_timepoints {
                        modes[[k, c, t]] *= std_c;
                    }
                }
            }
        }

        // center_frequencies: (K,)
        let n_iters = omega_result.len();
        let center_frequencies = if n_iters > 0 {
            Array1::from_vec(omega_result[n_iters - 1].clone())
        } else {
            Array1::zeros(num_modes)
        };

        let converged = residual_diff <= self.admm_config.tolerance;
        info!(
            modes_shape = ?[num_modes, self.num_channels, n_timepoints],
            center_frequencies = ?center_frequencies.as_slice(),
            num_iterations = n as u32,
            converged = converged,
            final_residual = residual_diff,
            tolerance = self.admm_config.tolerance,
            "MVMD decomposition completed"
        );

        MVMDResult {
            channels: self.channels.clone(),
            modes,
            center_frequencies,
            num_iterations: n as u32,
        }
    }

    /// Build the working data array (original channels + optional WGN) following
    /// the reference `na_mvmd.m` procedure.
    ///
    /// Returns `(working_data, c_aug, orig_stds)` where `orig_stds` is
    /// `Some(stds)` for NA-MVMD (caller must rescale output modes by `stds`)
    /// or `None` for Classic.
    fn build_augmented_data(&self) -> (Vec<Vec<f64>>, usize, Option<Vec<f64>>) {
        match &self.variant {
            MvmdVariant::Classic => (self.data.clone(), self.num_channels, None),
            MvmdVariant::NoiseAssisted {
                noise_channels,
                noise_std_ratio,
                seed,
            } => {
                let n_noise = *noise_channels;
                let c_aug = self.num_channels + n_noise;

                // Compute per-channel sample std (ddof=1) and normalize each
                // channel to unit variance — matches `s = std(signal,0,2);
                // signal = signal ./ s;` in the reference na_mvmd.m.
                let orig_stds: Vec<f64> = self
                    .data
                    .iter()
                    .map(|ch| channel_std(ch).max(1e-30))
                    .collect();

                let mut augmented: Vec<Vec<f64>> = self
                    .data
                    .iter()
                    .zip(orig_stds.iter())
                    .map(|(ch, &s)| ch.iter().map(|&x| x / s).collect())
                    .collect();

                // Append WGN channels: `noise_amp * wgn(noise_channels, T, 0)`
                // wgn(m,n,0) = unit-variance WGN; noise_std_ratio matches noise_amp.
                let mut rng = ChaCha8Rng::seed_from_u64(*seed);
                let dist = Normal::new(0.0_f64, noise_std_ratio.max(1e-30)).unwrap();
                for _ in 0..n_noise {
                    let noise: Vec<f64> = (0..self.num_tpoints)
                        .map(|_| dist.sample(&mut rng))
                        .collect();
                    augmented.push(noise);
                }

                (augmented, c_aug, Some(orig_stds))
            }
        }
    }

    /// Transform a data slice (channels × time-points) to the frequency domain.
    /// Returns Array2<Complex64> with shape [c, num_tpoints+1].
    fn to_freq_domain_from_data(data: &[Vec<f64>], num_tpoints: usize) -> Array2<Complex64> {
        let c = data.len();
        let num_fpoints = num_tpoints + 1;
        let pad_left = num_tpoints / 2;
        let pad_right = num_tpoints - pad_left;
        let padded_len = num_tpoints + pad_left + pad_right;

        let mut planner = FftPlanner::<f64>::new();
        let fft = planner.plan_fft_forward(padded_len);

        let rows: Vec<Vec<Complex64>> = data
            .par_iter()
            .map(|channel_data| {
                let mut padded: Vec<Complex64> = Vec::with_capacity(padded_len);
                for i in (0..pad_left).rev() {
                    let idx = i.min(num_tpoints - 1);
                    padded.push(Complex64::new(channel_data[idx], 0.0));
                }
                for &val in channel_data {
                    padded.push(Complex64::new(val, 0.0));
                }
                for i in 0..pad_right {
                    let idx = (num_tpoints - 1 - i).max(0);
                    padded.push(Complex64::new(channel_data[idx], 0.0));
                }
                fft.process(&mut padded);
                padded[..num_fpoints].to_vec()
            })
            .collect();

        let flat: Vec<Complex64> = rows.into_iter().flatten().collect();
        Array2::from_shape_vec((c, num_fpoints), flat)
            .expect("shape mismatch in to_freq_domain_from_data")
    }

    /// Initialize center frequencies based on the chosen method
    fn initialize_omegas(&self, omega: &mut [f64], num_modes: usize) {
        match &self.init {
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
            FrequencyInit::Custom(freqs) => {
                for (i, w) in omega.iter_mut().enumerate() {
                    if i < freqs.len() {
                        // Ensure the provided frequencies are scaled to [0, 0.5]
                        // based on your sampling rate
                        *w = freqs[i] / self.sampling_rate;
                    }
                }
            }
        }
    }

    /// Transform signal to frequency domain with symmetric padding.
    /// Returns Array2<Complex64> with shape [num_channels, num_tpoints+1].
    fn to_freq_domain(&self) -> Array2<Complex64> {
        Self::to_freq_domain_from_data(&self.data, self.num_tpoints)
    }

    /// Transform frequency-domain modes back to time domain.
    /// `signal_hat` has shape [c_chan, fpoints] (C-contiguous rows).
    fn to_time_domain(&self, signal_hat: ndarray::ArrayView2<Complex64>) -> Vec<Vec<f64>> {
        let fpoints = signal_hat.ncols();
        let red_ft = fpoints - 1;
        let full_len = 2 * red_ft;

        let mut planner = FftPlanner::<f64>::new();
        let ifft = planner.plan_fft_inverse(full_len);

        // Parallel over channels: Fft plan is shared via Arc (Send+Sync).
        signal_hat
            .outer_iter()
            .into_par_iter()
            .map(|channel_hat| {
                let ch = channel_hat.as_slice().expect("non-contiguous row");
                let mut full_hat: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); full_len];

                full_hat[red_ft..(red_ft + red_ft)].copy_from_slice(&ch[..red_ft]);

                for i in 1..=red_ft {
                    full_hat[red_ft - i] = ch[i].conj();
                }

                let mut shifted: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); full_len];
                let mid = full_len / 2;
                for (i, val) in full_hat.iter().enumerate() {
                    let new_idx = (i + mid) % full_len;
                    shifted[new_idx] = *val;
                }

                ifft.process(&mut shifted);

                let start = red_ft / 2;
                let end = start + red_ft;
                let scale = 1.0 / full_len as f64;
                shifted[start..end].iter().map(|c| c.re * scale).collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_mvmd_basic() {
        let num_samples = 256;
        let t: Vec<f64> = (0..num_samples)
            .map(|i| i as f64 / num_samples as f64)
            .collect();

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

        assert_eq!(result.modes.shape(), &[2, 2, 256]);
        assert!(result.num_iterations > 0);
        assert!(result.num_iterations <= 500);
        assert_eq!(result.center_frequencies.len(), 2);
    }

    #[test]
    fn test_na_mvmd_output_shape_matches_classic() {
        // NA-MVMD should return modes for original channels only (no noise channels in output).
        let num_samples = 128;
        let t: Vec<f64> = (0..num_samples)
            .map(|i| i as f64 / num_samples as f64)
            .collect();

        let make_channel = |f1: f64, f2: f64| -> Vec<f64> {
            t.iter()
                .map(|&ti| (2.0 * PI * f1 * ti).sin() + 0.5 * (2.0 * PI * f2 * ti).sin())
                .collect()
        };

        let data = vec![make_channel(5.0, 20.0), make_channel(4.0, 18.0)];

        let classic = MVMD::new(data.clone(), 2000.0)
            .with_admm_config(ADMMConfig::new(1e-6, 0.0, 100))
            .decompose(2);

        let na = MVMD::new(data, 2000.0)
            .with_admm_config(ADMMConfig::new(1e-6, 0.0, 100))
            .with_variant(MvmdVariant::NoiseAssisted {
                noise_channels: 1,
                noise_std_ratio: 0.8,
                seed: 0xC0FFEE,
            })
            .decompose(2);

        // Output shape must match classic: (K=2, C=2, T=128) — no noise channels leaked.
        assert_eq!(na.modes.shape(), classic.modes.shape());
        assert_eq!(
            na.center_frequencies.len(),
            classic.center_frequencies.len()
        );
    }

    #[test]
    fn test_na_mvmd_modes_do_not_collapse() {
        // Regression: NA-MVMD must separate two well-spaced tones, not collapse all
        // modes to the same center frequency (the bug fixed by the raw-GCS ω-update).
        let num_samples = 512;
        let sampling_rate = 2000.0_f64;
        let t: Vec<f64> = (0..num_samples)
            .map(|i| i as f64 / sampling_rate)
            .collect();

        // Two channels, each a sum of 50 Hz + 200 Hz tones — normalized: 0.025 and 0.1.
        let make_channel = |a1: f64, a2: f64| -> Vec<f64> {
            t.iter()
                .map(|&ti| {
                    a1 * (2.0 * PI * 50.0 * ti).sin() + a2 * (2.0 * PI * 200.0 * ti).sin()
                })
                .collect()
        };
        let data = vec![make_channel(1.0, 0.5), make_channel(0.9, 0.6)];

        let result = MVMD::new(data, sampling_rate)
            .with_admm_config(ADMMConfig::new(1e-7, 1e-3, 300))
            .with_init(FrequencyInit::Exponential)
            .with_variant(MvmdVariant::NoiseAssisted {
                noise_channels: 1,
                noise_std_ratio: 0.8,
                seed: 42,
            })
            .decompose(2);

        let freqs = result.center_frequencies.as_slice().unwrap();
        let (f0, f1) = (freqs[0], freqs[1]);
        let gap = (f0 - f1).abs();

        assert!(
            gap > 0.04,
            "modes collapsed: center frequencies {f0:.4} and {f1:.4} differ by only {gap:.4} \
             (expected > 0.04; likely all ω pinned to the same mid-band)"
        );

        // Neither mode should be all-zero.
        for k in 0..2_usize {
            let mode_k = result.modes.index_axis(ndarray::Axis(0), k);
            let max_abs = mode_k.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
            assert!(
                max_abs > 1e-6,
                "mode {k} is effectively zero (max |sample| = {max_abs:.2e})"
            );
        }
    }

    #[test]
    fn test_na_mvmd_deterministic() {
        // Same seed → identical output across two runs.
        let num_samples = 64;
        let t: Vec<f64> = (0..num_samples)
            .map(|i| i as f64 / num_samples as f64)
            .collect();
        let ch: Vec<f64> = t.iter().map(|&ti| (2.0 * PI * 10.0 * ti).sin()).collect();
        let data = vec![ch.clone(), ch];

        let variant = MvmdVariant::NoiseAssisted {
            noise_channels: 1,
            noise_std_ratio: 0.5,
            seed: 42,
        };

        let r1 = MVMD::new(data.clone(), 2000.0)
            .with_admm_config(ADMMConfig::new(1e-5, 0.0, 50))
            .with_variant(variant.clone())
            .decompose(2);

        let r2 = MVMD::new(data, 2000.0)
            .with_admm_config(ADMMConfig::new(1e-5, 0.0, 50))
            .with_variant(variant)
            .decompose(2);

        for (v1, v2) in r1
            .center_frequencies
            .iter()
            .zip(r2.center_frequencies.iter())
        {
            assert!(
                (v1 - v2).abs() < 1e-12,
                "center frequencies not deterministic: {v1} vs {v2}"
            );
        }
    }
}
