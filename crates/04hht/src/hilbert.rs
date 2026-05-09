use anyhow::Result;
use ndarray::{Array3, s};
use num_complex::Complex64;
use scirs2_signal::hilbert::hilbert;
use utils::config::AppConfig;

/// Result of a Hilbert-Huang Transform applied to a set of IMF modes.
///
/// Modes tensor shape: [n_modes, n_channels, n_timepoints]
/// Envelope/inst_freq shape same.
pub struct HHTResult {
    /// Instantaneous amplitude (envelope) [n_modes, n_channels, n_timepoints]
    pub envelope: Vec<f64>,
    pub envelope_shape: [usize; 3],
    /// Instantaneous frequency (Hz) [n_modes, n_channels, n_timepoints]
    pub inst_freq: Vec<f64>,
    pub inst_freq_shape: [usize; 3],
}

fn compute_instantaneous_angular_freq(analytic: &[Complex64], sampling_rate: f64) -> Vec<f64> {
    let n = analytic.len();
    if n < 2 {
        return vec![0.0; n];
    }

    let mut omega = vec![0.0; n];
    let dt = 1.0 / sampling_rate;

    for i in 1..n {
        let phase_diff = (analytic[i] * analytic[i - 1].conj()).arg();
        omega[i] = phase_diff / dt;
    }

    omega[0] = omega[1];
    omega
}

/// Compute HHT from a modes array with shape [n_modes, n_channels, n_timepoints].
///
/// Takes modes in-memory as f64 Array3 — no HDF5 read or f32 conversion.
pub fn compute_hht(cfg: &AppConfig, modes: &Array3<f64>) -> Result<HHTResult> {
    let shape = modes.shape();
    let n_modes = shape[0];
    let n_channels = shape[1];
    let n_timepoints = shape[2];
    let sampling_rate = cfg.task_sampling_rate;

    let mut envelope_buf = vec![0f64; n_modes * n_channels * n_timepoints];
    let mut inst_freq_buf = vec![0f64; n_modes * n_channels * n_timepoints];

    for m in 0..n_modes {
        for c in 0..n_channels {
            let channel_signal: Vec<f64> = modes.slice(s![m, c, ..]).to_vec();
            let analytic = hilbert(&channel_signal)
                .map_err(|e| anyhow::anyhow!("hilbert failed mode={} ch={}: {}", m, c, e))?;

            let amp: Vec<f64> = analytic.iter().map(|z| z.norm()).collect();
            let i_omega = compute_instantaneous_angular_freq(&analytic, sampling_rate);

            let base = m * n_channels * n_timepoints + c * n_timepoints;
            envelope_buf[base..base + n_timepoints].copy_from_slice(&amp);
            inst_freq_buf[base..base + n_timepoints].copy_from_slice(&i_omega);
        }
    }

    // log1p compression must precede max-divide so it acts on the raw dynamic
    // range rather than on the already-squashed [0,1] values.
    if cfg.hht.hht_log_amp {
        for v in envelope_buf.iter_mut() {
            *v = v.ln_1p();
        }
    }

    if cfg.hht.hht_envelope_normalize {
        for c in 0..n_channels {
            let ch_max = (0..n_modes)
                .map(|m| {
                    let base = m * n_channels * n_timepoints + c * n_timepoints;
                    envelope_buf[base..base + n_timepoints]
                        .iter()
                        .cloned()
                        .fold(0.0f64, f64::max)
                })
                .fold(0.0f64, f64::max);

            if ch_max > 0.0 {
                let denom = ch_max + 1e-12;
                for m in 0..n_modes {
                    let base = m * n_channels * n_timepoints + c * n_timepoints;
                    for v in envelope_buf[base..base + n_timepoints].iter_mut() {
                        *v /= denom;
                    }
                }
            }
        }
    }

    for val in inst_freq_buf.iter_mut() {
        *val /= 2.0 * std::f64::consts::PI;
    }

    Ok(HHTResult {
        envelope: envelope_buf,
        envelope_shape: [n_modes, n_channels, n_timepoints],
        inst_freq: inst_freq_buf,
        inst_freq_shape: [n_modes, n_channels, n_timepoints],
    })
}
