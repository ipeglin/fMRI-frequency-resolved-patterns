use ndarray::{ArrayView2, parallel::prelude::*};
use rustfft::num_complex::Complex64;

/// Compute the largest eigenvalue of the frequency-smoothed coherence matrix
/// Σ_z(ω) via power iteration, using an implicit matrix-vector product that
/// avoids materialising the (C+N)×(C+N) matrix.
///
/// Scratch buffers `x`, `y`, `z_buf` must each have length `mode_spectra.nrows()`.
/// Passing pre-allocated scratch eliminates per-call heap allocation in the hot loop.
///
/// # Arguments
/// * `mode_spectra` — `[C+N, F]` complex mode spectrum for mode k at this ADMM iteration
/// * `s_diag`       — `[C+N][F]` smoothed auto-spectra (per-channel power, box-smoothed)
/// * `f`            — target frequency bin index
/// * `window`       — smoothing window size (odd, ≥1; number of freq bins averaged)
/// * `max_iters`    — power-iteration iteration cap
/// * `tol`          — relative eigenvalue change for early exit
/// * `init_vec`     — deterministic starting vector (length C+N, normalised by caller)
/// * `x`, `y`, `z_buf` — pre-allocated scratch (each length C+N)
///
/// # Returns
/// Largest eigenvalue of the normalised coherence matrix at bin `f`.
#[allow(clippy::too_many_arguments)]
pub fn power_iteration_smoothed_coherence(
    mode_spectra: ArrayView2<Complex64>,
    s_diag: &[Vec<f64>],
    f: usize,
    window: usize,
    max_iters: usize,
    tol: f64,
    init_vec: &[Complex64],
    x: &mut [Complex64],
    y: &mut [Complex64],
    z_buf: &mut [Complex64],
) -> f64 {
    let c_aug = mode_spectra.nrows();
    let num_fpoints = mode_spectra.ncols();
    let half_w = (window / 2) as isize;

    // Clamp window to valid freq range around f
    let f_lo = (f as isize - half_w).max(0) as usize;
    let f_hi = (f as isize + half_w).min(num_fpoints as isize - 1) as usize;
    let actual_w = f_hi - f_lo + 1;

    // Precompute diagonal normaliser: d_i = 1 / sqrt(S_ii(f)), or 0 if power ≈ 0
    // S_ii(f) is already smoothed in s_diag.
    let eps = 1e-30_f64;
    let d: Vec<f64> = (0..c_aug)
        .map(|c| {
            let p = s_diag[c][f];
            if p > eps { 1.0 / p.sqrt() } else { 0.0 }
        })
        .collect();

    // Initialise x from init_vec
    x.clone_from_slice(init_vec);

    let inv_w = 1.0 / actual_w as f64;
    let mut lambda = 0.0_f64;

    for _ in 0..max_iters {
        // z = D · x
        for (c, (zc, &xc)) in z_buf.iter_mut().zip(x.iter()).enumerate() {
            *zc = xc.scale(d[c]);
        }

        // y = (1/W) Σ_{ω'∈win} u(ω') * (u(ω')^H · z),  then D · y
        y.iter_mut().for_each(|v| *v = Complex64::new(0.0, 0.0));
        for fw in f_lo..=f_hi {
            let s: Complex64 = (0..c_aug)
                .map(|c| mode_spectra[[c, fw]].conj() * z_buf[c])
                .fold(Complex64::new(0.0, 0.0), |a, b| a + b);
            for c in 0..c_aug {
                y[c] += mode_spectra[[c, fw]] * s;
            }
        }
        for (c, yv) in y.iter_mut().enumerate() {
            *yv = yv.scale(inv_w * d[c]);
        }

        // Rayleigh quotient
        let lambda_new: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi.conj() * yi).re)
            .sum();

        // Normalise y → new x
        let norm: f64 = y.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
        if norm < eps {
            return 0.0;
        }
        let inv_norm = 1.0 / norm;
        x.iter_mut().zip(y.iter()).for_each(|(xi, &yi)| *xi = yi.scale(inv_norm));

        if lambda > eps && (lambda_new - lambda).abs() / lambda.abs() < tol {
            return lambda_new.max(0.0);
        }
        lambda = lambda_new;
    }

    lambda.max(0.0)
}

/// Compute per-channel smoothed auto-spectra using a rectangular (box) window.
///
/// Returns a `[C][F]` array where `out[c][f]` is the average of
/// `|mode_spectra[c][f']|²` over `f' ∈ [f−W/2, f+W/2]`.
pub fn smoothed_auto_spectra(mode_spectra: ArrayView2<Complex64>, window: usize) -> Vec<Vec<f64>> {
    let num_fpoints = mode_spectra.ncols();
    let half_w = (window / 2) as isize;

    mode_spectra
        .outer_iter()
        .into_par_iter()
        .map(|ch| {
            let abs2: Vec<f64> = ch.iter().map(|z| z.norm_sqr()).collect();
            (0..num_fpoints)
                .map(|f| {
                    let f_lo = (f as isize - half_w).max(0) as usize;
                    let f_hi = (f as isize + half_w).min(num_fpoints as isize - 1) as usize;
                    let sum: f64 = abs2[f_lo..=f_hi].iter().sum();
                    sum / (f_hi - f_lo + 1) as f64
                })
                .collect()
        })
        .collect()
}
