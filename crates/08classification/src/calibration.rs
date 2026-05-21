//! Probability calibration: Platt scaling and isotonic regression.
//!
//! **Platt scaling** fits a 1-D logistic regression `σ(a·s + b)` on the
//! calibration set. Best for small sets (≤ 1000 samples) — the two-parameter
//! sigmoid keeps variance low when data is scarce.
//!
//! **Isotonic regression** (Pool-Adjacent Violators) fits a monotone step
//! function non-parametrically. Better for large sets (> 1000 samples) where
//! the sigmoid assumption can be overly restrictive.
//!
//! Use `CalibratorKind::fit_auto` to automatically select between them based
//! on calibration set size.

use anyhow::{Result, bail};

// ---------------------------------------------------------------------------
// Platt scaling
// ---------------------------------------------------------------------------

/// 1-D logistic regression calibrator.
///
/// Fits `σ(a·score + b) ≈ P(y = 1 | score)` via damped Newton-Raphson on
/// Platt's smoothed targets (`t+ = (N++1)/(N++2)`, `t- = 1/(N-+2)`).
#[derive(Debug, Clone, Copy)]
pub struct PlattScaler {
    pub a: f32,
    pub b: f32,
}

impl PlattScaler {
    /// Identity calibration: `σ(s)` (`a = 1, b = 0`). Fallback when fit fails.
    pub fn identity() -> Self {
        Self { a: 1.0, b: 0.0 }
    }

    /// Fit `(a, b)` so that `σ(a · score + b) ≈ P(y = 1 | score)`.
    ///
    /// Returns `Self::identity()` when the calibration set cannot support a
    /// meaningful logistic fit: fewer than 4 samples, single class present,
    /// non-finite scores, or score variance below `1e-12`.
    pub fn fit(scores: &[f32], y: &[i32]) -> Result<Self> {
        if scores.len() != y.len() {
            bail!(
                "platt fit: length mismatch ({} vs {})",
                scores.len(),
                y.len()
            );
        }
        let n = scores.len();
        if n < 4 {
            return Ok(Self::identity());
        }
        let n_pos = y.iter().filter(|&&v| v == 1).count();
        let n_neg = n - n_pos;
        if n_pos == 0 || n_neg == 0 {
            return Ok(Self::identity());
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Ok(Self::identity());
        }
        let mean: f64 = scores.iter().map(|&s| s as f64).sum::<f64>() / n as f64;
        let var: f64 = scores
            .iter()
            .map(|&s| (s as f64 - mean).powi(2))
            .sum::<f64>()
            / n as f64;
        if var < 1e-12 {
            return Ok(Self::identity());
        }

        // Platt's smoothed targets.
        let hi = (n_pos as f64 + 1.0) / (n_pos as f64 + 2.0);
        let lo = 1.0 / (n_neg as f64 + 2.0);
        let t: Vec<f64> = y.iter().map(|&v| if v == 1 { hi } else { lo }).collect();
        let s: Vec<f64> = scores.iter().map(|&v| v as f64).collect();

        let mut a = 0.0f64;
        let mut b = (n_neg as f64 + 1.0).ln() - (n_pos as f64 + 1.0).ln();
        let max_iter = 100;
        let lambda_init = 1e-3;
        let mut lambda = lambda_init;

        let nll = |a: f64, b: f64| -> f64 {
            let mut acc = 0.0;
            for i in 0..n {
                let f = a * s[i] + b;
                let lse = if f >= 0.0 {
                    f + (1.0 + (-f).exp()).ln()
                } else {
                    (1.0 + f.exp()).ln()
                };
                acc += lse - t[i] * f;
            }
            acc
        };

        let mut prev_loss = nll(a, b);
        for _ in 0..max_iter {
            let (mut g_a, mut g_b) = (0.0f64, 0.0f64);
            let (mut h_aa, mut h_bb, mut h_ab) = (0.0f64, 0.0f64, 0.0f64);
            for i in 0..n {
                let f = a * s[i] + b;
                let p = 1.0 / (1.0 + (-f).exp());
                let r = p * (1.0 - p);
                g_a += (p - t[i]) * s[i];
                g_b += p - t[i];
                h_aa += r * s[i] * s[i];
                h_bb += r;
                h_ab += r * s[i];
            }

            let mut accepted = false;
            for _ in 0..10 {
                let h_aa_d = h_aa + lambda;
                let h_bb_d = h_bb + lambda;
                let det = h_aa_d * h_bb_d - h_ab * h_ab;
                if det <= 0.0 || !det.is_finite() {
                    lambda *= 4.0;
                    continue;
                }
                let da = (-g_a * h_bb_d + g_b * h_ab) / det;
                let db = (g_a * h_ab - g_b * h_aa_d) / det;
                let new_a = a + da;
                let new_b = b + db;
                let new_loss = nll(new_a, new_b);
                if new_loss.is_finite() && new_loss < prev_loss {
                    a = new_a;
                    b = new_b;
                    prev_loss = new_loss;
                    lambda = (lambda * 0.5).max(1e-9);
                    accepted = true;
                    break;
                }
                lambda *= 4.0;
            }
            if !accepted {
                break;
            }
            if g_a.abs() < 1e-7 && g_b.abs() < 1e-7 {
                break;
            }
        }

        Ok(Self {
            a: a as f32,
            b: b as f32,
        })
    }

    pub fn transform(&self, score: f32) -> f32 {
        sigmoid(self.a * score + self.b)
    }

    pub fn transform_slice(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.transform(s)).collect()
    }
}

fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

// ---------------------------------------------------------------------------
// Isotonic regression (PAV)
// ---------------------------------------------------------------------------

/// Non-parametric monotone calibrator via Pool-Adjacent Violators (PAV).
///
/// Fits a monotonically non-decreasing step function mapping raw scores to
/// calibrated probabilities. Inference uses linear interpolation between
/// adjacent training breakpoints. Output is clamped to `[0, 1]`.
///
/// Preferred over Platt scaling when the calibration set is large (> 1000
/// samples): makes no sigmoid-shape assumption and is asymptotically
/// consistent under any monotone calibration relationship.
#[derive(Debug, Clone)]
pub struct IsotonicRegressor {
    /// Training scores in ascending sorted order.
    scores: Vec<f32>,
    /// PAV-calibrated probability for each training score.
    /// Monotonically non-decreasing. Same length as `scores`.
    probs: Vec<f32>,
}

impl IsotonicRegressor {
    /// Identity fallback: linear ramp from 0→1 over [0,1].
    pub fn identity() -> Self {
        Self {
            scores: vec![0.0, 1.0],
            probs: vec![0.0, 1.0],
        }
    }

    /// Fit PAV calibration. Returns `identity()` on degenerate inputs
    /// (< 4 samples, single class, any non-finite score).
    pub fn fit(scores: &[f32], y: &[i32]) -> Self {
        assert_eq!(scores.len(), y.len());
        let n = scores.len();
        if n < 4 {
            return Self::identity();
        }
        let n_pos = y.iter().filter(|&&v| v == 1).count();
        if n_pos == 0 || n_pos == n {
            return Self::identity();
        }
        if scores.iter().any(|s| !s.is_finite()) {
            return Self::identity();
        }

        // Sort by score ascending.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| {
            scores[a]
                .partial_cmp(&scores[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // PAV: stack of (label_sum, count) blocks. Merge back whenever the
        // left block's mean exceeds the right block's mean (violation).
        let mut sums: Vec<f64> = Vec::with_capacity(n);
        let mut counts: Vec<usize> = Vec::with_capacity(n);

        for &i in &idx {
            sums.push(y[i] as f64);
            counts.push(1);
            // Merge while left mean > right mean.
            loop {
                let len = sums.len();
                if len < 2 {
                    break;
                }
                let left_mean = sums[len - 2] / counts[len - 2] as f64;
                let right_mean = sums[len - 1] / counts[len - 1] as f64;
                if left_mean > right_mean {
                    let rs = sums.pop().unwrap();
                    let rc = counts.pop().unwrap();
                    *sums.last_mut().unwrap() += rs;
                    *counts.last_mut().unwrap() += rc;
                } else {
                    break;
                }
            }
        }

        // Expand blocks back into per-sample calibrated probs (sorted order).
        let mut sorted_probs: Vec<f32> = Vec::with_capacity(n);
        for (s, c) in sums.iter().zip(counts.iter()) {
            let p = (s / *c as f64).clamp(0.0, 1.0) as f32;
            for _ in 0..*c {
                sorted_probs.push(p);
            }
        }

        let sorted_scores: Vec<f32> = idx.iter().map(|&i| scores[i]).collect();

        Self {
            scores: sorted_scores,
            probs: sorted_probs,
        }
    }

    /// Map a raw score to a calibrated probability via linear interpolation
    /// on the PAV step function.
    pub fn transform(&self, score: f32) -> f32 {
        let n = self.scores.len();
        if n == 0 {
            return 0.5;
        }
        if score <= self.scores[0] {
            return self.probs[0];
        }
        if score >= self.scores[n - 1] {
            return self.probs[n - 1];
        }
        // Find insertion point: first index where scores[pos] > score.
        let pos = self.scores.partition_point(|&s| s <= score);
        let pos = pos.clamp(1, n - 1);
        let s0 = self.scores[pos - 1];
        let s1 = self.scores[pos];
        let p0 = self.probs[pos - 1];
        let p1 = self.probs[pos];
        if (s1 - s0).abs() < 1e-9 {
            return (p0 + p1) * 0.5;
        }
        let t = (score - s0) / (s1 - s0);
        (p0 + t * (p1 - p0)).clamp(0.0, 1.0)
    }

    pub fn transform_slice(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.transform(s)).collect()
    }
}

// ---------------------------------------------------------------------------
// Unified calibrator
// ---------------------------------------------------------------------------

/// Selects between Platt scaling and isotonic regression based on sample count.
///
/// - `n > 1000` → `IsotonicRegressor` (non-parametric, lower bias on large sets)
/// - `n ≤ 1000` → `PlattScaler` (lower variance on small sets)
#[derive(Debug, Clone)]
pub enum CalibratorKind {
    Platt(PlattScaler),
    Isotonic(IsotonicRegressor),
}

impl CalibratorKind {
    /// Automatically select and fit the appropriate calibrator.
    pub fn fit_auto(scores: &[f32], y: &[i32]) -> Self {
        if scores.len() > 1000 {
            Self::Isotonic(IsotonicRegressor::fit(scores, y))
        } else {
            Self::Platt(PlattScaler::fit(scores, y).unwrap_or(PlattScaler::identity()))
        }
    }

    pub fn transform(&self, score: f32) -> f32 {
        match self {
            Self::Platt(p) => p.transform(score),
            Self::Isotonic(r) => r.transform(score),
        }
    }

    pub fn transform_slice(&self, scores: &[f32]) -> Vec<f32> {
        scores.iter().map(|&s| self.transform(s)).collect()
    }

    /// Short name for JSON serialization.
    pub fn method_name(&self) -> &'static str {
        match self {
            Self::Platt(_) => "platt",
            Self::Isotonic(_) => "isotonic",
        }
    }

    /// Returns `(a, b)` for Platt; `(NaN, NaN)` for isotonic.
    pub fn platt_params(&self) -> (f32, f32) {
        match self {
            Self::Platt(p) => (p.a, p.b),
            Self::Isotonic(_) => (f32::NAN, f32::NAN),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn platt_fits_identity_like_when_scores_already_calibrated() {
        let mut scores = Vec::new();
        let mut y = Vec::new();
        for i in 0..100 {
            let x = (i as f32) / 99.0;
            scores.push(x);
            y.push(if x > 0.5 { 1 } else { 0 });
        }
        let p = PlattScaler::fit(&scores, &y).unwrap();
        let mid = p.transform(0.5);
        assert!((mid - 0.5).abs() < 0.15, "mid = {}", mid);
        assert!(p.transform(0.99) > 0.7);
        assert!(p.transform(0.01) < 0.3);
    }

    #[test]
    fn platt_handles_single_class_safely() {
        let scores = vec![0.1, 0.2, 0.3];
        let y = vec![1, 1, 1];
        let p = PlattScaler::fit(&scores, &y).unwrap();
        assert!((p.a - 1.0).abs() < 1e-6);
        assert!((p.b - 0.0).abs() < 1e-6);
    }

    #[test]
    fn platt_rescales_overconfident_scores() {
        let scores = vec![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        let y = vec![0, 1, 0, 1, 1, 0, 1, 0];
        let p = PlattScaler::fit(&scores, &y).unwrap();
        let p0 = p.transform(0.0);
        let p1 = p.transform(1.0);
        assert!(p0 > 0.2 && p0 < 0.8);
        assert!(p1 > 0.2 && p1 < 0.8);
    }

    #[test]
    fn isotonic_monotone_on_perfect_data() {
        // Perfectly separable: all positives have score > 0.5.
        let scores: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let y: Vec<i32> = scores
            .iter()
            .map(|&s| if s > 0.5 { 1 } else { 0 })
            .collect();
        let reg = IsotonicRegressor::fit(&scores, &y);
        // Calibrated prob should be non-decreasing.
        for i in 1..reg.probs.len() {
            assert!(
                reg.probs[i] >= reg.probs[i - 1] - 1e-6,
                "monotonicity violated at {} ({} < {})",
                i,
                reg.probs[i],
                reg.probs[i - 1]
            );
        }
        // Low score → low prob, high score → high prob.
        assert!(reg.transform(0.1) < 0.5);
        assert!(reg.transform(0.9) > 0.5);
    }

    #[test]
    fn isotonic_identity_on_single_class() {
        let scores = vec![0.1f32, 0.5, 0.9];
        let y = vec![1i32, 1, 1];
        let reg = IsotonicRegressor::fit(&scores, &y);
        // Should fall back to identity (linear 0→1).
        assert!((reg.transform(0.5) - 0.5).abs() < 0.1);
    }

    #[test]
    fn isotonic_pav_merges_violations() {
        // Non-monotone raw labels: PAV must pool.
        // scores: 0.1 0.2 0.3 0.4 0.5
        // labels:  1   0   0   1   1
        // Without pooling: probs = [1, 0, 0, 1, 1] (violations at 0→1)
        // PAV merges until monotone.
        let scores = vec![0.1f32, 0.2, 0.3, 0.4, 0.5];
        let y = vec![1i32, 0, 0, 1, 1];
        let reg = IsotonicRegressor::fit(&scores, &y);
        for i in 1..reg.probs.len() {
            assert!(
                reg.probs[i] >= reg.probs[i - 1] - 1e-6,
                "monotonicity violated at {}",
                i
            );
        }
    }

    #[test]
    fn calibrator_dispatches_by_size() {
        // ≤ 1000 → Platt
        let scores: Vec<f32> = (0..50).map(|i| i as f32 / 49.0).collect();
        let y: Vec<i32> = scores
            .iter()
            .map(|&s| if s > 0.5 { 1 } else { 0 })
            .collect();
        match CalibratorKind::fit_auto(&scores, &y) {
            CalibratorKind::Platt(_) => {}
            CalibratorKind::Isotonic(_) => panic!("expected Platt for n=50"),
        }

        // > 1000 → Isotonic
        let scores: Vec<f32> = (0..1001).map(|i| i as f32 / 1000.0).collect();
        let y: Vec<i32> = scores
            .iter()
            .map(|&s| if s > 0.5 { 1 } else { 0 })
            .collect();
        match CalibratorKind::fit_auto(&scores, &y) {
            CalibratorKind::Isotonic(_) => {}
            CalibratorKind::Platt(_) => panic!("expected Isotonic for n=1001"),
        }
    }
}
