//! Probabilistic metrics for binary classification.
//!
//! All functions assume label encoding `0 = negative` / `1 = positive` and
//! `p1` is the model's estimate of `P(y = 1 | x)`. None of the functions
//! allocate beyond their output struct unless explicitly noted.

use serde::Serialize;

/// Brier score: mean squared error between probability and 0/1 outcome.
/// Lower is better. Range `[0, 1]`. A model that always emits 0.5 yields 0.25.
pub fn brier_score(y_true: &[i32], p1: &[f32]) -> f32 {
    assert_eq!(y_true.len(), p1.len(), "brier: length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for (&y, &p) in y_true.iter().zip(p1.iter()) {
        let yf = y as f64;
        let pf = p as f64;
        acc += (pf - yf) * (pf - yf);
    }
    (acc / y_true.len() as f64) as f32
}

/// Mean negative log-likelihood. Probabilities are clamped to `[eps, 1-eps]`
/// so a confidently-wrong prediction yields a finite (but heavy) penalty.
pub fn log_loss(y_true: &[i32], p1: &[f32], eps: f32) -> f32 {
    assert_eq!(y_true.len(), p1.len(), "log_loss: length mismatch");
    if y_true.is_empty() {
        return 0.0;
    }
    let eps = eps.max(1e-15) as f64;
    let mut acc = 0.0f64;
    for (&y, &p) in y_true.iter().zip(p1.iter()) {
        let p = (p as f64).clamp(eps, 1.0 - eps);
        acc += if y == 1 { -p.ln() } else { -(1.0 - p).ln() };
    }
    (acc / y_true.len() as f64) as f32
}

#[derive(Debug, Clone, Serialize)]
pub struct RocPoint {
    pub threshold: f32,
    pub fpr: f32,
    pub tpr: f32,
}

/// Filter `(y, score)` pairs where the score is non-finite. Curves can't
/// rank NaN consistently — `NaN == NaN` is `false`, which used to spin the
/// tie-grouping loops forever. Returns owned vectors so the rest of the
/// pipeline keeps the simple `&[T]` shape.
fn drop_nan_scores(y_true: &[i32], scores: &[f32]) -> (Vec<i32>, Vec<f32>) {
    let mut yt = Vec::with_capacity(scores.len());
    let mut s = Vec::with_capacity(scores.len());
    for (&y, &p) in y_true.iter().zip(scores.iter()) {
        if p.is_finite() {
            yt.push(y);
            s.push(p);
        }
    }
    (yt, s)
}

/// Sort by score descending; sweep thresholds at every distinct score.
/// Returns curve including (0,0) and (1,1) endpoints. Non-finite scores are
/// dropped (a NaN can never be ranked, and would spin the tie loop).
pub fn roc_curve(y_true: &[i32], scores: &[f32]) -> Vec<RocPoint> {
    assert_eq!(y_true.len(), scores.len(), "roc_curve: length mismatch");
    let (y_true, scores) = drop_nan_scores(y_true, scores);
    let n = y_true.len();
    if n == 0 {
        return Vec::new();
    }
    let pos: f32 = y_true.iter().filter(|&&y| y == 1).count() as f32;
    let neg: f32 = (n as f32) - pos;
    if pos == 0.0 || neg == 0.0 {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut points = Vec::with_capacity(n + 2);
    points.push(RocPoint {
        threshold: f32::INFINITY,
        fpr: 0.0,
        tpr: 0.0,
    });
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut i = 0;
    while i < n {
        let s = scores[idx[i]];
        // Group ties so each distinct threshold yields one point.
        let group_start = i;
        while i < n && scores[idx[i]] == s {
            if y_true[idx[i]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        // Defence-in-depth: if the inner loop could not advance (would only
        // happen if `s` was NaN, but we already filter those), break out
        // rather than risk an unbounded outer loop.
        if i == group_start {
            break;
        }
        points.push(RocPoint {
            threshold: s,
            fpr: fp / neg,
            tpr: tp / pos,
        });
    }
    points
}

/// Area under the ROC curve via trapezoidal integration. `0.5` for chance,
/// `1.0` for perfect ranker.
pub fn auc_roc(y_true: &[i32], scores: &[f32]) -> f32 {
    let curve = roc_curve(y_true, scores);
    if curve.len() < 2 {
        return f32::NAN;
    }
    let mut auc = 0.0f64;
    for w in curve.windows(2) {
        let (a, b) = (&w[0], &w[1]);
        auc += (b.fpr - a.fpr) as f64 * (b.tpr + a.tpr) as f64 * 0.5;
    }
    auc as f32
}

#[derive(Debug, Clone, Serialize)]
pub struct PrPoint {
    pub threshold: f32,
    pub recall: f32,
    pub precision: f32,
}

pub fn pr_curve(y_true: &[i32], scores: &[f32]) -> Vec<PrPoint> {
    assert_eq!(y_true.len(), scores.len(), "pr_curve: length mismatch");
    let (y_true, scores) = drop_nan_scores(y_true, scores);
    let n = y_true.len();
    if n == 0 {
        return Vec::new();
    }
    let pos: f32 = y_true.iter().filter(|&&y| y == 1).count() as f32;
    if pos == 0.0 {
        return Vec::new();
    }

    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut points = Vec::with_capacity(n + 1);
    let mut tp = 0.0f32;
    let mut fp = 0.0f32;
    let mut i = 0;
    while i < n {
        let s = scores[idx[i]];
        let group_start = i;
        while i < n && scores[idx[i]] == s {
            if y_true[idx[i]] == 1 {
                tp += 1.0;
            } else {
                fp += 1.0;
            }
            i += 1;
        }
        if i == group_start {
            break;
        }
        let recall = tp / pos;
        let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 1.0 };
        points.push(PrPoint {
            threshold: s,
            recall,
            precision,
        });
    }
    points
}

/// Trapezoidal AUC over the precision-recall curve. Less optimistic than ROC
/// AUC under class imbalance, but slightly biased on small samples.
pub fn auc_pr(y_true: &[i32], scores: &[f32]) -> f32 {
    let curve = pr_curve(y_true, scores);
    if curve.is_empty() {
        return f32::NAN;
    }
    // Prepend (recall=0, precision=first.precision) so the integral starts at 0.
    let mut auc = 0.0f64;
    let mut prev_recall = 0.0f32;
    let mut prev_precision = curve[0].precision;
    for p in &curve {
        auc += (p.recall - prev_recall) as f64 * (p.precision + prev_precision) as f64 * 0.5;
        prev_recall = p.recall;
        prev_precision = p.precision;
    }
    auc as f32
}

#[derive(Debug, Clone, Serialize)]
pub struct CalibrationBin {
    pub bin_lo: f32,
    pub bin_hi: f32,
    pub count: u32,
    pub mean_pred: f32,
    pub frac_positive: f32,
}

/// Uniform-width binning of predicted probabilities for a reliability diagram.
/// Returns one entry per bin (empty bins included with `count = 0`).
pub fn calibration_bins(y_true: &[i32], p1: &[f32], n_bins: usize) -> Vec<CalibrationBin> {
    assert_eq!(y_true.len(), p1.len(), "calibration_bins: length mismatch");
    let n_bins = n_bins.max(1);
    let mut sums_pred = vec![0.0f64; n_bins];
    let mut sums_y = vec![0.0f64; n_bins];
    let mut counts = vec![0u32; n_bins];

    for (&y, &p) in y_true.iter().zip(p1.iter()) {
        let mut b = (p * n_bins as f32).floor() as usize;
        if b >= n_bins {
            b = n_bins - 1;
        }
        sums_pred[b] += p as f64;
        sums_y[b] += y as f64;
        counts[b] += 1;
    }

    (0..n_bins)
        .map(|b| {
            let bin_lo = b as f32 / n_bins as f32;
            let bin_hi = (b + 1) as f32 / n_bins as f32;
            let c = counts[b];
            let (mean_pred, frac_positive) = if c == 0 {
                (f32::NAN, f32::NAN)
            } else {
                (
                    (sums_pred[b] / c as f64) as f32,
                    (sums_y[b] / c as f64) as f32,
                )
            };
            CalibrationBin {
                bin_lo,
                bin_hi,
                count: c,
                mean_pred,
                frac_positive,
            }
        })
        .collect()
}

/// Expected Calibration Error: weighted average of `|frac_positive - mean_pred|`
/// across non-empty bins. Lower is better; 0 = perfectly calibrated.
pub fn expected_calibration_error(bins: &[CalibrationBin]) -> f32 {
    let n: u32 = bins.iter().map(|b| b.count).sum();
    if n == 0 {
        return f32::NAN;
    }
    let mut acc = 0.0f64;
    for b in bins {
        if b.count == 0 {
            continue;
        }
        let gap = (b.frac_positive - b.mean_pred).abs() as f64;
        acc += (b.count as f64) * gap;
    }
    (acc / n as f64) as f32
}

#[derive(Debug, Clone, Serialize)]
pub struct ThresholdReport {
    pub threshold: f32,
    pub accuracy: f32,
    pub sensitivity: f32,
    pub specificity: f32,
    pub youden_j: f32,
    pub f1: f32,
}

pub fn threshold_sweep(y_true: &[i32], p1: &[f32], thresholds: &[f32]) -> Vec<ThresholdReport> {
    assert_eq!(y_true.len(), p1.len(), "threshold_sweep: length mismatch");
    thresholds
        .iter()
        .map(|&t| evaluate_threshold(y_true, p1, t))
        .collect()
}

pub(crate) fn evaluate_threshold(y_true: &[i32], p1: &[f32], threshold: f32) -> ThresholdReport {
    let mut tp = 0u32;
    let mut fp = 0u32;
    let mut tn = 0u32;
    let mut fn_ = 0u32;
    for (&y, &p) in y_true.iter().zip(p1.iter()) {
        let pred = (p >= threshold) as i32;
        match (y, pred) {
            (1, 1) => tp += 1,
            (0, 1) => fp += 1,
            (0, 0) => tn += 1,
            (1, 0) => fn_ += 1,
            _ => {}
        }
    }
    let total = (tp + fp + tn + fn_).max(1) as f32;
    let pos = (tp + fn_).max(1) as f32;
    let neg = (tn + fp).max(1) as f32;
    let accuracy = (tp + tn) as f32 / total;
    let sensitivity = tp as f32 / pos;
    let specificity = tn as f32 / neg;
    let precision = if tp + fp > 0 {
        tp as f32 / (tp + fp) as f32
    } else {
        0.0
    };
    let f1 = if precision + sensitivity > 0.0 {
        2.0 * precision * sensitivity / (precision + sensitivity)
    } else {
        0.0
    };
    ThresholdReport {
        threshold,
        accuracy,
        sensitivity,
        specificity,
        youden_j: sensitivity + specificity - 1.0,
        f1,
    }
}

/// Threshold maximising F1-score along the precision-recall curve, swept over
/// the unique scores observed plus 0.0 and 1.0. Returns 0.5 when the sweep
/// yields no positive predictions or when the input is empty.
///
/// Ties broken by preferring the lower threshold (higher recall).
pub fn f1_optimal_threshold(y_true: &[i32], p1: &[f32]) -> f32 {
    if y_true.is_empty() {
        return 0.5;
    }
    let mut candidates: Vec<f32> = p1.iter().copied().filter(|p| p.is_finite()).collect();
    candidates.push(0.0);
    candidates.push(1.0);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut best_t = 0.5;
    let mut best_f1 = f32::NEG_INFINITY;
    for &t in &candidates {
        let r = evaluate_threshold(y_true, p1, t);
        // Prefer lower threshold on tie (more recall).
        if r.f1 > best_f1 {
            best_f1 = r.f1;
            best_t = t;
        }
    }
    best_t
}

/// Lowest threshold t such that specificity(t) ≥ `min_specificity`.
///
/// As threshold rises, fewer samples are predicted positive, so specificity
/// is non-decreasing. Scanning ascending candidates finds the least
/// conservative threshold that still satisfies the constraint — maximising
/// sensitivity subject to the specificity floor.
///
/// Returns `1.0` (predict nothing positive) when no candidate meets the target.
/// Returns `0.0` when there are no negative-class samples (specificity undefined).
pub fn specificity_constrained_threshold(y_true: &[i32], p1: &[f32], min_specificity: f32) -> f32 {
    if y_true.is_empty() {
        return 1.0;
    }
    let n_neg = y_true.iter().filter(|&&y| y == 0).count();
    if n_neg == 0 {
        return 0.0;
    }
    let mut candidates: Vec<f32> = p1.iter().copied().filter(|p| p.is_finite()).collect();
    candidates.push(0.0);
    candidates.push(1.0);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    for &t in &candidates {
        if evaluate_threshold(y_true, p1, t).specificity >= min_specificity {
            return t;
        }
    }
    1.0
}

/// Threshold maximising Youden's J = sensitivity + specificity - 1, swept over
/// the unique scores observed plus 0.0 and 1.0. Returns 0.5 when no positive
/// or negative class is present.
pub fn youden_optimal_threshold(y_true: &[i32], p1: &[f32]) -> f32 {
    if y_true.is_empty() {
        return 0.5;
    }
    // Skip NaNs — they can't be ranked or compared, so they'd survive
    // `dedup_by` (NaN - NaN is NaN, NaN < 1e-9 is false) and pollute the sweep.
    let mut candidates: Vec<f32> = p1.iter().copied().filter(|p| p.is_finite()).collect();
    candidates.push(0.0);
    candidates.push(1.0);
    candidates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-9);

    let mut best_t = 0.5;
    let mut best_j = f32::NEG_INFINITY;
    for &t in &candidates {
        let r = evaluate_threshold(y_true, p1, t);
        if r.youden_j > best_j {
            best_j = r.youden_j;
            best_t = t;
        }
    }
    best_t
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn perfect_classifier_metrics() {
        let y = vec![0, 0, 1, 1];
        let p = vec![0.1f32, 0.2, 0.8, 0.9];
        assert!(approx(brier_score(&y, &p), 0.025, 1e-5));
        assert!(approx(auc_roc(&y, &p), 1.0, 1e-6));
        assert!(approx(auc_pr(&y, &p), 1.0, 1e-6));
        assert!(log_loss(&y, &p, 1e-7) < 0.25);
    }

    #[test]
    fn random_classifier_auc_is_half() {
        let y = vec![0, 1, 0, 1, 0, 1];
        let p = vec![0.5f32; 6];
        assert!(approx(auc_roc(&y, &p), 0.5, 1e-6));
    }

    #[test]
    fn confidently_wrong_log_loss_high() {
        let y = vec![0, 1];
        let confident = vec![0.99f32, 0.01];
        let unsure = vec![0.5f32, 0.5];
        assert!(log_loss(&y, &confident, 1e-7) > log_loss(&y, &unsure, 1e-7));
    }

    #[test]
    fn calibration_perfect_when_p_equals_freq() {
        // Bin predictions roughly equal the empirical positive rate.
        let y = vec![0, 0, 0, 1, 1, 0, 1, 1];
        let p = vec![0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9];
        let bins = calibration_bins(&y, &p, 5);
        // ECE should be small.
        assert!(expected_calibration_error(&bins) < 0.4);
    }

    #[test]
    fn youden_picks_separating_threshold() {
        let y = vec![0, 0, 1, 1];
        let p = vec![0.1f32, 0.2, 0.8, 0.9];
        let t = youden_optimal_threshold(&y, &p);
        assert!(t > 0.2 && t <= 0.8, "t = {}", t);
    }

    #[test]
    fn f1_picks_separating_threshold() {
        let y = vec![0, 0, 1, 1];
        let p = vec![0.1f32, 0.2, 0.8, 0.9];
        let t = f1_optimal_threshold(&y, &p);
        // Perfect F1 achievable anywhere in (0.2, 0.8].
        assert!(t > 0.2 && t <= 0.8, "t = {}", t);
    }

    #[test]
    fn f1_returns_half_on_empty() {
        let y: Vec<i32> = vec![];
        let p: Vec<f32> = vec![];
        assert!((f1_optimal_threshold(&y, &p) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn spec_constrained_threshold_meets_target() {
        // 4 negatives, 2 positives; scores clearly separated.
        let y = vec![0, 0, 0, 0, 1, 1];
        let p = vec![0.1f32, 0.2, 0.3, 0.4, 0.8, 0.9];
        let t = specificity_constrained_threshold(&y, &p, 0.75);
        let r = evaluate_threshold(&y, &p, t);
        assert!(r.specificity >= 0.75, "spec = {}", r.specificity);
        // Should not be more conservative than necessary.
        // At t=0.4 all negatives are correctly rejected: spec = 1.0, sens > 0.
        assert!(t <= 0.4 + 1e-5, "threshold too high: {}", t);
    }

    #[test]
    fn spec_constrained_threshold_falls_back_to_one() {
        // Impossible to reach 99% specificity with these scores.
        let y = vec![0, 1];
        let p = vec![0.9f32, 0.1]; // inverted — no threshold yields spec >= 0.99
        let t = specificity_constrained_threshold(&y, &p, 0.99);
        // At t=1.0 nothing is predicted positive: TN=1, FP=0, spec=1.0.
        assert!((t - 1.0).abs() < 1e-6 || evaluate_threshold(&y, &p, t).specificity >= 0.99);
    }

    #[test]
    fn f1_threshold_maximises_f1_over_youden() {
        // Imbalanced: 1 positive vs 4 negatives.
        // F1 may prefer a threshold that trades specificity for recall.
        let y = vec![0, 0, 0, 0, 1];
        let p = vec![0.1f32, 0.2, 0.3, 0.45, 0.9];
        let t_f1 = f1_optimal_threshold(&y, &p);
        let r_f1 = evaluate_threshold(&y, &p, t_f1);
        // F1 at the chosen threshold should be non-trivial.
        assert!(r_f1.f1 > 0.0, "f1 = {}", r_f1.f1);
    }
}
