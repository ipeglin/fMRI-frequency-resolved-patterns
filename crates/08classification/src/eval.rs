//! Probabilistic KNN evaluator.
//!
//! Three-way stratified row-wise split, z-score using train stats, fit KNN
//! with the configured distance, then **emit per-sample probabilities** rather
//! than hard labels:
//!
//! * `p1_raw` – raw KNN vote-share for class 1 (anhedonic).
//! * `p1_cal` – Platt-scaled probability fit on the test (calibration) split
//!   and applied to val (held-out evaluation set).
//!
//! For each split we compute Brier score, log loss, AUC-ROC, AUC-PR, expected
//! calibration error, a uniform-binned reliability table, a threshold sweep,
//! and the Youden-optimal threshold. We retain the legacy `accuracy /
//! sensitivity / specificity / confusion_matrix` block so existing notebooks
//! keep working — they're now reported at the Youden threshold rather than
//! 0.5, with a parallel `*_at_0_5` block for direct comparison to the
//! pre-refactor numbers.
//!
//! A sibling `*_subject_probs.csv` is dumped next to each JSON for downstream
//! python plotting (reliability diagram, subject-rank uncertainty plot).

use anyhow::Result;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use tracing::info;
use utils::bids_filename::BidsFilename;

use crate::classifiers::{
    DistanceMetric, KNN, KnnConfig, RandomForestWrapper, accuracy, confusion_matrix_binary,
    sensitivity_from_cm, specificity_from_cm,
};
use crate::dataset::{FeatureSource, Label};
use crate::metrics::{
    BootstrapCi, CalibrationBin, ThresholdReport, auc_pr, auc_roc, bootstrap_ci, brier_score,
    brier_skill_score, calibration_bins, calibration_slope_intercept, expected_calibration_error,
    f1_optimal_threshold, log_loss, permutation_pvalue_auc, specificity_constrained_threshold,
    threshold_sweep, youden_optimal_threshold,
};
use crate::normalizer::ZScoreNormalizer;
use crate::pca::PcaReducer;
use crate::splits::{
    balance_train_indices, split_rows_stratified_new, split_subjects_stratified,
    subject_kfold_splits,
};

const SEED: u64 = 42;
const SWEEP_THRESHOLDS: &[f32] = &[0.3, 0.4, 0.5, 0.6, 0.7];
const N_CALIBRATION_BINS: usize = 10;
const LOGLOSS_EPS: f32 = 1e-7;
const N_BOOTSTRAP: usize = 1000;
const N_PERMUTATIONS: usize = 1000;
/// Minimum specificity for the `at_spec90` operating point.
const TARGET_SPECIFICITY: f32 = 0.90;

#[derive(Debug, Serialize)]
struct HardReport {
    threshold: f32,

    accuracy: f32,
    sensitivity: f32,
    specificity: f32,

    precision: f32,
    npv: f32,
    f1_score: f32,
    mcc: f32,

    confusion_matrix: [[u32; 2]; 2],
}

#[derive(Debug, Serialize)]
struct ProbabilisticReport {
    brier: BootstrapCi,
    log_loss: BootstrapCi,
    auc_roc: BootstrapCi,
    auc_pr: BootstrapCi,
    brier_skill_score: BootstrapCi,
    calibration_slope: f32,
    calibration_intercept: f32,
    auc_roc_perm_pvalue: f32,
    expected_calibration_error: f32,
    calibration_bins: Vec<CalibrationBin>,
    threshold_sweep: Vec<ThresholdReport>,
    youden_threshold: f32,
    f1_threshold: f32,
}

#[derive(Debug, Serialize)]
struct SplitReport {
    n_samples: usize,
    at_0_5: HardReport,
    at_youden: HardReport,
    at_f1: HardReport,
    at_spec90: HardReport,
    probabilistic: ProbabilisticReport,
}

#[derive(Debug, Serialize)]
struct SplitEntry {
    subjects: Vec<String>,
    n_controls: usize,
    n_anhedonic: usize,
}

#[derive(Debug, Serialize)]
struct SplitManifest {
    train: SplitEntry,
    calibration: SplitEntry,
    holdout: SplitEntry,
}

#[derive(Debug, Clone, Serialize)]
struct PerSamplePrediction {
    subject: String,
    leaf: Option<String>,
    roi: Option<usize>,
    y_true: i32,
    p1: f32,
}

#[derive(Debug, Serialize)]
struct ClassificationReport {
    analysis: String,
    source: String,
    split_seed: u64,
    classifier: String,
    num_neighbors: usize,
    metric: String,
    distance_weighted: bool,
    n_train: usize,
    /// Number of PCA components used. None = no PCA (full 1920-dim vectors).
    pca_components: Option<usize>,
    /// Number of trees (random forest only; None for KNN).
    n_trees: Option<usize>,
    calibration: SplitReport,
    holdout: SplitReport,
    calibration_predictions: Vec<PerSamplePrediction>,
    holdout_predictions: Vec<PerSamplePrediction>,
    split_manifest: SplitManifest,
}

struct FoldOutput {
    calibration: SplitReport,
    holdout: SplitReport,
    calibration_predictions: Vec<PerSamplePrediction>,
    holdout_predictions: Vec<PerSamplePrediction>,
}

#[derive(Debug, Serialize)]
struct KFoldFoldReport {
    fold: usize,
    n_train: usize,
    n_calibration: usize,
    n_holdout: usize,
    holdout: SplitReport,
    split_manifest: SplitManifest,
}

#[derive(Debug, Serialize)]
struct AggregatedMetrics {
    mean_auc_roc: f32,
    std_auc_roc: f32,
    mean_auc_pr: f32,
    std_auc_pr: f32,
    mean_brier: f32,
    std_brier: f32,
    mean_log_loss: f32,
    std_log_loss: f32,
}

#[derive(Debug, Serialize)]
struct KFoldClassificationReport {
    analysis: String,
    source: String,
    split_seed: u64,
    classifier: String,
    num_neighbors: usize,
    metric: String,
    distance_weighted: bool,
    pca_components: Option<usize>,
    n_trees: Option<usize>,
    k_folds: usize,
    folds: Vec<KFoldFoldReport>,
    aggregated: AggregatedMetrics,
}

fn split_entry(indices: &[usize], groups: &[String], ys: &[Label]) -> SplitEntry {
    let mut seen = std::collections::BTreeMap::new();
    for &i in indices {
        seen.entry(parse_subject(&groups[i])).or_insert(ys[i]);
    }
    let n_controls = seen.values().filter(|&&l| l == Label::Control).count();
    let n_anhedonic = seen.values().filter(|&&l| l == Label::Anhedonic).count();
    SplitEntry {
        subjects: seen.into_keys().collect(),
        n_controls,
        n_anhedonic,
    }
}

/// Group strings produced by `dataset.rs` look like
/// `sub-NDARxxxx[_<leaf>]_roiNNN`. Subjects don't contain `_`, so the first
/// `_` (or `_roi` when there's no leaf) marks the subject boundary.
fn parse_group(g: &str) -> (String, Option<String>, Option<usize>) {
    let (prefix, roi) = match g.rfind("_roi") {
        Some(i) => {
            let roi = g[i + 4..].parse::<usize>().ok();
            (&g[..i], roi)
        }
        None => (g, None),
    };
    let (subject, leaf) = match prefix.find('_') {
        Some(i) => (prefix[..i].to_string(), Some(prefix[i + 1..].to_string())),
        None => (prefix.to_string(), None),
    };
    (subject, leaf, roi)
}

fn parse_subject(g: &str) -> String {
    parse_group(g).0
}

type SplitArrays = (
    Vec<Vec<f32>>,
    Vec<i32>,
    Vec<Vec<f32>>,
    Vec<i32>,
    Vec<Vec<f32>>,
    Vec<i32>,
);

/// Fill pre-allocated split vectors by cloning rows from `xs`/`ys` by index.
/// Takes references so `xs`/`ys` can be reused across multiple folds.
fn fill_splits(
    train_idx: &[usize],
    calib_idx: &[usize],
    holdout_idx: &[usize],
    xs: &[Vec<f32>],
    ys: &[Label],
) -> SplitArrays {
    let mut x_train = vec![Vec::<f32>::new(); train_idx.len()];
    let mut y_train = vec![0i32; train_idx.len()];
    let mut x_calib = vec![Vec::<f32>::new(); calib_idx.len()];
    let mut y_calib = vec![0i32; calib_idx.len()];
    let mut x_holdout = vec![Vec::<f32>::new(); holdout_idx.len()];
    let mut y_holdout = vec![0i32; holdout_idx.len()];
    for (slot, &i) in train_idx.iter().enumerate() {
        x_train[slot] = xs[i].clone();
        y_train[slot] = ys[i].as_i32();
    }
    for (slot, &i) in calib_idx.iter().enumerate() {
        x_calib[slot] = xs[i].clone();
        y_calib[slot] = ys[i].as_i32();
    }
    for (slot, &i) in holdout_idx.iter().enumerate() {
        x_holdout[slot] = xs[i].clone();
        y_holdout[slot] = ys[i].as_i32();
    }
    (x_train, y_train, x_calib, y_calib, x_holdout, y_holdout)
}

/// Normalize, optionally reduce via PCA, fit KNN, calibrate, and compute metrics.
#[allow(clippy::too_many_arguments)]
fn compute_knn(
    mut x_train: Vec<Vec<f32>>,
    y_train: Vec<i32>,
    mut x_calib: Vec<Vec<f32>>,
    y_calib: Vec<i32>,
    mut x_holdout: Vec<Vec<f32>>,
    y_holdout: Vec<i32>,
    calib_parsed: Vec<(String, Option<String>, Option<usize>)>,
    holdout_parsed: Vec<(String, Option<String>, Option<usize>)>,
    num_neighbors: usize,
    metric: DistanceMetric,
    pca_n_components: Option<usize>,
) -> Result<FoldOutput> {
    let normalizer = ZScoreNormalizer::fit_f32(&x_train);
    normalizer.transform_f32_inplace(&mut x_train);
    normalizer.transform_f32_inplace(&mut x_calib);
    normalizer.transform_f32_inplace(&mut x_holdout);

    if let Some(k) = pca_n_components {
        let reducer = PcaReducer::fit(&x_train, k)?;
        x_train = reducer.transform(&x_train);
        x_calib = reducer.transform(&x_calib);
        x_holdout = reducer.transform(&x_holdout);
    }

    let mut knn = KNN::new(KnnConfig {
        num_neighbors,
        metric,
        distance_weighted: true,
        mahalanobis_shrinkage: 0.0,
    });
    knn.fit(x_train, y_train)?;

    let classes = knn.classes().to_vec();
    let pos_idx = p1_index(&classes)
        .ok_or_else(|| anyhow::anyhow!("positive class label `1` missing from training data"))?;

    let p1_calib: Vec<f32> = knn
        .predict_proba_batch(&x_calib)?
        .into_iter()
        .map(|row| row[pos_idx])
        .collect();
    let p1_holdout: Vec<f32> = knn
        .predict_proba_batch(&x_holdout)?
        .into_iter()
        .map(|row| row[pos_idx])
        .collect();
    drop(x_calib);
    drop(x_holdout);

    let calib_youden_t = youden_optimal_threshold(&y_calib, &p1_calib);
    let calib_f1_t = f1_optimal_threshold(&y_calib, &p1_calib);
    let calib_spec90_t = specificity_constrained_threshold(&y_calib, &p1_calib, TARGET_SPECIFICITY);

    Ok(FoldOutput {
        calibration: SplitReport {
            n_samples: y_calib.len(),
            at_0_5: hard_report_at(&y_calib, &p1_calib, 0.5),
            at_youden: hard_report_at(&y_calib, &p1_calib, calib_youden_t),
            at_f1: hard_report_at(&y_calib, &p1_calib, calib_f1_t),
            at_spec90: hard_report_at(&y_calib, &p1_calib, calib_spec90_t),
            probabilistic: prob_report(&y_calib, &p1_calib),
        },
        holdout: SplitReport {
            n_samples: y_holdout.len(),
            at_0_5: hard_report_at(&y_holdout, &p1_holdout, 0.5),
            at_youden: hard_report_at(&y_holdout, &p1_holdout, calib_youden_t),
            at_f1: hard_report_at(&y_holdout, &p1_holdout, calib_f1_t),
            at_spec90: hard_report_at(&y_holdout, &p1_holdout, calib_spec90_t),
            probabilistic: prob_report(&y_holdout, &p1_holdout),
        },
        calibration_predictions: build_predictions(calib_parsed, &y_calib, &p1_calib),
        holdout_predictions: build_predictions(holdout_parsed, &y_holdout, &p1_holdout),
    })
}

/// Normalize, optionally reduce via PCA, fit RF, and compute metrics.
#[allow(clippy::too_many_arguments)]
fn compute_rf(
    mut x_train: Vec<Vec<f32>>,
    y_train: Vec<i32>,
    mut x_calib: Vec<Vec<f32>>,
    y_calib: Vec<i32>,
    mut x_holdout: Vec<Vec<f32>>,
    y_holdout: Vec<i32>,
    calib_parsed: Vec<(String, Option<String>, Option<usize>)>,
    holdout_parsed: Vec<(String, Option<String>, Option<usize>)>,
    n_trees: usize,
    pca_n_components: Option<usize>,
    feature_subsample_ratio: f32,
) -> Result<FoldOutput> {
    let normalizer = ZScoreNormalizer::fit_f32(&x_train);
    normalizer.transform_f32_inplace(&mut x_train);
    normalizer.transform_f32_inplace(&mut x_calib);
    normalizer.transform_f32_inplace(&mut x_holdout);

    if let Some(k) = pca_n_components {
        let reducer = PcaReducer::fit(&x_train, k)?;
        x_train = reducer.transform(&x_train);
        x_calib = reducer.transform(&x_calib);
        x_holdout = reducer.transform(&x_holdout);
    }

    let rf = RandomForestWrapper::fit(&x_train, &y_train, n_trees, SEED, feature_subsample_ratio)?;
    drop(x_train);

    let classes = rf.classes.clone();
    let pos_idx = p1_index(&classes).ok_or_else(|| {
        anyhow::anyhow!("RF eval: positive class label `1` missing from training data")
    })?;

    let p1_calib: Vec<f32> = rf
        .predict_proba_batch(&x_calib)?
        .into_iter()
        .map(|row| row[pos_idx])
        .collect();
    let p1_holdout: Vec<f32> = rf
        .predict_proba_batch(&x_holdout)?
        .into_iter()
        .map(|row| row[pos_idx])
        .collect();
    drop(x_calib);
    drop(x_holdout);

    let calib_youden_t = youden_optimal_threshold(&y_calib, &p1_calib);
    let calib_f1_t = f1_optimal_threshold(&y_calib, &p1_calib);
    let calib_spec90_t = specificity_constrained_threshold(&y_calib, &p1_calib, TARGET_SPECIFICITY);

    Ok(FoldOutput {
        calibration: SplitReport {
            n_samples: y_calib.len(),
            at_0_5: hard_report_at(&y_calib, &p1_calib, 0.5),
            at_youden: hard_report_at(&y_calib, &p1_calib, calib_youden_t),
            at_f1: hard_report_at(&y_calib, &p1_calib, calib_f1_t),
            at_spec90: hard_report_at(&y_calib, &p1_calib, calib_spec90_t),
            probabilistic: prob_report(&y_calib, &p1_calib),
        },
        holdout: SplitReport {
            n_samples: y_holdout.len(),
            at_0_5: hard_report_at(&y_holdout, &p1_holdout, 0.5),
            at_youden: hard_report_at(&y_holdout, &p1_holdout, calib_youden_t),
            at_f1: hard_report_at(&y_holdout, &p1_holdout, calib_f1_t),
            at_spec90: hard_report_at(&y_holdout, &p1_holdout, calib_spec90_t),
            probabilistic: prob_report(&y_holdout, &p1_holdout),
        },
        calibration_predictions: build_predictions(calib_parsed, &y_calib, &p1_calib),
        holdout_predictions: build_predictions(holdout_parsed, &y_holdout, &p1_holdout),
    })
}

fn aggregate_kfold_metrics(folds: &[KFoldFoldReport]) -> AggregatedMetrics {
    let n = folds.len() as f32;
    let auc_rocs: Vec<f32> = folds
        .iter()
        .map(|f| f.holdout.probabilistic.auc_roc.point)
        .collect();
    let auc_prs: Vec<f32> = folds
        .iter()
        .map(|f| f.holdout.probabilistic.auc_pr.point)
        .collect();
    let briers: Vec<f32> = folds
        .iter()
        .map(|f| f.holdout.probabilistic.brier.point)
        .collect();
    let log_losses: Vec<f32> = folds
        .iter()
        .map(|f| f.holdout.probabilistic.log_loss.point)
        .collect();

    let mean_f = |v: &[f32]| v.iter().sum::<f32>() / n;
    let std_f = |v: &[f32], m: f32| (v.iter().map(|x| (x - m).powi(2)).sum::<f32>() / n).sqrt();

    let m_roc = mean_f(&auc_rocs);
    let m_pr = mean_f(&auc_prs);
    let m_brier = mean_f(&briers);
    let m_ll = mean_f(&log_losses);

    AggregatedMetrics {
        mean_auc_roc: m_roc,
        std_auc_roc: std_f(&auc_rocs, m_roc),
        mean_auc_pr: m_pr,
        std_auc_pr: std_f(&auc_prs, m_pr),
        mean_brier: m_brier,
        std_brier: std_f(&briers, m_brier),
        mean_log_loss: m_ll,
        std_log_loss: std_f(&log_losses, m_ll),
    }
}

fn hard_report_at(y_true: &[i32], p1: &[f32], threshold: f32) -> HardReport {
    let preds: Vec<i32> = p1.iter().map(|&p| (p >= threshold) as i32).collect();
    let cm = confusion_matrix_binary(y_true, &preds);

    let tn_f = cm[0][0] as f32;
    let fp_f = cm[0][1] as f32;
    let fn_f = cm[1][0] as f32;
    let tp_f = cm[1][1] as f32;

    let precision = if tp_f + fp_f > 0.0 {
        tp_f / (tp_f + fp_f)
    } else {
        0.0
    };
    let sensitivity = if tp_f + fn_f > 0.0 {
        tp_f / (tp_f + fn_f)
    } else {
        0.0
    };
    let npv = if tn_f + fn_f > 0.0 {
        tn_f / (tn_f + fn_f)
    } else {
        0.0
    };

    let f1_score = if precision + sensitivity > 0.0 {
        2.0 * (precision * sensitivity) / (precision + sensitivity)
    } else {
        0.0
    };

    let mcc_denominator = ((tp_f + fp_f) * (tp_f + fn_f) * (tn_f + fp_f) * (tn_f + fn_f)).sqrt();
    let mcc = if mcc_denominator > 0.0 {
        ((tp_f * tn_f) - (fp_f * fn_f)) / mcc_denominator
    } else {
        0.0
    };

    HardReport {
        threshold,
        accuracy: accuracy(y_true, &preds),
        sensitivity: sensitivity_from_cm(&cm),
        specificity: specificity_from_cm(&cm),
        precision,
        npv,
        f1_score,
        mcc,
        confusion_matrix: cm,
    }
}

fn prob_report(y_true: &[i32], p1: &[f32]) -> ProbabilisticReport {
    let bins = calibration_bins(y_true, p1, N_CALIBRATION_BINS);
    let ece = expected_calibration_error(&bins);
    let (cal_slope, cal_intercept) = calibration_slope_intercept(y_true, p1);
    ProbabilisticReport {
        brier: bootstrap_ci(y_true, p1, brier_score, N_BOOTSTRAP, SEED),
        log_loss: bootstrap_ci(
            y_true,
            p1,
            |y, s| log_loss(y, s, LOGLOSS_EPS),
            N_BOOTSTRAP,
            SEED + 1,
        ),
        auc_roc: bootstrap_ci(y_true, p1, auc_roc, N_BOOTSTRAP, SEED + 2),
        auc_pr: bootstrap_ci(y_true, p1, auc_pr, N_BOOTSTRAP, SEED + 3),
        brier_skill_score: bootstrap_ci(y_true, p1, brier_skill_score, N_BOOTSTRAP, SEED + 4),
        calibration_slope: cal_slope,
        calibration_intercept: cal_intercept,
        auc_roc_perm_pvalue: permutation_pvalue_auc(y_true, p1, N_PERMUTATIONS, SEED + 5),
        expected_calibration_error: ece,
        calibration_bins: bins,
        threshold_sweep: threshold_sweep(y_true, p1, SWEEP_THRESHOLDS),
        youden_threshold: youden_optimal_threshold(y_true, p1),
        f1_threshold: f1_optimal_threshold(y_true, p1),
    }
}

fn p1_index(classes: &[i32]) -> Option<usize> {
    classes.iter().position(|&c| c == 1)
}

fn build_predictions(
    parsed_groups: Vec<(String, Option<String>, Option<usize>)>,
    y_true: &[i32],
    p1: &[f32],
) -> Vec<PerSamplePrediction> {
    parsed_groups
        .into_iter()
        .enumerate()
        .map(|(j, (subject, leaf, roi))| PerSamplePrediction {
            subject,
            leaf,
            roi,
            y_true: y_true[j],
            p1: p1[j],
        })
        .collect()
}

fn write_subject_probs_csv<'a>(
    path: &Path,
    predictions: impl IntoIterator<Item = &'a PerSamplePrediction>,
) -> Result<()> {
    let mut out = String::from("subject\tleaf\troi\ty_true\tp1\n");
    for p in predictions {
        out.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\n",
            p.subject,
            p.leaf.as_deref().unwrap_or(""),
            p.roi.map(|r| r.to_string()).unwrap_or_default(),
            p.y_true,
            p.p1,
        ));
    }
    fs::write(path, out)?;
    Ok(())
}

/// Shared KNN pipeline — takes pre-computed split indices and runs
/// normalization, optional PCA, KNN fit, calibration, metric computation, and output.
///
/// When `pca_n_components` is `Some(k)`, features are projected to k dims after
/// z-score normalisation and results are written to `results_dir/pca_{k}/`.
#[allow(clippy::too_many_arguments)]
fn run_knn_pipeline(
    train_idx: Vec<usize>,
    calibration_idx: Vec<usize>,
    holdout_idx: Vec<usize>,
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: Option<usize>,
) -> Result<()> {
    let train_entry = split_entry(&train_idx, groups, &ys);
    let calib_entry = split_entry(&calibration_idx, groups, &ys);
    let holdout_entry = split_entry(&holdout_idx, groups, &ys);
    let calib_parsed: Vec<(String, Option<String>, Option<usize>)> = calibration_idx
        .iter()
        .map(|&i| parse_group(&groups[i]))
        .collect();
    let holdout_parsed: Vec<(String, Option<String>, Option<usize>)> = holdout_idx
        .iter()
        .map(|&i| parse_group(&groups[i]))
        .collect();

    let (x_train, y_train, x_calib, y_calib, x_holdout, y_holdout) =
        fill_splits(&train_idx, &calibration_idx, &holdout_idx, &xs, &ys);
    let n_train = x_train.len();
    drop(xs);
    drop(ys);

    let output = compute_knn(
        x_train,
        y_train,
        x_calib,
        y_calib,
        x_holdout,
        y_holdout,
        calib_parsed,
        holdout_parsed,
        num_neighbors,
        metric,
        pca_n_components,
    )?;

    info!(
        analysis,
        source = ?source,
        n_train,
        n_calibration = calibration_idx.len(),
        n_holdout = holdout_idx.len(),
        holdout_acc_0_5 = format!("{:.2}%", output.holdout.at_0_5.accuracy * 100.0),
        holdout_acc_youden = format!("{:.2}%", output.holdout.at_youden.accuracy * 100.0),
        holdout_acc_spec90 = format!("{:.2}%", output.holdout.at_spec90.accuracy * 100.0),
        holdout_spec_spec90 = format!("{:.2}%", output.holdout.at_spec90.specificity * 100.0),
        holdout_brier = format!("{:.4}", output.holdout.probabilistic.brier.point),
        holdout_logloss = format!("{:.4}", output.holdout.probabilistic.log_loss.point),
        holdout_auc_pr = format!("{:.4}", output.holdout.probabilistic.auc_pr.point),
        holdout_auc_roc = format!("{:.4}", output.holdout.probabilistic.auc_roc.point),
        holdout_ece = format!("{:.4}", output.holdout.probabilistic.expected_calibration_error),
        holdout_cm_youden = ?output.holdout.at_youden.confusion_matrix,
        "knn probabilistic results"
    );

    let effective_results_dir;
    let results_dir: &Path = if let Some(k) = pca_n_components {
        effective_results_dir = results_dir.join(format!("pca_{k}"));
        &effective_results_dir
    } else {
        results_dir
    };
    fs::create_dir_all(results_dir)?;
    let source_name = source.dir().to_string();
    let metric_name = metric.as_str().to_string();

    let report = ClassificationReport {
        analysis: analysis.to_string(),
        source: source_name.clone(),
        split_seed: SEED,
        classifier: "knn".to_string(),
        num_neighbors,
        metric: metric_name.clone(),
        distance_weighted: true,
        n_train,
        pca_components: pca_n_components,
        n_trees: None,
        calibration: output.calibration,
        holdout: output.holdout,
        calibration_predictions: output.calibration_predictions,
        holdout_predictions: output.holdout_predictions,
        split_manifest: SplitManifest {
            train: train_entry,
            calibration: calib_entry,
            holdout: holdout_entry,
        },
    };

    let mut run_counter = 0;
    let (json_path, csv_path) = loop {
        let base = BidsFilename::new()
            .with_pair("analysis", analysis)
            .with_pair("source", source_name.as_str())
            .with_pair("classifier", "knn")
            .with_pair("k", num_neighbors.to_string())
            .with_pair("metric", metric_name.as_str())
            .with_pair("run", format!("{:02}", run_counter).as_str());

        let json_filename = base
            .clone()
            .with_suffix("classification")
            .with_extension(".json")
            .to_filename();
        let csv_filename = base
            .with_suffix("subject_probs")
            .with_extension(".tsv")
            .to_filename();

        let json_path = results_dir.join(json_filename);
        let csv_path = results_dir.join(csv_filename);
        if !json_path.exists() && !csv_path.exists() {
            break (json_path, csv_path);
        }
        run_counter += 1;
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&json_path, json)?;

    write_subject_probs_csv(
        &csv_path,
        report
            .calibration_predictions
            .iter()
            .chain(report.holdout_predictions.iter()),
    )?;

    info!(
        json = %json_path.display(),
        csv = %csv_path.display(),
        "wrote classification report"
    );

    Ok(())
}

/// Stratified row-wise split, train-fit z-score, K-NN with the supplied
/// distance metric. Computes raw and calibrated per-sample probabilities and
/// reports the full probabilistic metric suite for both test and val.
///
/// `pca_n_components`: list of PCA dimensionalities to run alongside the
/// full-vector result. Empty slice = full vectors only. Each PCA run writes
/// to `results_dir/pca_{k}/`.
///
/// Takes `xs` and `ys` by value so the caller's row buffer can be released
/// as soon as we've drained it into train/test/val splits.
#[allow(clippy::too_many_arguments)]
pub fn eval_knn_three_way_split(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
) -> Result<()> {
    let (train_idx, calibration_idx, holdout_idx) = split_rows_stratified_new(&ys, SEED);
    let train_idx = balance_train_indices(&train_idx, &ys, SEED);

    // Full-vector run.
    run_knn_pipeline(
        train_idx.clone(),
        calibration_idx.clone(),
        holdout_idx.clone(),
        xs.clone(),
        ys.clone(),
        groups,
        num_neighbors,
        metric,
        analysis,
        source,
        results_dir,
        None,
    )?;

    // PCA-reduced runs.
    for &k in pca_n_components {
        run_knn_pipeline(
            train_idx.clone(),
            calibration_idx.clone(),
            holdout_idx.clone(),
            xs.clone(),
            ys.clone(),
            groups,
            num_neighbors,
            metric,
            analysis,
            source,
            results_dir,
            Some(k),
        )?;
    }

    Ok(())
}

/// Subject-disjoint split: all rows for a given subject land in exactly one
/// of train / calibration / holdout. Prevents KNN from retrieving same-subject
/// neighbors across the split boundary.
///
/// `pca_n_components`: same semantics as `eval_knn_three_way_split`.
#[allow(clippy::too_many_arguments)]
pub fn eval_knn_three_way_split_subject_aware(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
) -> Result<()> {
    // Derive subject→label map from ys + groups (first occurrence wins).
    let mut subject_label: HashMap<String, Label> = HashMap::new();
    for (group, &label) in groups.iter().zip(ys.iter()) {
        let subj = parse_subject(group);
        subject_label.entry(subj).or_insert(label);
    }

    let mut controls: Vec<String> = Vec::new();
    let mut anhedonics: Vec<String> = Vec::new();
    for (subj, label) in &subject_label {
        match label {
            Label::Control => controls.push(subj.clone()),
            Label::Anhedonic => anhedonics.push(subj.clone()),
        }
    }
    controls.sort();
    anhedonics.sort();

    let (train_s, calib_s, holdout_s) = split_subjects_stratified(&controls, &anhedonics, SEED);
    let train_set: HashSet<String> = train_s.into_iter().collect();
    let calib_set: HashSet<String> = calib_s.into_iter().collect();
    let holdout_set: HashSet<String> = holdout_s.into_iter().collect();

    let mut train_idx: Vec<usize> = Vec::new();
    let mut calib_idx: Vec<usize> = Vec::new();
    let mut holdout_idx: Vec<usize> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let subj = parse_subject(group);
        if train_set.contains(&subj) {
            train_idx.push(i);
        } else if calib_set.contains(&subj) {
            calib_idx.push(i);
        } else if holdout_set.contains(&subj) {
            holdout_idx.push(i);
        }
    }

    if train_idx.is_empty() {
        anyhow::bail!(
            "subject-stratified split produced empty training set for analysis `{analysis}` / source `{:?}`",
            source
        );
    }

    let train_idx = balance_train_indices(&train_idx, &ys, SEED);

    // Full-vector run.
    run_knn_pipeline(
        train_idx.clone(),
        calib_idx.clone(),
        holdout_idx.clone(),
        xs.clone(),
        ys.clone(),
        groups,
        num_neighbors,
        metric,
        analysis,
        source,
        results_dir,
        None,
    )?;

    // PCA-reduced runs.
    for &k in pca_n_components {
        run_knn_pipeline(
            train_idx.clone(),
            calib_idx.clone(),
            holdout_idx.clone(),
            xs.clone(),
            ys.clone(),
            groups,
            num_neighbors,
            metric,
            analysis,
            source,
            results_dir,
            Some(k),
        )?;
    }

    Ok(())
}

/// Shared Random Forest pipeline — takes pre-computed split indices and runs
/// normalization, optional PCA, RF fit, metric computation, and output.
///
/// When `pca_n_components` is `Some(k)`, results are written to `results_dir/pca_{k}/`.
#[allow(clippy::too_many_arguments)]
fn run_rf_pipeline(
    train_idx: Vec<usize>,
    calibration_idx: Vec<usize>,
    holdout_idx: Vec<usize>,
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    n_trees: usize,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: Option<usize>,
    feature_subsample_ratio: f32,
) -> Result<()> {
    let train_entry = split_entry(&train_idx, groups, &ys);
    let calib_entry = split_entry(&calibration_idx, groups, &ys);
    let holdout_entry = split_entry(&holdout_idx, groups, &ys);
    let calib_parsed: Vec<(String, Option<String>, Option<usize>)> = calibration_idx
        .iter()
        .map(|&i| parse_group(&groups[i]))
        .collect();
    let holdout_parsed: Vec<(String, Option<String>, Option<usize>)> = holdout_idx
        .iter()
        .map(|&i| parse_group(&groups[i]))
        .collect();

    let (x_train, y_train, x_calib, y_calib, x_holdout, y_holdout) =
        fill_splits(&train_idx, &calibration_idx, &holdout_idx, &xs, &ys);
    let n_train = x_train.len();
    drop(xs);
    drop(ys);

    let output = compute_rf(
        x_train,
        y_train,
        x_calib,
        y_calib,
        x_holdout,
        y_holdout,
        calib_parsed,
        holdout_parsed,
        n_trees,
        pca_n_components,
        feature_subsample_ratio,
    )?;

    info!(
        analysis,
        source = ?source,
        n_train,
        n_calibration = calibration_idx.len(),
        n_holdout = holdout_idx.len(),
        n_trees,
        holdout_acc_0_5 = format!("{:.2}%", output.holdout.at_0_5.accuracy * 100.0),
        holdout_acc_youden = format!("{:.2}%", output.holdout.at_youden.accuracy * 100.0),
        holdout_brier = format!("{:.4}", output.holdout.probabilistic.brier.point),
        holdout_auc_roc = format!("{:.4}", output.holdout.probabilistic.auc_roc.point),
        "rf probabilistic results"
    );

    let effective_results_dir;
    let results_dir: &Path = if let Some(k) = pca_n_components {
        effective_results_dir = results_dir.join(format!("pca_{k}"));
        &effective_results_dir
    } else {
        results_dir
    };
    fs::create_dir_all(results_dir)?;
    let source_name = source.dir().to_string();

    let report = ClassificationReport {
        analysis: analysis.to_string(),
        source: source_name.clone(),
        split_seed: SEED,
        classifier: "random_forest".to_string(),
        num_neighbors: 0,
        metric: "gini".to_string(),
        distance_weighted: false,
        n_train,
        pca_components: pca_n_components,
        n_trees: Some(n_trees),
        calibration: output.calibration,
        holdout: output.holdout,
        calibration_predictions: output.calibration_predictions,
        holdout_predictions: output.holdout_predictions,
        split_manifest: SplitManifest {
            train: train_entry,
            calibration: calib_entry,
            holdout: holdout_entry,
        },
    };

    let mut run_counter = 0;
    let (json_path, csv_path) = loop {
        let base = BidsFilename::new()
            .with_pair("analysis", analysis)
            .with_pair("source", source_name.as_str())
            .with_pair("classifier", "rf")
            .with_pair("trees", n_trees.to_string())
            .with_pair("run", format!("{:02}", run_counter).as_str());

        let json_filename = base
            .clone()
            .with_suffix("classification")
            .with_extension(".json")
            .to_filename();
        let csv_filename = base
            .with_suffix("subject_probs")
            .with_extension(".tsv")
            .to_filename();

        let json_path = results_dir.join(json_filename);
        let csv_path = results_dir.join(csv_filename);
        if !json_path.exists() && !csv_path.exists() {
            break (json_path, csv_path);
        }
        run_counter += 1;
    };

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&json_path, json)?;
    write_subject_probs_csv(
        &csv_path,
        report
            .calibration_predictions
            .iter()
            .chain(report.holdout_predictions.iter()),
    )?;
    info!(json = %json_path.display(), csv = %csv_path.display(), "wrote RF classification report");

    Ok(())
}

/// Stratified row-wise split with Random Forest ensemble.
///
/// Runs the full-vector pipeline once, then one PCA-reduced run per `k` in
/// `pca_n_components`. Each run writes to `results_dir/` or `results_dir/pca_{k}/`.
#[allow(clippy::too_many_arguments)]
pub fn eval_rf_three_way_split(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    n_trees: usize,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
) -> Result<()> {
    let (train_idx, calibration_idx, holdout_idx) = split_rows_stratified_new(&ys, SEED);
    let train_idx = balance_train_indices(&train_idx, &ys, SEED);

    run_rf_pipeline(
        train_idx.clone(),
        calibration_idx.clone(),
        holdout_idx.clone(),
        xs.clone(),
        ys.clone(),
        groups,
        n_trees,
        analysis,
        source,
        results_dir,
        None,
        0.0,
    )?;

    for &k in pca_n_components {
        run_rf_pipeline(
            train_idx.clone(),
            calibration_idx.clone(),
            holdout_idx.clone(),
            xs.clone(),
            ys.clone(),
            groups,
            n_trees,
            analysis,
            source,
            results_dir,
            Some(k),
            0.0,
        )?;
    }

    Ok(())
}

/// Subject-disjoint split with Random Forest ensemble.
///
/// Prevents same-subject rows from spanning split boundaries.
#[allow(clippy::too_many_arguments)]
pub fn eval_rf_three_way_split_subject_aware(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    n_trees: usize,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
) -> Result<()> {
    let mut subject_label: HashMap<String, Label> = HashMap::new();
    for (group, &label) in groups.iter().zip(ys.iter()) {
        subject_label.entry(parse_subject(group)).or_insert(label);
    }

    let mut controls: Vec<String> = Vec::new();
    let mut anhedonics: Vec<String> = Vec::new();
    for (subj, label) in &subject_label {
        match label {
            Label::Control => controls.push(subj.clone()),
            Label::Anhedonic => anhedonics.push(subj.clone()),
        }
    }
    controls.sort();
    anhedonics.sort();

    let (train_s, calib_s, holdout_s) = split_subjects_stratified(&controls, &anhedonics, SEED);
    let train_set: HashSet<String> = train_s.into_iter().collect();
    let calib_set: HashSet<String> = calib_s.into_iter().collect();
    let holdout_set: HashSet<String> = holdout_s.into_iter().collect();

    let mut train_idx: Vec<usize> = Vec::new();
    let mut calib_idx: Vec<usize> = Vec::new();
    let mut holdout_idx: Vec<usize> = Vec::new();
    for (i, group) in groups.iter().enumerate() {
        let subj = parse_subject(group);
        if train_set.contains(&subj) {
            train_idx.push(i);
        } else if calib_set.contains(&subj) {
            calib_idx.push(i);
        } else if holdout_set.contains(&subj) {
            holdout_idx.push(i);
        }
    }

    if train_idx.is_empty() {
        anyhow::bail!(
            "subject-stratified split produced empty training set for analysis `{analysis}` / source `{:?}`",
            source
        );
    }

    let train_idx = balance_train_indices(&train_idx, &ys, SEED);

    run_rf_pipeline(
        train_idx.clone(),
        calib_idx.clone(),
        holdout_idx.clone(),
        xs.clone(),
        ys.clone(),
        groups,
        n_trees,
        analysis,
        source,
        results_dir,
        None,
        0.0,
    )?;

    for &k in pca_n_components {
        run_rf_pipeline(
            train_idx.clone(),
            calib_idx.clone(),
            holdout_idx.clone(),
            xs.clone(),
            ys.clone(),
            groups,
            n_trees,
            analysis,
            source,
            results_dir,
            Some(k),
            0.0,
        )?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// K-fold subject-stratified evaluators
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn run_kfold_subject_aware<F>(
    xs: &[Vec<f32>],
    ys: &[Label],
    groups: &[String],
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
    k_folds: usize,
    classifier_name: &str,
    extra_pairs: &[(&str, String)],
    compute_fn: F,
) -> Result<()>
where
    F: Fn(
        Vec<Vec<f32>>,
        Vec<i32>,
        Vec<Vec<f32>>,
        Vec<i32>,
        Vec<Vec<f32>>,
        Vec<i32>,
        Vec<(String, Option<String>, Option<usize>)>,
        Vec<(String, Option<String>, Option<usize>)>,
        Option<usize>,
    ) -> Result<FoldOutput>,
{
    let mut subject_label: HashMap<String, Label> = HashMap::new();
    for (group, &label) in groups.iter().zip(ys.iter()) {
        subject_label.entry(parse_subject(group)).or_insert(label);
    }

    let mut controls: Vec<String> = Vec::new();
    let mut anhedonics: Vec<String> = Vec::new();
    for (subj, label) in &subject_label {
        match label {
            Label::Control => controls.push(subj.clone()),
            Label::Anhedonic => anhedonics.push(subj.clone()),
        }
    }
    controls.sort();
    anhedonics.sort();

    let kfold_splits = subject_kfold_splits(&controls, &anhedonics, k_folds, SEED);

    let pca_variants: Vec<Option<usize>> = std::iter::once(None)
        .chain(pca_n_components.iter().copied().map(Some))
        .collect();

    let source_name = source.dir().to_string();

    for pca in pca_variants {
        let mut fold_reports: Vec<KFoldFoldReport> = Vec::new();

        for (fold_idx, (train_s, calib_s, holdout_s)) in kfold_splits.iter().enumerate() {
            let train_set: HashSet<String> = train_s.iter().cloned().collect();
            let calib_set: HashSet<String> = calib_s.iter().cloned().collect();
            let holdout_set: HashSet<String> = holdout_s.iter().cloned().collect();

            let mut train_idx: Vec<usize> = Vec::new();
            let mut calib_idx: Vec<usize> = Vec::new();
            let mut holdout_idx: Vec<usize> = Vec::new();
            for (i, group) in groups.iter().enumerate() {
                let subj = parse_subject(group);
                if train_set.contains(&subj) {
                    train_idx.push(i);
                } else if calib_set.contains(&subj) {
                    calib_idx.push(i);
                } else if holdout_set.contains(&subj) {
                    holdout_idx.push(i);
                }
            }

            let train_idx = balance_train_indices(&train_idx, ys, SEED + fold_idx as u64);
            let train_entry = split_entry(&train_idx, groups, ys);
            let calib_entry = split_entry(&calib_idx, groups, ys);
            let holdout_entry = split_entry(&holdout_idx, groups, ys);
            let calib_parsed: Vec<_> = calib_idx.iter().map(|&i| parse_group(&groups[i])).collect();
            let holdout_parsed: Vec<_> = holdout_idx
                .iter()
                .map(|&i| parse_group(&groups[i]))
                .collect();
            let n_calib = calib_idx.len();
            let n_holdout = holdout_idx.len();

            let (x_train, y_train, x_calib, y_calib, x_holdout, y_holdout) =
                fill_splits(&train_idx, &calib_idx, &holdout_idx, xs, ys);
            let n_train = x_train.len();

            let output = compute_fn(
                x_train,
                y_train,
                x_calib,
                y_calib,
                x_holdout,
                y_holdout,
                calib_parsed,
                holdout_parsed,
                pca,
            )?;

            info!(
                analysis, classifier = classifier_name, fold = fold_idx, pca = ?pca,
                n_train, n_calib, n_holdout,
                holdout_auc_roc = format!("{:.4}", output.holdout.probabilistic.auc_roc.point),
                holdout_auc_pr = format!("{:.4}", output.holdout.probabilistic.auc_pr.point),
                holdout_brier = format!("{:.4}", output.holdout.probabilistic.brier.point),
                "k-fold fold result"
            );

            fold_reports.push(KFoldFoldReport {
                fold: fold_idx,
                n_train,
                n_calibration: n_calib,
                n_holdout,
                holdout: output.holdout,
                split_manifest: SplitManifest {
                    train: train_entry,
                    calibration: calib_entry,
                    holdout: holdout_entry,
                },
            });
        }

        let aggregated = aggregate_kfold_metrics(&fold_reports);
        info!(
            analysis, classifier = classifier_name, pca = ?pca, k_folds,
            mean_auc_roc = format!("{:.4}", aggregated.mean_auc_roc),
            std_auc_roc = format!("{:.4}", aggregated.std_auc_roc),
            mean_auc_pr = format!("{:.4}", aggregated.mean_auc_pr),
            std_auc_pr = format!("{:.4}", aggregated.std_auc_pr),
            "k-fold complete"
        );

        let report = KFoldClassificationReport {
            analysis: analysis.to_string(),
            source: source_name.clone(),
            split_seed: SEED,
            classifier: classifier_name.to_string(),
            num_neighbors: extra_pairs
                .iter()
                .find(|(k, _)| *k == "k")
                .and_then(|(_, v)| v.parse().ok())
                .unwrap_or(0),
            metric: extra_pairs
                .iter()
                .find(|(k, _)| *k == "metric")
                .map(|(_, v)| v.clone())
                .unwrap_or_default(),
            distance_weighted: classifier_name == "knn",
            pca_components: pca,
            n_trees: extra_pairs
                .iter()
                .find(|(k, _)| *k == "trees")
                .and_then(|(_, v)| v.parse().ok()),
            k_folds,
            folds: fold_reports,
            aggregated,
        };

        let effective_results_dir;
        let out_dir: &Path = if let Some(k) = pca {
            effective_results_dir = results_dir.join("kfold").join(format!("pca_{k}"));
            &effective_results_dir
        } else {
            effective_results_dir = results_dir.join("kfold");
            &effective_results_dir
        };
        fs::create_dir_all(out_dir)?;

        let mut run_counter = 0;
        let json_path = loop {
            let mut bids = BidsFilename::new()
                .with_pair("analysis", analysis)
                .with_pair("source", source_name.as_str())
                .with_pair("classifier", classifier_name)
                .with_pair("folds", k_folds.to_string());
            for (key, val) in extra_pairs {
                bids = bids.with_pair(*key, val.as_str());
            }
            let filename = bids
                .with_pair("run", format!("{:02}", run_counter).as_str())
                .with_suffix("kfold_classification")
                .with_extension(".json")
                .to_filename();
            let path = out_dir.join(filename);
            if !path.exists() {
                break path;
            }
            run_counter += 1;
        };

        fs::write(&json_path, serde_json::to_string_pretty(&report)?)?;
        info!(json = %json_path.display(), "wrote k-fold classification report");
    }

    Ok(())
}

/// Subject-disjoint k-fold CV with KNN.
///
/// Produces one aggregated JSON per PCA variant (plus full-vector) under
/// `results_dir/kfold/` and `results_dir/kfold/pca_{k}/`.
#[allow(clippy::too_many_arguments)]
pub fn eval_knn_kfold_subject_aware(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
    k_folds: usize,
) -> Result<()> {
    let extra = vec![
        ("k", num_neighbors.to_string()),
        ("metric", metric.as_str().to_string()),
    ];
    run_kfold_subject_aware(
        &xs,
        &ys,
        groups,
        analysis,
        source,
        results_dir,
        pca_n_components,
        k_folds,
        "knn",
        &extra,
        move |x_tr, y_tr, x_ca, y_ca, x_ho, y_ho, cp, hp, pca| {
            compute_knn(
                x_tr,
                y_tr,
                x_ca,
                y_ca,
                x_ho,
                y_ho,
                cp,
                hp,
                num_neighbors,
                metric,
                pca,
            )
        },
    )
}

/// Subject-disjoint k-fold CV with Random Forest.
///
/// Produces one aggregated JSON per PCA variant (plus full-vector) under
/// `results_dir/kfold/` and `results_dir/kfold/pca_{k}/`.
#[allow(clippy::too_many_arguments)]
pub fn eval_rf_kfold_subject_aware(
    xs: Vec<Vec<f32>>,
    ys: Vec<Label>,
    groups: &[String],
    n_trees: usize,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
    pca_n_components: &[usize],
    k_folds: usize,
) -> Result<()> {
    let extra = vec![("trees", n_trees.to_string())];
    run_kfold_subject_aware(
        &xs,
        &ys,
        groups,
        analysis,
        source,
        results_dir,
        pca_n_components,
        k_folds,
        "rf",
        &extra,
        move |x_tr, y_tr, x_ca, y_ca, x_ho, y_ho, cp, hp, pca| {
            compute_rf(
                x_tr, y_tr, x_ca, y_ca, x_ho, y_ho, cp, hp, n_trees, pca, 0.0,
            )
        },
    )
}
