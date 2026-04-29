//! Shared KNN evaluator: stratified row-wise train/test/val split, z-score
//! normalize using train stats, fit KNN with the configured distance, log
//! accuracy and confusion matrix for both test and val.

use anyhow::Result;
use serde::Serialize;
use std::fs;
use std::path::Path;
use tracing::info;
use utils::bids_filename::BidsFilename;

use crate::classifiers::{
    DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary, sensitivity_from_cm,
    specificity_from_cm,
};
use crate::dataset::{FeatureSource, Label};
use crate::normalizer::ZScoreNormalizer;
use crate::splits::split_rows_stratified_new;

const SEED: u64 = 42;

#[derive(Debug, Serialize)]
struct SplitReport {
    n_samples: usize,
    accuracy: f32,
    sensitivity: f32,
    specificity: f32,
    confusion_matrix: [[u32; 2]; 2],
}

#[derive(Debug, Serialize)]
struct SplitManifest {
    train: Vec<String>,
    test: Vec<String>,
    val: Vec<String>,
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
    test: SplitReport,
    val: SplitReport,
    split_manifest: SplitManifest,
}

fn to_f64(rows: &[Vec<f32>]) -> Vec<Vec<f64>> {
    rows.iter()
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect()
}

fn to_f32(rows: &[Vec<f64>]) -> Vec<Vec<f32>> {
    rows.iter()
        .map(|r| r.iter().map(|&v| v as f32).collect())
        .collect()
}

fn sorted_unique_subjects(indices: &[usize], groups: &[String]) -> Vec<String> {
    let mut seen = std::collections::BTreeSet::new();
    for &i in indices {
        seen.insert(groups[i].clone());
    }
    seen.into_iter().collect()
}

/// Stratified row-wise split, train-fit z-score, K-NN with the supplied
/// distance metric. Logs results tagged with `analysis` and `source`.
/// `groups` is the subject ID for each row (same length as `xs`/`ys`); used
/// to record which subjects landed in each split for reproducibility checks.
pub fn eval_knn_three_way_split(
    xs: &[Vec<f32>],
    ys: &[Label],
    groups: &[String],
    num_neighbors: usize,
    metric: DistanceMetric,
    analysis: &str,
    source: FeatureSource,
    results_dir: &Path,
) -> Result<()> {
    let (train_idx, test_idx, val_idx) = split_rows_stratified_new(ys, SEED);

    let take = |idx: &[usize]| -> (Vec<Vec<f32>>, Vec<i32>) {
        let mut x = Vec::with_capacity(idx.len());
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.push(xs[i].clone());
            y.push(ys[i].as_i32());
        }
        (x, y)
    };
    let (x_train, y_train) = take(&train_idx);
    let (x_test, y_test) = take(&test_idx);
    let (x_val, y_val) = take(&val_idx);

    let normalizer = ZScoreNormalizer::fit(&to_f64(&x_train));
    let x_train_n = to_f32(&normalizer.transform(&to_f64(&x_train)));
    let x_test_n = to_f32(&normalizer.transform(&to_f64(&x_test)));
    let x_val_n = to_f32(&normalizer.transform(&to_f64(&x_val)));

    let mut knn = KNN::new(KnnConfig {
        num_neighbors,
        metric,
        distance_weighted: false,
        mahalanobis_shrinkage: 0.0,
    });
    knn.fit(x_train_n, y_train.clone())?;

    let test_pred: Vec<i32> = x_test_n
        .iter()
        .map(|x| knn.predict(x).unwrap_or(-1))
        .collect();
    let val_pred: Vec<i32> = x_val_n
        .iter()
        .map(|x| knn.predict(x).unwrap_or(-1))
        .collect();

    let test_acc = accuracy(&y_test, &test_pred);
    let val_acc = accuracy(&y_val, &val_pred);
    let test_cm = confusion_matrix_binary(&y_test, &test_pred);
    let val_cm = confusion_matrix_binary(&y_val, &val_pred);

    info!(
        analysis,
        source = ?source,
        n_train = train_idx.len(),
        n_test = test_idx.len(),
        n_val = val_idx.len(),
        test_acc = format!("{:.2}%", test_acc * 100.0),
        val_acc = format!("{:.2}%", val_acc * 100.0),
        test_confusion = ?test_cm,
        val_confusion = ?val_cm,
        "knn results"
    );

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
        distance_weighted: false,
        n_train: train_idx.len(),
        test: SplitReport {
            n_samples: test_idx.len(),
            accuracy: test_acc,
            sensitivity: sensitivity_from_cm(&test_cm),
            specificity: specificity_from_cm(&test_cm),
            confusion_matrix: test_cm,
        },
        val: SplitReport {
            n_samples: val_idx.len(),
            accuracy: val_acc,
            sensitivity: sensitivity_from_cm(&val_cm),
            specificity: specificity_from_cm(&val_cm),
            confusion_matrix: val_cm,
        },
        split_manifest: SplitManifest {
            train: sorted_unique_subjects(&train_idx, groups),
            test: sorted_unique_subjects(&test_idx, groups),
            val: sorted_unique_subjects(&val_idx, groups),
        },
    };

    let mut run_counter = 0;
    let mut out_path;
    loop {
        let filename = BidsFilename::new()
            .with_pair("analysis", analysis)
            .with_pair("source", source_name.as_str())
            .with_pair("classifier", "knn")
            .with_pair("k", num_neighbors.to_string())
            .with_pair("metric", metric_name.as_str())
            .with_pair("run", &format!("{:02}", run_counter))
            .with_suffix("classification")
            .with_extension(".json")
            .to_filename();

        out_path = results_dir.join(filename);
        if !out_path.exists() {
            break;
        }
        run_counter += 1;
    }

    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&out_path, json)?;
    info!(path = %out_path.display(), "wrote classification report");

    Ok(())
}
