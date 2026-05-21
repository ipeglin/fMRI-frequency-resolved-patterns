//! Subject-stratified per-ROI classification using `baseline_averaged`
//! features. Splits at the subject level so a person never appears in both
//! train and test/val (no leakage across ROIs of the same subject).

use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

use crate::classifiers::{DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary};
use crate::dataset::{AnalysisKind, FeatureSource, Label, build_per_roi_dataset, load_labels};
use crate::normalizer::ZScoreNormalizer;
use crate::splits::split_subjects_stratified;

const SEED: u64 = 42;

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

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting subject-stratified classification");

    let subject_ids: HashSet<String> = fs::read_dir(&cfg.consolidated_data_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let p = e.path();
            if !p.is_dir() {
                return None;
            }
            Some(BidsSubjectId::parse(p.file_name()?.to_str()?).to_dir_name())
        })
        .collect();
    let mut subject_ids_vec: Vec<String> = subject_ids.into_iter().collect();
    subject_ids_vec.sort();
    let labels = load_labels(&cfg.consolidated_data_dir)?;

    let mut controls = Vec::new();
    let mut anhedonics = Vec::new();
    for s in &subject_ids_vec {
        match labels.get(s) {
            Some(Label::Control) => controls.push(s.clone()),
            Some(Label::Anhedonic) => anhedonics.push(s.clone()),
            None => {}
        }
    }
    let (train_s, test_s, val_s) = split_subjects_stratified(&controls, &anhedonics, SEED);
    let train_set: HashSet<String> = train_s.into_iter().collect();
    let test_set: HashSet<String> = test_s.into_iter().collect();
    let val_set: HashSet<String> = val_s.into_iter().collect();

    for source in [
        FeatureSource::Cwt,
        FeatureSource::Hht,
        FeatureSource::HhtSmoothed,
    ] {
        let (xs, ys, groups) = build_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids_vec,
            &labels,
            source,
            AnalysisKind::BaselineAveraged,
        )?;
        if xs.is_empty() {
            continue;
        }

        let mut buckets: HashMap<&'static str, (Vec<Vec<f32>>, Vec<i32>)> = HashMap::new();
        for ((row, label), subj) in xs.into_iter().zip(ys.into_iter()).zip(groups.into_iter()) {
            let key = if train_set.contains(&subj) {
                "train"
            } else if test_set.contains(&subj) {
                "test"
            } else if val_set.contains(&subj) {
                "val"
            } else {
                continue;
            };
            let entry = buckets.entry(key).or_default();
            entry.0.push(row);
            entry.1.push(label.as_i32());
        }
        let take = |k: &str| buckets.get(k).cloned().unwrap_or_default();
        let (x_train, y_train) = take("train");
        let (x_test, y_test) = take("test");
        let (x_val, y_val) = take("val");

        if x_train.is_empty() {
            info!(source = ?source, "no training data, skipping");
            continue;
        }

        let normalizer = ZScoreNormalizer::fit(&to_f64(&x_train));
        let x_train_n = to_f32(&normalizer.transform(&to_f64(&x_train)));
        let x_test_n = to_f32(&normalizer.transform(&to_f64(&x_test)));
        let x_val_n = to_f32(&normalizer.transform(&to_f64(&x_val)));

        let mut knn = KNN::new(KnnConfig {
            num_neighbors: cfg.classification.knn_num_neighbors,
            metric: DistanceMetric::Cosine,
            distance_weighted: false,
            mahalanobis_shrinkage: 0.0,
        });
        knn.fit(x_train_n, y_train.clone())?;

        let test_pred: Vec<i32> = x_test_n
            .iter()
            .map(|x| knn.predict(x).unwrap_or(0))
            .collect();
        let val_pred: Vec<i32> = x_val_n
            .iter()
            .map(|x| knn.predict(x).unwrap_or(0))
            .collect();

        info!(
            source = ?source,
            test_acc = format!("{:.2}%", accuracy(&y_test, &test_pred) * 100.0),
            val_acc = format!("{:.2}%", accuracy(&y_val, &val_pred) * 100.0),
            test_confusion = ?confusion_matrix_binary(&y_test, &test_pred),
            val_confusion = ?confusion_matrix_binary(&y_val, &val_pred),
            "subject stratified results"
        );
    }
    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "subject stratified done"
    );
    Ok(())
}
