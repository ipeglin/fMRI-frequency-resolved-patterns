//! Per-block ensemble: fit one KNN per face block, then majority-vote
//! predictions per subject. Splits subjects (groups) so the same person
//! never appears in both train and test/val. Reads `task_per_block` features.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::time::Instant;

use anyhow::Result;
use tracing::info;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

use crate::classifiers::{DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary};
use crate::dataset::{
    AnalysisKind, FeatureSource, Label, build_per_leaf_per_roi_dataset, load_labels,
};
use crate::normalizer::ZScoreNormalizer;
use crate::splits::split_groups_stratified;

const SEED: u64 = 42;

fn majority_vote(preds: &[i32], default: i32) -> i32 {
    let mut counts: HashMap<i32, usize> = HashMap::new();
    for &p in preds {
        *counts.entry(p).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, c)| c)
        .map(|(v, _)| v)
        .unwrap_or(default)
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

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting block ensemble (task_per_block) classification");

    let mut labels = load_labels(&cfg.consolidated_data_dir)?;
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
    labels.retain(|k, _| subject_ids.contains(k));
    let subject_ids_vec: Vec<String> = subject_ids.into_iter().collect();

    for source in [FeatureSource::Cwt, FeatureSource::Hht] {
        let by_block = build_per_leaf_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids_vec,
            &labels,
            source,
            AnalysisKind::TaskPerBlock,
        )?;
        if by_block.is_empty() {
            info!(source = ?source, "no blocks, skipping");
            continue;
        }

        // Subject group split is built once from the union of all blocks; same
        // subject always lands in the same partition across blocks.
        let mut all_groups: Vec<String> = Vec::new();
        let mut all_y_for_split: Vec<i32> = Vec::new();
        for (_, ys, groups) in by_block.values() {
            all_groups.extend(groups.iter().cloned());
            all_y_for_split.extend(ys.iter().map(|l| l.as_i32()));
        }
        let (train_g, test_g, val_g) = split_groups_stratified(&all_groups, &all_y_for_split, SEED);

        let mut subj_test_pred: HashMap<String, Vec<i32>> = HashMap::new();
        let mut subj_val_pred: HashMap<String, Vec<i32>> = HashMap::new();
        let mut subj_test_true: HashMap<String, Label> = HashMap::new();
        let mut subj_val_true: HashMap<String, Label> = HashMap::new();

        for (block_name, (xs, ys, groups)) in by_block {
            let mut x_train = Vec::new();
            let mut y_train_i32 = Vec::new();
            let mut x_test = Vec::new();
            let mut subj_test = Vec::new();
            let mut x_val = Vec::new();
            let mut subj_val = Vec::new();
            for ((row, label), subj) in xs.into_iter().zip(ys.into_iter()).zip(groups.into_iter()) {
                if train_g.contains(&subj) {
                    x_train.push(row);
                    y_train_i32.push(label.as_i32());
                } else if test_g.contains(&subj) {
                    x_test.push(row);
                    subj_test_true.insert(subj.clone(), label);
                    subj_test.push(subj);
                } else if val_g.contains(&subj) {
                    x_val.push(row);
                    subj_val_true.insert(subj.clone(), label);
                    subj_val.push(subj);
                }
            }
            if x_train.is_empty() {
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
            knn.fit(x_train_n, y_train_i32)?;

            for (row, subj) in x_test_n.iter().zip(subj_test.iter()) {
                let pred = knn.predict(row).unwrap_or(0);
                subj_test_pred.entry(subj.clone()).or_default().push(pred);
            }
            for (row, subj) in x_val_n.iter().zip(subj_val.iter()) {
                let pred = knn.predict(row).unwrap_or(0);
                subj_val_pred.entry(subj.clone()).or_default().push(pred);
            }
            info!(source = ?source, block = block_name, "block KNN fit");
        }

        let (mut y_t, mut p_t) = (Vec::new(), Vec::new());
        for (subj, preds) in subj_test_pred {
            p_t.push(majority_vote(&preds, 0));
            y_t.push(subj_test_true[&subj].as_i32());
        }
        let (mut y_v, mut p_v) = (Vec::new(), Vec::new());
        for (subj, preds) in subj_val_pred {
            p_v.push(majority_vote(&preds, 0));
            y_v.push(subj_val_true[&subj].as_i32());
        }

        info!(
            source = ?source,
            test_acc = format!("{:.2}%", accuracy(&y_t, &p_t) * 100.0),
            val_acc = format!("{:.2}%", accuracy(&y_v, &p_v) * 100.0),
            test_confusion = ?confusion_matrix_binary(&y_t, &p_t),
            val_confusion = ?confusion_matrix_binary(&y_v, &p_v),
            "block ensemble (subject majority vote) results"
        );
    }
    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "block ensemble done"
    );
    Ok(())
}
