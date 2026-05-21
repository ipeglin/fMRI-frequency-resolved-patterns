//! Analysis B (subject-stratified) — restAP baseline averaged: subject-disjoint
//! 70/15/15 split so no subject appears in both train and calibration/holdout.

use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, info};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

use crate::classifiers::DistanceMetric;
use crate::dataset::{AnalysisKind, build_per_roi_dataset, enabled_hammer_sources, load_labels};
use crate::eval::{
    eval_knn_kfold_subject_aware, eval_knn_three_way_split_subject_aware,
    eval_rf_kfold_subject_aware, eval_rf_three_way_split_subject_aware,
};

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting baseline (averaged) subject-stratified classification");

    let metric: DistanceMetric = cfg
        .classification
        .knn_metric
        .parse()
        .map_err(anyhow::Error::msg)
        .with_context(|| "invalid classification.knn_metric")?;

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

    let results_dir = cfg
        .resolved_classification_results_dir()
        .join("subject_stratified");

    for source in enabled_hammer_sources(cfg) {
        let (xs, ys, groups) = build_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids,
            &labels,
            source,
            AnalysisKind::BaselineAveraged,
        )?;
        if xs.is_empty() {
            debug!(source = ?source, "no samples, skipping");
            continue;
        }
        eval_knn_three_way_split_subject_aware(
            xs.clone(),
            ys.clone(),
            &groups,
            cfg.classification.knn_num_neighbors,
            metric,
            "baseline_averaged",
            source,
            &results_dir,
            &cfg.classification.pca_n_components,
        )?;
        eval_rf_three_way_split_subject_aware(
            xs.clone(),
            ys.clone(),
            &groups,
            cfg.classification.rf_n_trees,
            "baseline_averaged",
            source,
            &results_dir,
            &cfg.classification.pca_n_components,
        )?;
        eval_knn_kfold_subject_aware(
            xs.clone(),
            ys.clone(),
            &groups,
            cfg.classification.knn_num_neighbors,
            metric,
            "baseline_averaged",
            source,
            &results_dir,
            &cfg.classification.pca_n_components,
            cfg.classification.kfold_k,
        )?;
        eval_rf_kfold_subject_aware(
            xs,
            ys,
            &groups,
            cfg.classification.rf_n_trees,
            "baseline_averaged",
            source,
            &results_dir,
            &cfg.classification.pca_n_components,
            cfg.classification.kfold_k,
        )?;
    }

    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "baseline (averaged) subject-stratified done"
    );
    Ok(())
}
