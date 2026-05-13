//! Analysis F (subject-stratified) — restAP baseline image-resized: subject-disjoint
//! 70/15/15 split so no subject appears in both train and calibration/holdout.

use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, info};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

use crate::classifiers::DistanceMetric;
use crate::dataset::{
    AnalysisKind, FeatureSource, build_mean_dataset, build_per_roi_dataset, load_labels,
};
use crate::eval::eval_knn_three_way_split_subject_aware;

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting baseline (image-resized) subject-stratified classification");

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

    let results_dir = cfg.resolved_classification_results_dir().join("subject_stratified");

    for source in [
        FeatureSource::Ts,
        FeatureSource::Cwt,
        FeatureSource::Hht,
        FeatureSource::HhtRoiStratified,
        FeatureSource::HhtSmoothed,
        FeatureSource::HhtRoiStratifiedSmoothed,
    ] {
        let (xs, ys, groups) = build_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids,
            &labels,
            source,
            AnalysisKind::BaselineResized,
        )?;
        if xs.is_empty() {
            debug!(source = ?source, "no samples, skipping");
            continue;
        }
        eval_knn_three_way_split_subject_aware(
            xs,
            ys,
            &groups,
            cfg.classification.knn_num_neighbors,
            metric,
            "baseline_resized",
            source,
            &results_dir,
        )?;
    }

    for source in [
        FeatureSource::Ts,
        FeatureSource::Cwt,
        FeatureSource::Hht,
        FeatureSource::HhtRoiStratified,
        FeatureSource::HhtSmoothed,
        FeatureSource::HhtRoiStratifiedSmoothed,
    ] {
        let (xs, ys, groups) = build_mean_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids,
            &labels,
            source,
            AnalysisKind::BaselineResized,
        )?;
        if xs.is_empty() {
            debug!(source = ?source, "no samples, skipping");
            continue;
        }
        eval_knn_three_way_split_subject_aware(
            xs,
            ys,
            &groups,
            cfg.classification.knn_num_neighbors,
            metric,
            "baseline_resized_mean",
            source,
            &results_dir,
        )?;
    }

    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "baseline (image-resized) subject-stratified done"
    );
    Ok(())
}
