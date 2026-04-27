//! Analysis A — restAP baseline chunked: per-ROI rows from
//! `features/<src>/baseline_chunked/chunk_<i>` over both CWT and HHT.

use std::collections::HashSet;
use std::fs;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{debug, info};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

use crate::classifiers::DistanceMetric;
use crate::dataset::{AnalysisKind, FeatureSource, build_per_roi_dataset, load_labels};
use crate::eval::eval_knn_three_way_split;

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        "starting baseline (chunked) classification"
    );

    let metric: DistanceMetric = cfg
        .classification
        .knn_metric
        .parse()
        .map_err(anyhow::Error::msg)
        .with_context(|| "invalid classification.knn_metric")?;

    let mut labels = load_labels(&cfg.subject_filter_dir)?;
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
    info!(count = labels.len(), "valid subject labels loaded");

    for source in [FeatureSource::Cwt, FeatureSource::Hht] {
        let (xs, ys, _groups) = build_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids,
            &labels,
            source,
            AnalysisKind::BaselineChunked,
        )?;
        info!(
            source = ?source,
            samples = xs.len(),
            features = xs.first().map(|r| r.len()).unwrap_or(0),
            "built baseline_chunked dataset"
        );
        if xs.is_empty() {
            debug!(source = ?source, "no samples, skipping");
            continue;
        }
        eval_knn_three_way_split(
            &xs,
            &ys,
            cfg.classification.knn_num_neighbors,
            metric,
            "baseline_chunked",
            source,
            &cfg.classification_results_dir,
        )?;
    }

    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "baseline (chunked) done"
    );
    Ok(())
}
