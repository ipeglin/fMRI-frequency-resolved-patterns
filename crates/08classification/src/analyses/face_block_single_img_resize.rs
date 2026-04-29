//! Analysis G — hammerAP per face-block image-resized: per-ROI rows from
//! `features/<src>/task_per_block_resized/<block_name>` (one row per
//! subject × block × ROI) over both CWT and HHT. Upstream blocks were
//! bicubicly resized from `[224, T_block]` → `[224, 224]` before DenseNet.

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
    info!("starting task_per_block_resized classification");

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

    for source in [FeatureSource::Cwt, FeatureSource::Hht] {
        let (xs, ys, groups) = build_per_roi_dataset(
            &cfg.consolidated_data_dir,
            &subject_ids,
            &labels,
            source,
            AnalysisKind::TaskPerBlockResized,
        )?;
        info!(
            source = ?source,
            samples = xs.len(),
            features = xs.first().map(|r| r.len()).unwrap_or(0),
            "built task_per_block_resized dataset"
        );
        if xs.is_empty() {
            debug!(source = ?source, "no samples, skipping");
            continue;
        }
        eval_knn_three_way_split(
            &xs,
            &ys,
            &groups,
            cfg.classification.knn_num_neighbors,
            metric,
            "task_per_block_resized",
            source,
            &cfg.resolved_classification_results_dir(),
        )?;
    }

    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "task_per_block_resized done"
    );
    Ok(())
}
