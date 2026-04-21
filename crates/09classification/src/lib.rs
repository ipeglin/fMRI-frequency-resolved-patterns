mod classifiers;
mod dataset;

use std::collections::HashMap;
use std::time::Instant;

use anyhow::{Context, Result};
use tracing::{info, warn};
use utils::atlas::BrainAtlas;
use utils::config::AppConfig;

use crate::classifiers::{DistanceMetric, KNN, KnnConfig, accuracy, confusion_matrix_binary};
use crate::dataset::{
    FcFlatten, FcSelection, FcSpec, FeatureAggregation, FeatureSource, FeatureSpec,
    FeatureSubgroup, Label, build_dataset, build_fc_dataset, detect_n_rois, load_labels,
    load_subject_ids, read_roi_metadata,
};

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    // Disable HDF5 advisory file locking to match upstream crates (macOS/NFS).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        data_splitting_output_dir = %cfg.data_splitting_output_dir.display(),
        parcellated_ts_dir = %cfg.parcellated_ts_dir.display(),
        subject_filter_dir = %cfg.subject_filter_dir.display(),
        "starting subject classification"
    );

    // 1. Subject splits
    let train_ids = load_subject_ids(&cfg.data_splitting_output_dir.join("subjects_train.csv"))
        .context("loading training subjects")?;
    let test_ids = load_subject_ids(&cfg.data_splitting_output_dir.join("subjects_test.csv"))
        .context("loading test subjects")?;
    let val_ids = load_subject_ids(
        &cfg.data_splitting_output_dir
            .join("subjects_validation.csv"),
    )
    .context("loading validation subjects")?;

    info!(
        train = train_ids.len(),
        test = test_ids.len(),
        val = val_ids.len(),
        "loaded subject splits"
    );

    // 2. Labels: sub-<id> -> HealthyControl | Anhedonic
    let labels = load_labels(&cfg.subject_filter_dir).context("loading subject labels")?;
    info!(labeled_subjects = labels.len(), "loaded subject labels");

    // 3. Atlas: resolve target ROI indices (into FC channel ordering, 0..n_total)
    //    and readable labels (e.g. "17networks_LH_..._PFCv_...").
    let atlas = BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let atlas_rois = atlas.vpfc_mpfc_amy_ids();
    let atlas_roi_indices: Vec<usize> = atlas_rois.iter().map(|(i, _)| *i).collect();
    let atlas_roi_labels: Vec<String> = atlas_rois.iter().map(|(_, l)| l.clone()).collect();
    info!(
        n_target_rois = atlas_roi_indices.len(),
        rois = ?atlas_roi_labels,
        "resolved target ROIs from atlas"
    );

    let knn_cfg = KnnConfig {
        num_neighbors: 5,
        metric: DistanceMetric::Cosine,
        distance_weighted: true,
        mahalanobis_shrinkage: 1e-3,
    };

    // =========================================================================
    // DenseNet feature experiments
    // =========================================================================
    let source = FeatureSource::Cwt;
    let subgroup = FeatureSubgroup::WholeBand;

    // Aggregate: one mean vector per trial (collapses across the 28 ROIs).
    let mean_spec = FeatureSpec {
        source,
        subgroup: subgroup.clone(),
        aggregation: FeatureAggregation::Mean,
    };
    info!(spec = ?mean_spec, config = ?knn_cfg, "KNN — feature: mean across ROIs");
    run_feature_experiment(
        cfg,
        &train_ids,
        &test_ids,
        &val_ids,
        &labels,
        &mean_spec,
        &knn_cfg,
        "feat_mean",
    )?;

    // Concat all 28 per-ROI feature rows into one big vector per trial.
    let concat_spec = FeatureSpec {
        source,
        subgroup: subgroup.clone(),
        aggregation: FeatureAggregation::Concat,
    };
    info!(spec = ?concat_spec, "KNN — feature: all 28 ROIs concat");
    run_feature_experiment(
        cfg,
        &train_ids,
        &test_ids,
        &val_ids,
        &labels,
        &concat_spec,
        &knn_cfg,
        "feat_concat",
    )?;

    // Per-ROI: one classifier per ROI. Prefer labels stored in h5; fall back to atlas.
    let n_rois = detect_n_rois(&cfg.parcellated_ts_dir, &train_ids, source, &subgroup)
        .context("detecting n_rois from training data")?;
    let (h5_labels, _h5_indices) =
        read_roi_metadata(&cfg.parcellated_ts_dir, &train_ids, source, &subgroup)
            .unwrap_or_default();
    let per_roi_names: Vec<String> = (0..n_rois)
        .map(|i| {
            h5_labels
                .get(i)
                .cloned()
                .or_else(|| atlas_roi_labels.get(i).cloned())
                .unwrap_or_else(|| format!("roi_{}", i))
        })
        .collect();
    info!(n_rois = n_rois, "per-ROI sweep starting");

    for (roi, roi_label) in per_roi_names.iter().enumerate() {
        let spec = FeatureSpec {
            source,
            subgroup: subgroup.clone(),
            aggregation: FeatureAggregation::PerRoi(roi),
        };
        let tag = format!("feat_roi_{:02}_{}", roi, sanitize(roi_label));
        if let Err(e) = run_feature_experiment(
            cfg, &train_ids, &test_ids, &val_ids, &labels, &spec, &knn_cfg, &tag,
        ) {
            warn!(roi = roi, roi_label = %roi_label, error = %e, "per-ROI experiment failed");
        }
    }

    // =========================================================================
    // FC (functional connectivity) experiments — whole-band standardized, Fisher-Z
    // =========================================================================
    let fc_common_path = "fc/standardized".to_string();
    let fc_dataset = "fisher_z".to_string();

    // Full upper triangle: all 432 ROIs → 432*431/2 = 93096 features.
    let fc_full = FcSpec {
        group_path: fc_common_path.clone(),
        dataset: fc_dataset.clone(),
        selection: FcSelection::All,
        flatten: FcFlatten::UpperTriangle,
    };
    info!(spec = ?fc_full, "KNN — FC: full upper triangle");
    run_fc_experiment(
        cfg,
        &train_ids,
        &test_ids,
        &val_ids,
        &labels,
        &fc_full,
        &knn_cfg,
        "fc_full_tri",
    )?;

    // 28x28 submatrix upper triangle: focused on target ROIs → 28*27/2 = 378 features.
    let fc_subset = FcSpec {
        group_path: fc_common_path.clone(),
        dataset: fc_dataset.clone(),
        selection: FcSelection::SubsetRois(atlas_roi_indices.clone()),
        flatten: FcFlatten::UpperTriangle,
    };
    info!(spec = ?fc_subset, "KNN — FC: 28x28 subset upper triangle");
    run_fc_experiment(
        cfg,
        &train_ids,
        &test_ids,
        &val_ids,
        &labels,
        &fc_subset,
        &knn_cfg,
        "fc_subset_tri",
    )?;

    // 28 rows × 432 cols: seed-to-whole-brain → 28*432 = 12096 features.
    let fc_rows = FcSpec {
        group_path: fc_common_path.clone(),
        dataset: fc_dataset.clone(),
        selection: FcSelection::SubsetRois(atlas_roi_indices.clone()),
        flatten: FcFlatten::RowsFlat,
    };
    info!(spec = ?fc_rows, "KNN — FC: 28 seed rows");
    run_fc_experiment(
        cfg,
        &train_ids,
        &test_ids,
        &val_ids,
        &labels,
        &fc_rows,
        &knn_cfg,
        "fc_seed_rows",
    )?;

    let total_ms = run_start.elapsed().as_millis();
    info!(
        total_duration_ms = total_ms,
        "classification pipeline complete"
    );

    Ok(())
}

/// Strip characters that break log parsers / filesystem paths, so per-ROI tags
/// stay readable and stable across runs.
fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn run_feature_experiment(
    cfg: &AppConfig,
    train_ids: &[String],
    test_ids: &[String],
    val_ids: &[String],
    labels: &HashMap<String, Label>,
    spec: &FeatureSpec,
    knn_cfg: &KnnConfig,
    tag: &str,
) -> Result<()> {
    let (x_train, y_train) =
        build_dataset(&cfg.parcellated_ts_dir, train_ids, labels, spec).context("training set")?;
    let (x_test, y_test) =
        build_dataset(&cfg.parcellated_ts_dir, test_ids, labels, spec).context("test set")?;
    let (x_val, y_val) =
        build_dataset(&cfg.parcellated_ts_dir, val_ids, labels, spec).context("validation set")?;
    run_knn_on_splits(
        tag,
        knn_cfg,
        (x_train, y_train),
        (x_test, y_test),
        (x_val, y_val),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_fc_experiment(
    cfg: &AppConfig,
    train_ids: &[String],
    test_ids: &[String],
    val_ids: &[String],
    labels: &HashMap<String, Label>,
    spec: &FcSpec,
    knn_cfg: &KnnConfig,
    tag: &str,
) -> Result<()> {
    let (x_train, y_train) = build_fc_dataset(&cfg.parcellated_ts_dir, train_ids, labels, spec)
        .context("FC training set")?;
    let (x_test, y_test) =
        build_fc_dataset(&cfg.parcellated_ts_dir, test_ids, labels, spec).context("FC test set")?;
    let (x_val, y_val) = build_fc_dataset(&cfg.parcellated_ts_dir, val_ids, labels, spec)
        .context("FC validation set")?;
    run_knn_on_splits(
        tag,
        knn_cfg,
        (x_train, y_train),
        (x_test, y_test),
        (x_val, y_val),
    )
}

fn run_knn_on_splits(
    tag: &str,
    knn_cfg: &KnnConfig,
    train: (Vec<Vec<f32>>, Vec<i32>),
    test: (Vec<Vec<f32>>, Vec<i32>),
    val: (Vec<Vec<f32>>, Vec<i32>),
) -> Result<()> {
    let t = Instant::now();
    let (x_train, y_train) = train;
    let (x_test, y_test) = test;
    let (x_val, y_val) = val;

    if x_train.is_empty() {
        warn!(tag = tag, "training set empty, skipping");
        return Ok(());
    }
    let feat_dim = x_train[0].len();

    let mut knn = KNN::new(knn_cfg.clone());
    knn.fit(x_train, y_train)?;

    let y_test_pred = knn.predict_batch(&x_test)?;
    let y_val_pred = knn.predict_batch(&x_val)?;

    let test_acc = accuracy(&y_test, &y_test_pred);
    let val_acc = accuracy(&y_val, &y_val_pred);
    let test_cm = confusion_matrix_binary(&y_test, &y_test_pred);
    let val_cm = confusion_matrix_binary(&y_val, &y_val_pred);

    info!(
        tag = tag,
        feat_dim = feat_dim,
        n_train = knn.num_training_samples(),
        n_test = x_test.len(),
        n_val = x_val.len(),
        test_acc = test_acc,
        val_acc = val_acc,
        test_cm_tn_fp_fn_tp = ?[test_cm[0][0], test_cm[0][1], test_cm[1][0], test_cm[1][1]],
        val_cm_tn_fp_fn_tp = ?[val_cm[0][0], val_cm[0][1], val_cm[1][0], val_cm[1][1]],
        duration_ms = t.elapsed().as_millis(),
        "KNN experiment done"
    );

    Ok(())
}
