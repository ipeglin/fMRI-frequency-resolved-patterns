mod feature_extractor;
mod models;
pub mod preprocessing;
pub mod strategies;

use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

use anyhow::{Context, Result};
use tch::Tensor;
use tracing::{info, warn};
use utils::bids_filename::{BidsFilename, sort_bids_vec};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::{atlas::BrainAtlas, bids_filename::filter_directory_bids_files};

pub use feature_extractor::FeatureExtractor;
pub use strategies::{AnalysisCtx, FeatureSrc};

/// Run the DenseNet feature extraction pipeline over every subject HDF5 file
/// found in `cfg.consolidated_data_dir`. Per file, the BIDS task name selects
/// which analysis set runs (see `strategies::run_for_file`). All outputs are
/// written back into the same file under `features/<src>/<analysis>/...`.
pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        consolidated_data_dir = %cfg.consolidated_data_dir.display(),
        force = cfg.force,
        roi_selection_name = %cfg.roi_selection.name,
        image_fit = ?cfg.feature_extraction.image_fit,
        hht_log_amp = cfg.feature_extraction.hht_log_amp,
        "starting CNN feature extraction pipeline"
    );

    let weights_path = cfg.feature_extraction.cnn_weights_path.as_deref();
    let extractor = FeatureExtractor::new(weights_path, 1)?;
    match weights_path {
        Some(p) => {
            info!(weights = %p.display(), "DenseNet-201 initialised with pretrained weights")
        }
        None => info!("DenseNet-201 initialised with random weights"),
    }

    if cfg.roi_selection.is_empty() {
        anyhow::bail!(
            "no [roi_selection] configured — feature extraction requires a non-empty \
             cortical_regions and/or subcortical_regions list. Set [roi_selection] \
             in config.toml (e.g. name = \"vpfc_mpfc_amy\", \
             cortical_regions = [\"PFCv\", \"PFCm\"], subcortical_regions = [\"AMY\"])."
        );
    }

    let brain_atlas =
        BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let selected = brain_atlas.selected_rois(&cfg.roi_selection);
    if selected.is_empty() {
        anyhow::bail!(
            "[roi_selection] name=\"{}\" matched zero ROIs in atlas — check LUT paths \
             ({}, {}) and region names",
            cfg.roi_selection.name,
            cfg.cortical_atlas_lut.display(),
            cfg.subcortical_atlas_lut.display()
        );
    }
    let roi_indices: Vec<i64> = selected.iter().map(|r| r.row_index as i64).collect();
    let roi_labels: Vec<String> = selected.iter().map(|r| r.label.clone()).collect();
    let roi_matched_regions: Vec<String> =
        selected.iter().map(|r| r.matched_region.clone()).collect();
    let roi_selection_name = cfg.roi_selection.name.clone();
    let roi_selection_fingerprint = cfg.roi_selection.fingerprint();
    let roi_index_tensor = Tensor::from_slice(&roi_indices);
    let labels_joined = roi_labels.join(",");
    let matched_joined = roi_matched_regions.join(",");
    info!(
        n_target_rois = roi_indices.len(),
        rois = ?roi_labels,
        matched_regions = ?roi_matched_regions,
        roi_selection_name = %roi_selection_name,
        roi_selection_fingerprint = %roi_selection_fingerprint,
        "Using ROI-stratified configuration"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.consolidated_data_dir)?
        .filter_map(|e| e.ok())
        .filter_map(|e| {
            let path = e.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            let formatted = BidsSubjectId::parse(id).to_dir_name();
            Some((formatted, path))
        })
        .collect();

    let total_subjects = subjects.len();
    info!(num_subjects = total_subjects, "found subject directories");

    let mut subject_idx = 0;
    let mut error_count = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;
        let _subject_span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();

        let mut files: Vec<BidsFilename> = filter_directory_bids_files(dir, |bids| {
            let task = bids.get("task");
            task == Some("restAP") || task == Some("hammerAP")
        })
        .expect("Failed to read the directory");

        sort_bids_vec(&mut files, &["task", "run"], |key, a, b| {
            match key {
                "task" => {
                    // Define specific order for tasks
                    let priority = |v| match v {
                        "restAP" => 0,
                        "hammerAP" => 1,
                        _ => 2,
                    };
                    priority(a).cmp(&priority(b))
                }
                _ => {
                    // Default to standard lexicographical sort (works for padded "run" numbers)
                    a.cmp(b)
                }
            }
        });

        info!(num_files = files.len(), "processing subject");

        for file in &files {
            let path = file
                .try_to_path_buf()
                .context("BidsFilename has no path associated with it")?;
            let _file_span = tracing::info_span!("file", file = %path.display()).entered();

            let result: Result<()> = (|| {
                let bids = BidsFilename::parse(match path.file_name().and_then(|n| n.to_str()) {
                    Some(name) => name,
                    None => return Ok(()),
                });
                let task_name = bids.get("task").unwrap_or("unknown");

                let h5_file = hdf5::File::open_rw(&path)?;
                let ctx = AnalysisCtx {
                    extractor: &extractor,
                    fit: cfg.feature_extraction.image_fit,
                    hht_log_amp: cfg.feature_extraction.hht_log_amp,
                    roi_indices: &roi_indices,
                    roi_index_tensor: &roi_index_tensor,
                    roi_labels_joined: &labels_joined,
                    roi_matched_regions_joined: &matched_joined,
                    roi_selection_name: &roi_selection_name,
                    roi_selection_fingerprint: &roi_selection_fingerprint,
                    force: cfg.force,
                    subject_id: formatted_id,
                    task_name,
                };
                strategies::run_for_file(&ctx, &h5_file)
            })();

            if let Err(e) = result {
                error_count += 1;
                warn!(file = %path.display(), error = %e, "skipping file due to error");
            }
        }
    }

    if error_count > 0 {
        warn!(error_count, "some files were skipped due to errors");
    }
    info!(
        error_count,
        total_duration_ms = run_start.elapsed().as_millis() as u64,
        "feature extraction pipeline complete"
    );
    Ok(())
}
