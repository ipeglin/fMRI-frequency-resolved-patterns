mod feature_extractor;
mod models;

use std::{collections::BTreeMap, fs, path::PathBuf, time::Instant};

use anyhow::Result;
use atlas::BrainAtlas;
use config::{FeatureExtractionConfig, bids_filename::BidsFilename, bids_subject_id::BidsSubjectId};
use hdf5_io::{H5Attr, open_or_create_group, write_attrs, write_dataset};
use tch::{Kind, Tensor};
use tracing::{debug, info, warn};

pub use feature_extractor::FeatureExtractor;

// ---------------------------------------------------------------------------
// Pipeline entry point
// ---------------------------------------------------------------------------

pub fn run(cfg: &FeatureExtractionConfig) -> Result<()> {
    let run_start = Instant::now();

    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        bold_ts_dir = %cfg.bold_ts_dir.display(),
        force = cfg.force,
        "starting CNN feature extraction pipeline"
    );

    // num_classes=1 is a placeholder — classifier head unused for feature extraction.
    let weights_path = cfg.cnn_weights_path.as_deref();
    let extractor = FeatureExtractor::new(weights_path, 1)?;
    match weights_path {
        Some(p) => info!(weights = %p.display(), "DenseNet-201 initialised with pretrained weights"),
        None => info!("DenseNet-201 initialised with random weights"),
    }

    let brain_atlas =
        BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let roi_pairs = brain_atlas.vpfc_mpfc_amy_ids();
    let roi_indices: Vec<i64> = roi_pairs.iter().map(|(i, _)| *i as i64).collect();
    let roi_labels: Vec<String> = roi_pairs.iter().map(|(_, l)| l.clone()).collect();
    if roi_indices.is_empty() {
        anyhow::bail!(
            "no PFCv/PFCm/AMY ROIs matched in atlas — check LUT paths ({}, {})",
            cfg.cortical_atlas_lut.display(),
            cfg.subcortical_atlas_lut.display()
        );
    }
    let roi_index_tensor = Tensor::from_slice(&roi_indices);
    info!(
        n_target_rois = roi_indices.len(),
        rois = ?roi_labels,
        "selected target ROIs for feature extraction (vPFC + mPFC + AMY)"
    );

    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.bold_ts_dir)?
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

        let available_timeseries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .filter_map(|p| {
                if p.extension().and_then(|e| e.to_str()) == Some("h5") {
                    Some(p)
                } else {
                    None
                }
            })
            .collect();

        info!(
            subject = formatted_id,
            subject_idx = subject_idx,
            total_subjects = total_subjects,
            num_files = available_timeseries.len(),
            "processing subject"
        );

        for file_path in &available_timeseries {
            let file_result: anyhow::Result<()> = (|| {
                let bids =
                    BidsFilename::parse(match file_path.file_name().and_then(|n| n.to_str()) {
                        Some(name) => name,
                        None => return Ok(()),
                    });
                let task_name = bids.get("task").unwrap_or("unknown");

                // Open read-write: features are written back into the same file
                // under a new `features/` tree alongside cwt/, hht/, etc.
                let h5_file = hdf5_io::open_or_create(file_path)?;
                let features_root = open_or_create_group(&h5_file, "features", false)?;
                let labels_joined = roi_labels.join(",");

                ////////////////////////////////////////////////
                // CWT scalogram feature extraction          //
                // Source: cwt_standardized > cwt            //
                // Whole-band: <root>/whole-band/scalogram   //
                // Blocks:     <root>/blocks/block_N/scalogram
                ////////////////////////////////////////////////

                let cwt_root = h5_file
                    .group("cwt_standardized")
                    .or_else(|_| h5_file.group("cwt"));

                match cwt_root {
                    Err(_) => {
                        debug!(
                            subject = formatted_id,
                            task_name = task_name,
                            "no CWT group found, skipping (run cwt first)"
                        );
                    }
                    Ok(cwt_group) => {
                        let cwt_features_parent =
                            open_or_create_group(&features_root, "cwt", false)?;

                        // Whole-band
                        if let Ok(wb_group) = cwt_group.group("whole-band") {
                            process_cwt_subgroup(
                                &extractor,
                                &wb_group,
                                &cwt_features_parent,
                                "whole-band",
                                &roi_index_tensor,
                                &roi_indices,
                                &labels_joined,
                                cfg.force,
                                formatted_id,
                                task_name,
                            )?;
                        }

                        // Per-block
                        if let Ok(blocks_group) = cwt_group.group("blocks") {
                            let block_names: Vec<String> = blocks_group
                                .member_names()?
                                .into_iter()
                                .filter(|n| n.starts_with("block_"))
                                .collect();

                            let features_blocks_parent =
                                open_or_create_group(&cwt_features_parent, "blocks", false)?;

                            for block_name in &block_names {
                                let block_group = blocks_group.group(block_name)?;
                                process_cwt_subgroup(
                                    &extractor,
                                    &block_group,
                                    &features_blocks_parent,
                                    block_name,
                                    &roi_index_tensor,
                                    &roi_indices,
                                    &labels_joined,
                                    cfg.force,
                                    formatted_id,
                                    task_name,
                                )?;
                            }
                        }
                    }
                }

                //////////////////////////////////////////////////////////
                // HHT spectrogram feature extraction                   //
                // Source: hht                                          //
                // Whole-band: hht/whole-band/full_spectrum             //
                // Blocks:     hht/blocks/block_N/full_spectrum         //
                //////////////////////////////////////////////////////////

                match h5_file.group("hht") {
                    Err(_) => {
                        debug!(
                            subject = formatted_id,
                            task_name = task_name,
                            "no HHT group found, skipping (run hilbert first)"
                        );
                    }
                    Ok(hht_group) => {
                        let hht_features_parent =
                            open_or_create_group(&features_root, "hht", false)?;

                        if let Ok(wb_group) = hht_group.group("whole-band") {
                            process_hht_subgroup(
                                &extractor,
                                &wb_group,
                                &hht_features_parent,
                                "whole-band",
                                &roi_index_tensor,
                                &roi_indices,
                                &labels_joined,
                                cfg.force,
                                formatted_id,
                                task_name,
                            )?;
                        }

                        if let Ok(blocks_group) = hht_group.group("blocks") {
                            let block_names: Vec<String> = blocks_group
                                .member_names()?
                                .into_iter()
                                .filter(|n| n.starts_with("block_"))
                                .collect();

                            let features_blocks_parent =
                                open_or_create_group(&hht_features_parent, "blocks", false)?;

                            for block_name in &block_names {
                                let block_group = blocks_group.group(block_name)?;
                                process_hht_subgroup(
                                    &extractor,
                                    &block_group,
                                    &features_blocks_parent,
                                    block_name,
                                    &roi_index_tensor,
                                    &roi_indices,
                                    &labels_joined,
                                    cfg.force,
                                    formatted_id,
                                    task_name,
                                )?;
                            }
                        }
                    }
                }

                Ok(())
            })();

            if let Err(e) = file_result {
                error_count += 1;
                warn!(
                    subject = formatted_id,
                    file = %file_path.display(),
                    error = %e,
                    "skipping file due to error"
                );
            }
        }
    }

    if error_count > 0 {
        warn!(
            error_count = error_count,
            "some subjects/files were skipped due to errors"
        );
    }

    let total_duration_ms = run_start.elapsed().as_millis();
    info!(
        total_subjects = total_subjects,
        error_count = error_count,
        total_duration_ms = total_duration_ms,
        "feature extraction pipeline complete"
    );

    Ok(())
}

// ---------------------------------------------------------------------------
// Per-subgroup feature extraction (whole-band or single block)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn process_cwt_subgroup(
    extractor: &FeatureExtractor,
    source_group: &hdf5::Group,
    features_parent: &hdf5::Group,
    out_name: &str,
    roi_index_tensor: &Tensor,
    roi_indices: &[i64],
    labels_joined: &str,
    force: bool,
    subject: &str,
    task_name: &str,
) -> Result<()> {
    let ds = source_group.dataset("scalogram")?;
    let shape = ds.shape();
    let [n_rois, n_scales, n_timepoints] = match shape.as_slice() {
        &[a, b, c] => [a, b, c],
        _ => anyhow::bail!("unexpected scalogram shape {:?}", shape),
    };

    let max_idx = *roi_indices.iter().max().unwrap_or(&0);
    if (max_idx as usize) >= n_rois {
        anyhow::bail!(
            "target ROI index {} out of range for scalogram with {} rois",
            max_idx,
            n_rois
        );
    }

    if features_parent.group(out_name).is_ok() && !force {
        info!(
            subject = subject,
            task_name = task_name,
            subgroup = out_name,
            "CWT features already present, skipping (use --force to recompute)"
        );
        return Ok(());
    }

    let data_f64: Vec<f64> = ds.read_raw()?;
    let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();

    let scalogram_t = Tensor::from_slice(&data_f32)
        .reshape([n_rois as i64, n_scales as i64, n_timepoints as i64]);
    let scalogram_t = scalogram_t.index_select(0, roi_index_tensor);
    let n_rois_sel = roi_indices.len();

    info!(
        subject = subject,
        task_name = task_name,
        subgroup = out_name,
        n_rois = n_rois_sel,
        n_scales = n_scales,
        n_timepoints = n_timepoints,
        "extracting CWT scalogram features"
    );

    let extract_start = Instant::now();
    let (per_roi, mean) = extract_features_with_mean(extractor, &scalogram_t);
    let extract_duration_ms = extract_start.elapsed().as_millis();
    let feature_dim = mean.size1()?;

    info!(
        subject = subject,
        task_name = task_name,
        subgroup = out_name,
        feature_dim = feature_dim,
        extract_duration_ms = extract_duration_ms,
        "CWT scalogram features extracted"
    );

    write_feature_group(features_parent, out_name, &per_roi, &mean, labels_joined, force)?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn process_hht_subgroup(
    extractor: &FeatureExtractor,
    source_group: &hdf5::Group,
    features_parent: &hdf5::Group,
    out_name: &str,
    roi_index_tensor: &Tensor,
    roi_indices: &[i64],
    labels_joined: &str,
    force: bool,
    subject: &str,
    task_name: &str,
) -> Result<()> {
    let ds = source_group.dataset("full_spectrum")?;
    let shape = ds.shape();
    let [n_channels, n_freq_bins] = match shape.as_slice() {
        &[a, b] => [a, b],
        _ => anyhow::bail!("unexpected full_spectrum shape {:?}", shape),
    };

    let max_idx = *roi_indices.iter().max().unwrap_or(&0);
    if (max_idx as usize) >= n_channels {
        anyhow::bail!(
            "target ROI index {} out of range for HHT spectrum with {} channels",
            max_idx,
            n_channels
        );
    }

    if features_parent.group(out_name).is_ok() && !force {
        info!(
            subject = subject,
            task_name = task_name,
            subgroup = out_name,
            "HHT features already present, skipping (use --force to recompute)"
        );
        return Ok(());
    }

    let data_f64: Vec<f64> = ds.read_raw()?;
    let data_f32: Vec<f32> = data_f64.iter().map(|&v| v as f32).collect();

    let spectrum_t = Tensor::from_slice(&data_f32)
        .reshape([n_channels as i64, 1, n_freq_bins as i64]);
    let spectrum_t = spectrum_t.index_select(0, roi_index_tensor);
    let n_channels_sel = roi_indices.len();

    info!(
        subject = subject,
        task_name = task_name,
        subgroup = out_name,
        n_channels = n_channels_sel,
        n_freq_bins = n_freq_bins,
        "extracting HHT spectrogram features"
    );

    let extract_start = Instant::now();
    let (per_roi, mean) = extract_features_with_mean(extractor, &spectrum_t);
    let extract_duration_ms = extract_start.elapsed().as_millis();
    let feature_dim = mean.size1()?;

    info!(
        subject = subject,
        task_name = task_name,
        subgroup = out_name,
        feature_dim = feature_dim,
        extract_duration_ms = extract_duration_ms,
        "HHT spectrogram features extracted"
    );

    write_feature_group(features_parent, out_name, &per_roi, &mean, labels_joined, force)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Input preprocessing utilities
// ---------------------------------------------------------------------------

/// Convert a single ROI's CWT scalogram to a DenseNet-compatible image tensor.
///
/// Input:  `[n_scales, n_timepoints]` float tensor (power values, any range)
/// Output: `[1, 3, 224, 224]` float tensor, normalised to [0, 1], grayscale
///         replicated across all three channels.
///
/// The scalogram is min-max normalised per-image so that scale differences
/// between ROIs do not affect the spatial patterns the network learns.
pub fn scalogram_to_image(scalogram: &Tensor) -> Tensor {
    let min = scalogram.min();
    let max = scalogram.max();
    let normalised = (scalogram - &min) / (&max - &min + 1e-8);

    // [n_scales, n_timepoints] -> [1, 1, n_scales, n_timepoints]
    let img = normalised.unsqueeze(0).unsqueeze(0);

    // Resize to 224×224 using bilinear interpolation
    let img = img.upsample_bilinear2d(&[224, 224], false, None, None);

    // Replicate grayscale channel to RGB: [1, 1, 224, 224] -> [1, 3, 224, 224]
    img.expand(&[1, 3, 224, 224], false)
}

/// Convert a full trial's CWT scalogram to a batch of per-ROI image tensors.
///
/// Input:  `[n_rois, n_scales, n_timepoints]` float tensor
/// Output: `[n_rois, 3, 224, 224]` float tensor
///
/// Each ROI is normalised independently before resizing.
pub fn trial_scalogram_to_batch(scalogram: &Tensor) -> Tensor {
    let n_rois = scalogram.size()[0];

    let images: Vec<Tensor> = (0..n_rois)
        .map(|i| {
            let roi = scalogram.select(0, i); // [n_scales, n_timepoints]
            scalogram_to_image(&roi) // [1, 3, 224, 224]
        })
        .collect();

    Tensor::cat(&images, 0) // [n_rois, 3, 224, 224]
}

/// Extract trial-level features from a scalogram/spectrum by averaging over channels.
///
/// Input:  `[n_rois, n_scales, n_timepoints]` float tensor
/// Output: `[1920]` float tensor — mean DenseNet feature vector across ROIs/channels
///
/// Primary entry point for the KNN/SVM pipeline: call once per trial, collect
/// the resulting vectors into a matrix, then fit the classifier.
pub fn extract_trial_features(extractor: &FeatureExtractor, scalogram: &Tensor) -> Tensor {
    let batch = trial_scalogram_to_batch(scalogram); // [n_rois, 3, 224, 224]
    let features = extractor.extract_features(&batch); // [n_rois, 1920]
    features.mean_dim(Some([0i64].as_slice()), false, Kind::Float) // [1920]
}

/// Extract per-ROI features and the across-ROI mean in one pass.
///
/// Returns `(per_roi [n_rois, 1920], mean [1920])`, both f32 on CPU.
pub fn extract_features_with_mean(
    extractor: &FeatureExtractor,
    scalogram: &Tensor,
) -> (Tensor, Tensor) {
    let batch = trial_scalogram_to_batch(scalogram);
    let per_roi = extractor
        .extract_features(&batch)
        .to_kind(Kind::Float)
        .to_device(tch::Device::Cpu);
    let mean = per_roi.mean_dim(Some([0i64].as_slice()), false, Kind::Float);
    (per_roi, mean)
}

fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let flat = t.to_kind(Kind::Float).to_device(tch::Device::Cpu).contiguous();
    let numel = flat.numel();
    let mut out = vec![0f32; numel];
    flat.copy_data(&mut out, numel);
    out
}

/// Write a feature subgroup with per-ROI and mean datasets plus ROI-label metadata.
///
/// Layout:
///   <parent>/<name>/per_roi  [n_rois, feat_dim]
///   <parent>/<name>/mean     [feat_dim]
///   <parent>/<name>@labels   ROI IDs used for per_roi rows (comma-separated)
fn write_feature_group(
    parent: &hdf5::Group,
    name: &str,
    per_roi: &Tensor,
    mean: &Tensor,
    labels_joined: &str,
    force: bool,
) -> Result<()> {
    let group = open_or_create_group(parent, name, force)?;

    let per_roi_shape = per_roi.size();
    let (n_rois, feat_dim) = match per_roi_shape.as_slice() {
        &[r, d] => (r as usize, d as usize),
        _ => anyhow::bail!("unexpected per_roi feature shape {:?}", per_roi_shape),
    };

    let per_roi_buf = tensor_to_vec_f32(per_roi);
    let mean_buf = tensor_to_vec_f32(mean);

    write_dataset(&group, "per_roi", &per_roi_buf, &[n_rois, feat_dim], None)?;
    write_dataset(&group, "mean", &mean_buf, &[feat_dim], None)?;

    write_attrs(
        &group,
        &[
            H5Attr::string("labels", labels_joined),
            H5Attr::u32("n_rois", n_rois as u32),
            H5Attr::u32("feature_dim", feat_dim as u32),
        ],
    )?;

    Ok(())
}
