//! Five analysis strategies that turn upstream CWT/HHT spectra into DenseNet
//! feature vectors.
//!
//! restAP (full-run scalogram/spectrogram, time T_full):
//!   A) `baseline_chunked`  — split T_full into `CHUNK_COUNT` equal time chunks,
//!                            one DenseNet input per chunk per ROI.
//!   B) `baseline_averaged` — same chunks, then mean across chunks → one image.
//!
//! hammerAP (per face-block scalogram/spectrogram):
//!   C) `task_concat`       — shuffle face blocks (deterministic seed) and
//!                            concatenate along time → one image per ROI.
//!   D) `task_per_block`    — one DenseNet input per face block per ROI.
//!   E) `task_averaged`     — trim each block to `TASK_COMMON_BLOCK_W`, mean
//!                            across blocks → one image per ROI.
//!
//! All strategies share the same per-image preprocessing (`spectrum_to_image`)
//! and write outputs under `features/<src>/<analysis>/...` in the same HDF5
//! file. Existing groups are left in place unless `force` is set.

use anyhow::{Context, Result};
use hdf5::types::TypeDescriptor;
use std::time::Instant;
use tch::{Kind, Tensor};
use tracing::{debug, info};
use utils::config::ImageFitMode;
use utils::hdf5_io::{H5Attr, open_or_create_group, write_attrs, write_dataset};
use utils::roi_migration::check_roi_fingerprint;

use crate::FeatureExtractor;
use crate::preprocessing::{
    batch_spectrum_to_input, chunk_along_time, resize_and_mean_blocks, shuffled_concat,
    stack_and_mean, trim_and_mean_blocks,
};

pub const CHUNK_COUNT: i64 = 3;
pub const TASK_COMMON_BLOCK_W: i64 = 23;
pub const SHUFFLE_SEED: u64 = 42;

/// Which upstream spectrum source feeds the analysis.
#[derive(Debug, Clone, Copy)]
pub enum FeatureSrc {
    Cwt,
    Hht,
}

impl FeatureSrc {
    pub fn group_name(self) -> &'static str {
        match self {
            FeatureSrc::Cwt => "cwt",
            FeatureSrc::Hht => "hht",
        }
    }
}

/// Per-subject-per-file context passed into every analysis runner.
pub struct AnalysisCtx<'a> {
    pub extractor: &'a FeatureExtractor,
    pub fit: ImageFitMode,
    pub hht_log_amp: bool,
    /// Concat-row atlas indices for the configured `[roi_selection]`.
    pub roi_indices: &'a [i64],
    pub roi_index_tensor: &'a Tensor,
    pub roi_labels_joined: &'a str,
    /// Comma-joined `matched_region` per ROI, in the same order as `roi_indices`.
    pub roi_matched_regions_joined: &'a str,
    /// Human-readable identifier for the ROI selection (e.g. `"vpfc_mpfc_amy"`).
    pub roi_selection_name: &'a str,
    /// Stable fingerprint for the ROI selection. Compared against the same
    /// attr on upstream `_roi` HHT/MVMD groups; mismatch bails the file.
    pub roi_selection_fingerprint: &'a str,
    pub force: bool,
    pub subject_id: &'a str,
    pub task_name: &'a str,
}

impl AnalysisCtx<'_> {
    fn log_amp_for(&self, src: FeatureSrc) -> bool {
        matches!(src, FeatureSrc::Hht) && self.hht_log_amp
    }
}

// ---------------------------------------------------------------------------
// HDF5 readers
// ---------------------------------------------------------------------------

/// Read a 3D dataset as f32, regardless of whether it's stored as f32 or f64.
fn read_3d_as_f32(ds: &hdf5::Dataset) -> Result<(Tensor, [i64; 3])> {
    let shape = ds.shape();
    let [a, b, c] = match shape.as_slice() {
        &[a, b, c] => [a as i64, b as i64, c as i64],
        _ => anyhow::bail!("expected 3D dataset, got shape {:?}", shape),
    };
    let dtype = ds.dtype()?.to_descriptor()?;
    let buf: Vec<f32> = match dtype {
        TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
            let raw: Vec<f64> = ds.read_raw()?;
            raw.into_iter().map(|v| v as f32).collect()
        }
        TypeDescriptor::Float(hdf5::types::FloatSize::U4) => ds.read_raw()?,
        other => anyhow::bail!("unsupported dataset dtype {:?}", other),
    };
    let t = Tensor::from_slice(&buf).reshape([a, b, c]);
    Ok((t, [a, b, c]))
}

/// CWT restAP whole-run: `/03cwt/full_run_std` `[n_rois_all, 224, T_full]`.
/// Returns ROI-selected tensor `[n_target, 224, T_full]`.
fn load_cwt_full_run(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Option<Tensor>> {
    let cwt_root = match h5.group("03cwt") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let ds = match cwt_root.dataset("full_run_std") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (full, [n_all, _, _]) = read_3d_as_f32(&ds)?;
    validate_roi_range(ctx.roi_indices, n_all, "cwt full_run_std")?;
    Ok(Some(full.index_select(0, ctx.roi_index_tensor)))
}

/// CWT hammerAP face blocks: `/03cwt/blocks_std/<block_name>` per block.
/// Returns ordered list of (name, ROI-selected tensor `[n_target, 224, T_block]`).
fn load_cwt_blocks(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Vec<(String, Tensor)>> {
    let cwt_root = match h5.group("03cwt") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let blocks = match cwt_root.group("blocks_std") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let mut names: Vec<String> = blocks
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("block_"))
        .collect();
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let ds = blocks.dataset(&name)?;
        let (block, [n_all, _, _]) = read_3d_as_f32(&ds)?;
        validate_roi_range(ctx.roi_indices, n_all, "cwt blocks_std")?;
        out.push((name, block.index_select(0, ctx.roi_index_tensor)));
    }
    Ok(out)
}

/// HHT restAP whole-run: `/05hht/full_run_raw_roi/hilbert_spectrum`
/// `[n_target, 224, T_full]` (already ROI-selected upstream by 04mvmd / 05hilbert).
/// Bails on `roi_selection_fingerprint` mismatch — the upstream `_roi` group must
/// match the current `[roi_selection]` config.
fn load_hht_full_run(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Option<Tensor>> {
    let hht_root = match h5.group("05hht") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let sub = match hht_root.group("full_run_raw_roi") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    check_roi_fingerprint(
        &sub,
        ctx.roi_selection_fingerprint,
        "/05hht/full_run_raw_roi",
    )?;
    let ds = match sub.dataset("hilbert_spectrum") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (t, [n_rows, _, _]) = read_3d_as_f32(&ds)?;
    if (n_rows as usize) != ctx.roi_indices.len() {
        anyhow::bail!(
            "hht full_run_raw_roi rows {} != target ROI count {} — atlas mismatch",
            n_rows,
            ctx.roi_indices.len()
        );
    }
    Ok(Some(t))
}

/// HHT hammerAP face blocks: `/05hht/blocks_raw_roi/<block>/hilbert_spectrum`
/// (already ROI-selected upstream). Bails on `roi_selection_fingerprint`
/// mismatch on each block group.
fn load_hht_blocks(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Vec<(String, Tensor)>> {
    let hht_root = match h5.group("05hht") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let blocks = match hht_root.group("blocks_raw_roi") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let mut names: Vec<String> = blocks
        .member_names()?
        .into_iter()
        .filter(|n| n.starts_with("block_"))
        .collect();
    names.sort();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let g = blocks.group(&name)?;
        check_roi_fingerprint(
            &g,
            ctx.roi_selection_fingerprint,
            &format!("/05hht/blocks_raw_roi/{name}"),
        )?;
        let ds = match g.dataset("hilbert_spectrum") {
            Ok(d) => d,
            Err(_) => continue,
        };
        let (t, [n_rows, _, _]) = read_3d_as_f32(&ds)?;
        if (n_rows as usize) != ctx.roi_indices.len() {
            anyhow::bail!(
                "hht blocks_raw_roi/{} rows {} != target ROI count {}",
                name,
                n_rows,
                ctx.roi_indices.len()
            );
        }
        out.push((name, t));
    }
    Ok(out)
}

fn validate_roi_range(roi_indices: &[i64], n_available: i64, what: &str) -> Result<()> {
    if let Some(&max_idx) = roi_indices.iter().max() {
        if max_idx >= n_available {
            anyhow::bail!(
                "target ROI index {} out of range for {} (n_available={})",
                max_idx,
                what,
                n_available
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Feature extraction + write
// ---------------------------------------------------------------------------

/// Run a `[n_rois, F, T]` spectrum through preprocessing + DenseNet.
/// Returns `(per_roi [n_rois, 1920], mean [1920])` on CPU as f32.
fn extract(ctx: &AnalysisCtx, src: FeatureSrc, spec: &Tensor) -> (Tensor, Tensor) {
    let log_amp = ctx.log_amp_for(src);
    let batch = batch_spectrum_to_input(spec, log_amp, ctx.fit);
    let per_roi = ctx
        .extractor
        .extract_features(&batch)
        .to_kind(Kind::Float)
        .to_device(tch::Device::Cpu);
    let mean = per_roi.mean_dim(Some([0i64].as_slice()), false, Kind::Float);
    (per_roi, mean)
}

/// Variant of `extract` that forces an explicit `ImageFitMode`, ignoring
/// `ctx.fit`. Used only by the resize-baseline strategy so existing analyses
/// keep using the configured (pad) fit unchanged.
fn extract_with_fit(
    ctx: &AnalysisCtx,
    src: FeatureSrc,
    spec: &Tensor,
    fit: ImageFitMode,
) -> (Tensor, Tensor) {
    let log_amp = ctx.log_amp_for(src);
    let batch = batch_spectrum_to_input(spec, log_amp, fit);
    let per_roi = ctx
        .extractor
        .extract_features(&batch)
        .to_kind(Kind::Float)
        .to_device(tch::Device::Cpu);
    let mean = per_roi.mean_dim(Some([0i64].as_slice()), false, Kind::Float);
    (per_roi, mean)
}

fn tensor_to_vec_f32(t: &Tensor) -> Vec<f32> {
    let flat = t
        .to_kind(Kind::Float)
        .to_device(tch::Device::Cpu)
        .contiguous();
    let n = flat.numel();
    let mut buf = vec![0f32; n];
    flat.copy_data(&mut buf, n);
    buf
}

fn write_features(
    parent: &hdf5::Group,
    leaf_name: &str,
    per_roi: &Tensor,
    mean: &Tensor,
    ctx: &AnalysisCtx,
    analysis: &str,
) -> Result<()> {
    let group = open_or_create_group(parent, leaf_name, ctx.force)?;
    let per_roi_shape = per_roi.size();
    let (n_rois, feat_dim) = match per_roi_shape.as_slice() {
        &[r, d] => (r as usize, d as usize),
        _ => anyhow::bail!("unexpected per_roi shape {:?}", per_roi_shape),
    };
    if ctx.roi_indices.len() != n_rois {
        anyhow::bail!(
            "roi_indices.len {} != per_roi rows {}",
            ctx.roi_indices.len(),
            n_rois
        );
    }
    let per_roi_buf = tensor_to_vec_f32(per_roi);
    let mean_buf = tensor_to_vec_f32(mean);
    let roi_idx_u32: Vec<u32> = ctx.roi_indices.iter().map(|&i| i as u32).collect();
    write_dataset(&group, "per_roi", &per_roi_buf, &[n_rois, feat_dim], None)?;
    write_dataset(&group, "mean", &mean_buf, &[feat_dim], None)?;
    write_dataset(&group, "roi_indices", &roi_idx_u32, &[n_rois], None)?;
    write_attrs(
        &group,
        &[
            H5Attr::string("labels", ctx.roi_labels_joined),
            H5Attr::u32("n_rois", n_rois as u32),
            H5Attr::u32("feature_dim", feat_dim as u32),
            H5Attr::string("subject_id", ctx.subject_id),
            H5Attr::string("task", ctx.task_name),
            H5Attr::string("analysis", analysis),
            H5Attr::string("roi_selection_name", ctx.roi_selection_name),
            H5Attr::string("roi_selection_fingerprint", ctx.roi_selection_fingerprint),
            H5Attr::string("roi_matched_regions", ctx.roi_matched_regions_joined),
            H5Attr::string("image_fit", image_fit_label(ctx.fit)),
        ],
    )?;
    Ok(())
}

fn image_fit_label(fit: ImageFitMode) -> &'static str {
    match fit {
        ImageFitMode::Pad => "pad",
        ImageFitMode::Resize => "resize",
    }
}

fn analysis_root<'a>(
    features_root: &'a hdf5::Group,
    src: FeatureSrc,
    analysis: &str,
    force: bool,
) -> Result<hdf5::Group> {
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    open_or_create_group(&src_g, analysis, force)
}

fn already_done(parent: &hdf5::Group, name: &str, force: bool) -> bool {
    !force && parent.group(name).is_ok()
}

// ---------------------------------------------------------------------------
// Strategy A — restAP, baseline chunked
// ---------------------------------------------------------------------------

/// Split full-run spectrum into `CHUNK_COUNT` equal time chunks; one DenseNet
/// input per chunk per ROI. Each chunk written under
/// `features/<src>/baseline_chunked/chunk_<i>`.
pub fn run_baseline_chunked(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    full_spec: &Tensor,
) -> Result<()> {
    let analysis = "baseline_chunked";
    let root = analysis_root(features_root, src, analysis, ctx.force)?;
    let chunks = chunk_along_time(full_spec, CHUNK_COUNT);
    let started = Instant::now();
    for (i, chunk) in chunks.iter().enumerate() {
        let name = format!("chunk_{i}");
        if already_done(&root, &name, ctx.force) {
            debug!(src = src.group_name(), %name, "baseline_chunked: leaf exists, skipping");
            continue;
        }
        let (per_roi, mean) = extract(ctx, src, chunk);
        write_features(&root, &name, &per_roi, &mean, ctx, analysis)?;
    }
    info!(
        src = src.group_name(),
        n_chunks = CHUNK_COUNT,
        ms = started.elapsed().as_millis() as u64,
        "baseline_chunked done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy B — restAP, baseline averaged
// ---------------------------------------------------------------------------

/// Split full-run spectrum into `CHUNK_COUNT` chunks then mean across chunks
/// per ROI → one DenseNet image per ROI. Written under
/// `features/<src>/baseline_averaged`.
pub fn run_baseline_averaged(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    full_spec: &Tensor,
) -> Result<()> {
    let analysis = "baseline_averaged";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(
            src = src.group_name(),
            "baseline_averaged: exists, skipping"
        );
        return Ok(());
    }
    let chunks = chunk_along_time(full_spec, CHUNK_COUNT);
    let avg = stack_and_mean(&chunks);
    let started = Instant::now();
    let (per_roi, mean) = extract(ctx, src, &avg);
    write_features(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        ms = started.elapsed().as_millis() as u64,
        "baseline_averaged done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy F — restAP, baseline image-resized (no chunking)
// ---------------------------------------------------------------------------

/// Use the full-run spectrum directly (no time-axis chunking). Each ROI image
/// is bicubicly resized from `[224, T_full]` → `[224, 224]` by forcing
/// `ImageFitMode::Resize`, regardless of `ctx.fit`. One DenseNet image per
/// ROI. Written under `features/<src>/baseline_resized`.
pub fn run_baseline_resized(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    full_spec: &Tensor,
) -> Result<()> {
    let analysis = "baseline_resized";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(src = src.group_name(), "baseline_resized: exists, skipping");
        return Ok(());
    }
    let started = Instant::now();
    let (per_roi, mean) = extract_with_fit(ctx, src, full_spec, ImageFitMode::Resize);
    write_features_resized(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        ms = started.elapsed().as_millis() as u64,
        "baseline_resized done"
    );
    Ok(())
}

/// Mirror of `write_features` but hard-codes the `image_fit` attribute to
/// `"resize"` so the on-disk metadata reflects the actual fit applied by
/// `run_baseline_resized`, irrespective of `ctx.fit`.
fn write_features_resized(
    parent: &hdf5::Group,
    leaf_name: &str,
    per_roi: &Tensor,
    mean: &Tensor,
    ctx: &AnalysisCtx,
    analysis: &str,
) -> Result<()> {
    let group = open_or_create_group(parent, leaf_name, ctx.force)?;
    let per_roi_shape = per_roi.size();
    let (n_rois, feat_dim) = match per_roi_shape.as_slice() {
        &[r, d] => (r as usize, d as usize),
        _ => anyhow::bail!("unexpected per_roi shape {:?}", per_roi_shape),
    };
    if ctx.roi_indices.len() != n_rois {
        anyhow::bail!(
            "roi_indices.len {} != per_roi rows {}",
            ctx.roi_indices.len(),
            n_rois
        );
    }
    let per_roi_buf = tensor_to_vec_f32(per_roi);
    let mean_buf = tensor_to_vec_f32(mean);
    let roi_idx_u32: Vec<u32> = ctx.roi_indices.iter().map(|&i| i as u32).collect();
    write_dataset(&group, "per_roi", &per_roi_buf, &[n_rois, feat_dim], None)?;
    write_dataset(&group, "mean", &mean_buf, &[feat_dim], None)?;
    write_dataset(&group, "roi_indices", &roi_idx_u32, &[n_rois], None)?;
    write_attrs(
        &group,
        &[
            H5Attr::string("labels", ctx.roi_labels_joined),
            H5Attr::u32("n_rois", n_rois as u32),
            H5Attr::u32("feature_dim", feat_dim as u32),
            H5Attr::string("subject_id", ctx.subject_id),
            H5Attr::string("task", ctx.task_name),
            H5Attr::string("analysis", analysis),
            H5Attr::string("roi_selection_name", ctx.roi_selection_name),
            H5Attr::string("roi_selection_fingerprint", ctx.roi_selection_fingerprint),
            H5Attr::string("roi_matched_regions", ctx.roi_matched_regions_joined),
            H5Attr::string("image_fit", image_fit_label(ImageFitMode::Resize)),
        ],
    )?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy C — hammerAP, task concat (shuffled)
// ---------------------------------------------------------------------------

/// Deterministically shuffle face blocks then concatenate along time → one
/// DenseNet image per ROI. Written under `features/<src>/task_concat`.
pub fn run_task_concat(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    blocks: &[(String, Tensor)],
) -> Result<()> {
    let analysis = "task_concat";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(src = src.group_name(), "task_concat: exists, skipping");
        return Ok(());
    }
    if blocks.is_empty() {
        debug!(src = src.group_name(), "task_concat: no blocks");
        return Ok(());
    }
    let owned: Vec<Tensor> = blocks.iter().map(|(_, t)| t.shallow_clone()).collect();
    let concat = shuffled_concat(owned, SHUFFLE_SEED);
    let started = Instant::now();
    let (per_roi, mean) = extract(ctx, src, &concat);
    write_features(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        n_blocks = blocks.len(),
        concat_t = concat.size()[2],
        ms = started.elapsed().as_millis() as u64,
        "task_concat done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy D — hammerAP, per-block
// ---------------------------------------------------------------------------

/// One DenseNet input per face block per ROI. Written under
/// `features/<src>/task_per_block/<block_name>`.
pub fn run_task_per_block(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    blocks: &[(String, Tensor)],
) -> Result<()> {
    let analysis = "task_per_block";
    let root = analysis_root(features_root, src, analysis, ctx.force)?;
    let started = Instant::now();
    for (name, block) in blocks {
        if already_done(&root, name, ctx.force) {
            debug!(src = src.group_name(), %name, "task_per_block: exists, skipping");
            continue;
        }
        let (per_roi, mean) = extract(ctx, src, block);
        write_features(&root, name, &per_roi, &mean, ctx, analysis)?;
    }
    info!(
        src = src.group_name(),
        n_blocks = blocks.len(),
        ms = started.elapsed().as_millis() as u64,
        "task_per_block done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy E — hammerAP, task averaged across blocks
// ---------------------------------------------------------------------------

/// Trim each block's time axis to `TASK_COMMON_BLOCK_W`, mean across blocks
/// per ROI → one DenseNet image per ROI. Written under
/// `features/<src>/task_averaged`.
pub fn run_task_averaged(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    blocks: &[(String, Tensor)],
) -> Result<()> {
    let analysis = "task_averaged";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(src = src.group_name(), "task_averaged: exists, skipping");
        return Ok(());
    }
    if blocks.is_empty() {
        debug!(src = src.group_name(), "task_averaged: no blocks");
        return Ok(());
    }
    let block_tensors: Vec<Tensor> = blocks.iter().map(|(_, t)| t.shallow_clone()).collect();
    let usable: usize = block_tensors
        .iter()
        .filter(|b| b.size()[2] >= TASK_COMMON_BLOCK_W)
        .count();
    if usable == 0 {
        debug!(
            src = src.group_name(),
            "task_averaged: no blocks meet width >= {}", TASK_COMMON_BLOCK_W
        );
        return Ok(());
    }
    let avg = trim_and_mean_blocks(&block_tensors, TASK_COMMON_BLOCK_W);
    let started = Instant::now();
    let (per_roi, mean) = extract(ctx, src, &avg);
    write_features(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        usable_blocks = usable,
        ms = started.elapsed().as_millis() as u64,
        "task_averaged done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy G — hammerAP, per face-block image-resized
// ---------------------------------------------------------------------------

/// One DenseNet input per face block per ROI, but each block image is
/// bicubicly resized from `[224, T_block]` → `[224, 224]` instead of being
/// zero-padded. Written under
/// `features/<src>/task_per_block_resized/<block_name>`.
pub fn run_task_per_block_resized(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    blocks: &[(String, Tensor)],
) -> Result<()> {
    let analysis = "task_per_block_resized";
    let root = analysis_root(features_root, src, analysis, ctx.force)?;
    let started = Instant::now();
    for (name, block) in blocks {
        if already_done(&root, name, ctx.force) {
            debug!(src = src.group_name(), %name, "task_per_block_resized: exists, skipping");
            continue;
        }
        let (per_roi, mean) = extract_with_fit(ctx, src, block, ImageFitMode::Resize);
        write_features_resized(&root, name, &per_roi, &mean, ctx, analysis)?;
    }
    info!(
        src = src.group_name(),
        n_blocks = blocks.len(),
        ms = started.elapsed().as_millis() as u64,
        "task_per_block_resized done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Strategy H — hammerAP, image-resized blocks averaged
// ---------------------------------------------------------------------------

/// Bicubicly resize each block's raw spectrum to `224×224`, mean across
/// blocks per ROI, then run the averaged image through DenseNet. Written
/// under `features/<src>/task_averaged_resized`.
pub fn run_task_averaged_resized(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    blocks: &[(String, Tensor)],
) -> Result<()> {
    let analysis = "task_averaged_resized";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(src = src.group_name(), "task_averaged_resized: exists, skipping");
        return Ok(());
    }
    if blocks.is_empty() {
        debug!(src = src.group_name(), "task_averaged_resized: no blocks");
        return Ok(());
    }
    let block_tensors: Vec<Tensor> = blocks.iter().map(|(_, t)| t.shallow_clone()).collect();
    let avg = resize_and_mean_blocks(&block_tensors, 224, 224);
    let started = Instant::now();
    let (per_roi, mean) = extract_with_fit(ctx, src, &avg, ImageFitMode::Resize);
    write_features_resized(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        n_blocks = blocks.len(),
        ms = started.elapsed().as_millis() as u64,
        "task_averaged_resized done"
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Per-task driver
// ---------------------------------------------------------------------------

/// Dispatch the appropriate strategy set for a single subject file based on the
/// task name parsed from its BIDS filename.
///
/// `restAP` → A + B for both CWT and HHT (when available).
/// `hammerAP` → C + D + E for both CWT and HHT (when available).
pub fn run_for_file(ctx: &AnalysisCtx, h5: &hdf5::File) -> Result<()> {
    let features_root = open_or_create_group(h5, "07feature_extraction", false)
        .context("failed to open features/ root group")?;

    match ctx.task_name {
        "restAP" => {
            if let Some(spec) = load_cwt_full_run(h5, ctx)? {
                run_baseline_chunked(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
                run_baseline_averaged(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
                run_baseline_resized(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
            } else {
                debug!("restAP: no CWT full_run_std, skipping CWT analyses");
            }
            if let Some(spec) = load_hht_full_run(h5, ctx)? {
                run_baseline_chunked(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                run_baseline_averaged(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                run_baseline_resized(ctx, &features_root, FeatureSrc::Hht, &spec)?;
            } else {
                debug!("restAP: no HHT full_run, skipping HHT analyses");
            }
        }
        "hammerAP" => {
            let cwt_blocks = load_cwt_blocks(h5, ctx)?;
            if !cwt_blocks.is_empty() {
                run_task_concat(ctx, &features_root, FeatureSrc::Cwt, &cwt_blocks)?;
                run_task_per_block(ctx, &features_root, FeatureSrc::Cwt, &cwt_blocks)?;
                run_task_averaged(ctx, &features_root, FeatureSrc::Cwt, &cwt_blocks)?;
                run_task_per_block_resized(ctx, &features_root, FeatureSrc::Cwt, &cwt_blocks)?;
                run_task_averaged_resized(ctx, &features_root, FeatureSrc::Cwt, &cwt_blocks)?;
            } else {
                debug!("hammerAP: no CWT blocks_std, skipping CWT analyses");
            }
            let hht_blocks = load_hht_blocks(h5, ctx)?;
            if !hht_blocks.is_empty() {
                run_task_concat(ctx, &features_root, FeatureSrc::Hht, &hht_blocks)?;
                run_task_per_block(ctx, &features_root, FeatureSrc::Hht, &hht_blocks)?;
                run_task_averaged(ctx, &features_root, FeatureSrc::Hht, &hht_blocks)?;
                run_task_per_block_resized(ctx, &features_root, FeatureSrc::Hht, &hht_blocks)?;
                run_task_averaged_resized(ctx, &features_root, FeatureSrc::Hht, &hht_blocks)?;
            } else {
                debug!("hammerAP: no HHT blocks, skipping HHT analyses");
            }
        }
        other => {
            debug!(
                task = other,
                "unrecognized task, skipping feature extraction"
            );
        }
    }

    Ok(())
}
