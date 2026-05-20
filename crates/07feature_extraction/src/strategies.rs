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
use utils::frequency_bands;
use utils::hdf5_io::{H5Attr, open_or_create_group, write_attrs, write_dataset_old};
use utils::roi_migration::check_roi_fingerprint;

use crate::FeatureExtractor;
use crate::preprocessing::{
    batch_spectrum_to_input, chunk_along_time, quantize_2d_tensor, resize_and_mean_blocks,
    shuffled_concat, stack_and_mean, trim_and_mean_blocks,
};

pub const CHUNK_COUNT: i64 = 3;
pub const TASK_COMMON_BLOCK_W: i64 = 23;
pub const SHUFFLE_SEED: u64 = 42;

/// Trial types included in classification feature extraction. Shape blocks are
/// excluded here to keep 08classification scope identical while stage 03/04
/// emit all conditions for FC analysis.
const CLASSIFICATION_TRIAL_TYPES: &[&str] = &["face"];

/// Which upstream spectrum source feeds the analysis.
#[derive(Debug, Clone, Copy)]
pub enum FeatureSrc {
    Ts,
    Cwt,
    /// HHT from all-channel MVMD (`04hht/full_run_std`, `04hht/blocks_std`).
    /// ROI selection applied at load time, analogous to `Cwt`.
    Hht,
    /// HHT from ROI-stratified MVMD (`04hht/full_run_std_roi`, `04hht/blocks_std_roi`).
    /// Already ROI-selected upstream — no index_select at load time.
    HhtRoiStratified,
    /// Frequency-smoothed skeleton from all-channel MVMD.
    /// Huang's "lower frequency resolution" alternative: skeleton convolved with a
    /// per-segment frequency-axis moving average (time axis untouched).
    HhtSmoothed,
    /// Frequency-smoothed skeleton from ROI-stratified MVMD.
    HhtRoiStratifiedSmoothed,
    /// Instantaneous energy from all-channel MVMD: IE(t) = Σ_ω H²(ω,t).
    /// Collapses the frequency axis; encoded as a one-hot amplitude image for DenseNet.
    HhtIe,
    /// Instantaneous energy from ROI-stratified MVMD.
    HhtRoiStratifiedIe,
}

impl FeatureSrc {
    pub fn group_name(self) -> &'static str {
        match self {
            FeatureSrc::Ts => "ts",
            FeatureSrc::Cwt => "cwt",
            FeatureSrc::Hht => "hht",
            FeatureSrc::HhtRoiStratified => "hht_roi_stratified",
            FeatureSrc::HhtSmoothed => "hht_smoothed",
            FeatureSrc::HhtRoiStratifiedSmoothed => "hht_roi_stratified_smoothed",
            FeatureSrc::HhtIe => "hht_ie",
            FeatureSrc::HhtRoiStratifiedIe => "hht_roi_stratified_ie",
        }
    }
}

/// Per-subject-per-file context passed into every analysis runner.
pub struct AnalysisCtx<'a> {
    pub extractor: &'a FeatureExtractor,
    pub fit: ImageFitMode,
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
    /// fMRI sampling rate in Hz (= 1/TR). Used to derive the per-segment
    /// low-frequency floor `max(f_min, sampling_rate / n_t)` when scattering
    /// instantaneous frequencies into spectrum bins.
    pub sampling_rate: f64,
    /// When true, also produce frequency-smoothed HHT analyses (`hht_smoothed` /
    /// `hht_roi_smoothed`) alongside the skeleton.
    pub hht_smoothed: bool,
    /// When true, also produce instantaneous-energy analyses (`hht_ie` /
    /// `hht_roi_ie`) alongside the skeleton.
    pub hht_ie: bool,
    /// When false (default), skip `HhtRoiStratified*` feature sources — no
    /// `_roi` HDF5 groups were produced by stage 04.
    pub roi_stratified_decomposition: bool,
}

impl AnalysisCtx<'_> {
    /// Returns true when the spectrogram for `src` was pre-normalized by the
    /// producer crate and the consumer-side min-max step should be skipped.
    fn pre_normalized_for(src: FeatureSrc) -> bool {
        !matches!(src, FeatureSrc::Ts)
    }
}

// Note on `HhtIe` normalization: `compute_hht_ie` calls `quantize_2d_tensor`
// which performs its own global min-max before one-hot encoding, so the output
// is always a {0,1} tensor. Consumer-side min-max is a no-op on {0,1} values
// and is therefore safely skipped (`pre_normalized = true` for `HhtIe` /
// `HhtRoiStratifiedIe`).

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

/// Read a 2D dataset as f32, regardless of whether it's stored as f32 or f64.
fn read_2d_as_f32(ds: &hdf5::Dataset) -> Result<(Tensor, [i64; 2])> {
    let shape = ds.shape();
    let [a, b] = match shape.as_slice() {
        &[a, b] => [a as i64, b as i64],
        _ => anyhow::bail!("expected 3D dataset, got shape {:?}", shape),
    };
    debug!(
        shape = format!("({},{})", a, b),
        a = a,
        b = b,
        "read 2D as f32"
    );
    let dtype = ds.dtype()?.to_descriptor()?;
    let buf: Vec<f32> = match dtype {
        TypeDescriptor::Float(hdf5::types::FloatSize::U8) => {
            let raw: Vec<f64> = ds.read_raw()?;
            raw.into_iter().map(|v| v as f32).collect()
        }
        TypeDescriptor::Float(hdf5::types::FloatSize::U4) => ds.read_raw()?,
        other => anyhow::bail!("unsupported dataset dtype {:?}", other),
    };
    let t = Tensor::from_slice(&buf).reshape([a, b]);
    Ok((t, [a, b]))
}

/// restAP time series full-run: `/01fmri_parcellation/full_run_std` `[n_rois_all, 224, T_full]`.
/// Returns ROI-selected tensor `[n_target, 224, T_full]`.
fn load_resting_state_full_run(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Option<Tensor>> {
    let cwt_root = match h5.group("01fmri_parcellation") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let ds = match cwt_root.dataset("full_run_std") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (full, [n_all, _]) = read_2d_as_f32(&ds)?;
    validate_roi_range(ctx.roi_indices, n_all, "restAP full_run_std")?;

    let selected = full.index_select(0, ctx.roi_index_tensor);
    let min_val = selected.min().double_value(&[]);
    let max_val = selected.max().double_value(&[]);
    debug!("DEBUG RANGE: min={:.6?}, max={:.6?}", min_val, max_val);

    let sample_val = selected.get(0).get(0).double_value(&[]);
    debug!(
        "DEBUG SAMPLE: Local Index 0, Time 0 raw value = {:.6?}",
        sample_val
    );

    let quantized = quantize_2d_tensor(&selected, 224, false);

    // if ctx.subject_id == "sub-NDARINVAG388HJL" {
    //     let nonzero_indices = quantized.eq(1.0).nonzero();

    //     // Enumerate so we have 'i' (the index in the new tensor)
    //     // and 'original_roi_idx' (the actual ROI number)
    //     for (i, &original_roi_idx) in ctx.roi_indices.iter().enumerate() {
    //         // Search for 'i' (0, 1, 2...) in the quantized tensor
    //         let roi_mask = nonzero_indices.select(1, 0).eq(i as i64);

    //         let sum_as_tensor = roi_mask.sum(Kind::Int64);
    //         if sum_as_tensor.int64_value(&[]) > 0 {
    //             let roi_locations = nonzero_indices.index_select(0, &roi_mask.nonzero().squeeze());
    //             let mut locations_vec = Vec::new();

    //             for j in 0..roi_locations.size()[0] {
    //                 let amp = roi_locations.get(j).get(1).int64_value(&[]);
    //                 let time = roi_locations.get(j).get(2).int64_value(&[]);
    //                 locations_vec.push((time, amp));
    //             }

    //             locations_vec.sort_by_key(|&(time, _)| time);
    //             let num_to_log = locations_vec.len().min(20);

    //             warn!(
    //                 "VERIFICATION: Channel {:03} (Local Index {}) — First {} samples: {:?}",
    //                 original_roi_idx,
    //                 i,
    //                 num_to_log,
    //                 &locations_vec[..num_to_log]
    //             );
    //         }
    //     }
    // }

    Ok(Some(quantized))
}

/// CWT restAP full-run: `/03cwt/full_run_std` `[n_rois_all, 224, T_full]`.
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

/// CWT hammerAP face blocks: `/03cwt/blocks_std/{trial_type}/{block_name}` per block.
/// Returns ordered list of (name, ROI-selected tensor `[n_target, 224, T_block]`).
fn load_cwt_blocks(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Vec<(String, Tensor)>> {
    let cwt_root = match h5.group("03cwt") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let blocks_parent = match cwt_root.group("blocks_std") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    // CWT writes blocks_std/{trial_type}/{block_name}, not blocks_std/{block_name}.
    let mut trial_types: Vec<String> = blocks_parent
        .member_names()?
        .into_iter()
        .filter(|n| !n.starts_with("block_"))
        .collect();
    trial_types.retain(|t| CLASSIFICATION_TRIAL_TYPES.contains(&t.as_str()));
    let mut out = Vec::new();
    for trial_type in &trial_types {
        let trial_group = match blocks_parent.group(trial_type) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let mut names: Vec<String> = trial_group
            .member_names()?
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect();
        names.sort();
        for name in names {
            let ds = trial_group.dataset(&name)?;
            let (block, [n_all, _, _]) = read_3d_as_f32(&ds)?;
            validate_roi_range(ctx.roi_indices, n_all, "cwt blocks_std")?;
            out.push((name, block.index_select(0, ctx.roi_index_tensor)));
        }
    }
    Ok(out)
}

/// Build a scatter-style Hilbert spectrum image from instantaneous frequency and envelope.
///
/// Inputs `inst_freq` and `envelope` are both `[n_modes, n_ch, n_t]` f32 tensors.
/// Returns `[n_ch, 224, n_t]` f32 — same shape as the pre-computed `hilbert_spectrum`
/// dataset, so all downstream strategy code is unchanged.
///
/// The render grid is always 224 log-spaced bins over `[f_min, f_max)`, keeping
/// the frequency axis identical across all segment lengths so tensors are stackable.
/// Samples with IF below the per-segment floor `max(f_min, sampling_rate / n_t)`
/// are dropped in addition to the global `>= f_max` cut, preventing short segments
/// from populating frequency cells they cannot resolve (Huang 1998: lowest
/// resolvable frequency = 1/T).
///
/// Bin mapping: bin 0 = f_max (high), bin 223 = f_min (low).
/// Multiple modes at the same (channel, bin, time) resolved by element-wise max.
fn scatter_hht_spectrum(inst_freq: &Tensor, envelope: &Tensor, sampling_rate: f64) -> Tensor {
    let f_min_hz = frequency_bands::f_min();
    let f_max_hz = frequency_bands::f_max();
    let log_f_max = f_max_hz.ln();
    let log_ratio_denom = (f_min_hz / f_max_hz).ln();

    let sizes = inst_freq.size();
    let (n_modes, n_ch, n_t) = (sizes[0], sizes[1], sizes[2]);

    let f_lo = f_min_hz.max(sampling_rate / n_t as f64);

    let mut out = Tensor::zeros([n_ch, 224, n_t], (Kind::Float, inst_freq.device()));

    for m in 0..n_modes {
        let if_m = inst_freq.select(0, m); // [n_ch, n_t]
        let env_m = envelope.select(0, m); // [n_ch, n_t]

        let in_band = if_m.ge(f_lo).logical_and(&if_m.lt(f_max_hz));

        let if_clamped = if_m.clamp(1e-10, f_max_hz);
        let log_ratio = (if_clamped.log() - log_f_max) / log_ratio_denom;
        let bins = (log_ratio * 223.0)
            .round()
            .clamp(0.0, 223.0)
            .to_kind(Kind::Int64); // [n_ch, n_t]

        let env_masked = env_m * in_band.to_kind(Kind::Float);

        let mut mode_scatter = Tensor::zeros([n_ch, 224, n_t], (Kind::Float, inst_freq.device()));
        let _ = mode_scatter.scatter_(1, &bins.unsqueeze(1), &env_masked.unsqueeze(1));

        out = out.maximum(&mode_scatter);
    }

    out
}

/// Huang's "lower frequency resolution, time axis undisturbed" alternative to spatial smoothing.
///
/// Applies a moving average along the frequency axis only (dim 1). The kernel width
/// is derived per-segment from Huang's optimal cell count `N = round(n_t / 5)`:
/// `w = ceil(224 / N)`, forced odd, minimum 3 (Huang's "average over three adjacent cells").
/// For full runs (~488 TR) this gives w ≈ 3; for 23–24 TR blocks w = 45.
/// Because `round(23/5) == round(24/5) == 5`, both block lengths yield the same
/// kernel width so 23-TR and 24-TR block spectra have identical effective resolution.
fn smooth_hht_frequency(skeleton: &Tensor) -> Tensor {
    let n_t = skeleton.size()[2];
    let n_native = frequency_bands::hilbert_native_cells(n_t as usize) as i64;
    let mut w = ((224 + n_native - 1) / n_native).max(3);
    if w % 2 == 0 {
        w += 1;
    }
    skeleton
        .unsqueeze(0) // [1, n_ch, 224, n_t]
        .avg_pool2d([w, 1], [1, 1], [w / 2, 0], false, true, None)
        .squeeze_dim(0) // [n_ch, 224, n_t]
}

/// Compute instantaneous energy IE(t) = Σ_ω H²(ω, t) from a skeleton spectrum.
///
/// Input `[n_ch, 224, n_t]`. Squares each bin, sums over the frequency axis (dim 1)
/// → `[n_ch, n_t]` instantaneous power time-series. Encodes as a 224-level one-hot
/// amplitude image via `quantize_2d_tensor` → output `[n_ch, 224, n_t]` suitable for
/// the existing DenseNet input pipeline.
///
/// Note: H here is amplitude (log1p-compressed and per-channel normalized), not raw
/// amplitude. H² is therefore relative instantaneous energy on the processed scale.
/// Squaring a `[0, 1]` skeleton is not the same as H_raw² — but is consistent with
/// how the skeleton itself is used for classification.
fn compute_hht_ie(spectrum: &Tensor) -> Tensor {
    let ie =
        spectrum
            .pow_tensor_scalar(2.0)
            .sum_dim_intlist([1i64][..].as_ref(), false, Kind::Float); // [n_ch, n_t]
    quantize_2d_tensor(&ie, 224, false) // [n_ch, 224, n_t]
}

/// HHT restAP whole-run from all-channel MVMD: `/04hht/full_run_std/{instantaneous_frequency,envelope}`
/// Scatter-bins IF+envelope into `[n_all, 224, T_full]`, then ROI-selects.
fn load_hht_full_run(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Option<Tensor>> {
    let hht_root = match h5.group("04hht") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let sub = match hht_root.group("full_run_std") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let ds_if = match sub.dataset("instantaneous_frequency") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let ds_env = match sub.dataset("envelope") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (inst_freq, [_, n_ch, _]) = read_3d_as_f32(&ds_if)?; // [n_modes, n_ch, n_t]
    let (envelope, _) = read_3d_as_f32(&ds_env)?;
    validate_roi_range(ctx.roi_indices, n_ch, "hht full_run_std")?;
    let spec = scatter_hht_spectrum(&inst_freq, &envelope, ctx.sampling_rate); // [n_ch, 224, n_t]
    Ok(Some(spec.index_select(0, ctx.roi_index_tensor)))
}

/// HHT hammerAP face blocks from all-channel MVMD: `/04hht/blocks_std/{trial_type}/{block_name}/{if,env}`
/// Scatter-bins IF+envelope per block into `[n_all, 224, T_block]`, then ROI-selects.
fn load_hht_blocks(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Vec<(String, Tensor)>> {
    let hht_root = match h5.group("04hht") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let blocks_parent = match hht_root.group("blocks_std") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let mut trial_types: Vec<String> = blocks_parent
        .member_names()?
        .into_iter()
        .filter(|n| !n.starts_with("block_"))
        .collect();
    trial_types.retain(|t| CLASSIFICATION_TRIAL_TYPES.contains(&t.as_str()));
    let mut out = Vec::new();
    for trial_type in &trial_types {
        let trial_group = match blocks_parent.group(trial_type) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let mut names: Vec<String> = trial_group
            .member_names()?
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect();
        names.sort();
        for name in names {
            let g = trial_group.group(&name)?;
            let ds_if = match g.dataset("instantaneous_frequency") {
                Ok(d) => d,
                Err(_) => continue,
            };
            let ds_env = match g.dataset("envelope") {
                Ok(d) => d,
                Err(_) => continue,
            };
            let (inst_freq, [_, n_ch, _]) = read_3d_as_f32(&ds_if)?;
            let (envelope, _) = read_3d_as_f32(&ds_env)?;
            validate_roi_range(ctx.roi_indices, n_ch, "hht blocks_std")?;
            let spec = scatter_hht_spectrum(&inst_freq, &envelope, ctx.sampling_rate);
            out.push((name, spec.index_select(0, ctx.roi_index_tensor)));
        }
    }
    Ok(out)
}

/// HHT restAP whole-run from ROI-stratified MVMD: `/04hht/full_run_std_roi/{if,env}`
/// Already ROI-selected upstream; scatter-bins IF+envelope into `[n_target, 224, T_full]`.
fn load_hht_roi_full_run(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Option<Tensor>> {
    let hht_root = match h5.group("04hht") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let sub = match hht_root.group("full_run_std_roi") {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    check_roi_fingerprint(&sub, ctx.roi_selection_fingerprint)?;
    let ds_if = match sub.dataset("instantaneous_frequency") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let ds_env = match sub.dataset("envelope") {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    let (inst_freq, [_, n_rows, _]) = read_3d_as_f32(&ds_if)?; // [n_modes, n_roi, n_t]
    let (envelope, _) = read_3d_as_f32(&ds_env)?;
    if (n_rows as usize) != ctx.roi_indices.len() {
        anyhow::bail!(
            "hht_roi_stratified full_run_std_roi rows {} != target ROI count {} — atlas mismatch",
            n_rows,
            ctx.roi_indices.len()
        );
    }
    Ok(Some(scatter_hht_spectrum(
        &inst_freq,
        &envelope,
        ctx.sampling_rate,
    )))
}

/// HHT hammerAP face blocks from ROI-stratified MVMD:
/// `/04hht/blocks_std_roi/{trial_type}/{block_name}/{if,env}`
/// Already ROI-selected upstream; scatter-bins IF+envelope per block.
fn load_hht_roi_blocks(h5: &hdf5::File, ctx: &AnalysisCtx) -> Result<Vec<(String, Tensor)>> {
    let hht_root = match h5.group("04hht") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let blocks_parent = match hht_root.group("blocks_std_roi") {
        Ok(g) => g,
        Err(_) => return Ok(vec![]),
    };
    let mut trial_types: Vec<String> = blocks_parent
        .member_names()?
        .into_iter()
        .filter(|n| !n.starts_with("block_"))
        .collect();
    trial_types.retain(|t| CLASSIFICATION_TRIAL_TYPES.contains(&t.as_str()));
    let mut out = Vec::new();
    for trial_type in &trial_types {
        let trial_group = match blocks_parent.group(trial_type) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let mut names: Vec<String> = trial_group
            .member_names()?
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect();
        names.sort();
        for name in names {
            let g = trial_group.group(&name)?;
            check_roi_fingerprint(&g, ctx.roi_selection_fingerprint)?;
            let ds_if = match g.dataset("instantaneous_frequency") {
                Ok(d) => d,
                Err(_) => continue,
            };
            let ds_env = match g.dataset("envelope") {
                Ok(d) => d,
                Err(_) => continue,
            };
            let (inst_freq, [_, n_rows, _]) = read_3d_as_f32(&ds_if)?;
            let (envelope, _) = read_3d_as_f32(&ds_env)?;
            if (n_rows as usize) != ctx.roi_indices.len() {
                anyhow::bail!(
                    "hht_roi_stratified blocks_std_roi/{}/{} rows {} != target ROI count {}",
                    trial_type,
                    name,
                    n_rows,
                    ctx.roi_indices.len()
                );
            }
            out.push((
                name,
                scatter_hht_spectrum(&inst_freq, &envelope, ctx.sampling_rate),
            ));
        }
    }
    Ok(out)
}

fn validate_roi_range(roi_indices: &[i64], n_available: i64, what: &str) -> Result<()> {
    if let Some(&max_idx) = roi_indices.iter().max()
        && max_idx >= n_available
    {
        anyhow::bail!(
            "target ROI index {} out of range for {} (n_available={})",
            max_idx,
            what,
            n_available
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Feature extraction + write
// ---------------------------------------------------------------------------

/// Run a `[n_rois, F, T]` spectrum through preprocessing + DenseNet.
/// Returns `(per_roi [n_rois, 1920], mean [1920])` on CPU as f32.
fn extract(ctx: &AnalysisCtx, src: FeatureSrc, spec: &Tensor) -> (Tensor, Tensor) {
    let batch = batch_spectrum_to_input(spec, AnalysisCtx::pre_normalized_for(src), ctx.fit);
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
    let batch = batch_spectrum_to_input(spec, AnalysisCtx::pre_normalized_for(src), fit);
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
    write_dataset_old(&group, "per_roi", &per_roi_buf, &[n_rois, feat_dim], None)?;
    write_dataset_old(&group, "mean", &mean_buf, &[feat_dim], None)?;
    write_dataset_old(&group, "roi_indices", &roi_idx_u32, &[n_rois], None)?;
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

fn analysis_root(
    features_root: &hdf5::Group,
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
// Strategy A' — restAP, baseline chunked feature-space mean
// ---------------------------------------------------------------------------

/// Split full-run spectrum into `CHUNK_COUNT` time chunks, run DenseNet on each
/// chunk independently, then mean the resulting feature vectors across chunks
/// per ROI. Single-leaf write under `features/<src>/baseline_chunked_feature_mean`.
///
/// Differs from `baseline_averaged` (image-space mean): DenseNet sees each
/// chunk individually, then features are averaged. Because DenseNet is
/// nonlinear, mean(features(x_i)) != features(mean(x_i)).
pub fn run_baseline_chunked_feature_mean(
    ctx: &AnalysisCtx,
    features_root: &hdf5::Group,
    src: FeatureSrc,
    full_spec: &Tensor,
) -> Result<()> {
    let analysis = "baseline_chunked_feature_mean";
    let src_g = open_or_create_group(features_root, src.group_name(), false)?;
    if already_done(&src_g, analysis, ctx.force) {
        debug!(
            src = src.group_name(),
            "baseline_chunked_feature_mean: exists, skipping"
        );
        return Ok(());
    }
    let chunks = chunk_along_time(full_spec, CHUNK_COUNT);
    let started = Instant::now();

    let per_chunk_features: Vec<Tensor> = chunks
        .iter()
        .map(|chunk| {
            let (per_roi, _) = extract(ctx, src, chunk);
            per_roi
        })
        .collect();

    let stacked = Tensor::stack(&per_chunk_features, 0); // [n_chunks, n_rois, feat_dim]
    let per_roi = stacked.mean_dim(Some([0i64].as_slice()), false, Kind::Float);
    let mean = per_roi.mean_dim(Some([0i64].as_slice()), false, Kind::Float);

    write_features(&src_g, analysis, &per_roi, &mean, ctx, analysis)?;
    info!(
        src = src.group_name(),
        n_chunks = CHUNK_COUNT,
        ms = started.elapsed().as_millis() as u64,
        "baseline_chunked_feature_mean done"
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
    write_dataset_old(&group, "per_roi", &per_roi_buf, &[n_rois, feat_dim], None)?;
    write_dataset_old(&group, "mean", &mean_buf, &[feat_dim], None)?;
    write_dataset_old(&group, "roi_indices", &roi_idx_u32, &[n_rois], None)?;
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

/// Bicubicly resize each block's std spectrum to `224×224`, mean across
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
        debug!(
            src = src.group_name(),
            "task_averaged_resized: exists, skipping"
        );
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
                run_baseline_chunked_feature_mean(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
                run_baseline_averaged(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
                run_baseline_resized(ctx, &features_root, FeatureSrc::Cwt, &spec)?;
            } else {
                debug!("restAP: no CWT full_run_std, skipping CWT analyses");
            }
            if let Some(spec) = load_hht_full_run(h5, ctx)? {
                run_baseline_chunked(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                run_baseline_chunked_feature_mean(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                run_baseline_averaged(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                run_baseline_resized(ctx, &features_root, FeatureSrc::Hht, &spec)?;
                if ctx.hht_smoothed {
                    let spec_sm = smooth_hht_frequency(&spec);
                    run_baseline_chunked(ctx, &features_root, FeatureSrc::HhtSmoothed, &spec_sm)?;
                    run_baseline_chunked_feature_mean(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtSmoothed,
                        &spec_sm,
                    )?;
                    run_baseline_averaged(ctx, &features_root, FeatureSrc::HhtSmoothed, &spec_sm)?;
                    run_baseline_resized(ctx, &features_root, FeatureSrc::HhtSmoothed, &spec_sm)?;
                }
                if ctx.hht_ie {
                    let ie = compute_hht_ie(&spec);
                    run_baseline_chunked(ctx, &features_root, FeatureSrc::HhtIe, &ie)?;
                    run_baseline_chunked_feature_mean(ctx, &features_root, FeatureSrc::HhtIe, &ie)?;
                    run_baseline_averaged(ctx, &features_root, FeatureSrc::HhtIe, &ie)?;
                    run_baseline_resized(ctx, &features_root, FeatureSrc::HhtIe, &ie)?;
                }
            } else {
                debug!("restAP: no HHT full_run_std, skipping HHT analyses");
            }
            if ctx.roi_stratified_decomposition {
                if let Some(spec) = load_hht_roi_full_run(h5, ctx)? {
                    run_baseline_chunked(ctx, &features_root, FeatureSrc::HhtRoiStratified, &spec)?;
                    run_baseline_chunked_feature_mean(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &spec,
                    )?;
                    run_baseline_averaged(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &spec,
                    )?;
                    run_baseline_resized(ctx, &features_root, FeatureSrc::HhtRoiStratified, &spec)?;
                    if ctx.hht_smoothed {
                        let spec_sm = smooth_hht_frequency(&spec);
                        run_baseline_chunked(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &spec_sm,
                        )?;
                        run_baseline_chunked_feature_mean(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &spec_sm,
                        )?;
                        run_baseline_averaged(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &spec_sm,
                        )?;
                        run_baseline_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &spec_sm,
                        )?;
                    }
                    if ctx.hht_ie {
                        let ie = compute_hht_ie(&spec);
                        run_baseline_chunked(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie,
                        )?;
                        run_baseline_chunked_feature_mean(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie,
                        )?;
                        run_baseline_averaged(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie,
                        )?;
                        run_baseline_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie,
                        )?;
                    }
                } else {
                    debug!("restAP: no HHT full_run_std_roi, skipping HHT-ROI analyses");
                }
            }
            if let Some(spec) = load_resting_state_full_run(h5, ctx)? {
                run_baseline_chunked(ctx, &features_root, FeatureSrc::Ts, &spec)?;
                run_baseline_chunked_feature_mean(ctx, &features_root, FeatureSrc::Ts, &spec)?;
                run_baseline_resized(ctx, &features_root, FeatureSrc::Ts, &spec)?;
            } else {
                debug!("restAP: no full_run_std timeseries, skipping analysis")
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
                if ctx.hht_smoothed {
                    let sm_blocks: Vec<(String, Tensor)> = hht_blocks
                        .iter()
                        .map(|(n, t)| (n.clone(), smooth_hht_frequency(t)))
                        .collect();
                    run_task_concat(ctx, &features_root, FeatureSrc::HhtSmoothed, &sm_blocks)?;
                    run_task_per_block(ctx, &features_root, FeatureSrc::HhtSmoothed, &sm_blocks)?;
                    run_task_averaged(ctx, &features_root, FeatureSrc::HhtSmoothed, &sm_blocks)?;
                    run_task_per_block_resized(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtSmoothed,
                        &sm_blocks,
                    )?;
                    run_task_averaged_resized(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtSmoothed,
                        &sm_blocks,
                    )?;
                }
                if ctx.hht_ie {
                    let ie_blocks: Vec<(String, Tensor)> = hht_blocks
                        .iter()
                        .map(|(n, t)| (n.clone(), compute_hht_ie(t)))
                        .collect();
                    run_task_concat(ctx, &features_root, FeatureSrc::HhtIe, &ie_blocks)?;
                    run_task_per_block(ctx, &features_root, FeatureSrc::HhtIe, &ie_blocks)?;
                    run_task_averaged(ctx, &features_root, FeatureSrc::HhtIe, &ie_blocks)?;
                    run_task_per_block_resized(ctx, &features_root, FeatureSrc::HhtIe, &ie_blocks)?;
                    run_task_averaged_resized(ctx, &features_root, FeatureSrc::HhtIe, &ie_blocks)?;
                }
            } else {
                debug!("hammerAP: no HHT blocks_std, skipping HHT analyses");
            }
            if ctx.roi_stratified_decomposition {
                let hht_roi_blocks = load_hht_roi_blocks(h5, ctx)?;
                if !hht_roi_blocks.is_empty() {
                    run_task_concat(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &hht_roi_blocks,
                    )?;
                    run_task_per_block(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &hht_roi_blocks,
                    )?;
                    run_task_averaged(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &hht_roi_blocks,
                    )?;
                    run_task_per_block_resized(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &hht_roi_blocks,
                    )?;
                    run_task_averaged_resized(
                        ctx,
                        &features_root,
                        FeatureSrc::HhtRoiStratified,
                        &hht_roi_blocks,
                    )?;
                    if ctx.hht_smoothed {
                        let sm_roi_blocks: Vec<(String, Tensor)> = hht_roi_blocks
                            .iter()
                            .map(|(n, t)| (n.clone(), smooth_hht_frequency(t)))
                            .collect();
                        run_task_concat(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &sm_roi_blocks,
                        )?;
                        run_task_per_block(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &sm_roi_blocks,
                        )?;
                        run_task_averaged(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &sm_roi_blocks,
                        )?;
                        run_task_per_block_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &sm_roi_blocks,
                        )?;
                        run_task_averaged_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedSmoothed,
                            &sm_roi_blocks,
                        )?;
                    }
                    if ctx.hht_ie {
                        let ie_roi_blocks: Vec<(String, Tensor)> = hht_roi_blocks
                            .iter()
                            .map(|(n, t)| (n.clone(), compute_hht_ie(t)))
                            .collect();
                        run_task_concat(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie_roi_blocks,
                        )?;
                        run_task_per_block(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie_roi_blocks,
                        )?;
                        run_task_averaged(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie_roi_blocks,
                        )?;
                        run_task_per_block_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie_roi_blocks,
                        )?;
                        run_task_averaged_resized(
                            ctx,
                            &features_root,
                            FeatureSrc::HhtRoiStratifiedIe,
                            &ie_roi_blocks,
                        )?;
                    }
                } else {
                    debug!("hammerAP: no HHT blocks_std_roi, skipping HHT-ROI analyses");
                }
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
