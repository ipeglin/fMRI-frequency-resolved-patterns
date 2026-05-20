//! DenseNet-201 input preparation for time-frequency images.
//!
//! Standard torchvision DenseNet-201 expects per-channel normalized RGB:
//!   1. Pixel values in `[0, 1]` (float).
//!   2. Per-channel ImageNet mean/std applied — final tensor in roughly
//!      `[-2.1, +2.6]`.
//!
//! Pipeline per spectrogram (`[F, T]`, F=224 frequency bins):
//!   pre_normalized=true  → `[0, 1]` (from producer) → fit → replicate channel → ImageNet normalize
//!   pre_normalized=false → per-image min-max → `[0, 1]` → fit → replicate channel → ImageNet normalize
//!
//! CWT and HHT are pre-normalized at producer time (crates 03cwt / 05hilbert).
//! Raw timeseries (Ts) still requires consumer-side min-max.

use tch::{Kind, Tensor};
use tracing::debug;
use utils::config::ImageFitMode;

const TARGET_H: i64 = 224;
const TARGET_W: i64 = 224;
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Convert one grayscale spectrogram `[F, T]` to a DenseNet-ready tensor
/// `[1, 3, 224, 224]`.
///
/// `pre_normalized` signals that the spectrogram is already in `[0, 1]`
/// (normalized by the producer crate) and the consumer-side min-max step
/// should be skipped. Pass `false` for raw sources (e.g. Ts) that still
/// need per-image normalization here. `fit` controls how a non-`224×224`
/// spectrogram is coerced to the DenseNet input size.
pub fn spectrum_to_image(spec: &Tensor, pre_normalized: bool, fit: ImageFitMode) -> Tensor {
    let normalised = if pre_normalized {
        spec.shallow_clone()
    } else {
        let min = spec.min();
        let max = spec.max();
        (spec - &min) / (&max - &min + 1e-8)
    };

    let img = normalised.unsqueeze(0).unsqueeze(0);
    let img = match fit {
        ImageFitMode::Resize => img.upsample_bicubic2d([TARGET_H, TARGET_W], false, None, None),
        ImageFitMode::Pad => pad_to(&img, TARGET_H, TARGET_W),
    };
    let img = img.expand([1, 3, TARGET_H, TARGET_W], false).contiguous();

    let device = img.device();
    let mean = Tensor::from_slice(&IMAGENET_MEAN)
        .reshape([1, 3, 1, 1])
        .to_kind(Kind::Float)
        .to_device(device);
    let std = Tensor::from_slice(&IMAGENET_STD)
        .reshape([1, 3, 1, 1])
        .to_kind(Kind::Float)
        .to_device(device);
    (img - mean) / std
}

/// Stack per-ROI spectrograms into a single DenseNet input batch.
///
/// Input  `[n_rois, F, T]`. Output `[n_rois, 3, 224, 224]`.
pub fn batch_spectrum_to_input(
    rois_spec: &Tensor,
    pre_normalized: bool,
    fit: ImageFitMode,
) -> Tensor {
    let n = rois_spec.size()[0];
    let images: Vec<Tensor> = (0..n)
        .map(|i| spectrum_to_image(&rois_spec.select(0, i), pre_normalized, fit))
        .collect();
    Tensor::cat(&images, 0)
}

/// Right- and bottom-pad a `[1, 1, H, W]` tensor with zeros to `[1, 1, target_h, target_w]`.
/// Inputs already meeting/exceeding either target dim are passed through on that axis.
fn pad_to(img: &Tensor, target_h: i64, target_w: i64) -> Tensor {
    let dims = img.size();
    let cur_h = dims[2];
    let cur_w = dims[3];
    let pad_h = (target_h - cur_h).max(0);
    let pad_w = (target_w - cur_w).max(0);
    if pad_h == 0 && pad_w == 0 {
        return img.shallow_clone();
    }
    // `Tensor::pad` order is (left, right, top, bottom).
    img.pad([0, pad_w, 0, pad_h], "constant", Some(0.0))
}

/// Trim time axis to `n_chunks * (T / n_chunks)` and split into equal-width chunks.
///
/// Input `[n_rois, F, T]`. Output: vector of length `n_chunks`, each
/// `[n_rois, F, T / n_chunks]`. Trailing `T mod n_chunks` columns are dropped
/// from the tail so all chunks have identical width.
pub fn chunk_along_time(spec: &Tensor, n_chunks: i64) -> Vec<Tensor> {
    let dims = spec.size();
    let t = dims[2];
    let chunk_w = t / n_chunks;
    let trim_w = chunk_w * n_chunks;
    let trimmed = spec.narrow(2, 0, trim_w);
    (0..n_chunks)
        .map(|i| trimmed.narrow(2, i * chunk_w, chunk_w).contiguous())
        .collect()
}

/// Stack-and-mean a sequence of equally-shaped spectrograms.
pub fn stack_and_mean(specs: &[Tensor]) -> Tensor {
    Tensor::stack(specs, 0).mean_dim(Some([0i64].as_slice()), false, Kind::Float)
}

/// Concatenate per-block spectrograms along time after a deterministic shuffle.
///
/// Each `blocks[i]` shape `[n_rois, F, T_i]`. Output `[n_rois, F, sum(T_i)]`.
pub fn shuffled_concat(blocks: Vec<Tensor>, seed: u64) -> Tensor {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let mut shuffled = blocks;
    shuffled.shuffle(&mut rng);
    let refs: Vec<&Tensor> = shuffled.iter().collect();
    Tensor::cat(&refs, 2)
}

/// Trim each block's time axis to `target_w` (drop trailing columns), then
/// stack-mean across blocks. Blocks shorter than `target_w` are skipped.
pub fn trim_and_mean_blocks(blocks: &[Tensor], target_w: i64) -> Tensor {
    let trimmed: Vec<Tensor> = blocks
        .iter()
        .filter(|b| b.size()[2] >= target_w)
        .map(|b| b.narrow(2, 0, target_w).contiguous())
        .collect();
    stack_and_mean(&trimmed)
}

/// Bicubicly resize each per-ROI raw spectrum from `[F, T]` to
/// `[target_h, target_w]`. Input `[n_rois, F, T]` → output
/// `[n_rois, target_h, target_w]`. Used by resized-block strategies that need
/// to bring face blocks to a common 224×224 footprint before stack-mean.
pub fn resize_along_freq_time(spec: &Tensor, target_h: i64, target_w: i64) -> Tensor {
    spec.unsqueeze(1)
        .upsample_bicubic2d([target_h, target_w], false, None, None)
        .squeeze_dim(1)
}

/// Resize each block's raw spectrum to `[target_h, target_w]` then stack-mean
/// across blocks. Empty input returns an empty mean (caller should guard).
pub fn resize_and_mean_blocks(blocks: &[Tensor], target_h: i64, target_w: i64) -> Tensor {
    let resized: Vec<Tensor> = blocks
        .iter()
        .map(|b| resize_along_freq_time(b, target_h, target_w).contiguous())
        .collect();
    stack_and_mean(&resized)
}

/// Quantize a 1D or 2D time series to use `num_bins` amplitude quantization levels.
pub fn quantize_2d_tensor(input: &Tensor, num_bins: i64, per_channel: bool) -> Tensor {
    let size = input.size();
    let ndims = size.len();

    debug!(
        size = ?size,
        num_bins = ?num_bins,
        quantize_range_per_channel = per_channel,
        "quantizing tensor to image"
    );

    let (min, max) = if per_channel && ndims > 1 {
        (input.amin([-1], true), input.amax([-1], true))
    } else {
        (input.min(), input.max())
    };

    // Scale and Quantize
    let range = &max - &min;
    let range = range.masked_fill(&range.eq(0.0), 1.0); // Avoid division by zero

    let quantized_indices = (((input - &min) / range) * ((num_bins - 1) as f64))
        .floor()
        .to_kind(Kind::Int64)
        .clamp(0, num_bins - 1); // Clamp handles the "max" edge case

    // Transform to image representation using One-Hot
    // If input is (N,), one_hot shape is (N, 224)
    // If input is (C, N), one_hot shape is (C, N, 224)
    let one_hot = Tensor::one_hot(&quantized_indices, num_bins);

    // Transpose to treat the Bins as the Y-axis and Time as X-axis
    // This dynamically swaps the last two dimensions (Time and Bins)
    // Results in (224, N) or (C, 224, N)
    let image_tensor = one_hot.transpose(ndims as i64 - 1, ndims as i64);

    // Cast to float
    image_tensor.to_kind(Kind::Float)
}
