use anyhow::{Context, Result};
use std::path::Path;
use tch::{Device, Tensor, nn, nn::ModuleT};
use tracing::{info, warn};

use crate::models::DenseNet201;

/// Feature extractor wrapping DenseNet-201.
///
/// Intended use: feed CWT scalograms from the upstream pipeline, receive
/// 1920-dim feature vectors suitable for a KNN or SVM classifier.
pub struct FeatureExtractor {
    model: DenseNet201,
    vs: nn::VarStore,
}

impl FeatureExtractor {
    /// Create a new extractor.
    ///
    /// `num_classes` is used only when running full classification via
    /// `classify`. For pure feature extraction it can be set to any positive
    /// integer (e.g. `1`).
    ///
    /// If `weights_path` is provided the weights are loaded from a PyTorch
    /// state-dict file (`.pt`), a `.safetensors` file, or a tch-native `.ot`
    /// file. Loading is non-strict: extra keys in the file that do not match
    /// the model (e.g. `num_batches_tracked`) are skipped with a warning, and
    /// any model params missing from the file retain their random init.
    pub fn new(weights_path: Option<&Path>, num_classes: i64) -> Result<Self> {
        let device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(device);
        let model = DenseNet201::new(&vs.root(), num_classes);
        if let Some(path) = weights_path {
            load_pretrained(&mut vs, path)?;
        }
        Ok(Self { vs, model })
    }

    /// Extract a 1920-dim feature vector for each input image in the batch.
    ///
    /// Input:  `[batch, 3, 224, 224]` float tensor
    /// Output: `[batch, 1920]` float tensor
    ///
    /// Call `scalogram_to_image` or `trial_scalogram_to_batch` to convert
    /// upstream CWT data into the required format.
    pub fn extract_features(&self, xs: &Tensor) -> Tensor {
        let xs = xs.to_device(self.vs.device());
        tch::no_grad(|| self.model.forward_features(&xs, false))
    }

    /// Full forward pass — returns raw class logits `[batch, num_classes]`.
    pub fn classify(&self, xs: &Tensor) -> Tensor {
        let xs = xs.to_device(self.vs.device());
        tch::no_grad(|| self.model.forward_t(&xs, false))
    }
}

/// Load a torchvision-style DenseNet-201 state dict into `vs`.
///
/// Reads the file with `Tensor::load_multi_with_device` (works with PyTorch
/// `.pt` state-dict archives saved via `torch.save(model.state_dict(), path)`
/// as well as `.ot` / `.safetensors`). Keys are matched directly against the
/// VarStore's `.`-joined parameter names — torchvision's DenseNet naming
/// convention matches the paths built in `DenseNet201::new`.
///
/// Non-strict: extra keys in the file (e.g. `num_batches_tracked`, or the
/// 1000-class `classifier.*` head when we build with a different
/// `num_classes`) are skipped. Missing model params retain their random init.
fn load_pretrained(vs: &mut nn::VarStore, path: &Path) -> Result<()> {
    use std::collections::HashMap;

    let named = Tensor::read_safetensors(path)
        .with_context(|| format!("failed to read weights from {}", path.display()))?;

    let mut src: HashMap<String, Tensor> = named.into_iter().collect();
    let vars = vs.variables();

    let mut loaded = 0usize;
    let mut shape_mismatch: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    tch::no_grad(|| -> Result<()> {
        for (name, mut dst) in vars {
            match src.remove(&name) {
                Some(s) => {
                    if s.size() == dst.size() {
                        dst.copy_(&s);
                        loaded += 1;
                    } else {
                        shape_mismatch.push(format!(
                            "{} (file {:?} vs model {:?})",
                            name,
                            s.size(),
                            dst.size()
                        ));
                    }
                }
                None => missing.push(name),
            }
        }
        Ok(())
    })?;

    let unused: Vec<String> = src.into_keys().collect();

    info!(
        loaded = loaded,
        missing = missing.len(),
        unused = unused.len(),
        shape_mismatch = shape_mismatch.len(),
        path = %path.display(),
        "loaded DenseNet-201 weights"
    );

    if !shape_mismatch.is_empty() {
        warn!(
            keys = ?shape_mismatch,
            "weight shape mismatches — these params kept random init"
        );
    }
    if !missing.is_empty() {
        warn!(
            count = missing.len(),
            sample = ?missing.iter().take(5).collect::<Vec<_>>(),
            "model params not found in weights file — kept random init"
        );
    }
    if !unused.is_empty() {
        warn!(
            count = unused.len(),
            sample = ?unused.iter().take(5).collect::<Vec<_>>(),
            "weights file contained keys not in model — ignored"
        );
    }

    Ok(())
}
