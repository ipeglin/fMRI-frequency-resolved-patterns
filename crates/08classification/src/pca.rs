//! Truncated PCA via Gram-matrix simultaneous power iteration.
//!
//! Designed for the regime n_samples << n_features (typical for fMRI studies:
//! ~100–400 training samples, 1920-dimensional DenseNet embeddings). The
//! Gram matrix G = X_c·X_cᵀ is [n×n], so its eigendecomposition is O(n³)
//! not O(d³). Computing the top-k eigenvectors takes milliseconds.
//!
//! **Data-leakage contract:** always call `fit` on training data only, then
//! `transform` on calibration and holdout separately.

use anyhow::{Result, ensure};
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, StandardNormal};

const POWER_ITER: usize = 100;
const SEED: u64 = 42;

/// Truncated PCA fitted on a training set.
pub struct PcaReducer {
    /// Actual number of components (may be < requested if n_train is small).
    pub n_components: usize,
    /// Per-feature mean of training data. `[n_features]`
    mean: Vec<f32>,
    /// Principal component axes fitted on training data. `[n_components × n_features]`
    /// Row i is the i-th principal component direction (unit vector).
    components: Vec<Vec<f32>>,
}

impl PcaReducer {
    /// Fit PCA on `x_train` (already z-score normalised).
    ///
    /// `n_components` is clamped to `min(requested, n_train - 1, n_features)`.
    pub fn fit(x_train: &[Vec<f32>], n_components: usize) -> Result<Self> {
        let n = x_train.len();
        let d = x_train.first().map(|r| r.len()).unwrap_or(0);
        ensure!(n >= 2, "PCA: need at least 2 training samples, got {n}");
        ensure!(d > 0, "PCA: zero feature dimension");
        ensure!(n_components > 0, "PCA: n_components must be > 0");
        let k = n_components.min(n - 1).min(d);

        // Compute per-feature mean (in f64 for precision).
        let mut mean_f64 = vec![0.0f64; d];
        for row in x_train {
            for (j, &v) in row.iter().enumerate() {
                mean_f64[j] += v as f64;
            }
        }
        let inv_n = 1.0 / n as f64;
        for m in mean_f64.iter_mut() {
            *m *= inv_n;
        }

        // Center training data: X_c[i] = x_train[i] - mean.
        let x_c: Vec<Vec<f64>> = x_train
            .iter()
            .map(|row| {
                row.iter()
                    .zip(mean_f64.iter())
                    .map(|(&v, &m)| v as f64 - m)
                    .collect()
            })
            .collect();

        // Gram matrix G = X_c @ X_c.T  [n × n], symmetric.
        let mut g = vec![0.0f64; n * n];
        for i in 0..n {
            for j in i..n {
                let dot: f64 = x_c[i].iter().zip(x_c[j].iter()).map(|(&a, &b)| a * b).sum();
                g[i * n + j] = dot;
                g[j * n + i] = dot;
            }
        }

        // Simultaneous power iteration: V [n × k] (row-major: v[i*k + jj]).
        // Initialise with random columns, then orthonormalise.
        let mut rng = ChaCha8Rng::seed_from_u64(SEED);
        let normal = StandardNormal;
        let mut v: Vec<f64> = (0..n * k).map(|_| normal.sample(&mut rng)).collect();
        gram_schmidt(&mut v, n, k);

        for _ in 0..POWER_ITER {
            // v_new = G @ v  ([n×n] @ [n×k] → [n×k])
            let mut v_new = vec![0.0f64; n * k];
            for i in 0..n {
                for jj in 0..k {
                    let mut acc = 0.0f64;
                    for p in 0..n {
                        acc += g[i * n + p] * v[p * k + jj];
                    }
                    v_new[i * k + jj] = acc;
                }
            }
            gram_schmidt(&mut v_new, n, k);
            v = v_new;
        }

        // Convert eigenvectors (sample-space) to principal components (feature-space):
        // comp_jj = X_c.T @ v[:,jj], then normalise to unit length.
        let mean_f32: Vec<f32> = mean_f64.iter().map(|&m| m as f32).collect();
        let mut components = Vec::with_capacity(k);
        for jj in 0..k {
            let mut comp = vec![0.0f64; d];
            for i in 0..n {
                let vi = v[i * k + jj];
                for (j, c) in comp.iter_mut().enumerate() {
                    *c += x_c[i][j] * vi;
                }
            }
            let norm: f64 = comp.iter().map(|&c| c * c).sum::<f64>().sqrt();
            if norm > 1e-12 {
                for c in comp.iter_mut() {
                    *c /= norm;
                }
            }
            components.push(comp.into_iter().map(|c| c as f32).collect());
        }

        Ok(Self {
            n_components: k,
            mean: mean_f32,
            components,
        })
    }

    /// Project a batch of samples into PCA space.
    /// Input `[n_samples × n_features]` → output `[n_samples × n_components]`.
    pub fn transform(&self, x: &[Vec<f32>]) -> Vec<Vec<f32>> {
        x.iter().map(|row| self.transform_one(row)).collect()
    }

    fn transform_one(&self, row: &[f32]) -> Vec<f32> {
        self.components
            .iter()
            .map(|comp| {
                row.iter()
                    .zip(self.mean.iter())
                    .zip(comp.iter())
                    .map(|((&v, &m), &c)| (v - m) * c)
                    .sum::<f32>()
            })
            .collect()
    }
}

/// Classical Gram-Schmidt in-place on an `[n × k]` matrix (row-major: `v[i*k+jj]`).
/// Orthonormalises columns 0..k against each other.
fn gram_schmidt(v: &mut [f64], n: usize, k: usize) {
    for jj in 0..k {
        // Orthogonalise against all previous columns.
        for prev in 0..jj {
            let dot: f64 = (0..n).map(|i| v[i * k + jj] * v[i * k + prev]).sum();
            for i in 0..n {
                v[i * k + jj] -= dot * v[i * k + prev];
            }
        }
        // Normalise.
        let norm: f64 = (0..n).map(|i| v[i * k + jj].powi(2)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for i in 0..n {
                v[i * k + jj] /= norm;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pca_reduces_dimension() {
        let x: Vec<Vec<f32>> = (0..50)
            .map(|i| vec![i as f32, (i * 2) as f32, (i * 3) as f32, (i * 4) as f32])
            .collect();
        let reducer = PcaReducer::fit(&x, 2).unwrap();
        assert_eq!(reducer.n_components, 2);
        let projected = reducer.transform(&x);
        assert_eq!(projected.len(), 50);
        assert_eq!(projected[0].len(), 2);
    }

    #[test]
    fn pca_components_are_unit_vectors() {
        let x: Vec<Vec<f32>> = (0..30)
            .map(|i| {
                vec![
                    (i as f32).cos(),
                    (i as f32).sin(),
                    i as f32 * 0.1,
                    i as f32 * 0.3,
                ]
            })
            .collect();
        let reducer = PcaReducer::fit(&x, 2).unwrap();
        for comp in &reducer.components {
            let norm: f32 = comp.iter().map(|&c| c * c).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-4, "component norm = {norm}");
        }
    }

    #[test]
    fn pca_clamps_components_to_n_minus_one() {
        // 5 samples, 10 features, request 20 components → should get 4.
        let x: Vec<Vec<f32>> = (0..5)
            .map(|i| (0..10).map(|j| (i * j) as f32).collect())
            .collect();
        let reducer = PcaReducer::fit(&x, 20).unwrap();
        assert_eq!(reducer.n_components, 4);
    }

    #[test]
    fn pca_train_transform_shapes_consistent() {
        let n_train = 60;
        let d = 16;
        let k = 4;
        let x: Vec<Vec<f32>> = (0..n_train)
            .map(|i| (0..d).map(|j| (i * j) as f32 * 0.01).collect())
            .collect();
        let reducer = PcaReducer::fit(&x, k).unwrap();
        let out = reducer.transform(&x);
        assert_eq!(out.len(), n_train);
        assert_eq!(out[0].len(), k);
    }
}
