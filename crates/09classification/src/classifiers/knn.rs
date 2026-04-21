use std::collections::HashMap;

use anyhow::{Result, bail};
use ndarray::{Array1, Array2};

/// Distance metric used to rank neighbors.
///
/// `Cosine` is 1 - cos(θ), useful when feature magnitude varies across samples
/// (common for raw CNN embeddings).
///
/// `Mahalanobis` whitens features against the training-set covariance before
/// computing Euclidean distance. Robust when feature dimensions are correlated
/// or scaled very differently. Uses full covariance — requires `feat_dim^2`
/// memory and `O(n * feat_dim^2 + feat_dim^3)` fit time, so only practical up
/// to a few thousand dimensions.
///
/// `MahalanobisDiag` is the diagonal approximation (a.k.a. standardised
/// Euclidean): each dimension is divided by its training-set standard
/// deviation. Scales to arbitrary feat_dim at `O(n * feat_dim)` cost.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceMetric {
    Euclidean,
    Cosine,
    Mahalanobis,
    MahalanobisDiag,
}

#[derive(Debug, Clone)]
pub struct KnnConfig {
    pub num_neighbors: usize,
    pub metric: DistanceMetric,
    /// Weight neighbor votes by `1 / (dist + eps)`.
    pub distance_weighted: bool,
    /// Ledoit-Wolf-style shrinkage for Mahalanobis covariance:
    /// `S_reg = (1 - λ) * S + λ * diag(S_diag)`. Always applied with a small
    /// floor (`λ >= 1e-6`) so the matrix stays positive-definite. Ignored by
    /// non-Mahalanobis metrics.
    pub mahalanobis_shrinkage: f32,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self {
            num_neighbors: 3,
            metric: DistanceMetric::Euclidean,
            distance_weighted: false,
            mahalanobis_shrinkage: 1e-3,
        }
    }
}

/// Pre-computed whitening applied to both training and query vectors so that
/// distance reduces to plain Euclidean in the transformed space.
enum Whitening {
    /// Full transform: query' = inv_L · (query - mean).
    Full {
        mean: Array1<f32>,
        /// L^-1 where L is the lower Cholesky factor of the regularised
        /// covariance. Stored as [feat_dim, feat_dim] f32.
        inv_l: Array2<f32>,
    },
    /// Diagonal: scale per dim by 1 / std(d). Mean is irrelevant since it
    /// cancels out in pairwise differences.
    Diagonal { inv_std: Array1<f32> },
}

pub struct KNN {
    config: KnnConfig,
    train_x: Vec<Vec<f32>>,
    train_y: Vec<i32>,
    feat_dim: Option<usize>,
    whitening: Option<Whitening>,
}

impl KNN {
    pub fn new(config: KnnConfig) -> Self {
        Self {
            config,
            train_x: Vec::new(),
            train_y: Vec::new(),
            feat_dim: None,
            whitening: None,
        }
    }

    pub fn config(&self) -> &KnnConfig {
        &self.config
    }

    pub fn num_training_samples(&self) -> usize {
        self.train_x.len()
    }

    pub fn feat_dim(&self) -> Option<usize> {
        self.feat_dim
    }

    /// Store training samples. All rows of `x` must share one length; `y` must
    /// match `x.len()`. For Mahalanobis metrics this also builds the whitening
    /// transform and applies it to the stored training set.
    pub fn fit(&mut self, x: Vec<Vec<f32>>, y: Vec<i32>) -> Result<()> {
        if x.len() != y.len() {
            bail!("fit: x has {} rows but y has {}", x.len(), y.len());
        }
        if x.is_empty() {
            bail!("fit: training set is empty");
        }
        let dim = x[0].len();
        if dim == 0 {
            bail!("fit: feature vectors have length 0");
        }
        for (i, row) in x.iter().enumerate() {
            if row.len() != dim {
                bail!(
                    "fit: inconsistent feature dim at row {} ({} vs {})",
                    i,
                    row.len(),
                    dim
                );
            }
        }
        if self.config.num_neighbors == 0 {
            bail!("fit: num_neighbors must be >= 1");
        }
        if self.config.num_neighbors > x.len() {
            bail!(
                "fit: num_neighbors ({}) exceeds training set size ({})",
                self.config.num_neighbors,
                x.len()
            );
        }

        let (train_x, whitening) = match self.config.metric {
            DistanceMetric::Mahalanobis => {
                let w = build_full_whitening(&x, self.config.mahalanobis_shrinkage)?;
                let whitened = apply_whitening(&x, &w);
                (whitened, Some(w))
            }
            DistanceMetric::MahalanobisDiag => {
                let w = build_diag_whitening(&x, self.config.mahalanobis_shrinkage);
                let whitened = apply_whitening(&x, &w);
                (whitened, Some(w))
            }
            _ => (x, None),
        };

        self.train_x = train_x;
        self.train_y = y;
        self.feat_dim = Some(dim);
        self.whitening = whitening;
        Ok(())
    }

    pub fn predict(&self, x: &[f32]) -> Result<i32> {
        let Some(dim) = self.feat_dim else {
            bail!("predict: model not fitted");
        };
        if x.len() != dim {
            bail!("predict: feature dim mismatch ({} vs {})", x.len(), dim);
        }

        // Metric seen by the distance fn: Mahalanobis* reduces to Euclidean
        // after whitening, so we swap it out once here.
        let effective_metric = match self.config.metric {
            DistanceMetric::Mahalanobis | DistanceMetric::MahalanobisDiag => {
                DistanceMetric::Euclidean
            }
            m => m,
        };

        let query_owned: Vec<f32>;
        let query: &[f32] = match &self.whitening {
            Some(w) => {
                query_owned = whiten_vector(x, w);
                &query_owned
            }
            None => x,
        };

        let mut dists: Vec<(f32, i32)> = self
            .train_x
            .iter()
            .zip(self.train_y.iter())
            .map(|(row, &y)| (distance(query, row, effective_metric), y))
            .collect();

        let k = self.config.num_neighbors;
        dists.select_nth_unstable_by(k - 1, |a, b| {
            a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        let neighbors = &dists[..k];

        let mut votes: HashMap<i32, f32> = HashMap::new();
        for &(d, y) in neighbors {
            let w = if self.config.distance_weighted {
                1.0 / (d + 1e-8)
            } else {
                1.0
            };
            *votes.entry(y).or_insert(0.0) += w;
        }

        // Argmax over votes. Tie-break by smaller label id for determinism.
        votes
            .into_iter()
            .max_by(|a, b| {
                a.1.partial_cmp(&b.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| b.0.cmp(&a.0))
            })
            .map(|(label, _)| label)
            .ok_or_else(|| anyhow::anyhow!("predict: no neighbors"))
    }

    pub fn predict_batch(&self, xs: &[Vec<f32>]) -> Result<Vec<i32>> {
        xs.iter().map(|x| self.predict(x)).collect()
    }
}

fn distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean
        | DistanceMetric::Mahalanobis
        | DistanceMetric::MahalanobisDiag => {
            // Mahalanobis* reach here only if misrouted — treat as Euclidean.
            let mut acc = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                let d = x - y;
                acc += d * d;
            }
            acc.sqrt()
        }
        DistanceMetric::Cosine => {
            let mut dot = 0.0f32;
            let mut na = 0.0f32;
            let mut nb = 0.0f32;
            for (x, y) in a.iter().zip(b.iter()) {
                dot += x * y;
                na += x * x;
                nb += y * y;
            }
            let denom = (na.sqrt() * nb.sqrt()).max(1e-8);
            1.0 - dot / denom
        }
    }
}

// ---- Mahalanobis whitening --------------------------------------------------

/// Per-dim 1/std scaling. Any dim with std == 0 gets inv_std = 0, effectively
/// dropping it from the distance.
fn build_diag_whitening(x: &[Vec<f32>], shrinkage: f32) -> Whitening {
    let n = x.len();
    let d = x[0].len();
    let lam = shrinkage.max(1e-6) as f64;

    let mut mean = vec![0.0f64; d];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            mean[j] += v as f64;
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }

    let mut var = vec![0.0f64; d];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            let c = v as f64 - mean[j];
            var[j] += c * c;
        }
    }
    let denom = (n.saturating_sub(1)).max(1) as f64;
    for v in &mut var {
        *v = (1.0 - lam) * (*v / denom) + lam;
    }

    let inv_std: Array1<f32> = Array1::from_iter(var.iter().map(|&v| {
        if v > 0.0 {
            (1.0 / v.sqrt()) as f32
        } else {
            0.0
        }
    }));
    Whitening::Diagonal { inv_std }
}

/// Full Mahalanobis whitening via regularised covariance + Cholesky.
///
/// Regularisation: `S_reg = (1 - λ) * S + λ * diag(S)`. Shrinks off-diagonal
/// entries toward zero, preserves variance scale. Keeps S positive-definite
/// even when n <= feat_dim (which is common for fMRI).
fn build_full_whitening(x: &[Vec<f32>], shrinkage: f32) -> Result<Whitening> {
    let n = x.len();
    let d = x[0].len();
    let lam = shrinkage.clamp(1e-6, 1.0) as f64;

    // Mean.
    let mut mean_f64 = vec![0.0f64; d];
    for row in x {
        for (j, &v) in row.iter().enumerate() {
            mean_f64[j] += v as f64;
        }
    }
    for m in &mut mean_f64 {
        *m /= n as f64;
    }

    // Centered data as [n, d] f64.
    let mut centered = Array2::<f64>::zeros((n, d));
    for (i, row) in x.iter().enumerate() {
        for (j, &v) in row.iter().enumerate() {
            centered[[i, j]] = v as f64 - mean_f64[j];
        }
    }

    // Covariance S = (C^T C) / (n - 1).
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let mut cov = centered.t().dot(&centered);
    cov.mapv_inplace(|v| v / denom);

    // Regularise toward diagonal.
    let diag: Vec<f64> = (0..d).map(|i| cov[[i, i]]).collect();
    cov.mapv_inplace(|v| (1.0 - lam) * v);
    for i in 0..d {
        cov[[i, i]] += lam * diag[i];
    }

    // Cholesky: S_reg = L L^T (lower triangular).
    let l = cholesky_lower(cov).map_err(|e| {
        anyhow::anyhow!(
            "mahalanobis: covariance not positive-definite ({}); raise shrinkage",
            e
        )
    })?;

    // Invert lower-triangular L via forward substitution.
    let inv_l_f64 = inv_lower_triangular(&l)?;
    let inv_l = inv_l_f64.mapv(|v| v as f32);
    let mean = Array1::from_iter(mean_f64.iter().map(|&v| v as f32));
    Ok(Whitening::Full { mean, inv_l })
}

fn apply_whitening(x: &[Vec<f32>], w: &Whitening) -> Vec<Vec<f32>> {
    x.iter().map(|row| whiten_vector(row, w)).collect()
}

fn whiten_vector(x: &[f32], w: &Whitening) -> Vec<f32> {
    match w {
        Whitening::Diagonal { inv_std } => {
            x.iter().zip(inv_std.iter()).map(|(v, s)| v * s).collect()
        }
        Whitening::Full { mean, inv_l } => {
            // out = inv_l · (x - mean); inv_l is lower-triangular so inner
            // loop short-circuits at the diagonal.
            let d = x.len();
            let mut out = vec![0.0f32; d];
            for i in 0..d {
                let mut acc = 0.0f32;
                for j in 0..=i {
                    acc += inv_l[[i, j]] * (x[j] - mean[j]);
                }
                out[i] = acc;
            }
            out
        }
    }
}

/// Hand-rolled in-place Cholesky decomposition for small-to-medium matrices.
/// Returns the lower-triangular factor `L` such that `A = L L^T` (upper triangle zeroed).
fn cholesky_lower(mut a: Array2<f64>) -> Result<Array2<f64>> {
    let (rows, cols) = a.dim();
    if rows != cols {
        bail!("cholesky: non-square {}x{}", rows, cols);
    }
    let n = rows;
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[[i, j]];
            for k in 0..j {
                s -= a[[i, k]] * a[[j, k]];
            }
            if i == j {
                if s <= 0.0 || !s.is_finite() {
                    bail!("cholesky: diagonal {} not positive ({})", i, s);
                }
                a[[i, i]] = s.sqrt();
            } else {
                a[[i, j]] = s / a[[j, j]];
            }
        }
        for j in (i + 1)..n {
            a[[i, j]] = 0.0;
        }
    }
    Ok(a)
}

/// Forward-substitution inverse of a lower-triangular `L`. Returns `L^-1`
/// (also lower-triangular). Errors if any diagonal entry is zero.
fn inv_lower_triangular(l: &Array2<f64>) -> Result<Array2<f64>> {
    let n = l.nrows();
    let mut inv = Array2::<f64>::zeros((n, n));
    for col in 0..n {
        // Solve L · y = e_col for y, where e_col is the unit vector.
        for i in col..n {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for k in col..i {
                s -= l[[i, k]] * inv[[k, col]];
            }
            let diag = l[[i, i]];
            if diag == 0.0 {
                bail!("inv_lower_triangular: zero diagonal at {}", i);
            }
            inv[[i, col]] = s / diag;
        }
    }
    Ok(inv)
}

/// Fraction of correct predictions.
pub fn accuracy(y_true: &[i32], y_pred: &[i32]) -> f32 {
    if y_true.is_empty() {
        return 0.0;
    }
    let hits = y_true
        .iter()
        .zip(y_pred.iter())
        .filter(|(a, b)| a == b)
        .count();
    hits as f32 / y_true.len() as f32
}

/// 2x2 confusion matrix for binary labels {0, 1}.
/// Row = true, col = pred. Returns `[[TN, FP], [FN, TP]]`.
pub fn confusion_matrix_binary(y_true: &[i32], y_pred: &[i32]) -> [[u32; 2]; 2] {
    let mut cm = [[0u32; 2]; 2];
    for (&t, &p) in y_true.iter().zip(y_pred.iter()) {
        if let (0..=1, 0..=1) = (t, p) {
            cm[t as usize][p as usize] += 1;
        }
    }
    cm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linearly_separable_points() {
        let x = vec![
            vec![0.0, 0.0],
            vec![0.1, 0.0],
            vec![0.0, 0.1],
            vec![10.0, 10.0],
            vec![10.1, 10.0],
            vec![10.0, 10.1],
        ];
        let y = vec![0, 0, 0, 1, 1, 1];

        let mut knn = KNN::new(KnnConfig {
            num_neighbors: 3,
            metric: DistanceMetric::Euclidean,
            distance_weighted: false,
            mahalanobis_shrinkage: 1e-3,
        });
        knn.fit(x, y).unwrap();

        assert_eq!(knn.predict(&[0.05, 0.05]).unwrap(), 0);
        assert_eq!(knn.predict(&[9.9, 10.0]).unwrap(), 1);
    }

    #[test]
    fn cosine_distance_handles_magnitude() {
        let x = vec![
            vec![1.0, 0.0],
            vec![2.0, 0.0],
            vec![0.0, 1.0],
            vec![0.0, 5.0],
        ];
        let y = vec![0, 0, 1, 1];

        let mut knn = KNN::new(KnnConfig {
            num_neighbors: 1,
            metric: DistanceMetric::Cosine,
            distance_weighted: false,
            mahalanobis_shrinkage: 1e-3,
        });
        knn.fit(x, y).unwrap();

        assert_eq!(knn.predict(&[100.0, 0.0]).unwrap(), 0);
        assert_eq!(knn.predict(&[0.0, 100.0]).unwrap(), 1);
    }

    #[test]
    fn accuracy_and_confusion() {
        let t = vec![0, 1, 0, 1, 1];
        let p = vec![0, 1, 1, 1, 0];
        assert!((accuracy(&t, &p) - 0.6).abs() < 1e-6);
        let cm = confusion_matrix_binary(&t, &p);
        assert_eq!(cm, [[1, 1], [1, 2]]);
    }

    #[test]
    fn mahalanobis_diag_equalises_dim_scales() {
        // Two classes differ by equal-variance signal on *both* dims, but dim 0
        // has ~100x more noise. Plain Euclidean lets dim-0 noise dominate and
        // mis-classifies queries near the class 1 prototype. Diagonal
        // Mahalanobis rescales dim 0 down, restoring separation.
        let x = vec![
            // Class 0 around (0, 0) with dim-0 noise and a small dim-1 offset.
            vec![-100.0, -0.5],
            vec![100.0, -0.4],
            vec![-90.0, -0.3],
            vec![95.0, -0.6],
            // Class 1 around (0, +1) with same dim-0 noise.
            vec![-98.0, 0.7],
            vec![102.0, 0.8],
            vec![-93.0, 0.9],
            vec![97.0, 1.0],
        ];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut knn = KNN::new(KnnConfig {
            num_neighbors: 3,
            metric: DistanceMetric::MahalanobisDiag,
            distance_weighted: false,
            mahalanobis_shrinkage: 1e-3,
        });
        knn.fit(x, y).unwrap();

        // Query is near class-1 centroid on dim 1, dim-0 noise should not matter.
        assert_eq!(knn.predict(&[150.0, 0.9]).unwrap(), 1);
        assert_eq!(knn.predict(&[-150.0, -0.4]).unwrap(), 0);
    }

    #[test]
    fn mahalanobis_full_whitens_correlated_dims() {
        // Class 0 around origin, class 1 shifted along (1, 1). Dim 0 and dim 1
        // are highly correlated within each class, and the separating axis
        // runs along the principal direction. Full Mahalanobis rotates the
        // space so the decision boundary becomes axis-aligned.
        let x = vec![
            vec![-1.0, -1.05],
            vec![1.0, 0.95],
            vec![2.0, 2.05],
            vec![-2.0, -1.95],
            vec![4.0, 4.1],
            vec![6.0, 6.05],
            vec![7.0, 6.95],
            vec![5.0, 5.05],
        ];
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1];

        let mut knn = KNN::new(KnnConfig {
            num_neighbors: 3,
            metric: DistanceMetric::Mahalanobis,
            distance_weighted: false,
            mahalanobis_shrinkage: 1e-2,
        });
        knn.fit(x, y).unwrap();

        assert_eq!(knn.predict(&[0.5, 0.5]).unwrap(), 0);
        assert_eq!(knn.predict(&[5.5, 5.5]).unwrap(), 1);
    }

    #[test]
    fn cholesky_and_inverse_roundtrip() {
        let a = ndarray::array![[4.0, 2.0, 0.4], [2.0, 3.0, 0.5], [0.4, 0.5, 1.0],];
        let l = cholesky_lower(a.clone()).unwrap();
        let reconstructed = l.dot(&l.t());
        for i in 0..3 {
            for j in 0..3 {
                assert!((reconstructed[[i, j]] - a[[i, j]]).abs() < 1e-10);
            }
        }
        let inv_l = inv_lower_triangular(&l).unwrap();
        let product = inv_l.dot(&l);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((product[[i, j]] - expected).abs() < 1e-10);
            }
        }
    }
}
