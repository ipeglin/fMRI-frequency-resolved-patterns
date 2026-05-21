use anyhow::{Result, bail};
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::tree::decision_tree_classifier::{
    DecisionTreeClassifier, DecisionTreeClassifierParameters,
};

type DT = DecisionTreeClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>;

/// Bootstrap-bagged ensemble of `DecisionTreeClassifier` trees from smartcore.
///
/// Each tree is fit on a bootstrap resample of a random feature subset; probabilities
/// are averaged across the ensemble. Class column ordering follows `classes`, which
/// holds the sorted unique labels from the training set.
pub struct RandomForestWrapper {
    trees: Vec<(DT, Vec<usize>)>,
    /// Sorted unique class labels in the order corresponding to probability columns.
    pub classes: Vec<i32>,
}

fn rows_to_dense(x: &[Vec<f32>]) -> Result<DenseMatrix<f64>> {
    let rows: Vec<Vec<f64>> = x
        .iter()
        .map(|r| r.iter().map(|&v| v as f64).collect())
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
    DenseMatrix::from_2d_array(&refs)
        .map_err(|e| anyhow::anyhow!("DenseMatrix construction failed: {:?}", e))
}

fn bootstrap_indices(n: usize, rng: &mut ChaCha8Rng) -> Vec<usize> {
    (0..n).map(|_| rng.gen_range(0..n)).collect()
}

impl RandomForestWrapper {
    /// Fit a bagged ensemble of `n_trees` decision trees on `x_train` / `y_train`.
    ///
    /// Each tree is trained on a bootstrap resample using a random feature subset
    /// of size `sqrt(p)` when `feature_subsample_ratio <= 0.0`, or
    /// `floor(p * feature_subsample_ratio)` otherwise.
    pub fn fit(
        x_train: &[Vec<f32>],
        y_train: &[i32],
        n_trees: usize,
        seed: u64,
        feature_subsample_ratio: f32,
    ) -> Result<Self> {
        let n = x_train.len();
        if n == 0 {
            bail!("RandomForest: empty training set");
        }
        if n_trees == 0 {
            bail!("RandomForest: n_trees must be > 0");
        }

        let n_features = x_train[0].len();
        let m = if feature_subsample_ratio <= 0.0 {
            ((n_features as f64).sqrt().floor() as usize).max(1)
        } else {
            ((n_features as f32 * feature_subsample_ratio).floor() as usize)
                .max(1)
                .min(n_features)
        };

        let mut classes: Vec<i32> = y_train.to_vec();
        classes.sort_unstable();
        classes.dedup();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut trees = Vec::with_capacity(n_trees);

        for _ in 0..n_trees {
            let mut all_cols: Vec<usize> = (0..n_features).collect();
            all_cols.shuffle(&mut rng);
            all_cols.truncate(m);
            all_cols.sort_unstable();
            let col_idx = all_cols;

            let idx = bootstrap_indices(n, &mut rng);
            let x_boot: Vec<Vec<f32>> = idx
                .iter()
                .map(|&i| col_idx.iter().map(|&c| x_train[i][c]).collect())
                .collect();
            let y_boot: Vec<i32> = idx.iter().map(|&i| y_train[i]).collect();

            let x_mat = rows_to_dense(&x_boot)?;
            let params = DecisionTreeClassifierParameters::default();

            let tree = DT::fit(&x_mat, &y_boot, params)
                .map_err(|e| anyhow::anyhow!("DecisionTree fit error: {:?}", e))?;
            trees.push((tree, col_idx));
        }

        Ok(Self { trees, classes })
    }

    /// Average `predict_proba` across all trees.
    ///
    /// Returns `[n_samples × n_classes]` where columns correspond to `self.classes`
    /// in sorted order. Each tree receives only its own column subset; output is
    /// averaged across the ensemble.
    pub fn predict_proba_batch(&self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_samples = xs.len();
        let n_classes = self.classes.len();

        let mut acc = vec![vec![0.0f64; n_classes]; n_samples];

        for (tree, col_idx) in &self.trees {
            let x_sliced: Vec<Vec<f32>> = xs
                .iter()
                .map(|row| col_idx.iter().map(|&c| row[c]).collect())
                .collect();
            let x_mat = rows_to_dense(&x_sliced)?;
            let proba = tree
                .predict_proba(&x_mat)
                .map_err(|e| anyhow::anyhow!("DecisionTree predict_proba error: {:?}", e))?;

            for (i, row) in acc.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    *cell += proba.get((i, j));
                }
            }
        }

        let scale = 1.0 / self.trees.len() as f64;
        Ok(acc
            .into_iter()
            .map(|row| row.into_iter().map(|v| (v * scale) as f32).collect())
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize) -> (Vec<Vec<f32>>, Vec<i32>) {
        let xs: Vec<Vec<f32>> = (0..n)
            .map(|i| vec![i as f32, (i % 3) as f32, (i * 2) as f32])
            .collect();
        let ys: Vec<i32> = (0..n).map(|i| (i % 2) as i32).collect();
        (xs, ys)
    }

    #[test]
    fn rf_fit_and_predict_shape() {
        let (xs, ys) = make_data(40);
        let rf = RandomForestWrapper::fit(&xs, &ys, 5, 42, 0.0).unwrap();
        let proba = rf.predict_proba_batch(&xs).unwrap();
        assert_eq!(proba.len(), 40);
        assert_eq!(proba[0].len(), 2);
    }

    #[test]
    fn rf_probabilities_sum_to_one() {
        let (xs, ys) = make_data(30);
        let rf = RandomForestWrapper::fit(&xs, &ys, 10, 7, 0.0).unwrap();
        let proba = rf.predict_proba_batch(&xs[..5]).unwrap();
        for row in &proba {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row sum = {sum}");
        }
    }

    #[test]
    fn rf_classes_sorted() {
        let (xs, ys) = make_data(20);
        let rf = RandomForestWrapper::fit(&xs, &ys, 3, 99, 0.0).unwrap();
        assert_eq!(rf.classes, vec![0, 1]);
    }

    #[test]
    fn rf_trees_have_different_column_subsets() {
        // Use 20 features so m = floor(sqrt(20)) = 4; chance all 20 trees
        // pick identical subsets is negligible.
        let xs: Vec<Vec<f32>> = (0..60)
            .map(|i| (0..20).map(|j| (i * j) as f32).collect())
            .collect();
        let ys: Vec<i32> = (0..60).map(|i| (i % 2) as i32).collect();
        let rf = RandomForestWrapper::fit(&xs, &ys, 20, 42, 0.0).unwrap();
        let unique_subsets: std::collections::HashSet<Vec<usize>> =
            rf.trees.iter().map(|(_, cols)| cols.clone()).collect();
        assert!(
            unique_subsets.len() > 1,
            "all trees selected identical feature subsets"
        );
    }

    #[test]
    fn rf_feature_subsampling_probs_sum_to_one() {
        let (xs, ys) = make_data(40);
        let rf = RandomForestWrapper::fit(&xs, &ys, 10, 7, 0.0).unwrap();
        let proba = rf.predict_proba_batch(&xs[..5]).unwrap();
        for row in &proba {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4);
        }
    }
}
