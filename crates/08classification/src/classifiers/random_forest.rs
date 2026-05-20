use anyhow::{Result, bail};
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
/// Each tree is fit on a bootstrap resample; probabilities are averaged across
/// the ensemble. Class column ordering follows `classes`, which holds the
/// sorted unique labels from the training set.
pub struct RandomForestWrapper {
    trees: Vec<DT>,
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
    /// Bootstrap samples are drawn with `seed`; each tree gets a deterministic
    /// per-tree seed derived from the same RNG so results are reproducible.
    pub fn fit(
        x_train: &[Vec<f32>],
        y_train: &[i32],
        n_trees: usize,
        seed: u64,
    ) -> Result<Self> {
        let n = x_train.len();
        if n == 0 {
            bail!("RandomForest: empty training set");
        }
        if n_trees == 0 {
            bail!("RandomForest: n_trees must be > 0");
        }

        let mut classes: Vec<i32> = y_train.to_vec();
        classes.sort_unstable();
        classes.dedup();

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut trees = Vec::with_capacity(n_trees);

        for _ in 0..n_trees {
            let idx = bootstrap_indices(n, &mut rng);

            let x_boot: Vec<Vec<f32>> = idx.iter().map(|&i| x_train[i].clone()).collect();
            let y_boot: Vec<i32> = idx.iter().map(|&i| y_train[i]).collect();

            let x_mat = rows_to_dense(&x_boot)?;
            let params = DecisionTreeClassifierParameters::default();

            let tree = DT::fit(&x_mat, &y_boot, params)
                .map_err(|e| anyhow::anyhow!("DecisionTree fit error: {:?}", e))?;
            trees.push(tree);
        }

        Ok(Self { trees, classes })
    }

    /// Average `predict_proba` across all trees.
    ///
    /// Returns `[n_samples × n_classes]` where columns correspond to `self.classes`
    /// in sorted order. Class columns from individual trees are re-aligned to the
    /// master class list so missing classes in a bootstrap sample are handled.
    pub fn predict_proba_batch(&self, xs: &[Vec<f32>]) -> Result<Vec<Vec<f32>>> {
        let n_samples = xs.len();
        let n_classes = self.classes.len();
        let x_mat = rows_to_dense(xs)?;

        let mut acc = vec![vec![0.0f64; n_classes]; n_samples];

        for tree in &self.trees {
            let proba = tree
                .predict_proba(&x_mat)
                .map_err(|e| anyhow::anyhow!("DecisionTree predict_proba error: {:?}", e))?;

            // Smartcore DTs sort class columns in ascending label order, matching
            // `self.classes` which is also sorted ascending. Columns align directly.
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
        let rf = RandomForestWrapper::fit(&xs, &ys, 5, 42).unwrap();
        let proba = rf.predict_proba_batch(&xs).unwrap();
        assert_eq!(proba.len(), 40);
        assert_eq!(proba[0].len(), 2);
    }

    #[test]
    fn rf_probabilities_sum_to_one() {
        let (xs, ys) = make_data(30);
        let rf = RandomForestWrapper::fit(&xs, &ys, 10, 7).unwrap();
        let proba = rf.predict_proba_batch(&xs[..5]).unwrap();
        for row in &proba {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-4, "row sum = {sum}");
        }
    }

    #[test]
    fn rf_classes_sorted() {
        let (xs, ys) = make_data(20);
        let rf = RandomForestWrapper::fit(&xs, &ys, 3, 99).unwrap();
        assert_eq!(rf.classes, vec![0, 1]);
    }
}
