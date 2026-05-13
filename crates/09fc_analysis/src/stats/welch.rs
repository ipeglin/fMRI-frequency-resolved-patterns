use ndarray::{Array2, ArrayView3};

/// Edge-wise Welch t-statistic map from Fisher-Z subject stacks.
/// `z_stack`: [N_subj, C, C] in row-major order; `labels`: true = anhedonic.
/// Returns zeros for any edge if either group has fewer than 2 subjects.
pub fn welch_t_map(z_stack: ArrayView3<f64>, labels: &[bool]) -> Array2<f64> {
    let n = labels.len();
    let c = z_stack.shape()[1];
    let n_a = labels.iter().filter(|&&l| l).count();
    let n_c = n - n_a;

    let mut t_map = Array2::<f64>::zeros((c, c));
    if n_a < 2 || n_c < 2 {
        return t_map;
    }

    for i in 0..c {
        for j in i..c {
            let t = welch_t_edge(z_stack, labels, i, j, n, n_a, n_c);
            t_map[[i, j]] = t;
            t_map[[j, i]] = t;
        }
    }
    t_map
}

fn welch_t_edge(
    z_stack: ArrayView3<f64>,
    labels: &[bool],
    i: usize,
    j: usize,
    n: usize,
    n_a: usize,
    n_c: usize,
) -> f64 {
    let mut sum_a = 0.0f64;
    let mut sum_c = 0.0f64;
    for k in 0..n {
        let v = z_stack[[k, i, j]];
        if labels[k] {
            sum_a += v;
        } else {
            sum_c += v;
        }
    }
    let mean_a = sum_a / n_a as f64;
    let mean_c = sum_c / n_c as f64;

    let mut var_a = 0.0f64;
    let mut var_c = 0.0f64;
    for k in 0..n {
        let v = z_stack[[k, i, j]];
        let d = if labels[k] { v - mean_a } else { v - mean_c };
        if labels[k] {
            var_a += d * d;
        } else {
            var_c += d * d;
        }
    }
    let var_a = var_a / (n_a - 1) as f64;
    let var_c = var_c / (n_c - 1) as f64;

    let se = (var_a / n_a as f64 + var_c / n_c as f64).sqrt();
    if se == 0.0 {
        0.0
    } else {
        (mean_a - mean_c) / se
    }
}
