/// Benjamini-Hochberg FDR correction. Returns q-values in original index order.
pub fn bh_fdr(p_values: &[f64]) -> Vec<f64> {
    let n = p_values.len();
    if n == 0 {
        return vec![];
    }
    let mut indexed: Vec<(usize, f64)> =
        p_values.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let mut q = vec![1.0f64; n];
    let mut min_q = 1.0f64;
    for rank in (0..n).rev() {
        let (orig_idx, p) = indexed[rank];
        let q_val = (p * n as f64 / (rank + 1) as f64).min(1.0);
        min_q = min_q.min(q_val);
        q[orig_idx] = min_q;
    }
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bh_fdr_monotone() {
        let p = vec![0.001, 0.01, 0.05, 0.1, 0.5];
        let q = bh_fdr(&p);
        assert!(
            q.windows(2).all(|w| w[0] <= w[1]),
            "q-values not monotone: {:?}",
            q
        );
    }

    #[test]
    fn test_bh_fdr_empty() {
        assert!(bh_fdr(&[]).is_empty());
    }
}
