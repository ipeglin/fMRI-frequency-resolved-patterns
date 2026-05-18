use ndarray::{Array2, Array3, ArrayView3};
use rand::SeedableRng;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha20Rng;
use rayon::prelude::*;
use std::sync::Arc;

use super::fdr::bh_fdr;
use super::nbs::{find_components, max_component_size};
use super::welch::welch_t_map;

pub struct PermResult {
    pub obs_t: Array2<f64>,
    /// Per-edge two-sided permutation p-value.
    pub p_uncorr: Array2<f64>,
    /// FWER via max-|t| null distribution.
    pub p_fwer: Array2<f64>,
    /// BH-FDR on upper-triangle p_uncorr, mirrored symmetrically.
    pub q_fdr: Array2<f64>,
    /// True where significant NBS components (p < 0.05) exist.
    pub nbs_component_mask: Array2<bool>,
    /// One p-value per observed suprathreshold component.
    pub nbs_component_p: Vec<f64>,
    pub n_anhedonic: usize,
    pub n_control: usize,
}

pub fn run_permutation(
    z_stack: ArrayView3<f64>,
    labels: &[bool],
    n_perm: u32,
    seed: u64,
    primary_t: f64,
) -> PermResult {
    let c = z_stack.shape()[1];
    let obs_t = welch_t_map(z_stack, labels);
    let obs_t_abs = obs_t.mapv(f64::abs);
    let obs_supr = obs_t.mapv(|v| v.abs() >= primary_t);
    let obs_components = find_components(obs_supr.view());
    let obs_comp_sizes: Vec<usize> = obs_components.iter().map(|comp| comp.size()).collect();

    // Clone data into Arc so it can be shared across rayon threads.
    let z_arc: Arc<Array3<f64>> = Arc::new(z_stack.to_owned());
    let obs_t_abs_arc = Arc::new(obs_t_abs.clone());
    let labels_arc = Arc::new(labels.to_vec());

    let (count_extreme, max_t_nulls, max_comp_nulls): (Array2<u32>, Vec<f64>, Vec<usize>) =
        (0..n_perm)
            .into_par_iter()
            .map(|perm_idx| {
                let z = z_arc.view();
                let obs_abs = obs_t_abs_arc.view();

                let mut rng = ChaCha20Rng::seed_from_u64(
                    seed.wrapping_add((perm_idx as u64).wrapping_mul(6_364_136_223_846_793_005)),
                );
                let mut perm_labels = (*labels_arc).clone();
                perm_labels.shuffle(&mut rng);

                let t_null = welch_t_map(z, &perm_labels);
                let max_t = t_null.iter().map(|v| v.abs()).fold(f64::NEG_INFINITY, f64::max);
                let supr_null = t_null.mapv(|v| v.abs() >= primary_t);
                let max_comp = max_component_size(supr_null.view());

                let count_row: Array2<u32> = ndarray::Zip::from(&t_null)
                    .and(obs_abs)
                    .map_collect(|&tv, &ov| (tv.abs() >= ov) as u32);

                (count_row, max_t, max_comp)
            })
            .fold(
                || {
                    (
                        Array2::<u32>::zeros((c, c)),
                        Vec::<f64>::new(),
                        Vec::<usize>::new(),
                    )
                },
                |(mut counts, mut mts, mut mcs), (row, mt, mc)| {
                    counts += &row;
                    mts.push(mt);
                    mcs.push(mc);
                    (counts, mts, mcs)
                },
            )
            .reduce(
                || {
                    (
                        Array2::<u32>::zeros((c, c)),
                        Vec::<f64>::new(),
                        Vec::<usize>::new(),
                    )
                },
                |(mut c1, mut m1, mut mc1), (c2, m2, mc2)| {
                    c1 += &c2;
                    m1.extend(m2);
                    mc1.extend(mc2);
                    (c1, m1, mc1)
                },
            );

    let b = n_perm as f64;
    let p_uncorr = count_extreme.mapv(|v| (v as f64 + 1.0) / (b + 1.0));

    let p_fwer = obs_t_abs.mapv(|ov| {
        let exceed = max_t_nulls.iter().filter(|&&mt| mt >= ov).count();
        (exceed as f64 + 1.0) / (b + 1.0)
    });

    // BH-FDR on upper triangle.
    let upper_coords: Vec<(usize, usize)> =
        (0..c).flat_map(|i| ((i + 1)..c).map(move |j| (i, j))).collect();
    let upper_p: Vec<f64> = upper_coords.iter().map(|&(i, j)| p_uncorr[[i, j]]).collect();
    let upper_q = bh_fdr(&upper_p);
    let mut q_fdr = Array2::<f64>::ones((c, c));
    for (k, &(i, j)) in upper_coords.iter().enumerate() {
        q_fdr[[i, j]] = upper_q[k];
        q_fdr[[j, i]] = upper_q[k];
    }

    // NBS component p-values.
    let nbs_component_p: Vec<f64> = obs_comp_sizes
        .iter()
        .map(|&obs_size| {
            let exceed = max_comp_nulls.iter().filter(|&&mc| mc >= obs_size).count();
            (exceed as f64 + 1.0) / (b + 1.0)
        })
        .collect();

    // Mark edges in significant components (p < 0.05).
    let mut nbs_component_mask = Array2::<bool>::from_elem((c, c), false);
    for (idx, comp) in obs_components.iter().enumerate() {
        if nbs_component_p[idx] < 0.05 {
            for &(i, j) in &comp.edges {
                nbs_component_mask[[i, j]] = true;
                nbs_component_mask[[j, i]] = true;
            }
        }
    }

    let n_anhedonic = labels.iter().filter(|&&l| l).count();
    let n_control = labels.len() - n_anhedonic;

    PermResult {
        obs_t,
        p_uncorr,
        p_fwer,
        q_fdr,
        nbs_component_mask,
        nbs_component_p,
        n_anhedonic,
        n_control,
    }
}
