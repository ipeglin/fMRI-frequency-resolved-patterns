use anyhow::Result;
use std::path::Path;
use utils::hdf5_io::{H5Attr, open_or_create, open_or_create_group, write_attrs, write_dataset_old};

use crate::stats::permutation::PermResult;

#[allow(clippy::too_many_arguments)]
pub fn write_analysis_result(
    results_dir: &Path,
    task: &str,
    source: &str,
    level: &str,
    roi_suffix: &str,
    result: &PermResult,
    n_perm: u32,
    primary_t: f64,
    seed: u64,
) -> Result<()> {
    std::fs::create_dir_all(results_dir)?;
    let fname = format!("{task}_{source}_{level}{roi_suffix}.h5");
    let path = results_dir.join(fname);
    let file = open_or_create(&path)?;
    let grp = open_or_create_group(&file, "results", true)?;

    let c = result.obs_t.shape()[0];
    let shape = [c, c];

    write_dataset_old(&grp, "t_map", result.obs_t.as_slice().unwrap(), &shape, None)?;
    write_dataset_old(&grp, "p_uncorr", result.p_uncorr.as_slice().unwrap(), &shape, None)?;
    write_dataset_old(&grp, "p_fwer", result.p_fwer.as_slice().unwrap(), &shape, None)?;
    write_dataset_old(&grp, "q_fdr", result.q_fdr.as_slice().unwrap(), &shape, None)?;

    let mask_u8: Vec<u8> = result.nbs_component_mask.iter().map(|&b| b as u8).collect();
    write_dataset_old(&grp, "nbs_component_mask", &mask_u8, &shape, None)?;
    write_dataset_old(
        &grp,
        "nbs_component_p",
        &result.nbs_component_p,
        &[result.nbs_component_p.len()],
        None,
    )?;

    write_attrs(
        &grp,
        &[
            H5Attr::u32("n_permutations", n_perm),
            H5Attr::f64("primary_t_threshold", primary_t),
            H5Attr::string("permutation_seed", format!("{:#018x}", seed)),
            H5Attr::u32("n_anhedonic", result.n_anhedonic as u32),
            H5Attr::u32("n_control", result.n_control as u32),
            H5Attr::u32("n_channels", c as u32),
            H5Attr::string("task", task.to_string()),
            H5Attr::string("source", source.to_string()),
            H5Attr::string("level", level.to_string()),
        ],
    )?;

    Ok(())
}
