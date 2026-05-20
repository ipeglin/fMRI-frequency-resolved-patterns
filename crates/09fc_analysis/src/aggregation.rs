use anyhow::Result;
use ndarray::Array2;

/// Read a 2D f64 dataset from an HDF5 group. Returns None if group or dataset absent.
pub fn read_fc_matrix(
    file: &hdf5::File,
    group_path: &str,
    sub_group: Option<&str>,
    dataset: &str,
) -> Result<Option<Array2<f64>>> {
    let grp = match file.group(group_path) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let target = match sub_group {
        Some(sg) => match grp.group(sg) {
            Ok(g) => g,
            Err(_) => return Ok(None),
        },
        None => grp,
    };
    let ds = match target.dataset(dataset) {
        Ok(d) => d,
        Err(_) => return Ok(None),
    };
    Ok(Some(read_2d_f64(&ds)?))
}

/// Average FC matrices across `face/block_*` children under `base_path`.
#[allow(dead_code)]
/// `sub_group`: optional sub-path under each block (e.g. `"slow_5"`, `"mode_0"`).
/// `dataset`: dataset name within the (sub-)group (e.g. `"fisher_z"`, `"fisher_z_mean"`).
/// Returns None if the group or face sub-group is absent, or no block children exist.
pub fn aggregate_face_blocks(
    file: &hdf5::File,
    base_path: &str,
    sub_group: Option<&str>,
    dataset: &str,
) -> Result<Option<Array2<f64>>> {
    let root = match file.group(base_path) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };

    // Prefer face/ sub-group, fall back to direct block_* children.
    let block_parent = if let Ok(face) = root.group("face") {
        face
    } else {
        root
    };

    let block_names: Vec<String> = match block_parent.member_names() {
        Ok(names) => names
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect(),
        Err(_) => return Ok(None),
    };

    if block_names.is_empty() {
        return Ok(None);
    }

    let mut sum: Option<Array2<f64>> = None;
    let mut count: Option<Array2<f64>> = None;

    for block_name in &block_names {
        let block_group = match block_parent.group(block_name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let target = match sub_group {
            Some(sg) => match block_group.group(sg) {
                Ok(g) => g,
                Err(_) => continue,
            },
            None => block_group,
        };
        let ds = match target.dataset(dataset) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let mat = read_2d_f64(&ds)?;

        match sum.as_mut() {
            None => {
                let mut s = Array2::<f64>::zeros(mat.dim());
                let mut c = Array2::<f64>::zeros(mat.dim());
                ndarray::Zip::from(&mut s)
                    .and(&mut c)
                    .and(&mat)
                    .for_each(|sv, cv, &v| {
                        if !v.is_nan() {
                            *sv += v;
                            *cv += 1.0;
                        }
                    });
                sum = Some(s);
                count = Some(c);
            }
            Some(s) => {
                let c = count.as_mut().unwrap();
                ndarray::Zip::from(s)
                    .and(c)
                    .and(&mat)
                    .for_each(|sv, cv, &v| {
                        if !v.is_nan() {
                            *sv += v;
                            *cv += 1.0;
                        }
                    });
            }
        }
    }

    match (sum, count) {
        (Some(s), Some(c)) => {
            let mean = ndarray::Zip::from(&s)
                .and(&c)
                .map_collect(|&sv, &cv| if cv == 0.0 { f64::NAN } else { sv / cv });
            Ok(Some(mean))
        }
        _ => Ok(None),
    }
}

/// Average Fisher-z FC across `{condition}/block_*` children under `base_path`.
///
/// Generalisation of `aggregate_face_blocks`: pass `condition = "face"` to reproduce
/// original behaviour, or `"shape"` (or any future trial type) for the other condition.
pub fn aggregate_blocks_for_condition(
    file: &hdf5::File,
    base_path: &str,
    condition: &str,
    sub_group: Option<&str>,
    dataset: &str,
) -> Result<Option<Array2<f64>>> {
    let root = match file.group(base_path) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let block_parent = match root.group(condition) {
        Ok(g) => g,
        Err(_) => return Ok(None),
    };
    let block_names: Vec<String> = match block_parent.member_names() {
        Ok(names) => names
            .into_iter()
            .filter(|n| n.starts_with("block_"))
            .collect(),
        Err(_) => return Ok(None),
    };
    if block_names.is_empty() {
        return Ok(None);
    }
    let mut sum: Option<Array2<f64>> = None;
    let mut count: Option<Array2<f64>> = None;
    for block_name in &block_names {
        let block_group = match block_parent.group(block_name) {
            Ok(g) => g,
            Err(_) => continue,
        };
        let target = match sub_group {
            Some(sg) => match block_group.group(sg) {
                Ok(g) => g,
                Err(_) => continue,
            },
            None => block_group,
        };
        let ds = match target.dataset(dataset) {
            Ok(d) => d,
            Err(_) => continue,
        };
        let mat = read_2d_f64(&ds)?;
        match sum.as_mut() {
            None => {
                let mut s = Array2::<f64>::zeros(mat.dim());
                let mut c = Array2::<f64>::zeros(mat.dim());
                ndarray::Zip::from(&mut s)
                    .and(&mut c)
                    .and(&mat)
                    .for_each(|sv, cv, &v| {
                        if !v.is_nan() {
                            *sv += v;
                            *cv += 1.0;
                        }
                    });
                sum = Some(s);
                count = Some(c);
            }
            Some(s) => {
                let c = count.as_mut().unwrap();
                ndarray::Zip::from(s)
                    .and(c)
                    .and(&mat)
                    .for_each(|sv, cv, &v| {
                        if !v.is_nan() {
                            *sv += v;
                            *cv += 1.0;
                        }
                    });
            }
        }
    }
    match (sum, count) {
        (Some(s), Some(c)) => {
            let mean = ndarray::Zip::from(&s)
                .and(&c)
                .map_collect(|&sv, &cv| if cv == 0.0 { f64::NAN } else { sv / cv });
            Ok(Some(mean))
        }
        _ => Ok(None),
    }
}

/// Per-subject paired FC contrast: Fisher-z average of `cond_a` minus `cond_b`.
///
/// Both conditions are averaged within-subject first (using on-disk Fisher-z values
/// written by stage 06), then subtracted. Returns `None` if either condition is absent
/// or yields no blocks for this subject — caller should skip-with-warn.
pub fn aggregate_paired_contrast(
    file: &hdf5::File,
    base_path: &str,
    cond_a: &str,
    cond_b: &str,
    sub_group: Option<&str>,
    dataset: &str,
) -> Result<Option<Array2<f64>>> {
    let za = aggregate_blocks_for_condition(file, base_path, cond_a, sub_group, dataset)?;
    let zb = aggregate_blocks_for_condition(file, base_path, cond_b, sub_group, dataset)?;
    match (za, zb) {
        (Some(a), Some(b)) => Ok(Some(&a - &b)),
        _ => Ok(None),
    }
}

fn read_2d_f64(ds: &hdf5::Dataset) -> Result<Array2<f64>> {
    let shape = ds.shape();
    anyhow::ensure!(
        shape.len() == 2,
        "expected 2D dataset, got shape {:?}",
        shape
    );
    let raw: Vec<f64> = ds.read_raw()?;
    Ok(Array2::from_shape_vec((shape[0], shape[1]), raw)?)
}
