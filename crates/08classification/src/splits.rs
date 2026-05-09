use crate::dataset::Label as DatasetLabel;
use anyhow::{Context, Result};
use polars::prelude::*;
use rand::SeedableRng;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use std::collections::HashMap;
use std::{fs, path::Path};
use utils::polars_csv;

/// Write out a .csv with a list of subjects
pub fn write_subject_split_csvs<P: AsRef<Path>>(
    out_dir: P,
    train: &[String],
    test: &[String],
    val: &[String],
) -> Result<()> {
    let out = out_dir.as_ref();
    write_subject_set(out.join("subjects_train.tsv"), train)?;
    write_subject_set(out.join("subjects_test.tsv"), test)?;
    write_subject_set(out.join("subjects_validation.tsv"), val)?;
    Ok(())
}

fn write_subject_set<P: AsRef<Path>>(path: P, subjects: &[String]) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut sorted: Vec<String> = subjects.to_vec();
    sorted.sort();
    let df = DataFrame::new(vec![Column::new("subjectkey".into(), sorted)])?;
    polars_csv::write_tsv(&path, &df)
        .with_context(|| format!("failed to write {}", path.as_ref().display()))?;
    Ok(())
}

/// Subject-level split, with balance-truncation. Returns (train, test, val).
pub fn split_subjects_stratified(
    controls: &[String],
    anhedonic: &[String],
    seed: u64,
) -> (Vec<String>, Vec<String>, Vec<String>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let balanced_group_size = controls.len().min(anhedonic.len());

    let selected_controls: Vec<_> = controls
        .iter()
        .choose_multiple(&mut rng, balanced_group_size);
    let selected_anhedonic: Vec<_> = anhedonic
        .iter()
        .choose_multiple(&mut rng, balanced_group_size);

    let mut selected_subjects: Vec<String> = selected_controls
        .into_iter()
        .chain(selected_anhedonic)
        .cloned()
        .collect();
    selected_subjects.shuffle(&mut rng);

    let total = selected_subjects.len();
    let n_train = (total as f64 * 0.7).round() as usize;
    let n_val = (total as f64 * 0.15).round() as usize;

    let mut it = selected_subjects.into_iter();
    let train_set: Vec<String> = it.by_ref().take(n_train).collect();
    let val_set: Vec<String> = it.by_ref().take(n_val).collect();
    let test_set: Vec<String> = it.collect();

    (train_set, test_set, val_set)
}

/// Row-level, label-stratified split. Returns indices for (train, test, val).
pub fn split_rows_stratified_new(
    labels: &[DatasetLabel],
    seed: u64,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Group indices by their specific Label
    let mut label_groups: HashMap<DatasetLabel, Vec<usize>> = HashMap::new();
    for (i, &label) in labels.iter().enumerate() {
        label_groups.entry(label).or_default().push(i);
    }

    let mut train = Vec::new();
    let mut val = Vec::new();
    let mut test = Vec::new();

    // Process each group independently to maintain stratification
    for indices in label_groups.values_mut() {
        indices.shuffle(&mut rng);

        let len = indices.len() as f64;
        let n_train = (len * 0.7).round() as usize;
        let n_val = (len * 0.15).round() as usize;

        train.extend(&indices[..n_train]);
        val.extend(&indices[n_train..n_train + n_val]);
        test.extend(&indices[n_train + n_val..]);
    }

    // Final shuffle so the datasets aren't ordered by label
    train.shuffle(&mut rng);
    val.shuffle(&mut rng);
    test.shuffle(&mut rng);

    (train, test, val)
}

pub fn split_rows_stratified(labels: &[i32], seed: u64) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    let mut zeros = Vec::new();
    let mut ones = Vec::new();
    for (i, &l) in labels.iter().enumerate() {
        if l == 0 {
            zeros.push(i);
        } else {
            ones.push(i);
        }
    }

    zeros.shuffle(&mut rng);
    ones.shuffle(&mut rng);

    let z_train = (zeros.len() as f64 * 0.7).round() as usize;
    let z_val = (zeros.len() as f64 * 0.15).round() as usize;
    let o_train = (ones.len() as f64 * 0.7).round() as usize;
    let o_val = (ones.len() as f64 * 0.15).round() as usize;

    let mut train = Vec::new();
    train.extend(&zeros[..z_train]);
    train.extend(&ones[..o_train]);

    let mut val = Vec::new();
    val.extend(&zeros[z_train..z_train + z_val]);
    val.extend(&ones[o_train..o_train + o_val]);

    let mut test = Vec::new();
    test.extend(&zeros[z_train + z_val..]);
    test.extend(&ones[o_train + o_val..]);

    train.shuffle(&mut rng);
    val.shuffle(&mut rng);
    test.shuffle(&mut rng);

    (train, test, val)
}

/// Split for block ensemble. Groups typically are `(subject, roi)`.
/// Returns (train_groups, test_groups, val_groups).
pub fn split_groups_stratified<T: Clone + PartialEq>(
    group_ids: &[T],
    group_labels: &[i32],
    seed: u64,
) -> (Vec<T>, Vec<T>, Vec<T>) {
    // Unique groups and their label
    let mut unique_groups = Vec::new();
    for (id, label) in group_ids.iter().zip(group_labels.iter()) {
        if !unique_groups.iter().any(|(u_id, _)| u_id == id) {
            unique_groups.push((id.clone(), *label));
        }
    }
    let labels: Vec<i32> = unique_groups.iter().map(|(_, l)| *l).collect();
    let (tr_idx, te_idx, va_idx) = split_rows_stratified(&labels, seed);

    let train = tr_idx.iter().map(|&i| unique_groups[i].0.clone()).collect();
    let test = te_idx.iter().map(|&i| unique_groups[i].0.clone()).collect();
    let val = va_idx.iter().map(|&i| unique_groups[i].0.clone()).collect();

    (train, test, val)
}
