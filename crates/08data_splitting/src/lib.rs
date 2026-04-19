use std::{collections::HashSet, fs, path::Path, time::Instant};

use anyhow::{Context, Result};
use config::{DataSplitConfig, bids_subject_id::BidsSubjectId, polars_csv};
use polars::prelude::*;
use rand::SeedableRng;
use rand::seq::IteratorRandom;
use rand::seq::SliceRandom;
use rand_chacha::ChaCha8Rng;
use tracing::{debug, info, warn};

fn write_subject_set<P: AsRef<Path>>(path: P, subjects: &[String]) -> Result<()> {
    if let Some(parent) = path.as_ref().parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("failed to create {}", parent.display()))?;
    }
    let mut sorted: Vec<String> = subjects.to_vec();
    sorted.sort();
    let df = DataFrame::new(vec![Column::new("subjectkey".into(), sorted)])?;
    polars_csv::write_dataframe(&path, &df)
        .with_context(|| format!("failed to write {}", path.as_ref().display()))?;
    Ok(())
}

pub fn run(cfg: &DataSplitConfig) -> Result<()> {
    const TRAINING_FRAC: f64 = 0.7;
    const TESTING_FRAC: f64 = (1.0 - TRAINING_FRAC) / 2.0;
    const VALIDATION_FRAC: f64 = TESTING_FRAC;

    let run_start = Instant::now();

    info!(
        bold_ts_dir = %cfg.bold_ts_dir.display(),
        force = cfg.force,
        "starting data splitting pipeline"
    );

    let fmri_dir = &cfg.bold_ts_dir;
    let subject_directories: HashSet<String> = fs::read_dir(fmri_dir)?
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            Some(BidsSubjectId::parse(id).to_dir_name())
        })
        .collect();
    debug!(
        count = subject_directories.len(),
        "subject directories discovered"
    );

    let filter_dir = &cfg.subject_filter_dir;

    let controls_file = filter_dir.join("healthy_controls.csv");
    let control_subjects: Vec<String> = polars_csv::read_dataframe(&controls_file)
        .with_context(|| format!("failed to read {}", controls_file.display()))?
        .column("subjectkey")?
        .str()?
        .into_no_null_iter()
        .map(|s| BidsSubjectId::parse(s).to_dir_name())
        .collect();

    let anhedonic_file = filter_dir.join("anhedonic.csv");
    let anhedonic_subjects: Vec<String> = polars_csv::read_dataframe(&anhedonic_file)
        .with_context(|| format!("failed to read {}", anhedonic_file.display()))?
        .column("subjectkey")?
        .str()?
        .into_no_null_iter()
        .map(|s| BidsSubjectId::parse(s).to_dir_name())
        .collect();

    // Intersect category rosters with on-disk fMRI outputs: a subject is only
    // usable if it appears both in the clinical filter CSV and has processed data.
    let valid_controls: Vec<&String> = control_subjects
        .iter()
        .filter(|item| subject_directories.contains(item.as_str()))
        .collect();

    let valid_anhedonic: Vec<&String> = anhedonic_subjects
        .iter()
        .filter(|item| subject_directories.contains(item.as_str()))
        .collect();

    info!(
        controls = valid_controls.len(),
        anhedonic = valid_anhedonic.len(),
        "valid subjects identified"
    );

    // Balancing pins each group to the size of the smaller one, so the usable
    // dataset is 2 * min(controls, anhedonic). The smallest split (test or
    // validation) must still receive at least one subject after flooring.
    let balanced_group_size = valid_controls.len().min(valid_anhedonic.len());
    let balanced_total = 2 * balanced_group_size;
    let smallest_split_frac = TESTING_FRAC.min(VALIDATION_FRAC);
    let min_total_required = (1.0 / smallest_split_frac).ceil() as usize;

    if balanced_total < min_total_required {
        warn!(
            controls = valid_controls.len(),
            anhedonic = valid_anhedonic.len(),
            balanced_group_size = balanced_group_size,
            balanced_total = balanced_total,
            required = min_total_required,
            training_pct = TRAINING_FRAC * 100.0,
            testing_pct = TESTING_FRAC * 100.0,
            validation_pct = VALIDATION_FRAC * 100.0,
            "insufficient subjects after balancing to split into train/test/validation"
        );
        anyhow::bail!(
            "insufficient balanced dataset: {} controls + {} anhedonic → balanced total {}, need at least {} for {:.0}/{:.0}/{:.0} train/test/val split",
            valid_controls.len(),
            valid_anhedonic.len(),
            balanced_total,
            min_total_required,
            TRAINING_FRAC * 100.0,
            TESTING_FRAC * 100.0,
            VALIDATION_FRAC * 100.0,
        );
    }

    // Randomization for reproducability
    let seed = 42; // Just because
    let mut rng = ChaCha8Rng::seed_from_u64(seed);

    // Take N from smaller group, N at random from larger
    let selected_controls: Vec<_> = valid_controls
        .iter()
        .choose_multiple(&mut rng, balanced_group_size);
    let selected_anhedonic: Vec<_> = valid_anhedonic
        .iter()
        .choose_multiple(&mut rng, balanced_group_size);

    println!("Control Group: {:?}", selected_controls);
    println!("Anhedonic Group: {:?}", selected_anhedonic);

    let mut selected_subjects: Vec<String> = selected_controls
        .into_iter()
        .chain(selected_anhedonic)
        .map(|s| s.to_string()) // &&String -> String conversion
        .collect();

    // Split into train/test/validation using TRAINING_FRAC / TESTING_FRAC / VALIDATION_FRAC
    selected_subjects.shuffle(&mut rng);

    let total = selected_subjects.len();
    let n_train = (total as f64 * TRAINING_FRAC).round() as usize;
    let n_val = (total as f64 * VALIDATION_FRAC).round() as usize;

    let mut it = selected_subjects.into_iter();
    let train_set: Vec<String> = it.by_ref().take(n_train).collect();
    let val_set: Vec<String> = it.by_ref().take(n_val).collect();
    let test_set: Vec<String> = it.collect(); // Remaining items

    info!(
        train = train_set.len(),
        val = val_set.len(),
        test = test_set.len(),
        "Dataset split complete"
    );

    write_subject_set(&cfg.training_subjects_path, &train_set)?;
    write_subject_set(&cfg.validation_subjects_path, &val_set)?;
    write_subject_set(&cfg.test_subjects_path, &test_set)?;

    info!(
        train_file = %cfg.training_subjects_path.display(),
        val_file = %cfg.validation_subjects_path.display(),
        test_file = %cfg.test_subjects_path.display(),
        "Split subject keys written"
    );

    let total_duration_ms = run_start.elapsed().as_millis();
    info!(
        total_duration_ms = total_duration_ms,
        "data splitting pipeline complete"
    );

    Ok(())
}
