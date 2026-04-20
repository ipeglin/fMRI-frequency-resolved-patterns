mod classifiers;

use std::time::Instant;

use anyhow::{Context, Result};
use tracing::info;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::polars_csv;

use crate::classifiers::{KNN, KnnConfig};

pub fn run(cfg: &AppConfig) -> Result<()> {
    let run_start = Instant::now();

    info!(
        data_splitting_output_dir = %cfg.data_splitting_output_dir.display(),
        "starting subject classification",
    );

    // 1. Get training subjects
    let training_subjects_file = &cfg.data_splitting_output_dir.join("subjects_train.csv");
    let training_subjects: Vec<String> = polars_csv::read_dataframe(&training_subjects_file)
        .with_context(|| format!("failed to read {}", training_subjects_file.display()))?
        .column("*")?
        .str()?
        .into_no_null_iter()
        .map(|s| BidsSubjectId::parse(s).to_dir_name())
        .collect();
    info!(
        training_subjects = ?training_subjects,
        "found training subjects"
    );

    // 2a. Get feature vectors for training subjects
    let parcellated_ts_dir = &cfg.parcellated_ts_dir;
    // TODO: Get all feature maps for training subjects.

    // 2b. Optional: PCA

    // 3a. Initiate Classifiers
    let knn_classifier = KNN::from_training_data()?.with_config(KnnConfig { num_neighbors: 5 });

    // 3b. Initiate SVM

    // 4a. Train KNN

    // 4b. Train SVM

    // 5. Get test subjects
    let test_subjects_file = &cfg.data_splitting_output_dir.join("subjects_test.csv");

    // 6a. Test KNN

    // 6b. Test SVM

    // 7. Get validation subjects
    let validation_subjects_file = &cfg
        .data_splitting_output_dir
        .join("subjects_validation.csv");

    // 8a. Validation KNN

    // 8b. Validation SVM

    Ok(())
}
