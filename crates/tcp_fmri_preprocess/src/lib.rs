use anyhow::Result;
use config::{TCPfMRIPreprocessConfig, polars_csv};
use ndarray::{Array2, s};
use nifti_masker::{LabelsMasker, MaskerSignalConfig};
use polars::prelude::*;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, info_span, warn};

pub fn run(cfg: &TCPfMRIPreprocessConfig) -> Result<()> {
    let run_start = Instant::now();

    info!(
        fmri_dir = %cfg.fmri_dir.display(),
        filter_dir = %cfg.filter_dir.display(),
        output_dir = %cfg.output_dir.display(),
        cortical_atlas = %cfg.cortical_atlas.display(),
        subcortical_atlas = %cfg.subcortical_atlas.display(),
        dry_run = cfg.dry_run,
        force = cfg.force,
        "starting fMRI preprocessing pipeline"
    );

    // Load subject filter files
    let filter_dir = &cfg.filter_dir;
    let filtered_subjects = [
        filter_dir.join("healthy_controls.csv"),
        filter_dir.join("shaps_low_anhedonic.csv"),
        filter_dir.join("shaps_high_anhedonic.csv"),
        filter_dir.join("teps_anticipatory_anhedonic.csv"),
        filter_dir.join("teps_anticipatory_non_anhedonic.csv"),
        filter_dir.join("teps_anticipatory_anhedonic.csv"),
        filter_dir.join("teps_anticipatory_non_anhedonic.csv"),
    ];

    let dataframes: Vec<LazyFrame> = filtered_subjects
        .iter()
        .filter_map(|file| {
            polars_csv::read_dataframe(file)
                .map_err(|e| {
                    warn!(
                        file = %file.display(),
                        error = %e,
                        "failed to read filter file"
                    )
                })
                .ok()
                .map(|df| df.lazy())
        })
        .collect();

    let subjects = concat(dataframes, UnionArgs::default())?
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .collect()?;
    let subject_keys = subjects.column("subjectkey")?.str()?;
    let total_subjects = subject_keys.len();

    info!(
        total_subjects = total_subjects,
        filter_files_loaded = filtered_subjects.len(),
        "loaded subject keys"
    );

    // Validate atlases
    if !cfg.cortical_atlas.exists() || !cfg.subcortical_atlas.exists() {
        panic!("failed to locate atlases");
    }

    std::fs::create_dir_all(&cfg.output_dir)?;

    // Processing state
    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let error_count = 0usize;

    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let subject_idx = i + 1;
        let dir_name = parse_subject_directory_name(subject_key);
        let subject_dir = cfg.fmri_dir.join(&dir_name);

        // Create a span for the entire subject processing - this is the "wide event"
        let _subject_span = info_span!(
            "process_subject",
            subject_key = subject_key,
            subject_idx = subject_idx,
            total_subjects = total_subjects,
            subject_dir = %subject_dir.display(),
        )
        .entered();

        if !subject_dir.is_dir() {
            skipped_count += 1;
            warn!(
                subject_key = subject_key,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                reason = "missing_fmri_data",
                subject_dir = %subject_dir.display(),
                "skipping subject"
            );
            continue;
        }

        let mni_results_dir = subject_dir.join("MNINonLinear").join("Results");
        let files_to_preprocess = [
            // Harari-Hammer task
            mni_results_dir
                .join("task-hammerAP_run-01_bold")
                .join("task-hammerAP_run-01_bold.nii.gz"),
            // Resting state AP encoding
            mni_results_dir
                .join("task-restAP_run-01_bold")
                .join("task-restAP_run-01_bold.nii.gz"),
            mni_results_dir
                .join("task-restAP_run-02_bold")
                .join("task-restAP_run-02_bold.nii.gz"),
            // Resting state PA encoding
            mni_results_dir
                .join("task-restPA_run-01_bold")
                .join("task-restPA_run-01_bold.nii.gz"),
            mni_results_dir
                .join("task-restPA_run-02_bold")
                .join("task-restPA_run-02_bold.nii.gz"),
        ];

        for file_path in files_to_preprocess {
            if !file_path.exists() {
                skipped_count += 1;
                warn!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    reason = "missing_bold_file",
                    file_path = %file_path.display(),
                    "skipping subject file"
                );
                continue;
            }

            let task_name = file_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");

            // Check if output already exists (skip unless --force)
            let output_path = cfg
                .output_dir
                .join(subject_key)
                .join(format!("{}.h5", task_name));
            if output_path.exists() && !cfg.force {
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_preprocessed",
                    output_file = %output_path.display(),
                    "skipping file (already preprocessed, use --force to reprocess)"
                );
                continue;
            }

            let file_start = Instant::now();

            debug!(
                subject_key = subject_key,
                task_name = task_name,
                file_path = %file_path.display(),
                "starting parcellation"
            );

            // Signal preprocessing config: detrend and z-score standardize
            let signal_config = MaskerSignalConfig::with_defaults();

            // Cortical parcellation
            let cortical_start = Instant::now();
            let cortical_masker =
                LabelsMasker::with_config(&cfg.cortical_atlas, signal_config.clone())?;
            let cortical_bold = cortical_masker.fit_transform(&file_path)?;
            let cortical_duration_ms = cortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "cortical",
                n_rois = cortical_bold.shape()[0],
                n_timepoints = cortical_bold.shape()[1],
                duration_ms = cortical_duration_ms,
                atlas_path = %cfg.cortical_atlas.display(),
                detrend = signal_config.detrend,
                standardize = ?signal_config.standardize,
                "parcellation completed"
            );

            // Subcortical parcellation
            let subcortical_start = Instant::now();
            let subcortical_masker =
                LabelsMasker::with_config(&cfg.subcortical_atlas, signal_config.clone())?;
            let subcortical_bold = subcortical_masker.fit_transform(&file_path)?;
            let subcortical_duration_ms = subcortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "subcortical",
                n_rois = subcortical_bold.shape()[0],
                n_timepoints = subcortical_bold.shape()[1],
                duration_ms = subcortical_duration_ms,
                atlas_path = %cfg.subcortical_atlas.display(),
                detrend = signal_config.detrend,
                standardize = ?signal_config.standardize,
                "parcellation completed"
            );

            // Debug: Print first few values
            debug!(
                subject_key = subject_key,
                cortical_first_roi_first_5_timepoints = ?cortical_bold.slice(s![0, ..5]),
                subcortical_first_roi_first_5_timepoints = ?subcortical_bold.slice(s![0, ..5]),
                "timeseries sample values"
            );

            // Write output
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            let write_start = Instant::now();
            write_timeseries_h5(&output_path, &cortical_bold, &subcortical_bold)?;
            let write_duration_ms = write_start.elapsed().as_millis();

            let total_duration_ms = file_start.elapsed().as_millis();
            processed_count += 1;

            // Wide event: one comprehensive log per subject file processed
            info!(
                subject_key = subject_key,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                task_name = task_name,
                input_file = %file_path.display(),
                output_file = %output_path.display(),
                cortical_rois = cortical_bold.shape()[0],
                subcortical_rois = subcortical_bold.shape()[0],
                n_timepoints = cortical_bold.shape()[1],
                cortical_duration_ms = cortical_duration_ms,
                subcortical_duration_ms = subcortical_duration_ms,
                write_duration_ms = write_duration_ms,
                total_duration_ms = total_duration_ms,
                outcome = "success",
                "subject processed"
            );
        }
    }

    let run_duration_ms = run_start.elapsed().as_millis();

    // Final summary wide event
    info!(
        total_subjects = total_subjects,
        processed_count = processed_count,
        skipped_count = skipped_count,
        error_count = error_count,
        total_duration_ms = run_duration_ms,
        output_dir = %cfg.output_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI preprocessing pipeline completed"
    );

    Ok(())
}

fn parse_subject_directory_name(key: &str) -> String {
    format!("sub-{}", key.replace("_", ""))
}

fn write_timeseries_h5(
    path: &Path,
    cortical: &Array2<f32>,
    subcortical: &Array2<f32>,
) -> Result<()> {
    let file = hdf5::File::create(path)?;

    // Write cortical dataset
    let cortical_shape = cortical.shape();
    let cortical_standard = cortical.to_owned();
    let cortical_ds = file
        .new_dataset::<f32>()
        .shape([cortical_shape[0], cortical_shape[1]])
        .create("tcp_cortical")?;
    cortical_ds.write_raw(cortical_standard.as_slice().unwrap())?;

    // Write subcortical dataset
    let subcortical_shape = subcortical.shape();
    let subcortical_standard = subcortical.to_owned();
    let subcortical_ds = file
        .new_dataset::<f32>()
        .shape([subcortical_shape[0], subcortical_shape[1]])
        .create("tcp_subcortical")?;
    subcortical_ds.write_raw(subcortical_standard.as_slice().unwrap())?;

    Ok(())
}
