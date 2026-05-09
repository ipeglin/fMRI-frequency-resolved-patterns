mod nifti_masker;

use anyhow::Result;
use ndarray::{Array2, Axis, concatenate, s};
use nifti_masker::{LabelsMasker, MaskerSignalConfig, Standardize};
use polars::prelude::*;
use std::collections::HashMap;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info, info_span, warn};
use utils::bids_filename::{BidsFilename, find_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::hdf5_io::{H5Attr, open_or_create_group, write_attrs};
use utils::polars_csv;

/// Returns true when the `full_run_std` dataset is already present in the HDF5
/// file at `path`, meaning parcellation can be skipped for this file.
fn dataset_present(path: &Path) -> bool {
    hdf5::File::open(path)
        .ok()
        .and_then(|f| f.group("01fmri_parcellation").ok())
        .is_some_and(|g| g.dataset("full_run_std").is_ok())
}

/// Returns true when the `/metadata` group with `cohort` attribute is already present.
fn metadata_present(path: &Path) -> bool {
    hdf5::File::open(path)
        .ok()
        .and_then(|f| f.group("metadata").ok())
        .is_some_and(|g| g.attr("cohort").is_ok())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    let fmriprep_output_dir = &cfg.fmriprep_output_dir;
    match fs::read_dir(fmriprep_output_dir) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::NotFound => {
            error!(
                fmriprep_output_dir = %fmriprep_output_dir.display(),
                "Directory not found: {}. Make sure to have the disk connected, or connecting to the network drive",
                fmriprep_output_dir.display()
            );
            return Ok(());
        }
        Err(e) => panic!("Failed to read directory: {}", e),
    }

    let run_start = Instant::now();

    info!(
        fmriprep_output_dir = %cfg.fmriprep_output_dir.display(),
        filter_dir = %cfg.subject_filter_dir.display(),
        output_dir = %cfg.consolidated_data_dir.display(),
        cortical_atlas = %cfg.cortical_atlas.display(),
        subcortical_atlas = %cfg.subcortical_atlas.display(),
        force = cfg.force,
        standardize = cfg.parcellation.standardize,
        voxelwise_zscore = cfg.parcellation.voxelwise_zscore,
        "starting fMRI preprocessing pipeline"
    );

    let csv_dir = &cfg.csv_output_dir;
    let cohort_files: &[(&str, &str)] = &[
        ("desc-controls_subjects.tsv", "control"),
        ("desc-anhedonic_subjects.tsv", "anhedonic"),
    ];

    // Build cohort_map (subjectkey → label) and collect LazyFrames for subject list.
    let mut cohort_map: HashMap<String, &'static str> = HashMap::new();
    let mut dataframes: Vec<LazyFrame> = Vec::new();

    for (filename, cohort_label) in cohort_files {
        let file = csv_dir.join(filename);
        match polars_csv::read_tsv(&file) {
            Ok(df) => {
                if let Ok(col) = df.column("subjectkey")
                    && let Ok(s) = col.str()
                {
                    for key in s.into_no_null_iter() {
                        cohort_map.insert(key.to_string(), cohort_label);
                    }
                }
                dataframes.push(df.lazy());
            }
            Err(e) => {
                warn!(
                    file = %file.display(),
                    error = %e,
                    "failed to read filter file"
                );
            }
        }
    }

    let subjects = concat(dataframes, UnionArgs::default())?
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .sort(["subjectkey"], Default::default())
        .collect()?;
    let subject_keys = subjects.column("subjectkey")?.str()?;
    let total_subjects = subject_keys.len();

    info!(
        total_subjects = total_subjects,
        filter_files_loaded = cohort_files.len(),
        "loaded subject keys"
    );

    if !cfg.cortical_atlas.exists() || !cfg.subcortical_atlas.exists() {
        panic!("failed to locate atlases");
    }

    std::fs::create_dir_all(&cfg.consolidated_data_dir)?;
    std::fs::create_dir_all(&cfg.csv_output_dir)?;

    let csv_crate_prefix = BidsFilename::new()
        .with_pair("crate", "01")
        .with_extension(".csv")
        .with_directory(&cfg.csv_output_dir);

    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let error_count = 0usize;

    // Build the masker signal config once — both atlases share the same settings.
    //
    // standardize=ZscoreSample: per-ROI row z-score after extraction (cheap; matches
    // nilearn NiftiLabelsMasker(standardize="zscore_sample")). Applied inside
    // LabelsMasker::fit_transform via preprocess_signals.
    //
    // voxelwise_zscore: per-voxel z-score *before* parcellation (expensive opt-in).
    let std_mode = if cfg.parcellation.standardize {
        Standardize::ZscoreSample
    } else {
        Standardize::None
    };
    let masker_cfg = MaskerSignalConfig::default()
        .standardize(std_mode)
        .voxelwise_zscore(cfg.parcellation.voxelwise_zscore);

    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let subject_idx = i + 1;
        let subject_id = BidsSubjectId::parse(subject_key);
        let dir_name = subject_id.clone().to_dir_name();
        let subject_dir = fmriprep_output_dir.join(&dir_name);

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

        let mni_results_dir = subject_dir.join("func");

        let hammer_scan_files = find_bids_files(
            &mni_results_dir,
            &[
                ("task", "hammerAP"),
                ("space", "MNI152NLin2009cAsym"),
                ("res", "2"),
                ("desc", "preproc"),
            ],
            Some("bold"),
            Some(".nii.gz"),
        );
        let resting_scan_files = find_bids_files(
            &mni_results_dir,
            &[
                ("task", "restAP"),
                ("space", "MNI152NLin2009cAsym"),
                ("res", "2"),
                ("desc", "preproc"),
            ],
            Some("bold"),
            Some(".nii.gz"),
        );
        let files_to_preprocess: Vec<PathBuf> = hammer_scan_files
            .into_iter()
            .chain(resting_scan_files.into_iter())
            .collect();

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

            let bids_filename =
                BidsFilename::parse(file_path.file_name().and_then(|n| n.to_str()).unwrap_or(""));
            let task_name = bids_filename.get("task").unwrap_or("unknown");
            let output_stem = bids_filename.to_stem();

            let output_h5_path = cfg
                .consolidated_data_dir
                .join(BidsSubjectId::parse(subject_key).to_dir_name())
                .join(format!("{}.h5", output_stem));

            let _ = csv_crate_prefix
                .clone()
                .with_pair("sub", subject_id.as_bids_id())
                .with_pair("task", task_name)
                .with_pair("run", bids_filename.get("run").unwrap_or("unknown"));

            // Skip when both dataset and cohort metadata are present, unless --force.
            if !cfg.force
                && output_h5_path.exists()
                && dataset_present(&output_h5_path)
                && metadata_present(&output_h5_path)
            {
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_preprocessed",
                    output_file = %output_h5_path.display(),
                    "skipping file (full_run_std present, use --force to reprocess)"
                );
                continue;
            }

            // If parcellation is done but metadata is missing (e.g. pre-refactor file),
            // stamp the metadata without re-running parcellation.
            if !cfg.force
                && output_h5_path.exists()
                && dataset_present(&output_h5_path)
                && !metadata_present(&output_h5_path)
            {
                let cohort = cohort_map.get(subject_key).copied().unwrap_or("unknown");
                write_metadata(&output_h5_path, subject_key, cohort)?;
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    cohort,
                    "stamped missing cohort metadata on existing H5"
                );
                continue;
            }

            if let Some(parent) = output_h5_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            if cfg.force && output_h5_path.exists() {
                fs::remove_file(&output_h5_path)?;
            }

            let file_start = Instant::now();

            debug!(
                subject_key = subject_key,
                task_name = task_name,
                file_path = %file_path.display(),
                standardize = cfg.parcellation.standardize,
                voxelwise_zscore = cfg.parcellation.voxelwise_zscore,
                "starting parcellation"
            );

            let cortical_start = Instant::now();
            let cortical_masker =
                LabelsMasker::with_config(&cfg.cortical_atlas, masker_cfg.clone())?;
            let cortical_ts = cortical_masker.fit_transform(&file_path)?;
            let cortical_duration_ms = cortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "cortical",
                n_rois = cortical_ts.shape()[0],
                n_timepoints = cortical_ts.shape()[1],
                duration_ms = cortical_duration_ms,
                "parcellation completed"
            );

            let subcortical_start = Instant::now();
            let subcortical_masker =
                LabelsMasker::with_config(&cfg.subcortical_atlas, masker_cfg.clone())?;
            let subcortical_ts = subcortical_masker.fit_transform(&file_path)?;
            let subcortical_duration_ms = subcortical_start.elapsed().as_millis();

            debug!(
                subject_key = subject_key,
                atlas_type = "subcortical",
                n_rois = subcortical_ts.shape()[0],
                n_timepoints = subcortical_ts.shape()[1],
                duration_ms = subcortical_duration_ms,
                "parcellation completed"
            );

            debug!(
                subject_key = subject_key,
                cortical_first_roi_first_5 = ?cortical_ts.slice(s![0, ..5]),
                subcortical_first_roi_first_5 = ?subcortical_ts.slice(s![0, ..5]),
                "timeseries sample values"
            );

            let full_run_std =
                concatenate(Axis(0), &[cortical_ts.view(), subcortical_ts.view()])?;

            let cohort = cohort_map.get(subject_key).copied().unwrap_or("unknown");
            let write_start = Instant::now();
            write_dataset(
                &output_h5_path,
                &full_run_std,
                cfg.parcellation.standardize,
                subject_key,
                cohort,
            )?;
            let write_duration_ms = write_start.elapsed().as_millis();

            let total_duration_ms = file_start.elapsed().as_millis();
            processed_count += 1;

            info!(
                subject_key = subject_key,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                task_name = task_name,
                input_file = %file_path.display(),
                output_file = %output_h5_path.display(),
                n_rois = full_run_std.shape()[0],
                n_timepoints = full_run_std.shape()[1],
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

    info!(
        total_subjects = total_subjects,
        processed_count = processed_count,
        skipped_count = skipped_count,
        error_count = error_count,
        total_duration_ms = run_duration_ms,
        output_dir = %cfg.consolidated_data_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI preprocessing pipeline completed"
    );

    Ok(())
}

/// Write the concatenated ROI timeseries to `/01fmri_parcellation/full_run_std`
/// and stamp `/metadata` with `subjectkey` and `cohort` attributes.
///
/// The dataset name `full_run_std` is preserved regardless of the `standardize`
/// flag so downstream crates (03cwt, 04mvmd, 07feature_extraction) do not need
/// updating. A `standardized` attribute (u8 0/1) is written on the dataset to
/// make the processing mode discoverable for future stale-cache checks.
fn write_dataset(
    path: &Path,
    data: &Array2<f32>,
    standardized: bool,
    subjectkey: &str,
    cohort: &str,
) -> Result<()> {
    let file = if path.exists() {
        hdf5::File::open_rw(path)?
    } else {
        hdf5::File::create(path)?
    };

    let parc = match file.group("01fmri_parcellation") {
        Ok(g) => g,
        Err(_) => file.create_group("01fmri_parcellation")?,
    };

    let shape = data.shape();
    let ds = parc
        .new_dataset::<f32>()
        .shape([shape[0], shape[1]])
        .create("full_run_std")?;
    ds.write_raw(data.as_slice().unwrap())?;

    ds.new_attr::<u8>()
        .shape(())
        .create("standardized")?
        .as_writer()
        .write_scalar(&(standardized as u8))?;

    let meta = open_or_create_group(&file, "metadata", false)?;
    write_attrs(
        &meta,
        &[
            H5Attr::string("subjectkey", subjectkey),
            H5Attr::string("cohort", cohort),
        ],
    )?;

    Ok(())
}

/// Write (or update) only the `/metadata` group on an existing H5 file.
fn write_metadata(path: &Path, subjectkey: &str, cohort: &str) -> Result<()> {
    let file = hdf5::File::open_rw(path)?;
    let meta = open_or_create_group(&file, "metadata", false)?;
    write_attrs(
        &meta,
        &[
            H5Attr::string("subjectkey", subjectkey),
            H5Attr::string("cohort", cohort),
        ],
    )?;
    Ok(())
}

