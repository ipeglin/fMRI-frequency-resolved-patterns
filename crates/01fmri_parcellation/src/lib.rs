mod nifti_masker;

use anyhow::Result;
use ndarray::{Array2, Axis, concatenate, s};
use nifti_masker::{LabelsMasker, MaskerSignalConfig, Standardize, preprocess_signals};
use utils::bids_filename::{BidsFilename, find_bids_files};
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::polars_csv;
use polars::prelude::*;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{debug, error, info, info_span, warn};

/// Tracks which root-level datasets are absent from an existing HDF5 file.
///
/// Three groups are tracked independently:
///   - raw: parcellated signal with no preprocessing
///   - std: post-parcellation z-score standardization (per-ROI temporal z-score)
///   - voxel_zscore: voxel-wise z-score normalization applied before parcellation
///
/// Partial files can be extended without re-running the NIfTI masker.
#[derive(Debug)]
struct MissingDatasets {
    cortical_raw: bool,
    subcortical_raw: bool,
    timeseries_raw: bool,
    cortical_std: bool,
    subcortical_std: bool,
    timeseries_std: bool,
    cortical_voxel_zscore: bool,
    subcortical_voxel_zscore: bool,
    timeseries_voxel_zscore: bool,
}

impl MissingDatasets {
    /// All datasets need to be written (fresh file or --force).
    /// `voxelwise_zscore` controls whether the voxel-zscore group is included.
    fn all(voxelwise_zscore: bool) -> Self {
        Self {
            cortical_raw: true,
            subcortical_raw: true,
            timeseries_raw: true,
            cortical_std: true,
            subcortical_std: true,
            timeseries_std: true,
            cortical_voxel_zscore: voxelwise_zscore,
            subcortical_voxel_zscore: voxelwise_zscore,
            timeseries_voxel_zscore: voxelwise_zscore,
        }
    }

    /// True when no dataset needs to be written.
    fn all_present(&self) -> bool {
        !self.cortical_raw
            && !self.subcortical_raw
            && !self.timeseries_raw
            && !self.cortical_std
            && !self.subcortical_std
            && !self.timeseries_std
            && !self.cortical_voxel_zscore
            && !self.subcortical_voxel_zscore
            && !self.timeseries_voxel_zscore
    }

    /// True when any raw dataset is absent (masker must be run).
    fn needs_masker(&self) -> bool {
        self.cortical_raw || self.subcortical_raw || self.timeseries_raw
    }

    /// True when any post-parcellation standardized dataset is absent.
    fn needs_std(&self) -> bool {
        self.cortical_std || self.subcortical_std || self.timeseries_std
    }

    /// True when any voxel-wise z-score dataset is absent (masker must be re-run with voxelwise config).
    fn needs_voxel_zscore(&self) -> bool {
        self.cortical_voxel_zscore || self.subcortical_voxel_zscore || self.timeseries_voxel_zscore
    }
}

/// Inspect an existing HDF5 file and return which datasets are absent.
/// `voxelwise_zscore` controls whether the voxelzscore group is checked at all;
/// when false, those fields are always false so they never trigger computation.
fn check_missing_datasets(path: &Path, voxelwise_zscore: bool) -> MissingDatasets {
    match hdf5::File::open(path) {
        Ok(f) => MissingDatasets {
            cortical_raw: f.dataset("tcp_cortical_raw").is_err(),
            subcortical_raw: f.dataset("tcp_subcortical_raw").is_err(),
            timeseries_raw: f.dataset("tcp_timeseries_raw").is_err(),
            cortical_std: f.dataset("tcp_cortical_standardized").is_err(),
            subcortical_std: f.dataset("tcp_subcortical_standardized").is_err(),
            timeseries_std: f.dataset("tcp_timeseries_standardized").is_err(),
            cortical_voxel_zscore: voxelwise_zscore
                && f.dataset("tcp_cortical_voxelzscore").is_err(),
            subcortical_voxel_zscore: voxelwise_zscore
                && f.dataset("tcp_subcortical_voxelzscore").is_err(),
            timeseries_voxel_zscore: voxelwise_zscore
                && f.dataset("tcp_timeseries_voxelzscore").is_err(),
        },
        Err(_) => MissingDatasets::all(voxelwise_zscore),
    }
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    // Check that the fmri dir is even present.
    // If not, fail gracefully and inform the user that the
    // disk might not be connected, or the network disk is not opened.
    let fmriprep_output_dir = &cfg.fmriprep_output_dir;
    match fs::read_dir(fmriprep_output_dir) {
        Ok(_) => { /* Process entries */ }
        Err(e) if e.kind() == ErrorKind::NotFound => {
            error!(
                fmriprep_output_dir = %fmriprep_output_dir.display(),
                "Directory not found: {}. Make sure to have the disk connected, or connecting to the network drive", fmriprep_output_dir.display()
            );
            return Ok(());
        }
        Err(e) => panic!("Failed to read directory: {}", e),
    }

    let run_start = Instant::now();

    info!(
        fmriprep_output_dir = %cfg.fmriprep_output_dir.display(),
        filter_dir = %cfg.subject_filter_dir.display(),
        output_dir = %cfg.parcellated_ts_dir.display(),
        cortical_atlas = %cfg.cortical_atlas.display(),
        subcortical_atlas = %cfg.subcortical_atlas.display(),
        force = cfg.force,
        voxelwise_zscore = cfg.parcellation.voxelwise_zscore,
        "starting fMRI preprocessing pipeline"
    );

    // Load subject filter files
    let filter_dir = &cfg.subject_filter_dir;
    let filtered_subjects = [
        filter_dir.join("healthy_controls.csv"),
        filter_dir.join("anhedonic.csv"),
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

    std::fs::create_dir_all(&cfg.parcellated_ts_dir)?;

    // Processing state
    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let error_count = 0usize;

    for (i, subject_key) in subject_keys.into_iter().flatten().enumerate() {
        let subject_idx = i + 1;
        let dir_name = BidsSubjectId::parse(subject_key).to_dir_name();
        let subject_dir = fmriprep_output_dir.join(&dir_name);

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

            let output_path = cfg
                .parcellated_ts_dir
                .join(BidsSubjectId::parse(subject_key).to_dir_name())
                .join(format!("{}.h5", output_stem));

            // Determine which datasets still need to be written.
            //   --force  → treat all as missing (file will be removed and recreated)
            //   no file  → all missing (fresh create)
            //   file exists, no force → inspect per-dataset; only fill gaps
            let missing = if cfg.force || !output_path.exists() {
                MissingDatasets::all(cfg.parcellation.voxelwise_zscore)
            } else {
                check_missing_datasets(&output_path, cfg.parcellation.voxelwise_zscore)
            };

            if missing.all_present() {
                skipped_count += 1;
                info!(
                    subject_key = subject_key,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_preprocessed",
                    output_file = %output_path.display(),
                    "skipping file (all datasets present, use --force to reprocess)"
                );
                continue;
            }

            if output_path.exists() {
                debug!(
                    subject_key = subject_key,
                    task_name = task_name,
                    output_file = %output_path.display(),
                    missing_cortical_raw = missing.cortical_raw,
                    missing_subcortical_raw = missing.subcortical_raw,
                    missing_timeseries_raw = missing.timeseries_raw,
                    missing_cortical_std = missing.cortical_std,
                    missing_subcortical_std = missing.subcortical_std,
                    missing_timeseries_std = missing.timeseries_std,
                    "output file exists with missing datasets, appending"
                );
            }

            let file_start = Instant::now();

            debug!(
                subject_key = subject_key,
                task_name = task_name,
                file_path = %file_path.display(),
                run_masker = missing.needs_masker(),
                compute_std = missing.needs_std(),
                "starting parcellation"
            );

            // Run the NIfTI masker only when raw datasets need to be (re)written.
            // If only standardized datasets are missing we can read the raw arrays
            // directly from the existing HDF5, which is much faster.
            let mut cortical_duration_ms = 0u128;
            let mut subcortical_duration_ms = 0u128;

            let (cortical_raw, subcortical_raw) = if missing.needs_masker() {
                let raw_config = MaskerSignalConfig::default(); // detrend=false, standardize=None

                let cortical_start = Instant::now();
                let cortical_masker =
                    LabelsMasker::with_config(&cfg.cortical_atlas, raw_config.clone())?;
                let cortical_raw = cortical_masker.fit_transform(&file_path)?;
                cortical_duration_ms = cortical_start.elapsed().as_millis();

                debug!(
                    subject_key = subject_key,
                    atlas_type = "cortical",
                    n_rois = cortical_raw.shape()[0],
                    n_timepoints = cortical_raw.shape()[1],
                    duration_ms = cortical_duration_ms,
                    atlas_path = %cfg.cortical_atlas.display(),
                    "raw parcellation completed"
                );

                let subcortical_start = Instant::now();
                let subcortical_masker =
                    LabelsMasker::with_config(&cfg.subcortical_atlas, raw_config.clone())?;
                let subcortical_raw = subcortical_masker.fit_transform(&file_path)?;
                subcortical_duration_ms = subcortical_start.elapsed().as_millis();

                debug!(
                    subject_key = subject_key,
                    atlas_type = "subcortical",
                    n_rois = subcortical_raw.shape()[0],
                    n_timepoints = subcortical_raw.shape()[1],
                    duration_ms = subcortical_duration_ms,
                    atlas_path = %cfg.subcortical_atlas.display(),
                    "raw parcellation completed"
                );

                debug!(
                    subject_key = subject_key,
                    cortical_raw_first_roi_first_5_timepoints = ?cortical_raw.slice(s![0, ..5]),
                    subcortical_raw_first_roi_first_5_timepoints = ?subcortical_raw.slice(s![0, ..5]),
                    "raw timeseries sample values"
                );

                (cortical_raw, subcortical_raw)
            } else {
                // Raw datasets are already in the file; read them to compute std.
                debug!(
                    subject_key = subject_key,
                    task_name = task_name,
                    "reading existing raw datasets from HDF5 to derive standardized"
                );
                let existing = hdf5::File::open(&output_path)?;
                let cortical_raw: Array2<f32> = existing.dataset("tcp_cortical_raw")?.read_2d()?;
                let subcortical_raw: Array2<f32> =
                    existing.dataset("tcp_subcortical_raw")?.read_2d()?;
                (cortical_raw, subcortical_raw)
            };

            // Compute post-parcellation z-score standardized variants only when needed.
            let (cortical_std, subcortical_std) = if missing.needs_std() {
                let std_cfg = MaskerSignalConfig::default().standardize(Standardize::ZscoreSample);
                let c = preprocess_signals(&cortical_raw, &std_cfg);
                let s = preprocess_signals(&subcortical_raw, &std_cfg);
                (Some(c), Some(s))
            } else {
                (None, None)
            };

            // Run the masker with voxel-wise z-score normalization (applied before parcellation).
            let (cortical_voxel_zscore, subcortical_voxel_zscore) = if missing.needs_voxel_zscore()
            {
                let vz_config = MaskerSignalConfig::default().voxelwise_zscore(true);

                let cortical_start = Instant::now();
                let cortical_masker =
                    LabelsMasker::with_config(&cfg.cortical_atlas, vz_config.clone())?;
                let c = cortical_masker.fit_transform(&file_path)?;
                debug!(
                    subject_key = subject_key,
                    atlas_type = "cortical",
                    n_rois = c.shape()[0],
                    n_timepoints = c.shape()[1],
                    duration_ms = cortical_start.elapsed().as_millis(),
                    "voxel-wise z-score parcellation completed"
                );

                let subcortical_start = Instant::now();
                let subcortical_masker =
                    LabelsMasker::with_config(&cfg.subcortical_atlas, vz_config)?;
                let s = subcortical_masker.fit_transform(&file_path)?;
                debug!(
                    subject_key = subject_key,
                    atlas_type = "subcortical",
                    n_rois = s.shape()[0],
                    n_timepoints = s.shape()[1],
                    duration_ms = subcortical_start.elapsed().as_millis(),
                    "voxel-wise z-score parcellation completed"
                );

                (Some(c), Some(s))
            } else {
                (None, None)
            };

            // Prepare output directory.
            if let Some(parent) = output_path.parent() {
                std::fs::create_dir_all(parent)?;
            }

            // --force: remove the existing file so we start from a clean slate.
            // For the append path the file is opened in RW mode below.
            if cfg.force && output_path.exists() {
                fs::remove_file(&output_path)?;
            }

            let write_start = Instant::now();
            append_missing_datasets(
                &output_path,
                &missing,
                &cortical_raw,
                &subcortical_raw,
                cortical_std.as_ref(),
                subcortical_std.as_ref(),
                cortical_voxel_zscore.as_ref(),
                subcortical_voxel_zscore.as_ref(),
            )?;
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
                cortical_rois = cortical_raw.shape()[0],
                subcortical_rois = subcortical_raw.shape()[0],
                n_timepoints = cortical_raw.shape()[1],
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
        output_dir = %cfg.parcellated_ts_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI preprocessing pipeline completed"
    );

    Ok(())
}

/// Open an existing HDF5 file for appending (RW) or create a new one.
/// Writes only the datasets flagged as missing in `missing`.
///
/// Dataset naming convention:
///   tcp_cortical_raw             — per-atlas parcellated signal, no preprocessing
///   tcp_subcortical_raw          — per-atlas parcellated signal, no preprocessing
///   tcp_timeseries_raw           — cortical + subcortical concatenated, no preprocessing
///   tcp_cortical_standardized    — post-parcellation z-score (per-ROI temporal)
///   tcp_subcortical_standardized — post-parcellation z-score (per-ROI temporal)
///   tcp_timeseries_standardized  — cortical + subcortical concatenated, post-parcellation z-score
///   tcp_cortical_voxelzscore     — voxel-wise z-score applied before parcellation
///   tcp_subcortical_voxelzscore  — voxel-wise z-score applied before parcellation
///   tcp_timeseries_voxelzscore   — cortical + subcortical concatenated, voxel-wise z-score
fn append_missing_datasets(
    path: &Path,
    missing: &MissingDatasets,
    cortical_raw: &Array2<f32>,
    subcortical_raw: &Array2<f32>,
    cortical_std: Option<&Array2<f32>>,
    subcortical_std: Option<&Array2<f32>>,
    cortical_voxel_zscore: Option<&Array2<f32>>,
    subcortical_voxel_zscore: Option<&Array2<f32>>,
) -> Result<()> {
    let file = if path.exists() {
        hdf5::File::open_rw(path)?
    } else {
        hdf5::File::create(path)?
    };

    if missing.cortical_raw {
        write_2d_dataset(&file, cortical_raw, "tcp_cortical_raw")?;
    }
    if missing.subcortical_raw {
        write_2d_dataset(&file, subcortical_raw, "tcp_subcortical_raw")?;
    }
    if missing.timeseries_raw {
        let ts = concatenate(Axis(0), &[cortical_raw.view(), subcortical_raw.view()])?;
        write_2d_dataset(&file, &ts, "tcp_timeseries_raw")?;
    }

    if missing.cortical_std {
        if let Some(c) = cortical_std {
            write_2d_dataset(&file, c, "tcp_cortical_standardized")?;
        }
    }
    if missing.subcortical_std {
        if let Some(s) = subcortical_std {
            write_2d_dataset(&file, s, "tcp_subcortical_standardized")?;
        }
    }
    if missing.timeseries_std {
        if let (Some(c), Some(s)) = (cortical_std, subcortical_std) {
            let ts = concatenate(Axis(0), &[c.view(), s.view()])?;
            write_2d_dataset(&file, &ts, "tcp_timeseries_standardized")?;
        }
    }

    if missing.cortical_voxel_zscore {
        if let Some(c) = cortical_voxel_zscore {
            write_2d_dataset(&file, c, "tcp_cortical_voxelzscore")?;
        }
    }
    if missing.subcortical_voxel_zscore {
        if let Some(s) = subcortical_voxel_zscore {
            write_2d_dataset(&file, s, "tcp_subcortical_voxelzscore")?;
        }
    }
    if missing.timeseries_voxel_zscore {
        if let (Some(c), Some(s)) = (cortical_voxel_zscore, subcortical_voxel_zscore) {
            let ts = concatenate(Axis(0), &[c.view(), s.view()])?;
            write_2d_dataset(&file, &ts, "tcp_timeseries_voxelzscore")?;
        }
    }

    Ok(())
}

/// Write a single 2-D array as a named dataset into an open HDF5 file.
fn write_2d_dataset(file: &hdf5::File, data: &Array2<f32>, name: &str) -> Result<()> {
    let shape = data.shape();
    let ds = file
        .new_dataset::<f32>()
        .shape([shape[0], shape[1]])
        .create(name)?;
    ds.write_raw(data.as_slice().unwrap())?;
    Ok(())
}
