mod atlas;
mod timeseries;

use anyhow::Result;
use config::TCPfMRIProcessConfig;
use fc::ConnectivityMatrix;
use ndarray::{Array2, Array3};
use polars::prelude::*;
use signals::admm::ADMMConfig;
use signals::mvmd::{MVMDResult, MVMD};
use std::collections::{HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io};
use tracing::{debug, info, info_span, warn};

use atlas::{BrainAtlas, RoiType};
use timeseries::TimeseriesData;

pub fn run(cfg: &TCPfMRIProcessConfig) -> Result<()> {
    let run_start = Instant::now();

    info!(
        fmri_dir = %cfg.fmri_dir.display(),
        subjects = ?cfg.subject_file,
        output_dir = %cfg.output_dir.display(),
        cortical_atlas_lut = %cfg.cortical_atlas_lut.display(),
        subcortical_atlas_lut = %cfg.subcortical_atlas_lut.display(),
        force = cfg.force,
        "starting fMRI processing pipeline"
    );

    // Load subjects
    let fmri_dir = &cfg.fmri_dir;
    let available_subjects = get_directory_subject_directories(fmri_dir);
    let total_subjects_available = available_subjects.len();
    if total_subjects_available == 0 {
        panic!("no subject fMRI data found");
    }

    let target_subjects = cfg.subject_file.as_ref().map(read_subjects_file);

    let subjects_for_processing = filter_valid_subjects(available_subjects, target_subjects);

    if subjects_for_processing.is_empty() {
        panic!("no applicable subjects found for processing");
    }

    // Check if fMRI file available
    let required_scans = [String::from("task-hammerAP_run-01_bold.nii.h5")];
    let subjects_for_processing =
        filter_subjects_with_files(&cfg.fmri_dir, subjects_for_processing, &required_scans);

    let total_subjects = subjects_for_processing.len();

    info!(
        total_subjects_available = total_subjects_available,
        total_subjects_for_processing = total_subjects,
        "loaded subjects for processing",
    );

    let output_dir = &cfg.output_dir;
    if !output_dir.exists() {
        fs::create_dir_all(output_dir)?;
    }

    // Processing state
    let mut processed_count = 0usize;
    let mut skipped_count = 0usize;
    let mut error_count = 0usize;

    for (i, subject) in subjects_for_processing.iter().enumerate() {
        let subject_idx = i + 1;

        // Create a span for the entire subject processing
        let _subject_span = info_span!(
            "process_subject",
            subject = subject,
            subject_idx = subject_idx,
            total_subjects = total_subjects,
        )
        .entered();

        let fmri_subject_dir = cfg.fmri_dir.join(subject);

        for file in &required_scans {
            let file_start = Instant::now();

            // Load HDF5 file with parcellated timeseries
            let filepath = fmri_subject_dir.join(file);
            let file_name = filepath
                .file_stem()
                .and_then(|os| os.to_str())
                .map(|s| s.replace(".nii", ""));
            let task_name = file_name.as_deref().unwrap_or("unknown");

            // Check if subject already exists in output files (skip unless --force)
            let connectivity_path = output_dir.join(
                file_name
                    .as_ref()
                    .map_or("connectivity.h5".to_string(), |val| {
                        format!("{}__connectivity.h5", val)
                    }),
            );
            let subject_already_exists = subject_exists_in_h5(&connectivity_path, subject);
            if subject_already_exists && !cfg.force {
                skipped_count += 1;
                info!(
                    subject = subject,
                    subject_idx = subject_idx,
                    total_subjects = total_subjects,
                    task_name = task_name,
                    reason = "already_processed",
                    "skipping file (already exists in output, use --force to reprocess)"
                );
                continue;
            }

            let h5_file = match hdf5::File::open(&filepath) {
                Ok(f) => f,
                Err(e) => {
                    skipped_count += 1;
                    warn!(
                        subject = subject,
                        subject_idx = subject_idx,
                        total_subjects = total_subjects,
                        file = %filepath.display(),
                        error = %e,
                        reason = "failed_to_open_h5",
                        "skipping subject file"
                    );
                    continue;
                }
            };

            // Load cortical timeseries dataset (detrended + z-score standardized variant)
            let load_start = Instant::now();
            let cortical_ds = h5_file.dataset("tcp_cortical_detrended_standardized")?;
            let cortical_shape = cortical_ds.shape();
            let cortical_data = cortical_ds.read_2d::<f32>()?;

            // Load subcortical timeseries dataset (detrended + z-score standardized variant)
            let subcortical_ds = h5_file.dataset("tcp_subcortical_detrended_standardized")?;
            let subcortical_shape = subcortical_ds.shape();
            let subcortical_data = subcortical_ds.read_2d::<f32>()?;
            let load_duration_ms = load_start.elapsed().as_millis();

            debug!(
                subject = subject,
                task_name = task_name,
                file = %filepath.display(),
                cortical_shape = ?cortical_shape,
                subcortical_shape = ?subcortical_shape,
                load_duration_ms = load_duration_ms,
                "loaded parcellated timeseries"
            );

            // Get cortical ROIs by indices
            let cortical_lut = get_cortical_atlas_lut(&cfg.cortical_atlas_lut);
            let subcortical_lut = get_subcortical_atlas_lut(&cfg.subcortical_atlas_lut);
            let atlas = BrainAtlas::from_lut_maps(cortical_lut, subcortical_lut);

            // Get BOLD timeseries DataFrame
            let full_df = create_dual_atlas_dataframe(&cortical_data, &subcortical_data, &atlas)?;
            let n_rois = full_df.width();
            let n_timepoints = full_df.height();

            // Compute static functional connectivity matrix
            let fc_start = Instant::now();
            let corr_matrix = match ConnectivityMatrix::new(full_df.clone()) {
                Ok(m) => m,
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject,
                        subject_idx = subject_idx,
                        total_subjects = total_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "fc_computation_failed",
                        "skipping subject due to error"
                    );
                    continue;
                }
            };

            // Compute Fisher Z-transform
            let z_matrix = match corr_matrix.clone().into_fisher_z() {
                Ok(m) => m,
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject,
                        subject_idx = subject_idx,
                        total_subjects = total_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "fisher_transform_failed",
                        "skipping subject due to error"
                    );
                    continue;
                }
            };
            let fc_duration_ms = fc_start.elapsed().as_millis();

            // Extract connectivity data into arrays for HDF5
            let fc_labels = corr_matrix.labels.clone();
            let fc_corr_matrix = corr_matrix.to_ndarray()?;
            let fc_z_matrix = z_matrix.to_ndarray()?;

            // Write whole-signal connectivity to HDF5
            let fc_output_file = file_name
                .as_ref()
                .map_or("connectivity.h5".to_string(), |val| {
                    format!("{}__connectivity.h5", val)
                });
            let fc_connectivity_data = ConnectivityData {
                corr_matrix: fc_corr_matrix,
                z_matrix: fc_z_matrix,
                labels: fc_labels.clone(),
            };
            append_connectivity_h5(
                &output_dir.join(&fc_output_file),
                subject,
                &fc_connectivity_data,
                cfg.force,
            )?;

            debug!(
                subject = subject,
                task_name = task_name,
                n_rois = n_rois,
                fc_duration_ms = fc_duration_ms,
                output_file = fc_output_file,
                "computed whole-signal connectivity"
            );

            // Write BOLD timeseries to HDF5
            let timeseries_output_file = file_name
                .as_ref()
                .map_or("timeseries.h5".to_string(), |val| {
                    format!("{}__timeseries.h5", val)
                });

            let timeseries_data = match TimeseriesData::new(full_df.clone()) {
                Ok(ts) => ts,
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject,
                        subject_idx = subject_idx,
                        total_subjects = total_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "timeseries_extraction_failed",
                        "skipping subject due to error"
                    );
                    continue;
                }
            };

            append_timeseries_h5(
                &output_dir.join(&timeseries_output_file),
                subject,
                &timeseries_data,
                cfg.force,
            )?;

            debug!(
                subject = subject,
                task_name = task_name,
                n_rois = n_rois,
                n_timepoints = n_timepoints,
                output_file = timeseries_output_file,
                "saved BOLD timeseries"
            );

            // MVMD decomposition
            let mvmd_start = Instant::now();
            let admm_config = ADMMConfig::default();
            let mvmd = match MVMD::from_dataframe(&full_df, 1.0) {
                Ok(m) => m.with_admm_config(admm_config),
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject,
                        subject_idx = subject_idx,
                        total_subjects = total_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "mvmd_init_failed",
                        "skipping MVMD due to error"
                    );
                    continue;
                }
            };

            let num_modes = 10;
            let signal_decomposition = mvmd.decompose(num_modes);
            let mvmd_duration_ms = mvmd_start.elapsed().as_millis();

            let mvmd_output_file = file_name
                .as_ref()
                .map_or("mvmd_decomposition.h5".to_string(), |val| {
                    format!("{}__mvmd_decomposition.h5", val)
                });
            append_mvmd_results_h5(
                &output_dir.join(&mvmd_output_file),
                subject,
                &signal_decomposition,
                cfg.force,
            )?;

            debug!(
                subject = subject,
                task_name = task_name,
                num_modes = num_modes,
                mvmd_iterations = signal_decomposition.num_iterations,
                mvmd_duration_ms = mvmd_duration_ms,
                output_file = mvmd_output_file,
                "computed MVMD decomposition"
            );

            // Compute connectivity matrices for each MVMD mode
            let mode_fc_start = Instant::now();
            let mode_data = signal_decomposition.to_mode_dataframes()?;
            let num_modes = mode_data.len();
            let labels: Vec<String> = signal_decomposition.channels.clone();
            let num_rois = labels.len();

            let mut corr_matrices = Array3::<f64>::zeros((num_modes, num_rois, num_rois));
            let mut z_matrices = Array3::<f64>::zeros((num_modes, num_rois, num_rois));
            let mut center_frequencies = Vec::with_capacity(num_modes);

            for mode in mode_data {
                let mode_idx = mode.mode_index;
                center_frequencies.push(mode.center_frequency);

                let mode_corr = match ConnectivityMatrix::new(mode.timeseries) {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(
                            subject = subject,
                            mode_idx = mode_idx,
                            error = %e,
                            "failed to compute mode correlation matrix, filling with NaN"
                        );
                        // Fill this mode with NaN and continue
                        for i in 0..num_rois {
                            for j in 0..num_rois {
                                corr_matrices[[mode_idx, i, j]] = f64::NAN;
                                z_matrices[[mode_idx, i, j]] = f64::NAN;
                            }
                        }
                        continue;
                    }
                };

                // Extract correlation values into the 3D array
                let corr_arr = mode_corr.to_ndarray()?;
                corr_matrices
                    .slice_mut(ndarray::s![mode_idx, .., ..])
                    .assign(&corr_arr);

                // Compute Fisher Z-transform
                let mode_z = match mode_corr.into_fisher_z() {
                    Ok(m) => m,
                    Err(e) => {
                        warn!(
                            subject = subject,
                            mode_idx = mode_idx,
                            error = %e,
                            "failed to compute Fisher Z-transform for mode, filling with NaN"
                        );
                        for i in 0..num_rois {
                            for j in 0..num_rois {
                                z_matrices[[mode_idx, i, j]] = f64::NAN;
                            }
                        }
                        continue;
                    }
                };

                // Extract z-values into the 3D array
                let z_arr = mode_z.to_ndarray()?;
                z_matrices
                    .slice_mut(ndarray::s![mode_idx, .., ..])
                    .assign(&z_arr);
            }
            let mode_fc_duration_ms = mode_fc_start.elapsed().as_millis();

            // Write mode connectivity to HDF5
            let mode_fc_output_file = file_name
                .as_ref()
                .map_or("mode_connectivity.h5".to_string(), |val| {
                    format!("{}__mode_connectivity.h5", val)
                });
            let mode_connectivity_data = ModeConnectivityData {
                corr_matrices,
                z_matrices,
                center_frequencies,
                labels,
            };
            append_mode_connectivity_h5(
                &output_dir.join(&mode_fc_output_file),
                subject,
                &mode_connectivity_data,
                cfg.force,
            )?;

            debug!(
                subject = subject,
                task_name = task_name,
                num_modes = num_modes,
                mode_fc_duration_ms = mode_fc_duration_ms,
                output_file = mode_fc_output_file,
                "computed mode connectivity matrices"
            );

            let total_duration_ms = file_start.elapsed().as_millis();
            processed_count += 1;

            // Wide event: one comprehensive log per subject file processed
            info!(
                subject = subject,
                subject_idx = subject_idx,
                total_subjects = total_subjects,
                task_name = task_name,
                input_file = %filepath.display(),
                cortical_rois = cortical_shape[0],
                subcortical_rois = subcortical_shape[0],
                n_rois = n_rois,
                n_timepoints = n_timepoints,
                num_modes = num_modes,
                mvmd_iterations = signal_decomposition.num_iterations,
                load_duration_ms = load_duration_ms,
                fc_duration_ms = fc_duration_ms,
                mvmd_duration_ms = mvmd_duration_ms,
                mode_fc_duration_ms = mode_fc_duration_ms,
                total_duration_ms = total_duration_ms,
                outcome = "success",
                "subject processed"
            );
        }
    }

    let run_duration_ms = run_start.elapsed().as_millis();

    // Final summary wide event
    info!(
        total_subjects_available = total_subjects_available,
        total_subjects = total_subjects,
        processed_count = processed_count,
        skipped_count = skipped_count,
        error_count = error_count,
        total_duration_ms = run_duration_ms,
        output_dir = %cfg.output_dir.display(),
        outcome = if error_count == 0 { "success" } else { "completed_with_errors" },
        "fMRI processing pipeline completed"
    );

    Ok(())
}

fn get_directory_subject_directories(target_path: &PathBuf) -> Vec<String> {
    let prefix = "NDAR_INV";

    let dir_names: Vec<String> = fs::read_dir(target_path)
        .into_iter()
        .flatten()
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with(prefix) {
                        return Some(name.to_owned());
                    }
                }
            }
            None
        })
        .collect();

    dir_names
}

fn read_subjects_file(filename: &PathBuf) -> Vec<String> {
    let file = fs::File::open(filename).expect("failed to open the file");
    let reader = io::BufReader::new(file);

    reader
        .lines()
        .collect::<Result<Vec<String>, _>>()
        .expect("failed to read lines")
}

fn filter_valid_subjects(
    subjects_available: Vec<String>,
    target_subjects: Option<Vec<String>>,
) -> Vec<String> {
    match target_subjects {
        Some(targets) => {
            let available_set: HashSet<_> = subjects_available.iter().collect();
            targets
                .into_iter()
                .filter(|subject| available_set.contains(subject))
                .collect()
        }
        None => subjects_available,
    }
}

fn filter_subjects_with_files(
    fmri_dir: &PathBuf,
    subjects: Vec<String>,
    files: &[String],
) -> Vec<String> {
    subjects
        .into_iter()
        .filter(|subject| {
            let subject_dir = fmri_dir.join(subject);
            files.iter().all(|file| subject_dir.join(file).exists())
        })
        .collect()
}

fn get_cortical_atlas_lut(filename: &PathBuf) -> HashMap<String, u32> {
    let file = fs::File::open(filename).expect("Failed to open file");
    let reader = BufReader::new(file);
    let mut cortical_roi_map = HashMap::new();

    // Use a peekable iterator so we can skip headers
    let mut lines = reader.lines().peekable();

    while let Some(Ok(line)) = lines.next() {
        let line = line.trim();

        if line.is_empty() || !line.starts_with("17networks") {
            continue;
        }

        let roi_id = line.to_string();

        if let Some(Ok(params_line)) = lines.next() {
            let item_number_str = params_line
                .split_whitespace()
                .next()
                .expect("Parameter line empty");

            let item_number: u32 = item_number_str.parse().expect("Parse fail");

            let item_idx = item_number - 1;

            cortical_roi_map.insert(roi_id, item_idx);
        }
    }
    cortical_roi_map
}

fn get_subcortical_atlas_lut(filename: &PathBuf) -> HashMap<String, u32> {
    let subcortical_lut_file = fs::File::open(filename)
        .expect("Failed to open subcortical atlas file specified in config");

    let mut subcortical_roi_map = HashMap::new();
    let reader = BufReader::new(subcortical_lut_file);

    for (index, line_result) in reader.lines().enumerate() {
        let line = line_result.expect("Failed to read line from file");
        let roi_id = line.trim().to_string();

        let item_number = index as u32;

        subcortical_roi_map.insert(roi_id, item_number);
    }

    subcortical_roi_map
}

fn create_dual_atlas_dataframe(
    cortical_data: &Array2<f32>,
    subcortical_data: &Array2<f32>,
    atlas: &BrainAtlas,
) -> Result<DataFrame, PolarsError> {
    let columns: Vec<Column> = atlas
        .entries
        .iter()
        .map(|entry| {
            // Get a row view from the ndarray based on the ROI index
            // .row() returns an ArrayView1, which we convert to a slice for Polars
            let roi_signal_view = match &entry.metadata {
                RoiType::Cortical { .. } => cortical_data.row(entry.index as usize),
                RoiType::Subcortical { .. } => subcortical_data.row(entry.index as usize),
            };

            // Convert the ndarray view to a slice.
            // Note: This works because ndarray rows are typically contiguous.
            let roi_signal_slice = roi_signal_view
                .as_slice()
                .expect("Data in ndarray must be contiguous for Polars conversion");

            Series::new(entry.id.as_str().into(), roi_signal_slice).into()
        })
        .collect();

    DataFrame::new(columns)
}

/// Whole-signal connectivity results for HDF5 output
struct ConnectivityData {
    /// Correlation matrix: (N_ROIs x N_ROIs)
    corr_matrix: Array2<f64>,
    /// Fisher Z-transformed matrix: (N_ROIs x N_ROIs)
    z_matrix: Array2<f64>,
    /// ROI labels
    labels: Vec<String>,
}

/// Opens or creates an HDF5 file for appending subject data
fn open_or_create_h5(path: &Path) -> Result<hdf5::File> {
    if path.exists() {
        Ok(hdf5::File::open_rw(path)?)
    } else {
        Ok(hdf5::File::create(path)?)
    }
}

/// Checks if a subject group already exists in an HDF5 file
fn subject_exists_in_h5(path: &Path, subject: &str) -> bool {
    if !path.exists() {
        return false;
    }
    if let Ok(file) = hdf5::File::open(path) {
        file.group(subject).is_ok()
    } else {
        false
    }
}

/// Creates a subject group, optionally replacing an existing one if force is true.
fn create_subject_group(file: &hdf5::File, name: &str, force: bool) -> Result<hdf5::Group> {
    if force && file.group(name).is_ok() {
        file.unlink(name)?;
    }
    Ok(file.create_group(name)?)
}

/// Appends connectivity data for a subject.
fn append_connectivity_h5(
    path: &Path,
    subject: &str,
    data: &ConnectivityData,
    force: bool,
) -> Result<()> {
    let file = open_or_create_h5(path)?;

    let num_rois = data.corr_matrix.shape()[0];

    // Create subject group (replace if force=true)
    let group = create_subject_group(&file, subject, force)?;

    // Write correlation matrix: (N_ROIs x N_ROIs)
    let corr_ds = group
        .new_dataset::<f64>()
        .shape([num_rois, num_rois])
        .create("corr_matrix")?;
    corr_ds.write_raw(data.corr_matrix.as_slice().unwrap())?;

    // Write z-matrix: (N_ROIs x N_ROIs)
    let z_ds = group
        .new_dataset::<f64>()
        .shape([num_rois, num_rois])
        .create("z_matrix")?;
    z_ds.write_raw(data.z_matrix.as_slice().unwrap())?;

    // Write shared metadata to root if not already present
    let root = file.group("/")?;
    if root.attr("labels").is_err() {
        // ROI labels as comma-separated string (shared across all subjects)
        let labels_str = data.labels.join(",");
        let labels_unicode: hdf5::types::VarLenUnicode = labels_str.parse().unwrap();
        root.new_attr::<hdf5::types::VarLenUnicode>()
            .shape([1])
            .create("labels")?
            .write_raw(&[labels_unicode])?;

        // Number of ROIs (shared across all subjects)
        root.new_attr::<u32>()
            .shape([1])
            .create("num_rois")?
            .write_raw(&[num_rois as u32])?;
    }

    Ok(())
}

/// Mode connectivity results for HDF5 output
struct ModeConnectivityData {
    /// Correlation matrices for each mode: (K modes x N_ROIs x N_ROIs)
    corr_matrices: Array3<f64>,
    /// Fisher Z-transformed matrices for each mode: (K modes x N_ROIs x N_ROIs)
    z_matrices: Array3<f64>,
    /// Center frequency for each mode: (K,)
    center_frequencies: Vec<f64>,
    /// ROI/channel labels
    labels: Vec<String>,
}

/// Appends mode connectivity data for a subject.
fn append_mode_connectivity_h5(
    path: &Path,
    subject: &str,
    data: &ModeConnectivityData,
    force: bool,
) -> Result<()> {
    let file = open_or_create_h5(path)?;

    let num_modes = data.corr_matrices.shape()[0];
    let num_rois = data.corr_matrices.shape()[1];

    // Create subject group (replace if force=true)
    let group = create_subject_group(&file, subject, force)?;

    // Write correlation matrices: (K x N_ROIs x N_ROIs)
    let corr_ds = group
        .new_dataset::<f64>()
        .shape([num_modes, num_rois, num_rois])
        .create("corr_matrices")?;
    corr_ds.write_raw(data.corr_matrices.as_slice().unwrap())?;

    // Write z-matrices: (K x N_ROIs x N_ROIs)
    let z_ds = group
        .new_dataset::<f64>()
        .shape([num_modes, num_rois, num_rois])
        .create("z_matrices")?;
    z_ds.write_raw(data.z_matrices.as_slice().unwrap())?;

    // Write center frequencies as dataset: (K,)
    let cf_ds = group
        .new_dataset::<f64>()
        .shape([num_modes])
        .create("center_frequencies")?;
    cf_ds.write_raw(&data.center_frequencies)?;

    // Write per-subject metadata as attributes on subject group
    group
        .new_attr::<u32>()
        .shape([1])
        .create("num_modes")?
        .write_raw(&[num_modes as u32])?;

    // Write shared metadata to root if not already present
    let root = file.group("/")?;
    if root.attr("labels").is_err() {
        // ROI labels as comma-separated string (shared across all subjects)
        let labels_str = data.labels.join(",");
        let labels_unicode: hdf5::types::VarLenUnicode = labels_str.parse().unwrap();
        root.new_attr::<hdf5::types::VarLenUnicode>()
            .shape([1])
            .create("labels")?
            .write_raw(&[labels_unicode])?;

        // Number of ROIs (shared across all subjects)
        root.new_attr::<u32>()
            .shape([1])
            .create("num_rois")?
            .write_raw(&[num_rois as u32])?;
    }

    Ok(())
}

/// Appends BOLD timeseries data for a subject.
fn append_timeseries_h5(
    path: &Path,
    subject: &str,
    data: &TimeseriesData,
    force: bool,
) -> Result<()> {
    let file = open_or_create_h5(path)?;

    let n_timepoints = data.get_timepoint_count();
    let n_rois = data.get_channel_count();

    // Create subject group (replace if force=true)
    let group = create_subject_group(&file, subject, force)?;

    // Write timeseries: (T x N_ROIs)
    let ts_ds = group
        .new_dataset::<f64>()
        .shape([n_timepoints, n_rois])
        .create("timeseries")?;

    let as_array = data.to_ndarray()?;

    ts_ds.write_raw(
        as_array
            .as_slice()
            .expect("failed to convert array to slice"),
    )?;

    // Write per-subject metadata
    group
        .new_attr::<u32>()
        .shape([1])
        .create("n_timepoints")?
        .write_raw(&[n_timepoints as u32])?;

    // Write shared metadata to root if not already present
    let root = file.group("/")?;
    if root.attr("labels").is_err() {
        // ROI labels as comma-separated string (shared across all subjects)
        let labels_str = data.labels.join(",");
        let labels_unicode: hdf5::types::VarLenUnicode = labels_str.parse().unwrap();
        root.new_attr::<hdf5::types::VarLenUnicode>()
            .shape([1])
            .create("labels")?
            .write_raw(&[labels_unicode])?;

        // Number of ROIs (shared across all subjects)
        root.new_attr::<u32>()
            .shape([1])
            .create("num_rois")?
            .write_raw(&[n_rois as u32])?;
    }

    Ok(())
}

/// Appends MVMD results for a subject.
fn append_mvmd_results_h5(path: &Path, subject: &str, res: &MVMDResult, force: bool) -> Result<()> {
    let file = open_or_create_h5(path)?;

    // Create subject group (replace if force=true)
    let group = create_subject_group(&file, subject, force)?;

    // Write modes dataset: (K modes x C channels x T time-points)
    let modes_shape = res.modes.shape();
    let modes_ds = group
        .new_dataset::<f64>()
        .shape([modes_shape[0], modes_shape[1], modes_shape[2]])
        .create("modes")?;
    modes_ds.write_raw(res.modes.as_slice().unwrap())?;

    // Write center_frequencies dataset: (iter x K)
    let cf_shape = res.center_frequencies.shape();
    let cf_ds = group
        .new_dataset::<f64>()
        .shape([cf_shape[0], cf_shape[1]])
        .create("center_frequencies")?;
    cf_ds.write_raw(res.center_frequencies.as_slice().unwrap())?;

    // Write final_frequencies dataset: (K,)
    let ff_ds = group
        .new_dataset::<f64>()
        .shape([res.final_frequencies.len()])
        .create("final_frequencies")?;
    ff_ds.write_raw(res.final_frequencies.as_slice().unwrap())?;

    // Write num_iterations as attribute on subject group
    group
        .new_attr::<u32>()
        .shape([1])
        .create("num_iterations")?
        .write_raw(&[res.num_iterations])?;

    // Write shared metadata to root if not already present
    let root = file.group("/")?;
    if root.attr("channels").is_err() {
        // Channel/ROI labels as comma-separated string (shared across all subjects)
        let channels_str = res.channels.join(",");
        let channels_unicode: hdf5::types::VarLenUnicode = channels_str.parse().unwrap();
        root.new_attr::<hdf5::types::VarLenUnicode>()
            .shape([1])
            .create("channels")?
            .write_raw(&[channels_unicode])?;
    }

    Ok(())
}
