mod atlas;
mod timeseries;

use anyhow::Result;
use config::bids_filename::{BidsFilename, find_bids_files};
use config::bids_subject_id::BidsSubjectId;
use config::pipeline_config::FmriProcessConfig;
use fc::ConnectivityMatrix;
use ndarray::{Array2, Array3};
use polars::prelude::*;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{fs, io};
use tracing::{debug, info, info_span, warn};

use atlas::{BrainAtlas, RoiType};
use hdf5_io::{H5Attr, write_attrs, write_dataset};

pub fn run(cfg: &FmriProcessConfig) -> Result<()> {
    let run_start = Instant::now();

    info!(
        fmri_dir = %cfg.bold_ts_dir.display(),
        subjects = ?cfg.subject_file,
        output_dir = %cfg.output_dir.display(),
        cortical_atlas_lut = %cfg.cortical_atlas_lut.display(),
        subcortical_atlas_lut = %cfg.subcortical_atlas_lut.display(),
        force = cfg.force,
        "starting fMRI processing pipeline"
    );

    // Load subjects
    let bold_dir = &cfg.bold_ts_dir;
    let available_subjects = get_directory_subject_directories(bold_dir);
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
    let hammer_task_scan_pairs: &[(&str, &str)] = &[
        ("task", "hammerAP"),
        ("run", "01"),
        ("space", "MNI152NLin2009cAsym"),
        ("res", "2"),
        ("desc", "preproc"),
    ];
    let subject_hammer_map = map_subjects_with_files(
        &cfg.bold_ts_dir,
        subjects_for_processing,
        hammer_task_scan_pairs,
    );
    let total_hammer_subjects = subject_hammer_map.len();

    info!(
        total_subjects_with_hammer_available = total_subjects_available,
        total_subjects_with_hammer = total_hammer_subjects,
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

    for (i, (subject_id, hammer_files)) in subject_hammer_map.iter().enumerate() {
        let subject_idx = i + 1;

        // Create a span for the entire subject processing
        let _subject_span = info_span!(
            "process_subject",
            subject = subject_id,
            subject_idx = subject_idx,
            total_subjects = total_hammer_subjects,
        )
        .entered();

        for file_path in hammer_files {
            let file_start = Instant::now();

            // Derive task name by stripping the subject pair — shared across all subjects
            let task_name =
                BidsFilename::parse(file_path.file_name().and_then(|n| n.to_str()).unwrap_or(""))
                    .without(&["sub"])
                    .to_stem();
            let task_name = task_name.as_str();

            let fc_output_file = format!("{}_fc.h5", task_name);
            let timeseries_output_file = format!("{}_ts.h5", task_name);

            // TODO: Change this to use a manifest file written at the end of the processing pipeline
            // Check if subject already exists in output files (skip unless --force)
            let connectivity_path = output_dir.join(format!("{}_connectivity.h5", task_name));
            let subject_already_exists = subject_exists_in_h5(&connectivity_path, subject_id);
            if subject_already_exists && !cfg.force {
                skipped_count += 1;
                info!(
                    subject = subject_id,
                    subject_idx = subject_idx,
                    total_subjects = total_hammer_subjects,
                    task_name = task_name,
                    reason = "already_processed",
                    "skipping file (already exists in output, use --force to reprocess)"
                );
                continue;
            }

            // Open the HDF file
            let h5_file = match hdf5::File::open(&file_path) {
                Ok(f) => f,
                Err(e) => {
                    skipped_count += 1;
                    warn!(
                        subject = subject_id,
                        subject_idx = subject_idx,
                        total_subjects = total_hammer_subjects,
                        file = %file_path.display(),
                        error = %e,
                        reason = "failed_to_open_h5",
                        "skipping subject file"
                    );
                    continue;
                }
            };

            // Load cortical timeseries dataset
            let load_start = Instant::now();
            let cortical_ds = h5_file.dataset("tcp_cortical_raw")?;
            let cortical_shape = cortical_ds.shape();
            let cortical_data = cortical_ds.read_2d::<f32>()?;

            // Load subcortical timeseries dataset
            let subcortical_ds = h5_file.dataset("tcp_subcortical_raw")?;
            let subcortical_shape = subcortical_ds.shape();
            let subcortical_data = subcortical_ds.read_2d::<f32>()?;
            let load_duration_ms = load_start.elapsed().as_millis();

            debug!(
                subject = subject_id,
                task_name = task_name,
                file = %file_path.display(),
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

            ////////////////////////////
            // Whole-band Time Series //
            ////////////////////////////

            let ts_labels: Vec<String> = full_df
                .get_column_names()
                .iter()
                .map(|s| s.to_string())
                .collect();
            let ts_n_rois = full_df.width();
            let ts_n_tp = full_df.height();

            // Build flat (n_timepoints x n_rois) row-major buffer from full_df
            let mut ts_flat = vec![0f32; ts_n_tp * ts_n_rois];
            for (col_idx, col_name) in ts_labels.iter().enumerate() {
                if let Ok(col) = full_df.column(col_name.as_str()) {
                    if let Ok(values) = col.f32() {
                        for (row_idx, val) in values.iter().enumerate() {
                            ts_flat[row_idx * ts_n_rois + col_idx] = val.unwrap_or(f32::NAN);
                        }
                    }
                }
            }

            let ts_file = hdf5_io::open_or_create(&output_dir.join(&timeseries_output_file))?;
            let ts_group = hdf5_io::open_or_create_group(&ts_file, subject_id, cfg.force)?;

            write_dataset(
                &ts_group,
                "timeseries",
                &ts_flat,
                &[ts_n_tp, ts_n_rois],
                None,
            )?;

            // For each block group, build a DataFrame via the atlas and write as a single
            // dataset directly under the subject group (named after the block)
            let blocks_group = h5_file.group("blocks")?;
            for block_group in blocks_group.groups()? {
                let block_cortical_data = block_group.dataset("cortical_raw")?.read_2d::<f32>()?;
                let block_subcortical_data =
                    block_group.dataset("subcortical_raw")?.read_2d::<f32>()?;

                let block_df = create_dual_atlas_dataframe(
                    &block_cortical_data,
                    &block_subcortical_data,
                    &atlas,
                )?;
                let block_n_rois = block_df.width();
                let block_n_tp = block_df.height();

                let mut block_flat = vec![0f32; block_n_tp * block_n_rois];
                for (col_idx, col_name) in ts_labels.iter().enumerate() {
                    if let Ok(col) = block_df.column(col_name.as_str()) {
                        if let Ok(values) = col.f32() {
                            for (row_idx, val) in values.iter().enumerate() {
                                block_flat[row_idx * block_n_rois + col_idx] =
                                    val.unwrap_or(f32::NAN);
                            }
                        }
                    }
                }

                let block_name = block_group.name();
                let block_leaf = block_name.rsplit('/').next().unwrap_or(block_name.as_str());
                write_dataset(
                    &ts_group,
                    block_leaf,
                    &block_flat,
                    &[block_n_tp, block_n_rois],
                    None,
                )?;
            }

            let ts_root = ts_file.group("/")?;
            if ts_root.attr("labels").is_err() {
                write_attrs(
                    &ts_root,
                    &[
                        H5Attr::string("labels", ts_labels.join(",")),
                        H5Attr::u32("num_rois", ts_n_rois as u32),
                    ],
                )?;
            }

            debug!(
                subject = subject_id,
                task_name = task_name,
                n_rois = ts_n_rois,
                n_timepoints = ts_n_tp,
                output_file = timeseries_output_file,
                "saved BOLD timeseries"
            );

            ///////////////////////////////
            // Whole-band FC Computation //
            ///////////////////////////////

            // Compute static whole-band functional connectivity matrix
            let fc_wb_start = Instant::now();
            let corr_matrix_wb = match ConnectivityMatrix::new(full_df.clone()) {
                Ok(m) => m,
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject_id,
                        subject_idx = subject_idx,
                        total_subjects = total_hammer_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "fc_computation_failed",
                        "skipping subject due to error"
                    );
                    continue;
                }
            };

            // Compute Fisher Z-transform
            let z_matrix_wb = match corr_matrix_wb.clone().into_fisher_z() {
                Ok(m) => m,
                Err(e) => {
                    error_count += 1;
                    warn!(
                        subject = subject_id,
                        subject_idx = subject_idx,
                        total_subjects = total_hammer_subjects,
                        task_name = task_name,
                        error = %e,
                        reason = "fisher_transform_failed",
                        "skipping subject due to error"
                    );
                    continue;
                }
            };
            let fc_wb_duration_ms = fc_wb_start.elapsed().as_millis();

            // Write whole-band connectivity to HDF5
            let fc_output_path = output_dir.join(&fc_output_file);
            let fc_file = hdf5_io::open_or_create(&fc_output_path)?;
            let subject_fc_group = hdf5_io::open_or_create_group(&fc_file, subject_id, cfg.force)?;

            let fc_wb_labels = corr_matrix_wb.labels.clone();
            let fc_wb_corr_matrix = corr_matrix_wb.to_ndarray()?;
            let fc_wb_z_matrix = z_matrix_wb.to_ndarray()?;
            let num_rois_fc = fc_wb_corr_matrix.shape()[0];

            write_dataset(
                &subject_fc_group,
                "corr_matrix",
                fc_wb_corr_matrix.as_slice().unwrap(),
                &[num_rois_fc, num_rois_fc],
                None,
            )?;
            write_dataset(
                &subject_fc_group,
                "z_matrix",
                fc_wb_z_matrix.as_slice().unwrap(),
                &[num_rois_fc, num_rois_fc],
                None,
            )?;

            let fc_root = fc_file.group("/")?;
            if fc_root.attr("labels").is_err() {
                write_attrs(
                    &fc_root,
                    &[
                        H5Attr::string("labels", fc_wb_labels.join(",")),
                        H5Attr::u32("num_rois", num_rois_fc as u32),
                    ],
                )?;
            }

            debug!(
                subject = subject_id,
                task_name = task_name,
                n_rois = n_rois,
                fc_duration_ms = fc_wb_duration_ms,
                output_file = fc_output_file,
                "computed whole-signal connectivity"
            );

            //////////////////////////////////////////
            // Block-wise Whole-band FC Computation //
            //////////////////////////////////////////

            let group_path = "blocks";
            let trial_blocks = h5_file.group(group_path)?;
            println!("Datasets in group '{}':", group_path);

            // Iterate over datasets directly
            for block_group in trial_blocks.groups()? {
                let fc_wb_block_start = Instant::now();
                println!("Group: {}", block_group.name());

                let cortical_ds = block_group.dataset("cortical_raw")?;
                let cortical_data = cortical_ds.read_2d::<f32>()?;

                // Load subcortical timeseries dataset
                let subcortical_ds = block_group.dataset("subcortical_raw")?;
                let subcortical_data = subcortical_ds.read_2d::<f32>()?;

                let block_df =
                    create_dual_atlas_dataframe(&cortical_data, &subcortical_data, &atlas)?;
                let n_rois = block_df.width();
                // let n_timepoints = block_df.height();

                // Compute static whole-band functional connectivity matrix
                let block_corr_matrix_wb = match ConnectivityMatrix::new(block_df.clone()) {
                    Ok(m) => m,
                    Err(e) => {
                        error_count += 1;
                        warn!(
                            subject = subject_id,
                            subject_idx = subject_idx,
                            total_subjects = total_hammer_subjects,
                            task_name = task_name,
                            error = %e,
                            reason = "fc_computation_failed",
                            "skipping subject due to error"
                        );
                        continue;
                    }
                };

                // Compute Fisher Z-transform
                let block_z_matrix_wb = match block_corr_matrix_wb.clone().into_fisher_z() {
                    Ok(m) => m,
                    Err(e) => {
                        error_count += 1;
                        warn!(
                            subject = subject_id,
                            subject_idx = subject_idx,
                            total_subjects = total_hammer_subjects,
                            task_name = task_name,
                            error = %e,
                            reason = "fisher_transform_failed",
                            "skipping subject due to error"
                        );
                        continue;
                    }
                };

                // Extract connectivity data into arrays for HDF5
                let block_fc_wb_corr_matrix = block_corr_matrix_wb.to_ndarray()?;
                let block_fc_wb_z_matrix = block_z_matrix_wb.to_ndarray()?;

                // Write block connectivity to {subject_id}/{block_name} subgroup
                let block_name = block_group.name();
                let block_leaf = block_name.rsplit('/').next().unwrap_or(block_name.as_str());
                let block_fc_subgroup =
                    hdf5_io::open_or_create_group(&subject_fc_group, block_leaf, cfg.force)?;

                let block_num_rois = block_fc_wb_corr_matrix.shape()[0];
                write_dataset(
                    &block_fc_subgroup,
                    "corr_matrix",
                    block_fc_wb_corr_matrix.as_slice().unwrap(),
                    &[block_num_rois, block_num_rois],
                    None,
                )?;
                write_dataset(
                    &block_fc_subgroup,
                    "z_matrix",
                    block_fc_wb_z_matrix.as_slice().unwrap(),
                    &[block_num_rois, block_num_rois],
                    None,
                )?;

                let fc_wb_block_duration_ms = fc_wb_block_start.elapsed().as_millis();

                debug!(
                    subject = subject_id,
                    task_name = task_name,
                    n_rois = n_rois,
                    block_name = block_group.name(),
                    fc_block_duration_ms = fc_wb_block_duration_ms,
                    output_file = fc_output_file,
                    "computed block-wise whole-signal connectivity"
                );
            }

            // // Compute connectivity matrices for each MVMD mode
            // let mode_fc_start = Instant::now();
            // let mode_data = signal_decomposition.to_mode_dataframes()?;
            // let num_modes = mode_data.len();
            // let labels: Vec<String> = signal_decomposition.channels.clone();
            // let num_rois = labels.len();

            // let mut corr_matrices = Array3::<f64>::zeros((num_modes, num_rois, num_rois));
            // let mut z_matrices = Array3::<f64>::zeros((num_modes, num_rois, num_rois));
            // let mut center_frequencies = Vec::with_capacity(num_modes);

            // for mode in mode_data {
            //     let mode_idx = mode.mode_index;
            //     center_frequencies.push(mode.center_frequency);

            //     let mode_corr = match ConnectivityMatrix::new(mode.timeseries) {
            //         Ok(m) => m,
            //         Err(e) => {
            //             warn!(
            //                 subject = subject_id,
            //                 mode_idx = mode_idx,
            //                 error = %e,
            //                 "failed to compute mode correlation matrix, filling with NaN"
            //             );
            //             // Fill this mode with NaN and continue
            //             for i in 0..num_rois {
            //                 for j in 0..num_rois {
            //                     corr_matrices[[mode_idx, i, j]] = f64::NAN;
            //                     z_matrices[[mode_idx, i, j]] = f64::NAN;
            //                 }
            //             }
            //             continue;
            //         }
            //     };

            //     // Extract correlation values into the 3D array
            //     let corr_arr = mode_corr.to_ndarray()?;
            //     corr_matrices
            //         .slice_mut(ndarray::s![mode_idx, .., ..])
            //         .assign(&corr_arr);

            //     // Compute Fisher Z-transform
            //     let mode_z = match mode_corr.into_fisher_z() {
            //         Ok(m) => m,
            //         Err(e) => {
            //             warn!(
            //                 subject = subject_id,
            //                 mode_idx = mode_idx,
            //                 error = %e,
            //                 "failed to compute Fisher Z-transform for mode, filling with NaN"
            //             );
            //             for i in 0..num_rois {
            //                 for j in 0..num_rois {
            //                     z_matrices[[mode_idx, i, j]] = f64::NAN;
            //                 }
            //             }
            //             continue;
            //         }
            //     };

            //     // Extract z-values into the 3D array
            //     let z_arr = mode_z.to_ndarray()?;
            //     z_matrices
            //         .slice_mut(ndarray::s![mode_idx, .., ..])
            //         .assign(&z_arr);
            // }
            // let mode_fc_duration_ms = mode_fc_start.elapsed().as_millis();

            // // Write mode connectivity to HDF5
            // let mode_fc_output_file = format!("{}_mode_connectivity.h5", task_name);
            // {
            //     let mode_file = hdf5_io::open_or_create(&output_dir.join(&mode_fc_output_file))?;
            //     let mode_group = hdf5_io::open_or_create_group(&mode_file, subject_id, cfg.force)?;

            //     write_dataset(
            //         &mode_group,
            //         "corr_matrices",
            //         corr_matrices.as_slice().unwrap(),
            //         &[num_modes, num_rois, num_rois],
            //         None,
            //     )?;
            //     write_dataset(
            //         &mode_group,
            //         "z_matrices",
            //         z_matrices.as_slice().unwrap(),
            //         &[num_modes, num_rois, num_rois],
            //         None,
            //     )?;
            //     write_dataset(
            //         &mode_group,
            //         "center_frequencies",
            //         &center_frequencies,
            //         &[num_modes],
            //         None,
            //     )?;
            //     write_attrs(&mode_group, &[H5Attr::u32("num_modes", num_modes as u32)])?;

            //     let mode_root = mode_file.group("/")?;
            //     if mode_root.attr("labels").is_err() {
            //         write_attrs(
            //             &mode_root,
            //             &[
            //                 H5Attr::string("labels", labels.join(",")),
            //                 H5Attr::u32("num_rois", num_rois as u32),
            //             ],
            //         )?;
            //     }
            // }

            // debug!(
            //     subject = subject_id,
            //     task_name = task_name,
            //     num_modes = num_modes,
            //     mode_fc_duration_ms = mode_fc_duration_ms,
            //     output_file = mode_fc_output_file,
            //     "computed mode connectivity matrices"
            // );

            let total_duration_ms = file_start.elapsed().as_millis();
            processed_count += 1;

            // Wide event: one comprehensive log per subject file processed
            info!(
                subject = subject_id,
                subject_idx = subject_idx,
                total_subjects = total_hammer_subjects,
                task_name = task_name,
                input_file = %file_path.display(),
                cortical_rois = cortical_shape[0],
                subcortical_rois = subcortical_shape[0],
                n_rois = n_rois,
                n_timepoints = n_timepoints,
                // num_modes = num_modes,
                // mvmd_iterations = signal_decomposition.num_iterations,
                load_duration_ms = load_duration_ms,
                fc_duration_ms = fc_wb_duration_ms,
                // mvmd_duration_ms = mvmd_duration_ms,
                // mode_fc_duration_ms = mode_fc_duration_ms,
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
        total_subjects = total_hammer_subjects,
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
    fs::read_dir(target_path)
        .into_iter()
        .flatten()
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_dir() {
                if let Some(name) = path.file_name().and_then(|s| s.to_str()) {
                    if name.starts_with("sub-") {
                        return Some(name.to_owned());
                    }
                }
            }
            None
        })
        .collect()
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
                .map(|t| BidsSubjectId::parse(&t).to_dir_name())
                .filter(|dir_name| available_set.contains(dir_name))
                .collect()
        }
        None => subjects_available,
    }
}

fn filter_subjects_with_files(
    fmri_dir: &PathBuf,
    subjects: Vec<String>,
    required_pairs: &[(&str, &str)],
) -> Vec<String> {
    subjects
        .into_iter()
        .filter(|subject| {
            let subject_dir = fmri_dir.join(subject);
            !find_bids_files(&subject_dir, required_pairs, Some("bold"), Some(".h5")).is_empty()
        })
        .collect()
}

fn map_subjects_with_files(
    fmri_dir: &PathBuf,
    subjects: Vec<String>,
    required_pairs: &[(&str, &str)],
) -> BTreeMap<String, Vec<PathBuf>> {
    subjects
        .into_iter()
        .filter_map(|subject| {
            let fmri_subject_dir = fmri_dir.join(&subject);
            let task_files =
                find_bids_files(&fmri_subject_dir, required_pairs, Some("bold"), Some(".h5"));
            if task_files.is_empty() {
                return None;
            }
            Some((subject, task_files))
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

/// Checks if a subject group already exists in an HDF5 file.
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
