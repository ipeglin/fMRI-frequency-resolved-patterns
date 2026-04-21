use anyhow::Result;
use hdf5::types::VarLenUnicode;
use ndarray::{Array2, s};
use polars::prelude::*;
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    time::Instant,
};
use tracing::{info, warn};
use utils::bids_filename::BidsFilename;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;

const TR_SECONDS: f64 = 0.8;

struct BlockTimeseries {
    block_id: i32,
    trial_type: String,
    onset_s: f64,
    block_end_s: f64,
    cortical: Array2<f32>,
    subcortical: Array2<f32>,
    cortical_std: Option<Array2<f32>>,
    subcortical_std: Option<Array2<f32>>,
}

/// Condition-specific onset and duration lists for mixed block/event-related GLM modeling.
///
/// Both levels derive from the same `events_tsv_to_blocks` output:
/// - `trial_level`: one entry per individual stimulus presentation
/// - `block_level`: one entry per block (onset = first trial, duration = full block)
pub struct GlmConditions {
    /// condition -> (onset_s, duration_s) for individual stimulus events
    pub trial_level: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
    /// condition -> (onset_s, duration_s) for block-level regressors
    pub block_level: BTreeMap<String, (Vec<f64>, Vec<f64>)>,
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    let _run_start = Instant::now();

    // Disable HDF5 advisory file locking — required on macOS and some networked filesystems
    // where POSIX locks return EAGAIN (errno 35).
    unsafe { std::env::set_var("HDF5_USE_FILE_LOCKING", "FALSE") };

    info!(
        tcp_repo_dir = % cfg.tcp_repo_dir.display(),
        parcellated_ts_dir = %cfg.parcellated_ts_dir.display(),
        task_regressors_output_dir = %cfg.task_regressors_output_dir.display(),
        force = cfg.force,
        "starting fMRI trial segmentation"
    );

    // BTreeMap<formatted_subject_id, subject_dir>
    let subjects: BTreeMap<String, PathBuf> = fs::read_dir(&cfg.parcellated_ts_dir)?
        .filter_map(|entry_result| entry_result.ok())
        .filter_map(|entry| {
            let path = entry.path();
            if !path.is_dir() {
                return None;
            }
            let id = path.file_name()?.to_str()?;
            let formatted = BidsSubjectId::parse(id).to_dir_name();
            Some((formatted, path))
        })
        .collect();

    let total_subjects = subjects.len();
    info!(num_subjects = total_subjects, "found subject directories");

    let mut subject_idx = 0;

    for (formatted_id, dir) in &subjects {
        subject_idx += 1;
        let _subject_span = tracing::info_span!(
            "subject",
            subject = %formatted_id,
            subject_idx,
            total_subjects
        )
        .entered();
        let available_task_timeseries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|entry| entry.ok())
            .map(|entry| entry.path())
            .filter(|path| path.is_file())
            .filter_map(|path| {
                let name = path.file_name()?.to_str()?;
                if name.starts_with('.') || !name.ends_with(".h5") {
                    return None;
                }
                let parsed = BidsFilename::parse(name);
                if parsed.get("task") == Some("hammerAP") {
                    Some(path)
                } else {
                    info!(
                        file = %path.file_stem().and_then(|n| n.to_str())?,
                        "skipping non-hammerAP file"
                    );
                    None
                }
            })
            .collect();

        for file in &available_task_timeseries {
            let filename_without_extension = match file.file_stem().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            };

            let h5_name = match file.file_name().and_then(|n| n.to_str()) {
                Some(name) => name,
                None => continue,
            };
            let mut bids = BidsFilename::parse(h5_name).keep(&["sub", "task", "run"]);
            bids.suffix = None;
            bids.extension = None;
            let event_file_base = bids.to_stem();

            let mut events_bids = bids.clone();
            events_bids.suffix = Some("events".to_string());
            events_bids.extension = Some(".tsv".to_string());
            let event_file_name = events_bids.to_filename();
            let event_file = &cfg
                .tcp_repo_dir
                .join(formatted_id)
                .join("func")
                .join(event_file_name);
            let event_file = event_file.to_str().expect("File path could not be parsed");

            let event_blocks = events_tsv_to_blocks(event_file)?;

            // Extract and write GLM condition files (trial- and block-level)
            let glm_conditions = extract_glm_conditions(&event_blocks)?;
            let glm_subject_dir = cfg.task_regressors_output_dir.join(formatted_id);
            write_glm_conditions(&glm_subject_dir, &event_file_base, &glm_conditions)?;

            info!(
                file = %event_file_base,
                n_conditions = glm_conditions.trial_level.len(),
                glm_dir = %glm_subject_dir.display(),
                "wrote GLM condition onset/duration files"
            );

            let write_start = Instant::now();

            let h5_file = hdf5::File::open_rw(file)?;
            let blockwise_timeseries = get_timeseries_per_event_block(&event_blocks, &h5_file)?;
            write_blocks_h5(&h5_file, &blockwise_timeseries, cfg.force)?;

            let write_duration_ms = write_start.elapsed().as_millis();

            info!(
                file = %filename_without_extension,
                n_blocks = blockwise_timeseries.len(),
                write_duration_ms = write_duration_ms,
                "wrote block timeseries to HDF5"
            );
        }
    }

    Ok(())
}

fn get_timeseries_per_event_block(
    event_blocks: &DataFrame,
    h5_file: &hdf5::File,
) -> Result<Vec<BlockTimeseries>> {
    let cortical: Array2<f32> = h5_file.dataset("tcp_cortical_raw")?.read_2d()?;
    let subcortical: Array2<f32> = h5_file.dataset("tcp_subcortical_raw")?.read_2d()?;
    let n_timepoints = cortical.shape()[1];

    // Standardized datasets may not be present in older files.
    let cortical_std_full: Option<Array2<f32>> = h5_file
        .dataset("tcp_cortical_standardized")
        .ok()
        .and_then(|ds| ds.read_2d().ok());
    let subcortical_std_full: Option<Array2<f32>> = h5_file
        .dataset("tcp_subcortical_standardized")
        .ok()
        .and_then(|ds| ds.read_2d().ok());

    let block_ids = event_blocks.column("block_id")?.i32()?;
    let trial_types = event_blocks.column("trial_type")?.str()?;
    let onsets = event_blocks.column("onset")?.f64()?;
    let block_ends = event_blocks.column("block_end")?.f64()?;

    let mut result = Vec::new();

    for i in 0..event_blocks.height() {
        let block_id = match block_ids.get(i) {
            Some(v) => v,
            None => continue,
        };
        let trial_type = match trial_types.get(i) {
            Some(v) => v.to_string(),
            None => continue,
        };
        let onset = match onsets.get(i) {
            Some(v) => v,
            None => continue,
        };
        let block_end = match block_ends.get(i) {
            Some(v) => v,
            None => continue,
        };

        let start_idx = (onset / TR_SECONDS).floor() as usize;
        let end_idx = ((block_end / TR_SECONDS).ceil() as usize).min(n_timepoints);

        if start_idx >= end_idx || start_idx >= n_timepoints {
            warn!(
                block_id = block_id,
                onset_s = onset,
                block_end_s = block_end,
                start_idx = start_idx,
                end_idx = end_idx,
                n_timepoints = n_timepoints,
                "skipping block with invalid time range"
            );
            continue;
        }

        let cortical_block = cortical.slice(s![.., start_idx..end_idx]).to_owned();
        let subcortical_block = subcortical.slice(s![.., start_idx..end_idx]).to_owned();
        let cortical_std_block = cortical_std_full
            .as_ref()
            .map(|a| a.slice(s![.., start_idx..end_idx]).to_owned());
        let subcortical_std_block = subcortical_std_full
            .as_ref()
            .map(|a| a.slice(s![.., start_idx..end_idx]).to_owned());

        result.push(BlockTimeseries {
            block_id,
            trial_type,
            onset_s: onset,
            block_end_s: block_end,
            cortical: cortical_block,
            subcortical: subcortical_block,
            cortical_std: cortical_std_block,
            subcortical_std: subcortical_std_block,
        });
    }

    Ok(result)
}

fn write_blocks_h5(h5_file: &hdf5::File, blocks: &[BlockTimeseries], force: bool) -> Result<()> {
    let blocks_group = match h5_file.group("blocks") {
        Ok(g) => g,
        Err(_) => h5_file.create_group("blocks")?,
    };

    // Only create blocks_standardized group if any block has standardized data.
    let has_std = blocks.iter().any(|b| b.cortical_std.is_some());
    let blocks_std_group = if has_std {
        Some(match h5_file.group("blocks_standardized") {
            Ok(g) => g,
            Err(_) => h5_file.create_group("blocks_standardized")?,
        })
    } else {
        None
    };

    for block in blocks {
        let group_name = format!("block_{}", block.block_id);

        // --- raw blocks ---
        let skip_raw = !force && blocks_group.group(&group_name).is_ok();
        if !skip_raw {
            if force {
                // Remove existing group before recreating under force.
                let _ = blocks_group.unlink(&group_name);
            }
            let block_group = blocks_group.create_group(&group_name)?;

            let c_shape = block.cortical.shape();
            let c_ds = block_group
                .new_dataset::<f32>()
                .shape([c_shape[0], c_shape[1]])
                .create("cortical_raw")?;
            c_ds.write_raw(block.cortical.as_slice().unwrap())?;

            let s_shape = block.subcortical.shape();
            let s_ds = block_group
                .new_dataset::<f32>()
                .shape([s_shape[0], s_shape[1]])
                .create("subcortical_raw")?;
            s_ds.write_raw(block.subcortical.as_slice().unwrap())?;

            let trial_type_val: VarLenUnicode = block.trial_type.parse()?;
            block_group
                .new_attr::<VarLenUnicode>()
                .shape(())
                .create("trial_type")?
                .as_writer()
                .write_scalar(&trial_type_val)?;

            block_group
                .new_attr::<f64>()
                .shape(())
                .create("onset_s")?
                .as_writer()
                .write_scalar(&block.onset_s)?;

            block_group
                .new_attr::<f64>()
                .shape(())
                .create("block_end_s")?
                .as_writer()
                .write_scalar(&block.block_end_s)?;
        }

        // --- standardized blocks ---
        if let (Some(std_group), Some(cortical_std), Some(subcortical_std)) = (
            blocks_std_group.as_ref(),
            block.cortical_std.as_ref(),
            block.subcortical_std.as_ref(),
        ) {
            let skip_std = !force && std_group.group(&group_name).is_ok();
            if !skip_std {
                if force {
                    let _ = std_group.unlink(&group_name);
                }
                let std_block_group = std_group.create_group(&group_name)?;

                let c_shape = cortical_std.shape();
                let c_ds = std_block_group
                    .new_dataset::<f32>()
                    .shape([c_shape[0], c_shape[1]])
                    .create("cortical_standardized")?;
                c_ds.write_raw(cortical_std.as_slice().unwrap())?;

                let s_shape = subcortical_std.shape();
                let s_ds = std_block_group
                    .new_dataset::<f32>()
                    .shape([s_shape[0], s_shape[1]])
                    .create("subcortical_standardized")?;
                s_ds.write_raw(subcortical_std.as_slice().unwrap())?;

                let trial_type_val: VarLenUnicode = block.trial_type.parse()?;
                std_block_group
                    .new_attr::<VarLenUnicode>()
                    .shape(())
                    .create("trial_type")?
                    .as_writer()
                    .write_scalar(&trial_type_val)?;

                std_block_group
                    .new_attr::<f64>()
                    .shape(())
                    .create("onset_s")?
                    .as_writer()
                    .write_scalar(&block.onset_s)?;

                std_block_group
                    .new_attr::<f64>()
                    .shape(())
                    .create("block_end_s")?
                    .as_writer()
                    .write_scalar(&block.block_end_s)?;
            }
        }
    }

    Ok(())
}

/// Extract condition-specific onset/duration lists for GLM modeling from the block summary
/// produced by `events_tsv_to_blocks`.
///
/// **Trial level**: Each row's `trial_onset_list` / `trial_duration_list` nested lists are
/// flattened and grouped by `trial_type`.  `duration` values reflect stimulus presentation
/// only (as recorded in the BIDS events.tsv `duration` column — ISI/ITI are not included).
///
/// **Block level**: Each row's `onset` (= first cue / first trial onset) and `duration`
/// (= `fixEndTime.last - cueStartTime.first`, i.e. full block duration) are collected
/// and grouped by `trial_type`.
pub fn extract_glm_conditions(blocks: &DataFrame) -> Result<GlmConditions> {
    let trial_types = blocks.column("trial_type")?.str()?;
    let block_onsets = blocks.column("onset")?.f64()?;
    let block_durations = blocks.column("duration")?.f64()?;
    let trial_onset_lists = blocks.column("trial_onset_list")?.list()?;
    let trial_duration_lists = blocks.column("trial_duration_list")?.list()?;

    let mut trial_level: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();
    let mut block_level: BTreeMap<String, (Vec<f64>, Vec<f64>)> = BTreeMap::new();

    for i in 0..blocks.height() {
        let condition = match trial_types.get(i) {
            Some(c) => c.to_string(),
            None => continue,
        };
        let block_onset = match block_onsets.get(i) {
            Some(v) => v,
            None => continue,
        };
        let block_duration = match block_durations.get(i) {
            Some(v) => v,
            None => continue,
        };

        // Block-level regressor: one onset + duration per block
        let block_entry = block_level
            .entry(condition.clone())
            .or_insert_with(|| (Vec::new(), Vec::new()));
        block_entry.0.push(block_onset);
        block_entry.1.push(block_duration);

        // Trial-level regressors: individual stimulus onsets + durations within this block
        let onset_series = trial_onset_lists.get_as_series(i);
        let duration_series = trial_duration_lists.get_as_series(i);

        if let (Some(onsets_s), Some(durations_s)) = (onset_series, duration_series) {
            let onsets_ca = onsets_s.f64()?;
            let durations_ca = durations_s.f64()?;

            let trial_entry = trial_level
                .entry(condition)
                .or_insert_with(|| (Vec::new(), Vec::new()));

            for j in 0..onsets_ca.len() {
                if let (Some(onset), Some(dur)) = (onsets_ca.get(j), durations_ca.get(j)) {
                    trial_entry.0.push(onset);
                    trial_entry.1.push(dur);
                }
            }
        }
    }

    Ok(GlmConditions {
        trial_level,
        block_level,
    })
}

/// Write per-condition GLM onset/duration TSV files for a single run.
///
/// Produces two files per condition:
/// - `{run_key}_condition-{name}_trial.tsv`   — individual stimulus events
/// - `{run_key}_condition-{name}_block.tsv`   — block-level regressors
///
/// Each file has a header `onset_s\tduration_s` followed by one row per onset.
pub fn write_glm_conditions(dir: &Path, run_key: &str, conditions: &GlmConditions) -> Result<()> {
    fs::create_dir_all(dir)?;

    let write_pairs =
        |level_name: &str, level_data: &BTreeMap<String, (Vec<f64>, Vec<f64>)>| -> Result<()> {
            for (condition, (onsets, durations)) in level_data {
                let filename = format!("{}_condition-{}_{}.tsv", run_key, condition, level_name);
                let path = dir.join(&filename);
                let mut file = fs::File::create(&path)?;
                writeln!(file, "onset_s\tduration_s")?;
                for (&onset, &duration) in onsets.iter().zip(durations.iter()) {
                    writeln!(file, "{}\t{}", onset, duration)?;
                }
            }
            Ok(())
        };

    write_pairs("trial", &conditions.trial_level)?;
    write_pairs("block", &conditions.block_level)?;

    Ok(())
}

pub fn events_tsv_to_blocks(events_path: &str) -> Result<DataFrame> {
    let lf = LazyCsvReader::new(PlPath::from_str(events_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_null_values(Some(NullValues::AllColumnsSingle("n/a".into())))
        .with_ignore_errors(true)
        .finish()?;

    let blocks = lf
        .with_columns([col("cueStartTime").is_not_null().alias("block_start")])
        .with_columns([col("block_start")
            .cast(DataType::Int32)
            .cum_sum(false)
            .alias("block_id")])
        .group_by([col("block_id")])
        .agg([
            col("trial_type").first().alias("trial_type"),
            col("onset").first().alias("onset"),
            col("cueStartTime").first().alias("cue_onset"),
            col("cueEndTime").first().alias("cue_end"),
            col("fixEndTime").last().alias("block_end"),
            (col("fixEndTime").last() - col("onset").first()).alias("duration"),
            col("response_time").mean().alias("response_time_mean"),
            col("response_time").median().alias("response_time_median"),
            // Correct list aggregation in Rust Polars
            col("response_time").implode().alias("response_time_list"),
            col("onset").implode().alias("trial_onset_list"),
            col("duration").implode().alias("trial_duration_list"),
            col("accuracy_binarized").mean().alias("accuracy_mean"),
            col("stimLeft").implode().alias("stimLeft_list"),
            col("stimTop").implode().alias("stimTop_list"),
            col("stimRight").implode().alias("stimRight_list"),
            col("trial_type").count().alias("n_trials"),
        ])
        .sort(["onset"], SortMultipleOptions::default())
        .collect()?;

    Ok(blocks)
}
