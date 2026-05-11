use anyhow::Result;
use git2::Repository;
use polars::prelude::*;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::{info, warn};
use utils::annex;
use utils::bids_subject_id::BidsSubjectId;
use utils::config::AppConfig;
use utils::polars_csv;

#[derive(Error, Debug)]
pub enum TCPPreprocessError {
    #[error("File already exists: {0}")]
    AlreadyExists(String),
    #[error("File does not exist: {0}")]
    FileNotExist(String),
    #[error("Required file missing: {0}")]
    RequiredFileMissing(String),
}

fn write_sorted<P: AsRef<std::path::Path>>(path: P, df: &DataFrame) -> Result<()> {
    let sorted = df
        .clone()
        .lazy()
        .sort(
            ["subjectkey"],
            SortMultipleOptions {
                descending: vec![false],
                ..Default::default()
            },
        )
        .collect()?;
    polars_csv::write_tsv(path, &sorted)?;
    Ok(())
}

pub fn run(cfg: &AppConfig) -> Result<()> {
    info!("{:?}", cfg);

    ////////////////////////
    // Initialize dataset //
    ////////////////////////

    // Clone dataset
    let dataset_dir = &cfg.tcp_repo_dir;
    if !dataset_dir.is_dir() {
        let dataset_url = "https://github.com/OpenNeuroDatasets/ds005237.git";
        let local_path = dataset_dir;

        if !cfg.dry_run {
            if let Some(parent) = local_path.parent() {
                fs::create_dir_all(parent)?;
            }
            info!("Cloning {} into {}...", dataset_url, local_path.display());
            match Repository::clone(dataset_url, local_path) {
                Ok(_repo) => info!("Cloned successfully!"),
                Err(e) => panic!("failed to clone: {}", e),
            };
        } else {
            info!("Skipped cloning. Dry_run config detected")
        }
    }
    info!("TCP Dataset available on: {}", dataset_dir.display());

    // Validate and set annex remote
    if annex::validate_remote(dataset_dir, &cfg.tcp_annex_remote).is_err() {
        annex::enable_remote(dataset_dir, &cfg.tcp_annex_remote)?;
    }
    info!("Validated annex remote: {}", &cfg.tcp_annex_remote);

    // Validate dataset
    let phenotype_dir = dataset_dir.join("phenotype");
    let required_files: Vec<PathBuf> = vec![
        phenotype_dir.join("demos.tsv"),
        phenotype_dir.join("shaps01.tsv"),
        phenotype_dir.join("teps01.tsv"),
    ];
    required_files.iter().try_for_each(|file_path| {
        if !file_path.is_file() && !annex::is_broken_symlink(file_path) {
            return Err(TCPPreprocessError::RequiredFileMissing(format!(
                "{}",
                file_path.display()
            )));
        }
        // Continue to next iteration
        Ok(())
    })?;
    info!("All required files located: {:?}", required_files);

    /////////////////////////
    // Apply Demos Filters //
    /////////////////////////

    let filter_output_dir = &cfg.subject_filter_dir;

    fs::create_dir_all(filter_output_dir)?;

    // Check demos file is available
    let demos_path = dataset_dir.join("phenotype").join("demos.tsv");
    match annex::get_file_from_annex(dataset_dir, &demos_path) {
        Ok(_) => {
            info!("Fetched file from annex: {}", demos_path.display());
        }
        Err(e @ annex::AnnexError::AlreadyExists(_)) => {
            info!("{}", e);
        }
        Err(e @ annex::AnnexError::UnbrokenSymlink(_)) => {
            warn!("{}", e);
        }
        Err(e) => {
            panic!("{}", e);
        }
    };

    if !demos_path.exists() {
        panic!("could not find demos.tsv file");
    }

    let demos_path = demos_path.to_str().expect("File path could not be parsed"); // shadowing

    // Available demographics
    // let demos_df = LazyCsvReader::new(PlPath::from_str(demos_path))
    //     .with_separator(b',')
    //     .with_has_header(true)
    //     .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
    //     .with_ignore_errors(true)
    //     .with_encoding(CsvEncoding::LossyUtf8)
    //     .finish()?
    //     .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
    //     .select([col("subjectkey")])
    //     .collect()?;

    // General population
    let genpop_df = LazyCsvReader::new(PlPath::from_str(demos_path))
        .with_separator(b',')
        .with_has_header(true)
        .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
        .with_ignore_errors(true)
        .with_encoding(CsvEncoding::LossyUtf8)
        .finish()?
        .filter(col("Group").eq(lit("GenPop")))
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    // Major Depressive Disorder (Primary Diagnosis)
    // let primary_mdd_df = LazyCsvReader::new(PlPath::from_str(demos_path))
    //     .with_separator(b',')
    //     .with_has_header(true)
    //     .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
    //     .with_ignore_errors(true)
    //     .with_encoding(CsvEncoding::LossyUtf8)
    //     .finish()?
    //     .filter(col("Primary_Dx").str().contains(lit("MDD"), false))
    //     .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
    //     .select([col("subjectkey")])
    //     .collect()?;

    // // Major Depressive Disorder (Primary Diagnosis)
    // let secondary_mdd_df = LazyCsvReader::new(PlPath::from_str(demos_path))
    //     .with_separator(b',')
    //     .with_has_header(true)
    //     .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
    //     .with_ignore_errors(true)
    //     .with_encoding(CsvEncoding::LossyUtf8)
    //     .finish()?
    //     .filter(col("Non-Primary_Dx").str().contains(lit("MDD"), false))
    //     .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
    //     .select([col("subjectkey")])
    //     .collect()?;

    /////////////////////////
    // Apply SHAPS Filters //
    /////////////////////////

    // Check demos file is available
    let shaps_path = dataset_dir.join("phenotype").join("shaps01.tsv");
    match annex::get_file_from_annex(dataset_dir, &shaps_path) {
        Ok(_) => {
            info!("Fetched file from annex: {}", shaps_path.display());
        }
        Err(e @ annex::AnnexError::AlreadyExists(_)) => {
            info!("{}", e);
        }
        Err(e @ annex::AnnexError::UnbrokenSymlink(_)) => {
            warn!("{}", e);
        }
        Err(e) => {
            panic!("{}", e);
        }
    }

    if !shaps_path.exists() {
        panic!("could not find shaps01.tsv file");
    }

    let shaps_path = shaps_path.to_str().expect("File path could not be parsed"); // shadowing

    // Load SHAPS data and compute corrected scores from individual items
    // shaps[1-14]a items: 0 or 1 are valid scores, -9 means unanswered
    let shaps_item_cols: Vec<_> = (1..=14).map(|i| format!("shaps{}a", i)).collect();

    let shaps_valid_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .with_column(col("shaps_8a").alias("shaps8a")) // Fix typo in column 8 name
        .filter(col("shaps_total").neq(lit(999))) // Exclude subjects with invalid total
        .with_column(
            // Sum only valid scores (0 or 1), treating -9 as 0
            shaps_item_cols
                .iter()
                .map(|col_name| {
                    when(col(col_name).eq(lit(0)).or(col(col_name).eq(lit(1))))
                        .then(col(col_name))
                        .otherwise(lit(0))
                })
                .reduce(|acc, expr| acc + expr)
                .unwrap()
                .alias("shaps_computed_total"),
        )
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .select([col("subjectkey"), col("shaps_computed_total")])
        .collect()?;

    // Available SHAPS subjects
    // let shaps_df = shaps_valid_df
    //     .clone()
    //     .lazy()
    //     .select([col("subjectkey")])
    //     .collect()?;

    // Anhedonic subjects: computed scores are 3–14
    let shaps_anhedonic_df = shaps_valid_df
        .clone()
        .lazy()
        .filter(col("shaps_computed_total").gt_eq(lit(3)))
        .select([col("subjectkey")])
        .collect()?;

    ////////////////////////
    // Apply TEPS Filters //
    ////////////////////////

    // Check teps file is available
    let teps_path = dataset_dir.join("phenotype").join("teps01.tsv");
    match annex::get_file_from_annex(dataset_dir, &teps_path) {
        Ok(_) => {
            info!("Fetched file from annex: {}", teps_path.display());
        }
        Err(e @ annex::AnnexError::AlreadyExists(_)) => {
            info!("{}", e);
        }
        Err(e @ annex::AnnexError::UnbrokenSymlink(_)) => {
            warn!("{}", e);
        }
        Err(e) => {
            panic!("{}", e);
        }
    }

    if !teps_path.exists() {
        panic!("could not find teps01.tsv file");
    }

    let teps_path = teps_path.to_str().expect("File path could not be parsed"); // shadowing

    // Declare TEPS score categories
    // TCP teps[1-18] maps to ISTART score_teps_q[1-18]
    let anticipatory_cols =
        ["1", "3", "7", "11", "12", "14", "15", "16", "17", "18"].map(|i| format!("teps{}", i));
    let consummatory_cols =
        ["2", "4", "5", "6", "8", "9", "10", "13"].map(|i| format!("teps{}", i));

    // Load TEPS data with all score columns
    let mut select_exprs: Vec<Expr> = vec![col("subjectkey")];
    select_exprs.extend(
        anticipatory_cols
            .iter()
            .chain(consummatory_cols.iter())
            .map(col),
    );

    let teps_valid_df = LazyCsvReader::new(PlPath::from_str(teps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .select(select_exprs)
        .collect()?;

    // Available TEPS subjects
    // let teps_df = teps_valid_df
    //     .clone()
    //     .lazy()
    //     .select([col("subjectkey")])
    //     .collect()?;

    // Compute per-participant mean for anticipatory and consummatory scores.
    // Manually compute mean: sum valid scores and divide by count of non-null values
    let teps_scored_df = teps_valid_df
        .lazy()
        .with_columns([
            // Anticipatory mean
            (anticipatory_cols
                .iter()
                .map(|c| col(c).cast(DataType::Float64))
                .reduce(|acc, expr| acc + expr)
                .unwrap()
                / anticipatory_cols
                    .iter()
                    .map(|c| col(c).is_not_null().cast(DataType::Float64))
                    .reduce(|acc, expr| acc + expr)
                    .unwrap())
            .alias("teps_ant_mean"),
            // Consummatory mean
            (consummatory_cols
                .iter()
                .map(|c| col(c).cast(DataType::Float64))
                .reduce(|acc, expr| acc + expr)
                .unwrap()
                / consummatory_cols
                    .iter()
                    .map(|c| col(c).is_not_null().cast(DataType::Float64))
                    .reduce(|acc, expr| acc + expr)
                    .unwrap())
            .alias("teps_con_mean"),
        ])
        .select([
            col("subjectkey"),
            col("teps_ant_mean"),
            col("teps_con_mean"),
        ])
        // Drop participants where both means are null (all scores were invalid)
        .filter(
            col("teps_ant_mean")
                .is_not_null()
                .or(col("teps_con_mean").is_not_null()),
        )
        .collect()?;

    // Compute population-level mean and std for each subscale
    let teps_stats = teps_scored_df
        .clone()
        .lazy()
        .select([
            col("teps_ant_mean").mean().alias("ant_mean"),
            col("teps_ant_mean").std(1).alias("ant_std"),
            col("teps_con_mean").mean().alias("con_mean"),
            col("teps_con_mean").std(1).alias("con_std"),
        ])
        .collect()?;

    let ant_mean = teps_stats.column("ant_mean")?.f64()?.get(0).unwrap();
    let ant_std = teps_stats.column("ant_std")?.f64()?.get(0).unwrap();
    let con_mean = teps_stats.column("con_mean")?.f64()?.get(0).unwrap();
    let con_std = teps_stats.column("con_std")?.f64()?.get(0).unwrap();

    let ant_threshold = ant_mean - 2.0 * ant_std;
    let con_threshold = con_mean - 2.0 * con_std;

    info!(
        "TEPS-ANT stats: mean={:.2}, std={:.2}, anhedonia threshold={:.2}",
        ant_mean, ant_std, ant_threshold
    );
    info!(
        "TEPS-CON stats: mean={:.2}, std={:.2}, anhedonia threshold={:.2}",
        con_mean, con_std, con_threshold
    );

    // Anticipatory anhedonic: scoring more than 2 SD below mean on teps_ant_mean
    let teps_anticipatory_anhedonic_df = teps_scored_df
        .clone()
        .lazy()
        .filter(col("teps_ant_mean").lt(lit(ant_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    // Consummatory anhedonic: scoring more than 2 SD below mean on teps_con_mean
    let teps_consummatory_anhedonic_df = teps_scored_df
        .lazy()
        .filter(col("teps_con_mean").lt(lit(con_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    //////////////////////////////
    // Raw Data Availability    //
    //////////////////////////////

    // Scan the BIDS dataset root for subjects with anat/ and func/ directories.
    // BIDS subject folders are named sub-<id>. The demos.tsv subjectkey format for NDAR IDs
    // is NDAR_INVXXXXXXXX, which maps to sub-NDARINVXXXXXXXX in BIDS (underscore dropped).
    let mut subjects_with_bids_data: Vec<String> = Vec::new();
    let mut subjects_with_resting_task: Vec<String> = Vec::new();
    let mut subjects_with_hammer_task: Vec<String> = Vec::new();

    let resting_pat = if cfg.restap_run01_only {
        "task-restAP_run-01"
    } else {
        "task-restAP"
    };

    if let Ok(entries) = fs::read_dir(dataset_dir) {
        for entry in entries.flatten() {
            let dir_name = entry.file_name();
            let dir_name_str = dir_name.to_string_lossy();

            if !dir_name_str.starts_with("sub-") {
                continue;
            }

            let subjectkey = BidsSubjectId::parse(dir_name_str.as_ref()).to_subjectkey();

            let subject_path = entry.path();
            let anat_path = subject_path.join("anat");
            let func_path = subject_path.join("func");

            if anat_path.is_dir() && func_path.is_dir() {
                subjects_with_bids_data.push(subjectkey.clone());

                let func_names: Vec<String> = fs::read_dir(&func_path)
                    .map(|dir_entries| {
                        dir_entries
                            .flatten()
                            .map(|e| e.file_name().to_string_lossy().into_owned())
                            .collect()
                    })
                    .unwrap_or_default();

                if func_names.iter().any(|n| n.contains(resting_pat)) {
                    subjects_with_resting_task.push(subjectkey.clone());
                }
                if func_names.iter().any(|n| n.contains("task-hammerAP")) {
                    subjects_with_hammer_task.push(subjectkey);
                }
            }
        }
    }

    info!(
        "Subjects with BIDS data (anat+func): {}",
        subjects_with_bids_data.len()
    );
    info!(
        "Subjects with BIDS data + {} (inclusion criterion): {}",
        resting_pat,
        subjects_with_resting_task.len()
    );
    info!(
        "Subjects with BIDS data + task-hammerAP: {}",
        subjects_with_hammer_task.len()
    );

    let subjects_with_resting_task_df = DataFrame::new(vec![Column::new(
        "subjectkey".into(),
        subjects_with_resting_task,
    )])?;

    let subjects_with_hammer_task_df = DataFrame::new(vec![Column::new(
        "subjectkey".into(),
        subjects_with_hammer_task,
    )])?;

    /////////////////////
    // Combine Filters //
    /////////////////////

    // Union of all anhedonic subjects across SHAPS, TEPS-ANT, TEPS-CON
    let all_anhedonic_df = shaps_anhedonic_df
        .vstack(&teps_anticipatory_anhedonic_df)?
        .vstack(&teps_consummatory_anhedonic_df)?
        .lazy()
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .collect()?;

    // Healthy Controls: GenPop subjects not flagged anhedonic by any scale, with resting-state data
    let anhedonic_marker = all_anhedonic_df
        .clone()
        .lazy()
        .with_column(lit(true).alias("_anhedonic"))
        .collect()?;
    let healthy_controls_df = genpop_df
        .join(
            &anhedonic_marker,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Left),
            None,
        )?
        .lazy()
        .filter(col("_anhedonic").is_null())
        .drop(cols(["_anhedonic"]))
        .collect()?
        .join(
            &subjects_with_resting_task_df,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Inner),
            None,
        )?;

    write_sorted(
        filter_output_dir.join("desc-controls_subjects.tsv"),
        &healthy_controls_df,
    )?;

    // Anhedonic subjects: flagged by at least one scale, with resting-state data
    let anhedonic_df = all_anhedonic_df.join(
        &subjects_with_resting_task_df,
        ["subjectkey"],
        ["subjectkey"],
        JoinArgs::new(JoinType::Inner),
        None,
    )?;

    write_sorted(
        filter_output_dir.join("desc-anhedonic_subjects.tsv"),
        &anhedonic_df,
    )?;

    write_sorted(
        filter_output_dir.join("desc-hammer_subjects.tsv"),
        &subjects_with_hammer_task_df,
    )?;

    Ok(())
}
