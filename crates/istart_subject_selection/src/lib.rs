use anyhow::Result;
use config::ISTARTSubjectSelectionConfig;
use config::annex;
use config::polars_csv;
use git2::Repository;
use polars::prelude::*;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum ISTARTPreprocessError {
    #[error("File already exists: {0}")]
    AlreadyExists(String),
    #[error("File does not exist: {0}")]
    FileNotExist(String),
    #[error("Required file missing: {0}")]
    RequiredFileMissing(String),
}

pub fn run(cfg: &ISTARTSubjectSelectionConfig) -> Result<()> {
    info!("{:?}", cfg);

    ////////////////////////
    // Initialize dataset //
    ////////////////////////

    // Clone dataset
    let dataset_dir = &cfg.istart_dir;
    if !dataset_dir.is_dir() {
        let dataset_url = "https://github.com/OpenNeuroDatasets/ds004920.git";
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
    info!("ISTART Dataset available on: {}", dataset_dir.display());

    // Validate and set annex remote
    if annex::validate_remote(dataset_dir, &cfg.istart_annex_remote).is_err() {
        annex::enable_remote(dataset_dir, &cfg.istart_annex_remote)?;
    }
    info!("Validated annex remote: {}", &cfg.istart_annex_remote);

    // Validate dataset
    let phenotype_dir = dataset_dir.join("phenotype");
    let required_files: Vec<PathBuf> =
        vec![phenotype_dir.join("temporal_experience_of_pleasure_scale.tsv")];
    required_files.iter().try_for_each(|file_path| {
        if !file_path.is_file() && !annex::is_broken_symlink(file_path) {
            return Err(ISTARTPreprocessError::RequiredFileMissing(format!(
                "{}",
                file_path.display()
            )));
        }
        // Continue to next iteration
        Ok(())
    })?;
    info!("All required files located: {:?}", required_files);

    /////////////////////////
    // Apply TEPS Filters //
    /////////////////////////

    // Check teps file is available
    let teps_path = dataset_dir
        .join("phenotype")
        .join("temporal_experience_of_pleasure_scale.tsv");
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

    let filter_output_dir = cfg.output_dir.join("filters");
    fs::create_dir_all(&filter_output_dir)?;

    // Declare TEPS score categories
    let anticipatory_cols = ["1", "3", "7", "11", "12", "14", "15", "16", "17", "18"]
        .map(|i| format!("score_teps_q{}", i));
    let consummatory_cols =
        ["2", "4", "5", "6", "8", "9", "10", "13"].map(|i| format!("score_teps_q{}", i));

    // Load TEPS data once with all score columns
    let mut select_exprs: Vec<Expr> = vec![col("participant_id")];
    select_exprs.extend(
        anticipatory_cols
            .iter()
            .chain(consummatory_cols.iter())
            .map(|c| col(c)),
    );

    let teps_valid_df = LazyCsvReader::new(PlPath::from_str(teps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .unique(Some(cols(["participant_id"])), UniqueKeepStrategy::Any)
        .select(select_exprs)
        .collect()?;

    // Available TEPS subjects
    let teps_df = teps_valid_df
        .clone()
        .lazy()
        .select([col("participant_id")])
        .collect()?;
    polars_csv::write_dataframe(filter_output_dir.join("teps.csv"), &teps_df)?;

    // Compute per-participant mean for anticipatory and consummatory scores.
    // Cast score columns to f64 (treating "n/a" as null via ignore_errors) and
    // use horizontal mean which skips nulls automatically via as_struct().mean().
    let ant_exprs: Vec<Expr> = anticipatory_cols
        .iter()
        .map(|c| col(c).cast(DataType::Float64))
        .collect();
    let con_exprs: Vec<Expr> = consummatory_cols
        .iter()
        .map(|c| col(c).cast(DataType::Float64))
        .collect();

    let teps_scored_df = teps_valid_df
        .lazy()
        .with_columns([
            as_struct(ant_exprs).mean().alias("teps_ant_mean"),
            as_struct(con_exprs).mean().alias("teps_con_mean"),
        ])
        .select([
            col("participant_id"),
            col("teps_ant_mean"),
            col("teps_con_mean"),
        ])
        // Drop participants where both means are null (all scores were n/a)
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
        .select([col("participant_id")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_anticipatory_anhedonic.csv"),
        &teps_anticipatory_anhedonic_df,
    )?;

    // Anticipatory non-anhedonic: scoring at or above the threshold on teps_ant_mean
    let teps_anticipatory_non_anhedonic_df = teps_scored_df
        .clone()
        .lazy()
        .filter(col("teps_ant_mean").gt_eq(lit(ant_threshold)))
        .select([col("participant_id")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_anticipatory_non_anhedonic.csv"),
        &teps_anticipatory_non_anhedonic_df,
    )?;

    // Consummatory anhedonic: scoring more than 2 SD below mean on teps_con_mean
    let teps_consummatory_anhedonic_df = teps_scored_df
        .clone()
        .lazy()
        .filter(col("teps_con_mean").lt(lit(con_threshold)))
        .select([col("participant_id")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_consummatory_anhedonic.csv"),
        &teps_consummatory_anhedonic_df,
    )?;

    // Consummatory non-anhedonic: scoring at or above the threshold on teps_con_mean
    let teps_consummatory_non_anhedonic_df = teps_scored_df
        .lazy()
        .filter(col("teps_con_mean").gt_eq(lit(con_threshold)))
        .select([col("participant_id")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_consummatory_non_anhedonic.csv"),
        &teps_consummatory_non_anhedonic_df,
    )?;

    /////////////////////
    // Combine Filters //
    /////////////////////

    // Healthy Controls (Not defined yet)

    Ok(())
}
