use anyhow::Result;
use config::TCPSubjectSelectionConfig;
use config::annex;
use config::polars_csv;
use git2::Repository;
use polars::prelude::*;
use std::fs;
use std::path::PathBuf;
use thiserror::Error;
use tracing::{info, warn};

#[derive(Error, Debug)]
pub enum TCPPreprocessError {
    #[error("File already exists: {0}")]
    AlreadyExists(String),
    #[error("File does not exist: {0}")]
    FileNotExist(String),
    #[error("Required file missing: {0}")]
    RequiredFileMissing(String),
}

pub fn run(cfg: &TCPSubjectSelectionConfig) -> Result<()> {
    info!("{:?}", cfg);

    ////////////////////////
    // Initialize dataset //
    ////////////////////////

    // Clone dataset
    let dataset_dir = &cfg.tcp_dir;
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

    let filter_output_dir = cfg.output_dir.join("filters");
    fs::create_dir_all(&filter_output_dir)?;

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
    let demos_df = LazyCsvReader::new(PlPath::from_str(demos_path))
        .with_separator(b',')
        .with_has_header(true)
        .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
        .with_ignore_errors(true)
        .with_encoding(CsvEncoding::LossyUtf8)
        .finish()?
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(filter_output_dir.join("demos.csv"), &demos_df)?;

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

    polars_csv::write_dataframe(filter_output_dir.join("genpop.csv"), &genpop_df)?;

    // Major Depressive Disorder (Primary Diagnosis)
    let primary_mdd_df = LazyCsvReader::new(PlPath::from_str(demos_path))
        .with_separator(b',')
        .with_has_header(true)
        .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
        .with_ignore_errors(true)
        .with_encoding(CsvEncoding::LossyUtf8)
        .finish()?
        .filter(col("Primary_Dx").str().contains(lit("MDD"), false))
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(filter_output_dir.join("primary_mdd.csv"), &primary_mdd_df)?;

    // Major Depressive Disorder (Primary Diagnosis)
    let secondary_mdd_df = LazyCsvReader::new(PlPath::from_str(demos_path))
        .with_separator(b',')
        .with_has_header(true)
        .with_skip_rows(1) // Skip the garbage first row, treat row 2 as headers
        .with_ignore_errors(true)
        .with_encoding(CsvEncoding::LossyUtf8)
        .finish()?
        .filter(col("Non-Primary_Dx").str().contains(lit("MDD"), false))
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("secondary_mdd.csv"),
        &secondary_mdd_df,
    )?;

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

    // Available SHAPS
    let shaps_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("shaps_total").neq(lit(999)))
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(filter_output_dir.join("shaps.csv"), &shaps_df)?;

    // Non-anhedonic
    let non_anhedonic_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("shaps_total").neq(lit(999)))
        .filter(col("shaps_total").lt(lit(3))) // non-anhedonic scores are 0–2
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("shaps_non_anhedonic.csv"),
        &non_anhedonic_df,
    )?;

    // Anhedonic subjects
    let anhedonic_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("shaps_total").neq(lit(999)))
        .filter(col("shaps_total").gt_eq(lit(3))) // anhedonic scores are 3–14
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(filter_output_dir.join("shaps_anhedonic.csv"), &anhedonic_df)?;

    // Low-anhedonic subjects
    let low_anhedonic_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("shaps_total").neq(lit(999)))
        .filter(col("shaps_total").gt_eq(lit(3))) // low-anhedonic scores are 3–14
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("shaps_low_anhedonic.csv"),
        &low_anhedonic_df,
    )?;

    // High-anhedonic subjects
    let high_anhedonic_df = LazyCsvReader::new(PlPath::from_str(shaps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("shaps_total").neq(lit(999)))
        .filter(col("shaps_total").gt_eq(lit(3))) // low-anhedonic scores are 3–14
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any) // Get unique entries
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("shaps_high_anhedonic.csv"),
        &high_anhedonic_df,
    )?;

    /////////////////////////
    // Apply TEPS Filters //
    /////////////////////////

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

    // Load TEPS data once with scores, filtering out 999 values
    let teps_valid_df = LazyCsvReader::new(PlPath::from_str(teps_path))
        .with_separator(b'\t')
        .with_has_header(true)
        .with_ignore_errors(true)
        .finish()?
        .filter(col("teps_ap").neq(lit(999))) // Anticipatory Pleasure Score
        .filter(col("teps_cp").neq(lit(999))) // Consummatory Pleasure Score
        .unique(Some(cols(["subjectkey"])), UniqueKeepStrategy::Any)
        .select([col("subjectkey"), col("teps_ap"), col("teps_cp")])
        .collect()?;

    // Available TEPS subjects
    let teps_df = teps_valid_df
        .clone()
        .lazy()
        .select([col("subjectkey")])
        .collect()?;
    polars_csv::write_dataframe(filter_output_dir.join("teps.csv"), &teps_df)?;

    // Compute mean and std for anticipatory and consummatory pleasure scores
    let teps_stats = teps_valid_df
        .clone()
        .lazy()
        .select([
            col("teps_ap").mean().alias("ap_mean"),
            col("teps_ap").std(1).alias("ap_std"),
            col("teps_cp").mean().alias("cp_mean"),
            col("teps_cp").std(1).alias("cp_std"),
        ])
        .collect()?;

    let ap_mean = teps_stats.column("ap_mean")?.f64()?.get(0).unwrap();
    let ap_std = teps_stats.column("ap_std")?.f64()?.get(0).unwrap();
    let cp_mean = teps_stats.column("cp_mean")?.f64()?.get(0).unwrap();
    let cp_std = teps_stats.column("cp_std")?.f64()?.get(0).unwrap();

    let ap_threshold = ap_mean - 2.0 * ap_std;
    let cp_threshold = cp_mean - 2.0 * cp_std;

    info!(
        "TEPS-ANT stats: mean={:.2}, std={:.2}, anhedonia threshold={:.2}",
        ap_mean, ap_std, ap_threshold
    );
    info!(
        "TEPS-CON stats: mean={:.2}, std={:.2}, anhedonia threshold={:.2}",
        cp_mean, cp_std, cp_threshold
    );

    // Anticipatory anhedonic: scoring more than 2 SD below mean on teps_ap
    let teps_anticipatory_anhedonic_df = teps_valid_df
        .clone()
        .lazy()
        .filter(col("teps_ap").lt(lit(ap_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_anticipatory_anhedonic.csv"),
        &teps_anticipatory_anhedonic_df,
    )?;

    // Anticipatory non-anhedonic: scoring at or above the threshold on teps_ap
    let teps_anticipatory_non_anhedonic_df = teps_valid_df
        .clone()
        .lazy()
        .filter(col("teps_ap").gt_eq(lit(ap_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_anticipatory_non_anhedonic.csv"),
        &teps_anticipatory_non_anhedonic_df,
    )?;

    // Consummatory anhedonic: scoring more than 2 SD below mean on teps_cp
    let teps_consummatory_anhedonic_df = teps_valid_df
        .clone()
        .lazy()
        .filter(col("teps_cp").lt(lit(cp_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_consummatory_anhedonic.csv"),
        &teps_consummatory_anhedonic_df,
    )?;

    // Consummatory non-anhedonic: scoring at or above the threshold on teps_cp
    let teps_consummatory_non_anhedonic_df = teps_valid_df
        .lazy()
        .filter(col("teps_cp").gt_eq(lit(cp_threshold)))
        .select([col("subjectkey")])
        .collect()?;

    polars_csv::write_dataframe(
        filter_output_dir.join("teps_consummatory_non_anhedonic.csv"),
        &teps_consummatory_non_anhedonic_df,
    )?;

    /////////////////////
    // Combine Filters //
    /////////////////////

    // Healthy Controls (GenPop + SHAPS non-anhedonic + TEPS non-anhedonic on both subscales)
    let healthy_controls_df = genpop_df
        .join(
            &non_anhedonic_df,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Inner),
            None,
        )?
        .join(
            &teps_anticipatory_non_anhedonic_df,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Inner),
            None,
        )?
        .join(
            &teps_consummatory_non_anhedonic_df,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Inner),
            None,
        )?;

    polars_csv::write_dataframe(
        filter_output_dir.join("healthy_controls.csv"),
        &healthy_controls_df,
    )?;

    // Major Depressive Disorder
    let mdd_df = primary_mdd_df
        .join(
            &secondary_mdd_df,
            ["subjectkey"],
            ["subjectkey"],
            JoinArgs::new(JoinType::Full),
            None,
        )?
        .lazy()
        .with_column(coalesce(&[col("subjectkey"), col("subjectkey_right")]).alias("subjectkey"))
        .drop(cols(["subjectkey_right"]))
        .collect()?;

    polars_csv::write_dataframe(filter_output_dir.join("mdd.csv"), &mdd_df)?;

    Ok(())
}
