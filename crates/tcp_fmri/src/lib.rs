use anyhow::Result;
use config::{TCPfMRIPreprocessConfig, polars_csv};
use polars::prelude::*;
use tracing::{info, warn};

pub fn run(cfg: &TCPfMRIPreprocessConfig) -> Result<()> {
    info!("{:?}", cfg);

    ///////////////////
    // Load subjects //
    ///////////////////

    let filter_dir = &cfg.filter_dir;

    // Filter files
    let filtered_subjects = vec![
        filter_dir.join("healthy_controls.csv"),
        filter_dir.join("low_anhedonic.csv"),
        filter_dir.join("high_anhedonic.csv"),
    ];

    // Load dataframes
    let dataframes: Vec<LazyFrame> = filtered_subjects
        .iter()
        .filter_map(|file| {
            polars_csv::read_dataframe(file)
                .map_err(|e| warn!("Failed to read {}: {}", file.display(), e))
                .ok()
                .map(|df| df.lazy())
        })
        .collect();

    // Get subject key series
    let subjects = concat(dataframes, UnionArgs::default())?.collect()?;
    let subject_keys = subjects.column("subjectkey")?.str()?;

    //////////////////
    // Load atlases //
    //////////////////

    // Check if atlas files are present
    if !&cfg.cortical_atlas.exists() || !&cfg.subcortical_atlas.exists() {
        panic!("failed to locate atlases");
    }

    ///////////////////////////////
    // Process subject fMRI BOLD //
    ///////////////////////////////

    // Iterate over subject keys
    for subject_key in subject_keys.into_iter().flatten() {
        info!("processing subject key: {}", subject_key);

        // Parcellate fMRI data using atlases
    }

    // Write BOLD time series for each subject

    Ok(())
}
