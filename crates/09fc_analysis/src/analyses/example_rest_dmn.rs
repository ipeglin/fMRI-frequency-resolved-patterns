//! Placeholder: resting-state Default Mode Network FC group difference.
//!
//! Replace this citation with the specific paper once identified.
//! ROI: DMN parcels from Schaefer-400 17-network atlas (DefaultA + DefaultB + DefaultC networks).
//! Condition: resting-state full-run.
//! Test: cohort permutation t-test (anhedonic vs control) on broadband TS-FC and CWT slow bands.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use utils::atlas::{BrainAtlas, RoiType};
use utils::config::AppConfig;
use utils::frequency_bands::SLOW_BANDS;

use crate::aggregation::read_fc_matrix;
use crate::dispatch::{
    RunOneParams, enumerate_and_labels, extract_roi_submatrix, results_dir, run_one,
};

const TASK: &str = "example_rest_dmn";

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting {TASK}");

    let (subjects, labels) = enumerate_and_labels(cfg)?;
    let rdir = results_dir(cfg);
    std::fs::create_dir_all(&rdir)?;

    let atlas = BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let roi_idx = dmn_rois(&atlas);

    if roi_idx.is_empty() {
        warn!("{TASK}: resolved to empty ROI set — verify LUT region names against atlas");
        return Ok(());
    }
    info!("{TASK}: {} ROIs selected", roi_idx.len());

    let n_perm = cfg.fc_analysis.n_permutations;
    let primary_t = cfg.fc_analysis.nbs_primary_t;
    let seed = cfg.fc_analysis.permutation_seed;

    // TS broadband resting-state
    {
        let idx = Arc::new(roi_idx.clone());
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "ts",
                level: "full_run",
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = read_fc_matrix(f, "/06fc/ts/full_run_std", None, "fisher_z")?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // CWT per slow band resting-state
    for &(band, _, _) in SLOW_BANDS {
        let idx = Arc::new(roi_idx.clone());
        let group = format!("/06fc/cwt/full_run_std/{band}");
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "cwt",
                level: band,
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = read_fc_matrix(f, &group, None, "fisher_z")?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    info!(
        elapsed_ms = started.elapsed().as_millis() as u64,
        "{TASK} done"
    );
    Ok(())
}

/// Row indices for DMN: DefaultA + DefaultB + DefaultC cortical networks.
fn dmn_rois(atlas: &BrainAtlas) -> Vec<usize> {
    let mut idx = atlas.concat_row_indices(|e| {
        matches!(&e.metadata, RoiType::Cortical { network, .. }
            if network == "DefaultA" || network == "DefaultB" || network == "DefaultC")
    });
    idx.sort();
    idx.dedup();
    idx
}
