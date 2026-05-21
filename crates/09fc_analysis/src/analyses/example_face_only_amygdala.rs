//! Placeholder: face-block amygdala FC group difference.
//!
//! Replace this citation with the specific paper once identified.
//! ROI: bilateral amygdala (lAMY + mAMY, Tian S2 subcortical atlas).
//! Condition: face-block average (absolute, not contrasted).
//! Test: cohort permutation t-test (anhedonic vs control) on broadband TS-FC and CWT slow bands.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use utils::atlas::{BrainAtlas, RoiType};
use utils::config::AppConfig;
use utils::frequency_bands::SLOW_BANDS;

use crate::aggregation::aggregate_blocks_for_condition;
use crate::dispatch::{
    RunOneParams, enumerate_and_labels, extract_roi_submatrix, results_dir, run_one,
};

const TASK: &str = "example_face_only_amygdala";

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting {TASK}");

    let (subjects, labels) = enumerate_and_labels(cfg)?;
    let rdir = results_dir(cfg);
    std::fs::create_dir_all(&rdir)?;

    let atlas = BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let roi_idx = amygdala_rois(&atlas);

    if roi_idx.is_empty() {
        warn!("{TASK}: resolved to empty ROI set — verify LUT region names against atlas");
        return Ok(());
    }
    info!("{TASK}: {} ROIs selected", roi_idx.len());

    let n_perm = cfg.fc_analysis.n_permutations;
    let primary_t = cfg.fc_analysis.nbs_primary_t;
    let seed = cfg.fc_analysis.permutation_seed;

    // TS broadband face-block average
    {
        let idx = Arc::new(roi_idx.clone());
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "ts",
                level: "face_block_avg",
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = aggregate_blocks_for_condition(
                    f,
                    "/06fc/ts/blocks_std",
                    "face",
                    None,
                    "fisher_z",
                )?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // CWT per slow band face-block average
    for &(band, _, _) in SLOW_BANDS {
        let idx = Arc::new(roi_idx.clone());
        let level = format!("face_block_avg_{band}");
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "cwt",
                level: &level,
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = aggregate_blocks_for_condition(
                    f,
                    "/06fc/cwt/blocks_std",
                    "face",
                    Some(band),
                    "fisher_z",
                )?;
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

/// Row indices for bilateral amygdala (lAMY + mAMY, both hemispheres, Tian S2).
fn amygdala_rois(atlas: &BrainAtlas) -> Vec<usize> {
    let mut idx = atlas.concat_row_indices(|e| {
        matches!(&e.metadata, RoiType::Subcortical { region, .. }
            if region == "lAMY" || region == "mAMY")
    });
    idx.sort();
    idx.dedup();
    idx
}
