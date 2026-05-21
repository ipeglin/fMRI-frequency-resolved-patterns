//! Keedwell et al. (2005) replication: anhedonia-linked FC differences
//! in the face-vs-shape (sad-vs-neutral) contrast.
//!
//! Keedwell et al. found that Anhedonia severity correlated with altered
//! responses in amygdala, OFC, and medial PFC specifically *relative* to
//! neutral stimuli. This analysis tests whether that pattern is visible in
//! pairwise-FC contrasts (face-block mean minus shape-block mean per subject)
//! using a cohort permutation test (anhedonic vs control on the Δ).
//!
//! ROIs: bilateral amygdala (lAMY + mAMY, both hemispheres), OFC (LimbicB),
//! and medial PFC (LimbicB_PFCm + DefaultA_PFCm). Region names are exact
//! matches against the Tian S2 subcortical LUT and Schaefer-400 cortical LUT
//! loaded at runtime — verify with `utils::atlas::BrainAtlas::find_ids_by_metadata`
//! if the atlas files change.

use anyhow::Result;
use std::sync::Arc;
use std::time::Instant;
use tracing::{info, warn};
use utils::atlas::{BrainAtlas, RoiType};
use utils::config::AppConfig;
use utils::frequency_bands::SLOW_BANDS;

use crate::aggregation::{aggregate_blocks_for_condition, aggregate_paired_contrast};
use crate::dispatch::{
    RunOneParams, enumerate_and_labels, extract_roi_submatrix, results_dir, run_one,
};

const TASK: &str = "keedwell_face_vs_shape";

pub fn run(cfg: &AppConfig) -> Result<()> {
    let started = Instant::now();
    info!("starting {TASK}");

    let (subjects, labels) = enumerate_and_labels(cfg)?;
    let rdir = results_dir(cfg);
    std::fs::create_dir_all(&rdir)?;

    let atlas = BrainAtlas::from_lut_files(&cfg.cortical_atlas_lut, &cfg.subcortical_atlas_lut);
    let roi_idx = keedwell_rois(&atlas);

    if roi_idx.is_empty() {
        warn!("{TASK}: resolved to empty ROI set — verify LUT region names against atlas");
        return Ok(());
    }
    info!("{TASK}: {} ROIs selected", roi_idx.len());

    let n_perm = cfg.fc_analysis.n_permutations;
    let primary_t = cfg.fc_analysis.nbs_primary_t;
    let seed = cfg.fc_analysis.permutation_seed;
    let num_modes = cfg.hht.num_modes;

    // TS broadband face-vs-shape contrast
    {
        let idx = Arc::new(roi_idx.clone());
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "ts",
                level: "face_minus_shape",
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = aggregate_paired_contrast(
                    f,
                    "/06fc/ts/blocks_std",
                    "face",
                    "shape",
                    None,
                    "fisher_z",
                )?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // CWT per slow band
    for &(band, _, _) in SLOW_BANDS {
        let idx = Arc::new(roi_idx.clone());
        let level = format!("face_minus_shape_{band}");
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
                let full = aggregate_paired_contrast(
                    f,
                    "/06fc/cwt/blocks_std",
                    "face",
                    "shape",
                    Some(band),
                    "fisher_z",
                )?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // MVMD per mode
    for k in 0..num_modes {
        let sub = format!("mode_{k}");
        let level = format!("face_minus_shape_mode_{k}");
        let idx = Arc::new(roi_idx.clone());
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "mvmd",
                level: &level,
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = aggregate_paired_contrast(
                    f,
                    "/06fc/mvmd/blocks_std",
                    "face",
                    "shape",
                    Some(&sub),
                    "fisher_z",
                )?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // MVMD per slow band (fisher_z_mean — band-averaged across modes)
    for &(band, _, _) in SLOW_BANDS {
        let idx = Arc::new(roi_idx.clone());
        let level = format!("face_minus_shape_band_{band}");
        run_one(
            &RunOneParams {
                subjects: &subjects,
                labels: &labels,
                results_dir: &rdir,
                task: TASK,
                source: "mvmd_band",
                level: &level,
                roi_suffix: "",
                n_perm,
                primary_t,
                seed,
            },
            move |f| {
                let full = aggregate_paired_contrast(
                    f,
                    "/06fc/mvmd/blocks_std",
                    "face",
                    "shape",
                    Some(band),
                    "fisher_z_mean",
                )?;
                Ok(full.map(|m| extract_roi_submatrix(&m, &idx)))
            },
        )?;
    }

    // Face-only average (preserves comparability with prior hammerAP results)
    for &(band, _, _) in SLOW_BANDS {
        let idx = Arc::new(roi_idx.clone());
        let level = format!("face_only_{band}");
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

/// Row indices in the concatenated atlas matrix (cortical then subcortical) for
/// Keedwell-relevant regions: bilateral amygdala (lAMY + mAMY) + OFC + mPFC.
fn keedwell_rois(atlas: &BrainAtlas) -> Vec<usize> {
    // Bilateral amygdala — lateral and medial divisions, both hemispheres.
    // Region names follow Tian S2: "lAMY" and "mAMY".
    let mut idx = atlas.concat_row_indices(|e| {
        matches!(&e.metadata, RoiType::Subcortical { region, .. }
            if region == "lAMY" || region == "mAMY")
    });

    // Orbitofrontal cortex (LimbicB network)
    idx.extend(atlas.concat_row_indices(|e| {
        matches!(&e.metadata, RoiType::Cortical { network, region, .. }
            if network == "LimbicB" && region == "OFC")
    }));

    // Medial prefrontal cortex (LimbicB and DefaultA networks — both include vmPFC/sgACC)
    idx.extend(atlas.concat_row_indices(|e| {
        matches!(&e.metadata, RoiType::Cortical { network, region, .. }
            if region == "PFCm" && (network == "LimbicB" || network == "DefaultA"))
    }));

    idx.sort();
    idx.dedup();
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use utils::atlas::BrainAtlas;

    fn mock_atlas() -> BrainAtlas {
        let cortical = HashMap::from([
            ("17networks_LH_LimbicB_OFC_1".to_string(), 0u32),
            ("17networks_RH_LimbicB_OFC_1".to_string(), 1u32),
            ("17networks_LH_LimbicB_PFCm".to_string(), 2u32),
            ("17networks_RH_DefaultA_PFCm_1".to_string(), 3u32),
        ]);
        let subcortical = HashMap::from([
            ("lAMY-lh".to_string(), 0u32),
            ("lAMY-rh".to_string(), 1u32),
            ("mAMY-lh".to_string(), 2u32),
            ("mAMY-rh".to_string(), 3u32),
            ("pCAU-rh".to_string(), 4u32),
        ]);
        BrainAtlas::from_lut_maps(cortical, subcortical)
    }

    #[test]
    fn keedwell_rois_returns_expected_indices() {
        let atlas = mock_atlas();
        let idx = keedwell_rois(&atlas);
        // 4 cortical + 4 amygdala subcortical = 8, but deduped
        assert!(!idx.is_empty(), "should find at least some ROIs");
        // caudate (pCAU) should NOT be included
        let n_cortical = atlas.n_cortical;
        let caudate_row = n_cortical + 4; // index 4 in subcortical
        assert!(
            !idx.contains(&caudate_row),
            "caudate should not be in keedwell ROIs"
        );
        // all indices should be sorted
        let sorted: Vec<usize> = {
            let mut v = idx.clone();
            v.sort();
            v.dedup();
            v
        };
        assert_eq!(idx, sorted, "indices should be sorted and deduped");
    }
}
