pub mod annex;
pub mod bids_filename;
pub mod bids_subject_id;
pub mod pipeline_config;
pub mod polars_csv;

pub use pipeline_config::{
    CwtConfig, DataSplitConfig, FcConfig, FeatureExtractionConfig, FmriParcellationConfig,
    HilbertHuangConfig, MvmdConfig, TcpSubjectSelectionConfig, TrialSegmentationConfig,
};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Reserved top-level TOML key whose contents are merged into every sibling
/// sub-table as defaults. Per-section values override. Values can be any TOML
/// type (paths, strings, ints, bools); unknown keys are ignored by sections
/// that don't declare the field, so adding a new shared default is safe.
///
/// # Adding a new shared value
/// 1. Add the field to every sub-config struct that should consume it.
/// 2. Add the key under `[defaults]` in `config.toml`.
/// No changes to this crate are required.
pub const SHARED_DEFAULTS_KEY: &str = "defaults";

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
    #[serde(default)]
    pub tcp_subject_selection: TcpSubjectSelectionConfig,
    #[serde(default)]
    pub fmri_parcellation: FmriParcellationConfig,
    #[serde(default)]
    pub fmri_segment_trials: TrialSegmentationConfig,
    #[serde(default)]
    pub mvmd: MvmdConfig,
    #[serde(default)]
    pub cwt: CwtConfig,
    #[serde(default)]
    pub hilbert: HilbertHuangConfig,
    #[serde(default)]
    pub fc: FcConfig,
    #[serde(default)]
    pub feature_extraction: FeatureExtractionConfig,
    #[serde(default)]
    pub data_splitting: DataSplitConfig,
}

pub fn load_config(path: &Path) -> Result<AppConfig> {
    let s = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read config: {}", path.display()))?;
    parse_config(&s)
        .with_context(|| format!("Failed to load config: {}", path.display()))
}

fn parse_config(s: &str) -> Result<AppConfig> {
    let mut value: toml::Value = toml::from_str(s).context("Failed to parse TOML")?;
    apply_shared_defaults(&mut value);
    value.try_into().context("Failed to deserialize config")
}

/// Injects each key from the `[defaults]` table into every sibling sub-table
/// that does not already define it. Shallow merge only: nested tables inside
/// sub-configs are not deep-merged. Sub-config sections retain priority.
fn apply_shared_defaults(value: &mut toml::Value) {
    let Some(root) = value.as_table_mut() else {
        return;
    };
    let shared = match root.remove(SHARED_DEFAULTS_KEY) {
        Some(toml::Value::Table(t)) => t,
        _ => return,
    };
    for (_, sub) in root.iter_mut() {
        let Some(tbl) = sub.as_table_mut() else {
            continue;
        };
        for (k, v) in &shared {
            tbl.entry(k.clone()).or_insert_with(|| v.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_defaults_fill_missing_keys() {
        let toml = r#"
            [defaults]
            bold_ts_dir = "/shared/bold"

            [cwt]

            [fc]
        "#;
        let cfg = parse_config(toml).unwrap();
        assert_eq!(cfg.cwt.bold_ts_dir.to_str().unwrap(), "/shared/bold");
        assert_eq!(cfg.fc.bold_ts_dir.to_str().unwrap(), "/shared/bold");
    }

    #[test]
    fn per_section_overrides_shared() {
        let toml = r#"
            [defaults]
            bold_ts_dir = "/shared/bold"

            [cwt]
            bold_ts_dir = "/local/bold"
        "#;
        let cfg = parse_config(toml).unwrap();
        assert_eq!(cfg.cwt.bold_ts_dir.to_str().unwrap(), "/local/bold");
    }

    #[test]
    fn unknown_shared_keys_ignored_by_sections() {
        // `cortical_atlas_lut` is not a field on CwtConfig; serde drops it.
        let toml = r#"
            [defaults]
            bold_ts_dir = "/shared/bold"
            cortical_atlas_lut = "/atlas/ct.txt"

            [cwt]

            [feature_extraction]
            subcortical_atlas_lut = "/atlas/sub.txt"
        "#;
        let cfg = parse_config(toml).unwrap();
        assert_eq!(cfg.cwt.bold_ts_dir.to_str().unwrap(), "/shared/bold");
        assert_eq!(
            cfg.feature_extraction.cortical_atlas_lut.to_str().unwrap(),
            "/atlas/ct.txt"
        );
    }

    #[test]
    fn missing_defaults_table_is_noop() {
        let toml = r#"
            [cwt]
            bold_ts_dir = "/only/bold"
        "#;
        let cfg = parse_config(toml).unwrap();
        assert_eq!(cfg.cwt.bold_ts_dir.to_str().unwrap(), "/only/bold");
    }

    #[test]
    fn non_path_shared_values_work() {
        let toml = r#"
            [defaults]
            force = true

            [cwt]
            bold_ts_dir = "/b"

            [fc]
            bold_ts_dir = "/b"
            force = false
        "#;
        let cfg = parse_config(toml).unwrap();
        assert!(cfg.cwt.force);
        assert!(!cfg.fc.force);
    }
}
