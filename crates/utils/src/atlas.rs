use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Hemisphere {
    Left,
    Right,
}

/// Per-region hemisphere filter used in `RoiSelectionSpec`. `LH` / `RH`
/// restrict selection to a single hemisphere; bare region entries (no filter)
/// match both.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HemiFilter {
    LH,
    RH,
}

impl HemiFilter {
    fn matches(self, hemi: Hemisphere) -> bool {
        matches!(
            (self, hemi),
            (HemiFilter::LH, Hemisphere::Left) | (HemiFilter::RH, Hemisphere::Right)
        )
    }
}

/// A single entry in a `cortical_regions`, `cortical_networks`, or
/// `subcortical_regions` list. Either a bare region name (matches both
/// hemispheres) or an inline table with an explicit hemisphere pin.
///
/// In `config.toml`:
/// ```toml
/// cortical_regions = ["PFCv", { region = "PFCm", hemisphere = "LH" }]
/// ```
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
#[serde(untagged)]
pub enum RoiSpec {
    WithHemi { region: String, hemisphere: HemiFilter },
    Bare(String),
}

impl RoiSpec {
    /// Region name used for matching (the bare string or the `region` field).
    pub fn name(&self) -> &str {
        match self {
            RoiSpec::Bare(s) => s,
            RoiSpec::WithHemi { region, .. } => region,
        }
    }

    /// True when this spec entry matches `name` (exact) AND `hemi`.
    pub fn matches_name_and_hemi(&self, name: &str, hemi: Hemisphere) -> bool {
        match self {
            RoiSpec::Bare(s) => s == name,
            RoiSpec::WithHemi { region, hemisphere } => region == name && hemisphere.matches(hemi),
        }
    }

    /// True when this spec entry's name is a substring of `full_name` AND
    /// `hemi` is compatible (bare = any, WithHemi = specific).
    pub fn contains_name_and_hemi(&self, full_name: &str, hemi: Hemisphere) -> bool {
        match self {
            RoiSpec::Bare(s) => full_name.contains(s.as_str()),
            RoiSpec::WithHemi { region, hemisphere } => {
                full_name.contains(region.as_str()) && hemisphere.matches(hemi)
            }
        }
    }
}

impl From<&str> for RoiSpec {
    fn from(s: &str) -> Self {
        RoiSpec::Bare(s.to_string())
    }
}

impl From<String> for RoiSpec {
    fn from(s: String) -> Self {
        RoiSpec::Bare(s)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoiType {
    Cortical {
        network: String,
        region: String,
        parcel: Option<u32>,
    },
    Subcortical {
        region: String,
        subregion: Option<String>,
    },
}

#[derive(Debug, Clone)]
pub struct RoiEntry {
    pub id: String,
    pub index: u32,
    pub hemisphere: Hemisphere,
    pub metadata: RoiType,
}

#[derive(Debug, Clone)]
pub struct BrainAtlas {
    pub entries: Vec<RoiEntry>,
    pub n_cortical: usize,
    pub n_subcortical: usize,
}

impl BrainAtlas {
    pub fn from_lut_maps(
        cortical_map: HashMap<String, u32>,
        subcortical_map: HashMap<String, u32>,
    ) -> Self {
        let n_cortical = cortical_map.len();
        let n_subcortical = subcortical_map.len();
        let mut entries = Vec::new();

        for (id, index) in cortical_map {
            let parts: Vec<&str> = id.split('_').collect();
            if parts.len() < 3 {
                continue;
            }

            let hemisphere = if parts[1] == "LH" {
                Hemisphere::Left
            } else {
                Hemisphere::Right
            };
            let network = parts[2].to_string();

            let (region, parcel) = match parts.get(4) {
                Some(p_str) => match p_str.parse::<u32>() {
                    Ok(num) => (parts[3].to_string(), Some(num)),
                    Err(_) => (format!("{}_{}", parts[3], p_str), None),
                },
                None => (parts.get(3).unwrap_or(&"Unknown").to_string(), None),
            };

            entries.push(RoiEntry {
                id,
                index,
                hemisphere,
                metadata: RoiType::Cortical {
                    network,
                    region,
                    parcel,
                },
            });
        }

        for (id, index) in subcortical_map {
            let mut parts: Vec<&str> = id.split('-').collect();
            let hemi_str = parts.pop().unwrap_or("");
            let hemisphere = if hemi_str == "lh" {
                Hemisphere::Left
            } else {
                Hemisphere::Right
            };

            let region = parts[0].to_string();
            let subregion = parts.get(1).map(|s| s.to_string());

            entries.push(RoiEntry {
                id,
                index,
                hemisphere,
                metadata: RoiType::Subcortical { region, subregion },
            });
        }

        Self {
            entries,
            n_cortical,
            n_subcortical,
        }
    }

    pub fn from_lut_files(cortical_lut: &Path, subcortical_lut: &Path) -> Self {
        Self::from_lut_maps(
            load_cortical_lut(cortical_lut),
            load_subcortical_lut(subcortical_lut),
        )
    }

    pub fn find_indices<F>(&self, predicate: F) -> Vec<u32>
    where
        F: Fn(&RoiEntry) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(e))
            .map(|e| e.index)
            .collect()
    }

    pub fn get_network(&self, name: &str, hemi: Option<Hemisphere>) -> Vec<u32> {
        self.find_indices(|e| {
            if let RoiType::Cortical { network, .. } = &e.metadata {
                let match_name = network == name;
                let match_hemi = hemi.is_none_or(|h| h == e.hemisphere);
                match_name && match_hemi
            } else {
                false
            }
        })
    }

    pub fn find_cortical_by_region(&self, region_name: &str, hemi: Option<Hemisphere>) -> Vec<u32> {
        self.find_indices(|e| {
            if let RoiType::Cortical { region, .. } = &e.metadata {
                region == region_name && hemi.is_none_or(|h| h == e.hemisphere)
            } else {
                false
            }
        })
    }

    pub fn find_subcortical_by_region(
        &self,
        region_name: &str,
        hemi: Option<Hemisphere>,
    ) -> Vec<u32> {
        self.find_indices(|e| {
            if let RoiType::Subcortical { region, .. } = &e.metadata {
                region == region_name && hemi.is_none_or(|h| h == e.hemisphere)
            } else {
                false
            }
        })
    }

    pub fn find_ids_by_metadata<F>(&self, predicate: F) -> Vec<String>
    where
        F: Fn(&RoiEntry) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(e))
            .map(|e| e.id.clone())
            .collect()
    }

    /// Returns the row indices of the concatenated timeseries
    /// (cortical rows 0..n_cortical, then subcortical rows n_cortical..n_cortical+n_subcortical)
    /// matching the predicate. The row order follows the convention used by
    /// `tcp_timeseries_raw` / `tcp_timeseries_standardized` (cortical then subcortical concat).
    pub fn concat_row_indices<F>(&self, predicate: F) -> Vec<usize>
    where
        F: Fn(&RoiEntry) -> bool,
    {
        self.entries
            .iter()
            .filter(|e| predicate(e))
            .map(|e| match &e.metadata {
                RoiType::Cortical { .. } => e.index as usize,
                RoiType::Subcortical { .. } => self.n_cortical + e.index as usize,
            })
            .collect()
    }

    /// Resolve a `RoiSelectionSpec` against this atlas. Cortical entries are
    /// matched by intersection of `cortical_regions` (exact match against
    /// `RoiType::Cortical.region`, e.g. `"PFCm"`) and `cortical_networks`
    /// (exact match against `RoiType::Cortical.network`, e.g. `"LimbicA"`).
    /// An empty list on either axis means "no constraint on this axis"; if
    /// both lists are empty, no cortical rows are selected (i.e. cortical
    /// inclusion requires at least one axis to be specified). Subcortical
    /// regions match by substring (so `"AMY"` catches `lAMY-lh` and
    /// `mAMY-rh`). Returned rows are sorted by `row_index` for deterministic
    /// ordering downstream.
    pub fn selected_rois(&self, spec: &RoiSelectionSpec) -> Vec<SelectedRoi> {
        let cortical_active =
            !spec.cortical_regions.is_empty() || !spec.cortical_networks.is_empty();
        let mut out: Vec<SelectedRoi> = self
            .entries
            .iter()
            .filter_map(|e| match &e.metadata {
                RoiType::Cortical {
                    region, network, ..
                } => {
                    if !cortical_active {
                        return None;
                    }
                    let region_ok = spec.cortical_regions.is_empty()
                        || spec
                            .cortical_regions
                            .iter()
                            .any(|r| r.matches_name_and_hemi(region, e.hemisphere));
                    let network_ok = spec.cortical_networks.is_empty()
                        || spec
                            .cortical_networks
                            .iter()
                            .any(|n| n.matches_name_and_hemi(network, e.hemisphere));
                    if region_ok && network_ok {
                        Some(SelectedRoi {
                            row_index: e.index as usize,
                            label: e.id.clone(),
                            matched_region: region.clone(),
                            hemisphere: e.hemisphere,
                            kind: "cortical",
                        })
                    } else {
                        None
                    }
                }
                RoiType::Subcortical { region, .. } => spec
                    .subcortical_regions
                    .iter()
                    .find(|pat| pat.contains_name_and_hemi(region, e.hemisphere))
                    .map(|matched| SelectedRoi {
                        row_index: self.n_cortical + e.index as usize,
                        label: e.id.clone(),
                        matched_region: matched.name().to_string(),
                        hemisphere: e.hemisphere,
                        kind: "subcortical",
                    }),
            })
            .collect();
        out.sort_by_key(|r| r.row_index);
        out.dedup_by_key(|r| r.row_index);
        out
    }
}

/// Single ROI entry chosen by a `RoiSelectionSpec`. Carries the concat-row
/// index used by `tcp_timeseries_raw` / `tcp_timeseries_standardized` (cortical
/// rows 0..n_cortical, subcortical rows n_cortical..n_cortical+n_subcortical),
/// along with the region name that triggered selection so downstream code can
/// look up region origin per ROI.
#[derive(Debug, Clone)]
pub struct SelectedRoi {
    pub row_index: usize,
    pub label: String,
    pub matched_region: String,
    pub hemisphere: Hemisphere,
    pub kind: &'static str,
}

/// User-facing ROI selection spec. Source of truth for which atlas rows feed
/// the spec-dependent pipeline stages (07feature_extraction, 09fc_analysis).
/// Empty cortical+subcortical lists mean "no subset".
///
/// Each entry in `cortical_regions`, `cortical_networks`, and
/// `subcortical_regions` is either a bare region name (both hemispheres) or an
/// inline table `{ region = "...", hemisphere = "LH" }` for hemisphere-pinned
/// selection.
///
/// Cortical filtering combines `cortical_regions` and `cortical_networks` as
/// an intersection: a cortical ROI is included only when it matches every
/// axis that has a non-empty list. An empty list on a given axis means "no
/// constraint on this axis". Cortical inclusion requires at least one axis.
///
/// `stratified_decomposition` (default `false`): when `true`, stage 04 runs a
/// separate MVMD pass on the ROI-subset rows in addition to the full-signal
/// pass, producing `_roi` HDF5 groups. When `false` (the default), only the
/// full-signal multi-channel decomp is run and ROIs are extracted post-hoc
/// from the derived modes.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RoiSelectionSpec {
    /// Human-readable identifier used in HDF5 attrs and output dir naming.
    #[serde(default)]
    pub name: String,
    /// Exact match against `RoiType::Cortical.region` (e.g. `"PFCm"`).
    /// Each entry may optionally pin a hemisphere.
    #[serde(default)]
    pub cortical_regions: Vec<RoiSpec>,
    /// Exact match against `RoiType::Cortical.network` (e.g. `"LimbicA"`).
    /// Intersected with `cortical_regions` when both are non-empty.
    #[serde(default)]
    pub cortical_networks: Vec<RoiSpec>,
    /// Substring match against `RoiType::Subcortical.region` (e.g. `"AMY"`
    /// catches `lAMY` and `mAMY`). Each entry may optionally pin a hemisphere.
    #[serde(default)]
    pub subcortical_regions: Vec<RoiSpec>,
    /// Run a separate ROI-stratified MVMD decomposition in addition to the
    /// full-signal pass. Default `false` — ROIs extracted post-hoc.
    #[serde(default)]
    pub stratified_decomposition: bool,
}

impl RoiSelectionSpec {
    pub fn is_empty(&self) -> bool {
        self.cortical_regions.is_empty()
            && self.cortical_networks.is_empty()
            && self.subcortical_regions.is_empty()
    }

    /// Stable string identifier used for migration checks. Mismatch between
    /// stored fingerprint on an HDF5 group and the current config means the
    /// data was produced under a different selection and must be regenerated.
    ///
    /// Base format: `"{name}:{sorted_cort_names}|{sorted_subc_names}"`. Bare
    /// entries (no hemisphere) produce the same fingerprint as before this
    /// change — backward compatible with existing HDF5 attrs.
    ///
    /// Extended suffixes (only when used):
    /// - `|net={sorted_networks}` — when `cortical_networks` is non-empty
    /// - `|hemi=cort:{...};net:{...};subc:{...}` — when any entry pins a hemi
    /// - `|strat=1` — when `stratified_decomposition` is true
    pub fn fingerprint(&self) -> String {
        let sorted_names = |specs: &[RoiSpec]| -> Vec<String> {
            let mut v: Vec<String> = specs.iter().map(|r| r.name().to_string()).collect();
            v.sort();
            v
        };

        let cort = sorted_names(&self.cortical_regions);
        let subc = sorted_names(&self.subcortical_regions);
        let base = format!("{}:{}|{}", self.name, cort.join(","), subc.join(","));

        let mut fp = if self.cortical_networks.is_empty() {
            base
        } else {
            let nets = sorted_names(&self.cortical_networks);
            format!("{}|net={}", base, nets.join(","))
        };

        let has_hemi_pin = self
            .cortical_regions
            .iter()
            .chain(&self.cortical_networks)
            .chain(&self.subcortical_regions)
            .any(|r| matches!(r, RoiSpec::WithHemi { .. }));

        if has_hemi_pin {
            let encode_hemi_list = |specs: &[RoiSpec]| -> String {
                let mut parts: Vec<String> = specs
                    .iter()
                    .map(|r| match r {
                        RoiSpec::Bare(n) => n.clone(),
                        RoiSpec::WithHemi { region, hemisphere } => format!(
                            "{}@{}",
                            region,
                            match hemisphere {
                                HemiFilter::LH => "LH",
                                HemiFilter::RH => "RH",
                            }
                        ),
                    })
                    .collect();
                parts.sort();
                parts.join(",")
            };
            fp = format!(
                "{}|hemi=cort:{};net:{};subc:{}",
                fp,
                encode_hemi_list(&self.cortical_regions),
                encode_hemi_list(&self.cortical_networks),
                encode_hemi_list(&self.subcortical_regions),
            );
        }

        if self.stratified_decomposition {
            fp = format!("{}|strat=1", fp);
        }

        fp
    }
}

pub fn load_cortical_lut(filename: &Path) -> HashMap<String, u32> {
    let file = fs::File::open(filename).expect("Failed to open cortical atlas LUT");
    let reader = BufReader::new(file);
    let mut cortical_roi_map = HashMap::new();

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

pub fn load_subcortical_lut(filename: &Path) -> HashMap<String, u32> {
    let file = fs::File::open(filename).expect("Failed to open subcortical atlas LUT");
    let mut subcortical_roi_map = HashMap::new();
    let reader = BufReader::new(file);
    for (index, line_result) in reader.lines().enumerate() {
        let line = line_result.expect("Failed to read line from file");
        let roi_id = line.trim().to_string();
        subcortical_roi_map.insert(roi_id, index as u32);
    }
    subcortical_roi_map
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_atlas() -> BrainAtlas {
        let mut cort = HashMap::new();
        cort.insert("17networks_LH_LimbicA_PFCm_1".to_string(), 0);
        cort.insert("17networks_RH_LimbicA_PFCm_1".to_string(), 1);
        cort.insert("17networks_LH_LimbicB_PFCv_1".to_string(), 2);
        cort.insert("17networks_RH_LimbicB_PFCv_1".to_string(), 3);
        cort.insert("17networks_LH_DefaultA_PFCm_1".to_string(), 4);
        cort.insert("17networks_LH_DefaultB_PFCv_1".to_string(), 5);
        cort.insert("17networks_LH_DefaultA_pCun_1".to_string(), 6);

        let mut subc = HashMap::new();
        subc.insert("lAMY-lh".to_string(), 0);
        subc.insert("mAMY-rh".to_string(), 1);
        subc.insert("HIP-lh".to_string(), 2);

        BrainAtlas::from_lut_maps(cort, subc)
    }

    #[test]
    fn region_only_selects_all_networks_in_region() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec!["PFCm".into()],
            cortical_networks: vec![],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        let labels: Vec<&str> = sel.iter().map(|r| r.label.as_str()).collect();
        assert!(labels.contains(&"17networks_LH_LimbicA_PFCm_1"));
        assert!(labels.contains(&"17networks_RH_LimbicA_PFCm_1"));
        assert!(labels.contains(&"17networks_LH_DefaultA_PFCm_1"));
        assert_eq!(sel.len(), 3);
    }

    #[test]
    fn network_only_selects_all_regions_in_network() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec![],
            cortical_networks: vec!["LimbicA".into(), "LimbicB".into()],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        let labels: Vec<&str> = sel.iter().map(|r| r.label.as_str()).collect();
        assert!(labels.contains(&"17networks_LH_LimbicA_PFCm_1"));
        assert!(labels.contains(&"17networks_RH_LimbicA_PFCm_1"));
        assert!(labels.contains(&"17networks_LH_LimbicB_PFCv_1"));
        assert!(labels.contains(&"17networks_RH_LimbicB_PFCv_1"));
        assert_eq!(sel.len(), 4);
    }

    #[test]
    fn region_and_network_intersect() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec!["PFCv".into(), "PFCm".into()],
            cortical_networks: vec!["LimbicA".into(), "LimbicB".into()],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        assert_eq!(sel.len(), 4);
        for r in &sel {
            assert!(r.label.contains("Limbic"));
            assert!(r.label.contains("PFC"));
        }
    }

    #[test]
    fn empty_cortical_filters_match_no_cortical() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec![],
            cortical_networks: vec![],
            subcortical_regions: vec!["AMY".into()],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        assert_eq!(sel.len(), 2);
        assert!(sel.iter().all(|r| r.kind == "subcortical"));
    }

    #[test]
    fn hemi_filter_lh_excludes_rh_cortical() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec![RoiSpec::WithHemi {
                region: "PFCm".into(),
                hemisphere: HemiFilter::LH,
            }],
            cortical_networks: vec![],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        assert!(sel.iter().all(|r| r.hemisphere == Hemisphere::Left));
        assert!(sel.iter().any(|r| r.label.contains("LH")));
        assert!(sel.iter().all(|r| !r.label.contains("RH")));
    }

    #[test]
    fn hemi_filter_subcortical_rh_only() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec![],
            cortical_networks: vec![],
            subcortical_regions: vec![RoiSpec::WithHemi {
                region: "AMY".into(),
                hemisphere: HemiFilter::RH,
            }],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        assert_eq!(sel.len(), 1);
        assert_eq!(sel[0].label, "mAMY-rh");
        assert_eq!(sel[0].hemisphere, Hemisphere::Right);
    }

    #[test]
    fn hemi_bare_and_pinned_mix() {
        let atlas = fixture_atlas();
        let spec = RoiSelectionSpec {
            name: "t".into(),
            cortical_regions: vec![
                RoiSpec::Bare("PFCv".into()),
                RoiSpec::WithHemi {
                    region: "PFCm".into(),
                    hemisphere: HemiFilter::LH,
                },
            ],
            cortical_networks: vec![],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        let sel = atlas.selected_rois(&spec);
        // PFCv: both hemis (2); PFCm LH only (2 networks in fixture)
        let pfcv: Vec<_> = sel.iter().filter(|r| r.matched_region == "PFCv").collect();
        let pfcm: Vec<_> = sel.iter().filter(|r| r.matched_region == "PFCm").collect();
        assert_eq!(pfcv.len(), 2);
        assert!(pfcm.iter().all(|r| r.hemisphere == Hemisphere::Left));
    }

    #[test]
    fn fingerprint_backward_compat_when_networks_empty() {
        let spec = RoiSelectionSpec {
            name: "vpfc_mpfc_amy".into(),
            cortical_regions: vec!["PFCm".into(), "PFCv".into()],
            cortical_networks: vec![],
            subcortical_regions: vec!["AMY".into()],
            stratified_decomposition: false,
        };
        assert_eq!(spec.fingerprint(), "vpfc_mpfc_amy:PFCm,PFCv|AMY");
    }

    #[test]
    fn fingerprint_appends_sorted_networks_when_set() {
        let spec = RoiSelectionSpec {
            name: "limbic_pfc".into(),
            cortical_regions: vec!["PFCm".into()],
            cortical_networks: vec!["LimbicB".into(), "LimbicA".into()],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        assert_eq!(spec.fingerprint(), "limbic_pfc:PFCm||net=LimbicA,LimbicB");
    }

    #[test]
    fn fingerprint_hemi_segment_added_when_pinned() {
        let spec = RoiSelectionSpec {
            name: "test".into(),
            cortical_regions: vec![
                "PFCv".into(),
                RoiSpec::WithHemi {
                    region: "PFCm".into(),
                    hemisphere: HemiFilter::LH,
                },
            ],
            cortical_networks: vec![],
            subcortical_regions: vec!["AMY".into()],
            stratified_decomposition: false,
        };
        let fp = spec.fingerprint();
        assert!(fp.contains("|hemi="));
        assert!(fp.contains("PFCm@LH"));
        assert!(fp.contains("PFCv"));
    }

    #[test]
    fn fingerprint_strat_suffix() {
        let spec = RoiSelectionSpec {
            name: "test".into(),
            cortical_regions: vec!["PFCm".into()],
            cortical_networks: vec![],
            subcortical_regions: vec![],
            stratified_decomposition: true,
        };
        assert!(spec.fingerprint().ends_with("|strat=1"));
    }

    #[test]
    fn fingerprint_no_strat_suffix_when_false() {
        let spec = RoiSelectionSpec {
            name: "test".into(),
            cortical_regions: vec!["PFCm".into()],
            cortical_networks: vec![],
            subcortical_regions: vec![],
            stratified_decomposition: false,
        };
        assert!(!spec.fingerprint().contains("|strat="));
    }

    #[test]
    fn is_empty_requires_all_three_lists_empty() {
        let mut spec = RoiSelectionSpec::default();
        assert!(spec.is_empty());
        spec.cortical_networks = vec!["LimbicA".into()];
        assert!(!spec.is_empty());
    }
}
