use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hemisphere {
    Left,
    Right,
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
                        || spec.cortical_regions.iter().any(|r| r == region);
                    let network_ok = spec.cortical_networks.is_empty()
                        || spec.cortical_networks.iter().any(|n| n == network);
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
                    .find(|pat| region.contains(pat.as_str()))
                    .map(|matched| SelectedRoi {
                        row_index: self.n_cortical + e.index as usize,
                        label: e.id.clone(),
                        matched_region: matched.clone(),
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
/// the spec-dependent pipeline stages (04mvmd `_roi`, 05hilbert `_roi`,
/// 06fc `_roi`, 07feature_extraction). Empty cortical+subcortical lists mean
/// "no subset" — currently only relevant for future "all ROIs" mode.
///
/// Cortical filtering combines `cortical_regions` and `cortical_networks` as
/// an intersection: a cortical ROI is included only when it matches every
/// axis that has a non-empty list. An empty list on a given axis means "no
/// constraint on this axis" (e.g. empty `cortical_networks` = all networks
/// allowed). Cortical inclusion still requires at least one axis to be
/// specified — leaving both empty selects no cortical rows.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RoiSelectionSpec {
    /// Human-readable identifier used in HDF5 attrs and output dir naming.
    #[serde(default)]
    pub name: String,
    /// Exact match against `RoiType::Cortical.region` (e.g. `"PFCm"`).
    #[serde(default)]
    pub cortical_regions: Vec<String>,
    /// Exact match against `RoiType::Cortical.network` (e.g. `"LimbicA"`).
    /// Intersected with `cortical_regions` when both are non-empty.
    #[serde(default)]
    pub cortical_networks: Vec<String>,
    /// Substring match against `RoiType::Subcortical.region` (e.g. `"AMY"`
    /// catches `lAMY` and `mAMY`).
    #[serde(default)]
    pub subcortical_regions: Vec<String>,
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
    /// Format: `"{name}:{cortical_regions}|{subcortical_regions}"` when
    /// `cortical_networks` is empty (preserves fingerprints for configs that
    /// never opt into network filtering). When networks are specified the
    /// suffix `|net={cortical_networks}` is appended.
    pub fn fingerprint(&self) -> String {
        let mut cort = self.cortical_regions.clone();
        cort.sort();
        let mut subc = self.subcortical_regions.clone();
        subc.sort();
        let base = format!("{}:{}|{}", self.name, cort.join(","), subc.join(","));
        if self.cortical_networks.is_empty() {
            base
        } else {
            let mut nets = self.cortical_networks.clone();
            nets.sort();
            format!("{}|net={}", base, nets.join(","))
        }
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
        };
        let sel = atlas.selected_rois(&spec);
        assert_eq!(sel.len(), 2);
        assert!(sel.iter().all(|r| r.kind == "subcortical"));
    }

    #[test]
    fn fingerprint_backward_compat_when_networks_empty() {
        let spec = RoiSelectionSpec {
            name: "vpfc_mpfc_amy".into(),
            cortical_regions: vec!["PFCm".into(), "PFCv".into()],
            cortical_networks: vec![],
            subcortical_regions: vec!["AMY".into()],
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
        };
        assert_eq!(spec.fingerprint(), "limbic_pfc:PFCm||net=LimbicA,LimbicB");
    }

    #[test]
    fn is_empty_requires_all_three_lists_empty() {
        let mut spec = RoiSelectionSpec::default();
        assert!(spec.is_empty());
        spec.cortical_networks = vec!["LimbicA".into()];
        assert!(!spec.is_empty());
    }
}
