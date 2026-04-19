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
                let match_hemi = hemi.map_or(true, |h| h == e.hemisphere);
                match_name && match_hemi
            } else {
                false
            }
        })
    }

    pub fn find_cortical_by_region(&self, region_name: &str, hemi: Option<Hemisphere>) -> Vec<u32> {
        self.find_indices(|e| {
            if let RoiType::Cortical { region, .. } = &e.metadata {
                region == region_name && hemi.map_or(true, |h| h == e.hemisphere)
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
                region == region_name && hemi.map_or(true, |h| h == e.hemisphere)
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

    /// Row indices for the subset of interest: vPFC (PFCv), mPFC (PFCm) cortical
    /// plus amygdala (AMY) subcortical ROIs, across both hemispheres.
    pub fn vpfc_mpfc_amy_indices(&self) -> Vec<usize> {
        let mut idxs: Vec<usize> = self
            .concat_row_indices(|e| match &e.metadata {
                RoiType::Cortical { region, .. } => region == "PFCv" || region == "PFCm",
                RoiType::Subcortical { region, .. } => region.contains("AMY"),
            });
        idxs.sort_unstable();
        idxs.dedup();
        idxs
    }

    pub fn vpfc_mpfc_amy_ids(&self) -> Vec<(usize, String)> {
        let mut pairs: Vec<(usize, String)> = self
            .entries
            .iter()
            .filter(|e| match &e.metadata {
                RoiType::Cortical { region, .. } => region == "PFCv" || region == "PFCm",
                RoiType::Subcortical { region, .. } => region.contains("AMY"),
            })
            .map(|e| {
                let row = match &e.metadata {
                    RoiType::Cortical { .. } => e.index as usize,
                    RoiType::Subcortical { .. } => self.n_cortical + e.index as usize,
                };
                (row, e.id.clone())
            })
            .collect();
        pairs.sort_by_key(|(i, _)| *i);
        pairs
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
