use std::collections::HashMap;

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
    pub id: String, // original ID (e.g., "lAMY-rh")
    pub index: u32, // The index in [roi][time] array
    pub hemisphere: Hemisphere,
    pub metadata: RoiType,
}

pub struct BrainAtlas {
    pub entries: Vec<RoiEntry>,
}

impl BrainAtlas {
    pub fn from_lut_maps(
        cortical_map: HashMap<String, u32>,
        subcortical_map: HashMap<String, u32>,
    ) -> Self {
        let mut entries = Vec::new();

        // Convert Cortical
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

            // Handle trailing parcel numbers vs anatomical regions
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

        // Convert Subcortical
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

        Self { entries }
    }
}

impl BrainAtlas {
    /// Generic filter: find all indices matching a condition
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
}

// Network regions
impl BrainAtlas {
    /// Search specifically for a cortical network (e.g., "DefaultA")
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
}

// Anatomical regions
impl BrainAtlas {
    /// Find indices by cortical region name (e.g., "PFCv", "PFCm")
    pub fn find_cortical_by_region(&self, region_name: &str, hemi: Option<Hemisphere>) -> Vec<u32> {
        self.find_indices(|e| {
            if let RoiType::Cortical { region, .. } = &e.metadata {
                region == region_name && hemi.map_or(true, |h| h == e.hemisphere)
            } else {
                false
            }
        })
    }

    /// Find indices by subcortical region name (e.g., "lAMY", "mAMY")
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
}

impl BrainAtlas {
    /// Returns the original IDs for a specific search
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
}
