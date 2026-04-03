use std::{
    fs,
    path::{Path, PathBuf},
};

/// A structured representation of a BIDS-style `key-value` filename.
///
/// Format: `key1-val1_key2-val2_suffix.ext`
///
/// Each underscore-separated token containing a `-` is a key-value entity.
/// A trailing token without `-` is the suffix (e.g., `bold`, `events`).
/// Extensions are preserved separately, with `.nii.gz` treated as a unit.
#[derive(Debug, Clone, PartialEq)]
pub struct BidsFilename {
    pub pairs: Vec<(String, String)>,
    pub suffix: Option<String>,
    pub extension: Option<String>,
}

impl BidsFilename {
    /// Parse a filename string into its BIDS components.
    pub fn parse(s: &str) -> Self {
        let (stem, ext) = strip_bids_extension(s);

        let mut pairs = Vec::new();
        let mut suffix = None;

        for token in stem.split('_').filter(|t| !t.is_empty()) {
            match token.find('-') {
                Some(pos) => pairs.push((token[..pos].to_string(), token[pos + 1..].to_string())),
                None => suffix = Some(token.to_string()),
            }
        }

        Self {
            pairs,
            suffix,
            extension: if ext.is_empty() {
                None
            } else {
                Some(ext.to_string())
            },
        }
    }

    /// Returns the value for `key`, or `None` if not present.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.pairs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }

    /// Returns `true` if the filename contains `key` with exactly `value`.
    pub fn matches_pair(&self, key: &str, value: &str) -> bool {
        self.pairs.iter().any(|(k, v)| k == key && v == value)
    }

    /// Returns a new `BidsFilename` keeping only the specified keys (preserves suffix and extension).
    pub fn keep(&self, keys: &[&str]) -> Self {
        Self {
            pairs: self
                .pairs
                .iter()
                .filter(|(k, _)| keys.contains(&k.as_str()))
                .cloned()
                .collect(),
            suffix: self.suffix.clone(),
            extension: self.extension.clone(),
        }
    }

    /// Returns a new `BidsFilename` with the specified keys removed.
    pub fn without(&self, keys: &[&str]) -> Self {
        Self {
            pairs: self
                .pairs
                .iter()
                .filter(|(k, _)| !keys.contains(&k.as_str()))
                .cloned()
                .collect(),
            suffix: self.suffix.clone(),
            extension: self.extension.clone(),
        }
    }

    /// Reconstructs the filename stem without extension.
    ///
    /// Guarantees no leading, trailing, or consecutive `_` separators.
    pub fn to_stem(&self) -> String {
        let mut parts: Vec<String> = self
            .pairs
            .iter()
            .map(|(k, v)| format!("{}-{}", k, v))
            .collect();
        if let Some(s) = &self.suffix {
            parts.push(s.clone());
        }
        parts.join("_")
    }

    /// Reconstructs the full filename (stem + extension if present).
    pub fn to_filename(&self) -> String {
        match &self.extension {
            Some(ext) => format!("{}{}", self.to_stem(), ext),
            None => self.to_stem(),
        }
    }
}

impl std::fmt::Display for BidsFilename {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_filename())
    }
}

/// Find all regular files directly under `dir` whose BIDS filename matches all
/// `required_pairs`, an optional `suffix`, and an optional file `extension`.
///
/// Does not recurse into subdirectories. Returns an empty `Vec` if `dir` cannot
/// be read.
pub fn find_bids_files(
    dir: &Path,
    required_pairs: &[(&str, &str)],
    suffix: Option<&str>,
    extension: Option<&str>,
) -> Vec<PathBuf> {
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };

    entries
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_file())
        .filter(|p| {
            let name = match p.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => return false,
            };
            let bids = BidsFilename::parse(name);
            required_pairs.iter().all(|(k, v)| bids.matches_pair(k, v))
                && suffix.map_or(true, |s| bids.suffix.as_deref() == Some(s))
                && extension.map_or(true, |e| bids.extension.as_deref() == Some(e))
        })
        .collect()
}

fn strip_bids_extension(s: &str) -> (&str, &str) {
    if let Some(stem) = s.strip_suffix(".nii.gz") {
        return (stem, ".nii.gz");
    }
    if let Some(idx) = s.rfind('.') {
        return (&s[..idx], &s[idx..]);
    }
    (s, "")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_full_bold_filename() {
        let f = BidsFilename::parse(
            "sub-NDARINVAL101MH2_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
        );
        assert_eq!(f.get("sub"), Some("NDARINVAL101MH2"));
        assert_eq!(f.get("task"), Some("hammerAP"));
        assert_eq!(f.get("run"), Some("01"));
        assert_eq!(f.get("space"), Some("MNI152NLin2009cAsym"));
        assert_eq!(f.get("res"), Some("2"));
        assert_eq!(f.get("desc"), Some("preproc"));
        assert_eq!(f.suffix.as_deref(), Some("bold"));
        assert_eq!(f.extension.as_deref(), Some(".h5"));
    }

    #[test]
    fn parse_nii_gz_extension() {
        let f = BidsFilename::parse(
            "sub-ABC_task-restAP_run-02_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz",
        );
        assert_eq!(f.extension.as_deref(), Some(".nii.gz"));
        assert_eq!(f.suffix.as_deref(), Some("bold"));
    }

    #[test]
    fn without_sub_strips_subject() {
        let stem = BidsFilename::parse(
            "sub-NDARINVAL101MH2_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
        )
        .without(&["sub"])
        .to_stem();
        assert_eq!(
            stem,
            "task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold"
        );
    }

    #[test]
    fn keep_filters_to_specified_keys() {
        let stem = BidsFilename::parse(
            "sub-NDARINVAL101MH2_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold",
        )
        .keep(&["sub", "task", "run"])
        .to_stem();
        assert_eq!(stem, "sub-NDARINVAL101MH2_task-hammerAP_run-01_bold");
    }

    #[test]
    fn to_stem_no_double_underscores() {
        let f = BidsFilename {
            pairs: vec![("task".into(), "hammerAP".into()), ("run".into(), "01".into())],
            suffix: Some("bold".into()),
            extension: None,
        };
        assert_eq!(f.to_stem(), "task-hammerAP_run-01_bold");
    }

    #[test]
    fn matches_pair_works() {
        let f = BidsFilename::parse("sub-ABC_task-hammerAP_run-01_bold.h5");
        assert!(f.matches_pair("task", "hammerAP"));
        assert!(!f.matches_pair("task", "restAP"));
    }

    #[test]
    fn display_renders_full_filename() {
        let f = BidsFilename::parse(
            "sub-NDARINVAL101MH2_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5",
        );
        assert_eq!(
            format!("{}", f),
            "sub-NDARINVAL101MH2_task-hammerAP_run-01_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.h5"
        );
    }
}
