use std::{
    cmp::Ordering,
    fmt, fs,
    path::{Path, PathBuf},
};

/// A structured representation of a BIDS-style `key-value` filename.
///
/// Format: `key1-val1_key2-val2_suffix.ext`
///
/// Each underscore-separated token containing a `-` is a key-value entity.
/// A trailing token without `-` is the suffix (e.g., `bold`, `events`).
/// Extensions are preserved separately, with `.nii.gz` treated as a unit.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct BidsFilename {
    pub pairs: Vec<(String, String)>,
    pub suffix: Option<String>,
    pub extension: Option<String>,
    pub path: Option<PathBuf>,
    pub parent_directory: Option<PathBuf>,
}

impl fmt::Display for BidsFilename {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // self.to_path_buf() returns:
        // - "path/to/file.nii.gz" if parent_directory is set
        // - "file.nii.gz" if parent_directory is None
        write!(f, "{}", self.to_path_buf().display())
    }
}

impl BidsFilename {
    /// Creates an empty BidsFilename.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder method to add a key-value pair.
    pub fn with_pair<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.pairs.push((key.into(), value.into()));
        self
    }

    /// Builder method to set the suffix.
    pub fn with_suffix<S: Into<String>>(mut self, suffix: S) -> Self {
        self.suffix = Some(suffix.into());
        self
    }

    /// Builder method to set the extension.
    pub fn with_extension<E: Into<String>>(mut self, extension: E) -> Self {
        self.extension = Some(extension.into());
        self
    }

    /// Reorders the internal pairs based on the provided list of keys.
    /// Keys not in the list will be moved to the end.
    pub fn reorder_by_keys(&mut self, order: &[&str]) {
        self.pairs.sort_by(|(a, _), (b, _)| {
            let pos_a = order.iter().position(|&k| k == a).unwrap_or(usize::MAX);
            let pos_b = order.iter().position(|&k| k == b).unwrap_or(usize::MAX);
            pos_a.cmp(&pos_b)
        });
    }

    /// Parse a filename string into its BIDS components.
    pub fn parse(s: &str) -> Self {
        let path = Path::new(s);

        // Check if the filepath itself exists
        let (stored_path, parent_dir) = if path.exists() {
            (
                Some(path.to_path_buf()),
                path.parent().map(|p| p.to_path_buf()),
            )
        } else {
            (None, None)
        };

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
            path: stored_path,
            parent_directory: parent_dir,
        }
    }

    pub fn from_path_buf(path: &Path) -> Self {
        let s = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
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
            path: Some(path.to_path_buf()),
            parent_directory: path.parent().map(|p| p.to_path_buf()),
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
            path: self.path.clone(),
            parent_directory: self
                .path
                .as_ref()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf()),
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
            path: self.path.clone(),
            parent_directory: self
                .path
                .as_ref()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf()),
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

    /// Reconstructs the full path by joining the parent directory and the BIDS filename.
    /// If no parent directory is set, it returns the filename as a relative path.
    pub fn to_path_buf(&self) -> PathBuf {
        let filename = self.to_filename();

        match &self.parent_directory {
            Some(dir) => dir.join(filename),
            None => PathBuf::from(filename),
        }
    }

    /// This enforces that the user must specify a path for file operations.
    pub fn try_to_path_buf(&self) -> Option<PathBuf> {
        self.parent_directory
            .as_ref()
            .map(|dir| dir.join(self.to_filename()))
    }

    /// Updates the parent directory in-place.
    pub fn set_directory<P: AsRef<Path>>(&mut self, path: P) {
        self.parent_directory = Some(path.as_ref().to_path_buf());
    }

    /// Returns a new version of the struct with a different parent directory (Builder pattern).
    pub fn with_directory<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.parent_directory = Some(path.as_ref().to_path_buf());
        self
    }

    /// Checks if the file represented by this BidsFilename exists on the filesystem.
    /// Note: This checks the reconstructed path (parent_directory + filename).
    pub fn exists(&self) -> bool {
        self.to_path_buf().exists()
    }

    /// Checks if the *original* path (if one was provided via from_path_buf) exists.
    pub fn original_exists(&self) -> bool {
        self.path.as_ref().is_some_and(|p| p.exists())
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
                && suffix.is_none_or(|s| bids.suffix.as_deref() == Some(s))
                && extension.is_none_or(|e| bids.extension.as_deref() == Some(e))
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

pub fn filter_directory_bids_files<F>(
    dir: &Path,
    predicate: F,
) -> Result<Vec<BidsFilename>, Box<dyn std::error::Error>>
where
    F: Fn(&BidsFilename) -> bool,
{
    let files = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_file())
        .filter_map(|path| {
            let bids = BidsFilename::from_path_buf(&path);
            if predicate(&bids) { Some(bids) } else { None }
        })
        .collect();

    Ok(files)
}

pub fn sort_bids_vec<F>(files: &mut [BidsFilename], sort_keys: &[&str], custom_compare: F)
where
    F: Fn(&str, &str, &str) -> Ordering,
    // Arguments: (key, value_a, value_b)
{
    files.sort_by(|a, b| {
        for &key in sort_keys {
            let val_a = a.get(key);
            let val_b = b.get(key);

            match (val_a, val_b) {
                (Some(v_a), Some(v_b)) => {
                    let res = custom_compare(key, v_a, v_b);
                    if res != Ordering::Equal {
                        return res;
                    }
                }
                // Handle cases where one file is missing a key
                (Some(_), None) => return Ordering::Less,
                (None, Some(_)) => return Ordering::Greater,
                (None, None) => continue,
            }
        }
        Ordering::Equal
    });
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
            pairs: vec![
                ("task".into(), "hammerAP".into()),
                ("run".into(), "01".into()),
            ],
            suffix: Some("bold".into()),
            extension: None,
            path: None,
            parent_directory: None,
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
