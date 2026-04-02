use std::path::{Path, PathBuf};

/// A structured BIDS subject identifier, normalizing between the three forms
/// this codebase encounters:
///
/// | Form             | Example                  |
/// |------------------|--------------------------|
/// | BIDS directory   | `sub-NDARINVXXXXXXXX`    |
/// | BIDS entity value (filenames) | `NDARINVXXXXXXXX` |
/// | NDA subjectkey   | `NDAR_INVXXXXXXXX`       |
///
/// The canonical internal representation is the BIDS entity value (no `sub-`
/// prefix, no underscores). All three input forms are accepted by [`parse`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BidsSubjectId(String);

impl BidsSubjectId {
    /// Parse from any of the three forms:
    /// - `sub-NDARINVXXXXXXXX` (BIDS directory name)
    /// - `NDARINVXXXXXXXX` (BIDS entity value, used inside filenames)
    /// - `NDAR_INVXXXXXXXX` (NDA subjectkey, e.g. from `demos.tsv`)
    pub fn parse(s: &str) -> Self {
        // Strip leading "sub-" if present
        let without_prefix = s.strip_prefix("sub-").unwrap_or(s);

        // Normalise NDA subjectkey: NDAR_INV... -> NDARINV...
        // The underscore always appears at position 4 in NDAR IDs.
        let canonical = if without_prefix.starts_with("NDAR_") {
            let mut owned = without_prefix.to_string();
            owned.remove(4); // remove the '_' at index 4
            owned
        } else {
            without_prefix.to_string()
        };

        Self(canonical)
    }

    /// Returns the BIDS entity value as used inside filenames (e.g. `NDARINVXXXXXXXX`).
    pub fn as_bids_id(&self) -> &str {
        &self.0
    }

    /// Returns the BIDS directory name (e.g. `sub-NDARINVXXXXXXXX`).
    pub fn to_dir_name(&self) -> String {
        format!("sub-{}", self.0)
    }

    /// Returns the NDA subjectkey form (e.g. `NDAR_INVXXXXXXXX`).
    ///
    /// For NDAR IDs the underscore is re-inserted at position 4. For any other
    /// prefix the value is returned unchanged.
    pub fn to_subjectkey(&self) -> String {
        if self.0.starts_with("NDAR") && self.0.len() > 4 {
            format!("NDAR_{}", &self.0[4..])
        } else {
            self.0.clone()
        }
    }

    /// Returns the path to this subject's directory within a BIDS dataset root.
    pub fn to_dir(&self, bids_root: &Path) -> PathBuf {
        bids_root.join(self.to_dir_name())
    }
}

impl std::fmt::Display for BidsSubjectId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_dir_name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_from_bids_dir_name() {
        let id = BidsSubjectId::parse("sub-NDARINVAL101MH2");
        assert_eq!(id.as_bids_id(), "NDARINVAL101MH2");
        assert_eq!(id.to_dir_name(), "sub-NDARINVAL101MH2");
        assert_eq!(id.to_subjectkey(), "NDAR_INVAL101MH2");
    }

    #[test]
    fn parse_from_bids_entity_value() {
        let id = BidsSubjectId::parse("NDARINVAL101MH2");
        assert_eq!(id.as_bids_id(), "NDARINVAL101MH2");
        assert_eq!(id.to_dir_name(), "sub-NDARINVAL101MH2");
        assert_eq!(id.to_subjectkey(), "NDAR_INVAL101MH2");
    }

    #[test]
    fn parse_from_subjectkey() {
        let id = BidsSubjectId::parse("NDAR_INVAL101MH2");
        assert_eq!(id.as_bids_id(), "NDARINVAL101MH2");
        assert_eq!(id.to_dir_name(), "sub-NDARINVAL101MH2");
        assert_eq!(id.to_subjectkey(), "NDAR_INVAL101MH2");
    }

    #[test]
    fn all_three_forms_are_equivalent() {
        let from_dir = BidsSubjectId::parse("sub-NDARINVAL101MH2");
        let from_entity = BidsSubjectId::parse("NDARINVAL101MH2");
        let from_key = BidsSubjectId::parse("NDAR_INVAL101MH2");
        assert_eq!(from_dir, from_entity);
        assert_eq!(from_entity, from_key);
    }

    #[test]
    fn display_shows_dir_name() {
        let id = BidsSubjectId::parse("NDAR_INVAL101MH2");
        assert_eq!(format!("{}", id), "sub-NDARINVAL101MH2");
    }

    #[test]
    fn to_dir_appends_under_root() {
        let id = BidsSubjectId::parse("NDAR_INVAL101MH2");
        let root = Path::new("/data/ds005237");
        assert_eq!(
            id.to_dir(root),
            PathBuf::from("/data/ds005237/sub-NDARINVAL101MH2")
        );
    }
}
