use anyhow::{anyhow, Result};
use hdf5::types::VarLenUnicode;

use crate::hdf5_io::{H5Attr, write_attrs};

/// Read a string-valued attribute, returning `None` if absent.
fn read_str_attr(loc: &hdf5::Location, name: &str) -> Result<Option<String>> {
    let Ok(attr) = loc.attr(name) else {
        return Ok(None);
    };
    let s = attr
        .read_raw::<VarLenUnicode>()?
        .into_iter()
        .next()
        .map(|v| v.to_string());
    Ok(s)
}

/// Read a u32-valued attribute, returning `None` if absent.
fn read_u32_attr(loc: &hdf5::Location, name: &str) -> Result<Option<u32>> {
    let Ok(attr) = loc.attr(name) else {
        return Ok(None);
    };
    let v = attr.read_raw::<u32>()?.into_iter().next();
    Ok(v)
}

/// Mirror the ROI-tracking attrs (`roi_selection_fingerprint`, `roi_selection_name`,
/// `roi_matched_regions`, `roi_labels`, `n_rois`) from `src` onto `dest`. Skips
/// individual attrs that are missing on `src` and skips writes for attrs that
/// already exist on `dest`. Used by 05hilbert / 06fc / 07feature_extraction so
/// that downstream fingerprint checks remain valid through the pipeline.
pub fn propagate_roi_attrs(src: &hdf5::Location, dest: &hdf5::Location) -> Result<()> {
    let mut attrs: Vec<H5Attr> = Vec::new();
    for name in ["roi_selection_fingerprint", "roi_selection_name", "roi_matched_regions", "roi_labels"] {
        if dest.attr(name).is_ok() {
            continue;
        }
        if let Some(v) = read_str_attr(src, name)? {
            attrs.push(H5Attr::string(name, v));
        }
    }
    if dest.attr("n_rois").is_err() {
        if let Some(v) = read_u32_attr(src, "n_rois")? {
            attrs.push(H5Attr::u32("n_rois", v));
        }
    }
    if !attrs.is_empty() {
        write_attrs(dest, &attrs)?;
    }
    Ok(())
}

/// Reads the `roi_selection_fingerprint` attribute from an HDF5 location and
/// compares it against the expected value. Returns `Ok(())` on match.
///
/// On mismatch (or missing attr) returns an error directing the user to rerun
/// the producing stage with `--force` so the ROI-dependent group is rewritten
/// against the current `[roi_selection]` config.
///
/// Used by 04mvmd / 05hilbert / 06fc / 07feature_extraction to refuse loading
/// stale `_roi` groups produced under a different ROI selection.
pub fn check_roi_fingerprint(loc: &hdf5::Location, expected: &str, group_path: &str) -> Result<()> {
    let attr = match loc.attr("roi_selection_fingerprint") {
        Ok(a) => a,
        Err(_) => {
            return Err(anyhow!(
                "ROI selection fingerprint missing on existing group `{group_path}` — \
                 produced by an older pipeline before fingerprint tracking. \
                 Rerun the producing stage with `--force` to regenerate against \
                 the current `[roi_selection]` config."
            ));
        }
    };
    let stored: String = attr
        .read_raw::<VarLenUnicode>()?
        .into_iter()
        .next()
        .map(|s| s.to_string())
        .unwrap_or_default();
    if stored != expected {
        return Err(anyhow!(
            "ROI selection fingerprint mismatch on `{group_path}`: stored=`{stored}`, \
             expected=`{expected}`. The data was produced under a different `[roi_selection]` \
             config. Rerun the producing stage with `--force` to regenerate."
        ));
    }
    Ok(())
}
