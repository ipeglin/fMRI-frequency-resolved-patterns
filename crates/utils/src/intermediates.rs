use anyhow::Result;
use polars::prelude::DataFrame;
use std::path::PathBuf;

use crate::config::AppConfig;
use crate::polars_csv::write_tsv;

/// Returns the stage-specific sub-directory under `intermediates_output_dir`.
pub fn stage_dir(cfg: &AppConfig, stage: &str) -> PathBuf {
    cfg.intermediates_output_dir.join(stage)
}

/// Write `df` as a tab-separated file at `intermediates_output_dir/<stage>/<filename>`.
///
/// No-op when `cfg.dump_intermediates` is false. Creates parent directories as
/// needed. `filename` should be a BIDS-ish basename including extension, e.g.
/// `sub-NDARXXX_task-hammerAP_run-01_desc-modes_summary.tsv`.
pub fn dump_tsv(cfg: &AppConfig, stage: &str, filename: &str, df: &DataFrame) -> Result<()> {
    if !cfg.dump_intermediates {
        return Ok(());
    }
    let dir = stage_dir(cfg, stage);
    std::fs::create_dir_all(&dir)?;
    let path = dir.join(filename);
    write_tsv(&path, df)
        .map_err(|e| anyhow::anyhow!("failed to write TSV {}: {}", path.display(), e))?;
    Ok(())
}
