use anyhow::Result;
use config::TCPfMRIProcessConfig;
use std::time::Instant;
use tracing::info;

pub fn run(cfg: &TCPfMRIProcessConfig) -> Result<()> {
    let run_start = Instant::now();

    info!(
        fmri_dir = %cfg.fmri_dir.display(),
        subjects = ?cfg.subjects,
        output_dir = %cfg.output_dir.display(),
        "starting fMRI processing pipeline"
    );

    Ok(())
}
