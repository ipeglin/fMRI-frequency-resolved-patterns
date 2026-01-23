use anyhow::Result;
use config::TCPfMRIPreprocessConfig;
use tracing::info;

pub fn run(cfg: &TCPfMRIPreprocessConfig) -> Result<()> {
    info!("{:?}", cfg);

    Ok(())
}
