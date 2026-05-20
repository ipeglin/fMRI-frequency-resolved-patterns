use anyhow::Result;
use tracing::info;
use utils::config::AppConfig;

mod aggregation;
pub(crate) mod analyses;
pub(crate) mod dispatch;
mod io;
pub mod stats;

pub fn run(cfg: &AppConfig) -> Result<()> {
    let results_dir = dispatch::results_dir(cfg);
    std::fs::create_dir_all(&results_dir)?;

    info!(
        results_dir = %results_dir.display(),
        "global roi_selection is no longer used by stage 09; per-analysis ROIs apply"
    );

    analyses::keedwell_face_vs_shape::run(cfg)?;
    analyses::example_rest_dmn::run(cfg)?;
    analyses::example_face_only_amygdala::run(cfg)?;

    info!("FC analysis complete. Results in {}", results_dir.display());
    Ok(())
}
