use anyhow::Result;
use config::TCPPreprocessConfig;

pub fn run(cfg: &TCPPreprocessConfig) -> Result<()> {
    print_config(cfg);

    Ok(())
}

fn print_config(cfg: &TCPPreprocessConfig) {
    println!("TPC Preprocessing:");
    println!("  fMRI Dir: {}", cfg.fmri_dir.display());
    println!("  TCP Dir: {}", cfg.tcp_dir.display());
    println!("  Output Dir: {}", cfg.output_dir.display());
    println!("  Filters: {:?}", cfg.filters);
    println!("  Dry run: {}", cfg.dry_run);
}
