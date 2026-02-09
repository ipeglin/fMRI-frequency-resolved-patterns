use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use config::{
    AppConfig, ISTARTSubjectSelectionConfig, TCPSubjectSelectionConfig, TCPfMRIPreprocessConfig,
    TCPfMRIProcessConfig, load_config,
};
use std::path::PathBuf;
use tracing::info;
use tracing_subscriber::{EnvFilter, fmt, prelude::*};

#[derive(Debug, Clone, Copy, ValueEnum, Default)]
enum LogFormat {
    #[default]
    Pretty,
    Json,
    Compact,
}

#[derive(Debug, Parser)]
#[command(name = "masters", version, about = "Preprocess + Process Pipeline")]
struct Cli {
    /// Path to a config TOML file
    #[arg(long, global = true, default_value = "config.toml")]
    config: PathBuf,

    /// Logging filter, e.g. 'info', 'debug', 'trace', 'myproj=debug'
    #[arg(long, global = true, default_value = "info")]
    log_level: String,

    /// Log output format
    #[arg(long, global = true, value_enum, default_value = "pretty")]
    log_format: LogFormat,

    #[command(subcommand)]
    cmd: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    TcpSelectSubjects {
        #[arg(long)]
        tcp_dir: Option<PathBuf>,

        #[arg(long)]
        tcp_annex_remote: Option<String>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long, value_delimiter = ',')]
        filters: Option<Vec<String>>,

        #[arg(long)]
        dry_run: Option<bool>,
    },
    TcpFmriPreprocess {
        #[arg(long)]
        fmri_dir: Option<PathBuf>,

        #[arg(long)]
        filter_dir: Option<PathBuf>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long)]
        cortical_atlas: Option<PathBuf>,

        #[arg(long)]
        subcortical_atlas: Option<PathBuf>,

        #[arg(long)]
        dry_run: Option<bool>,

        /// Force reprocessing of subjects that already have preprocessed output
        #[arg(long, short = 'f')]
        force: bool,
    },
    TcpFmriProcess {
        #[arg(long)]
        fmri_dir: Option<PathBuf>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long)]
        cortical_lut: Option<PathBuf>,

        #[arg(long)]
        subcortical_lut: Option<PathBuf>,

        #[arg(long)]
        subject_file: Option<PathBuf>,

        /// Force reprocessing of subjects that already exist in output files
        #[arg(long, short = 'f')]
        force: bool,
    },
    IstartSelectSubjects {
        #[arg(long)]
        istart_dir: Option<PathBuf>,

        #[arg(long)]
        istart_annex_remote: Option<String>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long)]
        dry_run: Option<bool>,
    },
}

fn init_logging(level: &str, format: LogFormat) {
    let filter = EnvFilter::new(level);

    match format {
        LogFormat::Json => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().json().with_file(true).with_line_number(true))
                .init();
        }
        LogFormat::Compact => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().compact())
                .init();
        }
        LogFormat::Pretty => {
            tracing_subscriber::registry()
                .with(filter)
                .with(fmt::layer().pretty())
                .init();
        }
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    init_logging(&cli.log_level, cli.log_format);

    let cfg = load_config(&cli.config).unwrap_or_else(|e| {
        eprintln!(
            "Warning: Failed to load config from {}: {}",
            cli.config.display(),
            e
        );
        eprintln!("Using default configuration values");
        AppConfig {
            tcp_subject_selection: TCPSubjectSelectionConfig::default(),
            tcp_fmri_preprocess: TCPfMRIPreprocessConfig::default(),
            tcp_fmri_process: TCPfMRIProcessConfig::default(),
            istart_subject_selection: ISTARTSubjectSelectionConfig::default(),
        }
    });

    info!(
        config_path = %cli.config.display(),
        log_level = %cli.log_level,
        version = env!("CARGO_PKG_VERSION"),
        "starting pipeline"
    );

    match cli.cmd {
        Command::TcpSelectSubjects {
            tcp_dir,
            tcp_annex_remote,
            output_dir,
            filters,
            dry_run,
        } => {
            let mut p = cfg.tcp_subject_selection;

            if let Some(v) = tcp_dir {
                p.tcp_dir = v
            };
            if let Some(v) = output_dir {
                p.output_dir = v
            };
            if let Some(v) = tcp_annex_remote {
                p.tcp_annex_remote = v
            };

            if let Some(ref v) = filters
                && !v.is_empty()
            {
                p.filters = filters;
            }

            if let Some(v) = dry_run {
                p.dry_run = v
            };

            tcp_subject_selection::run(&p)
        }
        Command::TcpFmriPreprocess {
            fmri_dir,
            filter_dir,
            output_dir,
            cortical_atlas,
            subcortical_atlas,
            dry_run,
            force,
        } => {
            let mut p = cfg.tcp_fmri_preprocess;

            if let Some(v) = fmri_dir {
                p.fmri_dir = v
            };
            if let Some(v) = filter_dir {
                p.filter_dir = v
            };
            if let Some(v) = output_dir {
                p.output_dir = v
            };

            if let Some(v) = cortical_atlas {
                p.cortical_atlas = v
            };
            if let Some(v) = subcortical_atlas {
                p.subcortical_atlas = v
            };

            if let Some(v) = dry_run {
                p.dry_run = v
            };
            if force {
                p.force = true
            };

            tcp_fmri_preprocess::run(&p)
        }
        Command::TcpFmriProcess {
            fmri_dir,
            output_dir,
            cortical_lut,
            subcortical_lut,
            subject_file,
            force,
        } => {
            let mut p = cfg.tcp_fmri_process;

            if let Some(v) = fmri_dir {
                p.fmri_dir = v
            };
            if let Some(v) = output_dir {
                p.output_dir = v
            };
            if let Some(v) = cortical_lut {
                p.cortical_atlas_lut = v
            };
            if let Some(v) = subcortical_lut {
                p.subcortical_atlas_lut = v
            };
            if let Some(v) = subject_file {
                p.subject_file = Some(v)
            };
            if force {
                p.force = true
            };

            tcp_fmri_process::run(&p)
        }
        Command::IstartSelectSubjects {
            istart_dir,
            istart_annex_remote,
            output_dir,
            dry_run,
        } => {
            let mut p = cfg.istart_subject_selection;

            if let Some(v) = istart_dir {
                p.istart_dir = v
            };
            if let Some(v) = output_dir {
                p.output_dir = v
            };
            if let Some(v) = istart_annex_remote {
                p.istart_annex_remote = v
            };

            if let Some(v) = dry_run {
                p.dry_run = v
            };

            istart_subject_selection::run(&p)
        }
    }
}
