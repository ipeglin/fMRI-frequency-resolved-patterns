use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use utils::config::{AppConfig, load_config};
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
    SelectSubjects {
        #[arg(long)]
        tcp_repo_dir: Option<PathBuf>,

        #[arg(long)]
        tcp_annex_remote: Option<String>,

        #[arg(long)]
        subject_filter_dir: Option<PathBuf>,

        #[arg(long)]
        dry_run: Option<bool>,
    },
    ParcellateBold {
        #[arg(long)]
        fmriprep_output_dir: Option<PathBuf>,

        #[arg(long)]
        subject_filter_dir: Option<PathBuf>,

        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        cortical_atlas: Option<PathBuf>,

        #[arg(long)]
        subcortical_atlas: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,

        /// Apply voxel-wise z-score normalization before parcellation.
        #[arg(long)]
        voxelwise_zscore: bool,
    },
    SegmentTrials {
        #[arg(long)]
        tcp_repo_dir: Option<PathBuf>,

        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        task_regressors_output_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    DecomposeMvmd {
        #[arg(long)]
        tcp_repo_dir: Option<PathBuf>,

        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        num_modes: Option<u8>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Cwt {
        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Hht {
        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Fc {
        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    #[cfg(feature = "feature-extraction")]
    FeatureExtraction {
        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        cortical_lut: Option<PathBuf>,

        #[arg(long)]
        subcortical_lut: Option<PathBuf>,

        #[arg(long)]
        cnn_weights: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    SplitData {
        #[arg(long)]
        subject_filter_dir: Option<PathBuf>,

        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        training_subjects: Option<PathBuf>,

        #[arg(long)]
        test_subjects: Option<PathBuf>,

        #[arg(long)]
        validation_subjects: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Classify {
        #[arg(long)]
        parcellated_ts_dir: Option<PathBuf>,

        #[arg(long)]
        training_subjects: Option<PathBuf>,

        #[arg(long)]
        test_subjects: Option<PathBuf>,

        #[arg(long)]
        validation_subjects: Option<PathBuf>,
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

    let mut cfg = load_config(&cli.config).unwrap_or_else(|e| {
        eprintln!(
            "Warning: Failed to load config from {}: {}",
            cli.config.display(),
            e
        );
        eprintln!("Using default configuration values");
        AppConfig::default()
    });

    info!(
        config_path = %cli.config.display(),
        log_level = %cli.log_level,
        version = env!("CARGO_PKG_VERSION"),
        "starting pipeline"
    );

    match cli.cmd {
        Command::SelectSubjects {
            tcp_repo_dir,
            tcp_annex_remote,
            subject_filter_dir,
            dry_run,
        } => {
            if let Some(v) = tcp_repo_dir {
                cfg.tcp_repo_dir = v;
            }
            if let Some(v) = subject_filter_dir {
                cfg.subject_filter_dir = v;
            }
            if let Some(v) = tcp_annex_remote {
                cfg.tcp_annex_remote = v;
            }
            if let Some(v) = dry_run {
                cfg.dry_run = v;
            }

            tcp_subject_selection::run(&cfg)
        }
        Command::ParcellateBold {
            fmriprep_output_dir,
            subject_filter_dir,
            parcellated_ts_dir,
            cortical_atlas,
            subcortical_atlas,
            force,
            voxelwise_zscore,
        } => {
            if let Some(v) = fmriprep_output_dir {
                cfg.fmriprep_output_dir = v;
            }
            if let Some(v) = subject_filter_dir {
                cfg.subject_filter_dir = v;
            }
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = cortical_atlas {
                cfg.cortical_atlas = v;
            }
            if let Some(v) = subcortical_atlas {
                cfg.subcortical_atlas = v;
            }
            if force {
                cfg.force = true;
            }
            if voxelwise_zscore {
                cfg.parcellation.voxelwise_zscore = true;
            }

            fmri_parcellation::run(&cfg)
        }
        Command::SegmentTrials {
            tcp_repo_dir,
            parcellated_ts_dir,
            task_regressors_output_dir,
            force,
        } => {
            if let Some(v) = tcp_repo_dir {
                cfg.tcp_repo_dir = v;
            }
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = task_regressors_output_dir {
                cfg.task_regressors_output_dir = v;
            }
            if force {
                cfg.force = true;
            }

            fmri_segment_trials::run(&cfg)
        }
        Command::DecomposeMvmd {
            tcp_repo_dir,
            parcellated_ts_dir,
            num_modes,
            force,
        } => {
            if let Some(v) = tcp_repo_dir {
                cfg.tcp_repo_dir = v;
            }
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = num_modes {
                cfg.mvmd.num_modes = v as usize;
            }
            if force {
                cfg.force = true;
            }

            mvmd::run(&cfg)
        }
        Command::Cwt { parcellated_ts_dir, force } => {
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if force {
                cfg.force = true;
            }

            cwt::run(&cfg)
        }
        Command::Hht { parcellated_ts_dir, force } => {
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if force {
                cfg.force = true;
            }

            hilbert::run(&cfg)
        }
        Command::Fc { parcellated_ts_dir, force } => {
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if force {
                cfg.force = true;
            }

            fc::run(&cfg)
        }
        #[cfg(feature = "feature-extraction")]
        Command::FeatureExtraction {
            parcellated_ts_dir,
            cortical_lut,
            subcortical_lut,
            cnn_weights,
            force,
        } => {
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = cortical_lut {
                cfg.cortical_atlas_lut = v;
            }
            if let Some(v) = subcortical_lut {
                cfg.subcortical_atlas_lut = v;
            }
            if let Some(v) = cnn_weights {
                cfg.feature_extraction.cnn_weights_path = Some(v);
            }
            if force {
                cfg.force = true;
            }

            feature_extraction::run(&cfg)
        }
        Command::SplitData {
            subject_filter_dir,
            parcellated_ts_dir,
            training_subjects,
            test_subjects,
            validation_subjects,
            force,
        } => {
            if let Some(v) = subject_filter_dir {
                cfg.subject_filter_dir = v;
            }
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = training_subjects {
                cfg.training_subjects_path = v;
            }
            if let Some(v) = test_subjects {
                cfg.test_subjects_path = v;
            }
            if let Some(v) = validation_subjects {
                cfg.validation_subjects_path = v;
            }
            if force {
                cfg.force = true;
            }

            data_splitting::run(&cfg)
        }
        Command::Classify {
            parcellated_ts_dir,
            training_subjects,
            test_subjects,
            validation_subjects,
        } => {
            if let Some(v) = parcellated_ts_dir {
                cfg.parcellated_ts_dir = v;
            }
            if let Some(v) = training_subjects {
                cfg.training_subjects_path = v;
            }
            if let Some(v) = test_subjects {
                cfg.test_subjects_path = v;
            }
            if let Some(v) = validation_subjects {
                cfg.validation_subjects_path = v;
            }

            classification::run(&cfg)
        }
    }
}
