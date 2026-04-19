use anyhow::Result;
use clap::{Parser, Subcommand, ValueEnum};
use config::{
    AppConfig, CwtConfig, DataSplitConfig, FcConfig, FmriParcellationConfig, HilbertHuangConfig,
    MvmdConfig, TcpSubjectSelectionConfig, TrialSegmentationConfig, load_config,
    pipeline_config::FeatureExtractionConfig,
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
    SelectSubjects {
        #[arg(long)]
        tcp_dir: Option<PathBuf>,

        #[arg(long)]
        tcp_annex_remote: Option<String>,

        #[arg(long)]
        output_dir: Option<PathBuf>,

        #[arg(long)]
        dry_run: Option<bool>,
    },
    ParcellateBold {
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

        /// Force reprocessing of subjects that already have preprocessed output
        #[arg(long, short = 'f')]
        force: bool,

        /// Apply voxel-wise z-score normalization before parcellation.
        /// Produces additional HDF5 datasets (tcp_cortical_voxelzscore, etc.).
        /// Does not imply --force; existing voxelzscore datasets are skipped.
        #[arg(long)]
        voxelwise_zscore: bool,
    },
    SegmentTrials {
        #[arg(long)]
        tcp_dir: Option<PathBuf>,

        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

        #[arg(long)]
        glm_output_dir: Option<PathBuf>,

        /// Force reprocessing of blocks that already exist in output files
        #[arg(long, short = 'f')]
        force: bool,
    },
    DecomposeMvmd {
        #[arg(long)]
        tcp_dir: Option<PathBuf>,

        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

        #[arg(long)]
        num_modes: Option<u8>,

        /// Force reprocessing of subjects that already exist in output files
        #[arg(long, short = 'f')]
        force: bool,
    },
    Cwt {
        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Hht {
        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    Fc {
        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    },
    #[cfg(feature = "feature-extraction")]
    FeatureExtraction {
        #[arg(long)]
        bold_ts_dir: Option<PathBuf>,

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
        bold_ts_dir: Option<PathBuf>,

        #[arg(long)]
        training_subjects: Option<PathBuf>,

        #[arg(long)]
        test_subjects: Option<PathBuf>,

        #[arg(long)]
        validation_subjects: Option<PathBuf>,

        #[arg(long, short = 'f')]
        force: bool,
    }, // TcpFmriProcess {
       //     #[arg(long)]
       //     bold_ts_dir: Option<PathBuf>,

       //     #[arg(long)]
       //     output_dir: Option<PathBuf>,

       //     #[arg(long)]
       //     cortical_lut: Option<PathBuf>,

       //     #[arg(long)]
       //     subcortical_lut: Option<PathBuf>,

       //     #[arg(long)]
       //     subject_file: Option<PathBuf>,

       //     /// Force reprocessing of subjects that already exist in output files
       //     #[arg(long, short = 'f')]
       //     force: bool,
       // },
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
            tcp_subject_selection: TcpSubjectSelectionConfig::default(),
            fmri_parcellation: FmriParcellationConfig::default(),
            fmri_segment_trials: TrialSegmentationConfig::default(),
            mvmd: MvmdConfig::default(),
            cwt: CwtConfig::default(),
            hilbert: HilbertHuangConfig::default(),
            fc: FcConfig::default(),
            feature_extraction: FeatureExtractionConfig::default(),
            data_splitting: DataSplitConfig::default(),
            // fmri_process: FmriProcessConfig::default(),
        }
    });

    info!(
        config_path = %cli.config.display(),
        log_level = %cli.log_level,
        version = env!("CARGO_PKG_VERSION"),
        "starting pipeline"
    );

    match cli.cmd {
        Command::SelectSubjects {
            tcp_dir,
            tcp_annex_remote,
            output_dir,
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

            if let Some(v) = dry_run {
                p.dry_run = v
            };

            tcp_subject_selection::run(&p)
        }
        Command::ParcellateBold {
            fmri_dir,
            filter_dir,
            output_dir,
            cortical_atlas,
            subcortical_atlas,
            force,
            voxelwise_zscore,
        } => {
            let mut p = cfg.fmri_parcellation;

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

            if force {
                p.force = true
            };
            if voxelwise_zscore {
                p.voxelwise_zscore = true
            };

            fmri_parcellation::run(&p)
        }
        Command::SegmentTrials {
            tcp_dir,
            bold_ts_dir,
            glm_output_dir,
            force,
        } => {
            let mut p = cfg.fmri_segment_trials;

            if let Some(v) = tcp_dir {
                p.tcp_dir = v
            }

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if let Some(v) = glm_output_dir {
                p.glm_output_dir = v
            }

            if force {
                p.force = true
            }

            fmri_segment_trials::run(&p)
        }
        Command::DecomposeMvmd {
            tcp_dir,
            bold_ts_dir,
            num_modes,
            force,
        } => {
            let mut p = cfg.mvmd;

            if let Some(v) = tcp_dir {
                p.tcp_dir = v
            }

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if let Some(v) = num_modes {
                p.num_modes = v as usize
            }

            if force {
                p.force = true
            };

            mvmd::run(&p)
        }
        Command::Cwt { bold_ts_dir, force } => {
            let mut p = cfg.cwt;

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if force {
                p.force = true
            }

            cwt::run(&p)
        }
        Command::Hht { bold_ts_dir, force } => {
            let mut p = cfg.hilbert;

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if force {
                p.force = true
            }

            hilbert::run(&p)
        }
        Command::Fc { bold_ts_dir, force } => {
            let mut p = cfg.fc;

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if force {
                p.force = true
            }

            fc::run(&p)
        }
        #[cfg(feature = "feature-extraction")]
        Command::FeatureExtraction {
            bold_ts_dir,
            cortical_lut,
            subcortical_lut,
            cnn_weights,
            force,
        } => {
            let mut p = cfg.feature_extraction;

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if let Some(v) = cortical_lut {
                p.cortical_atlas_lut = v
            }

            if let Some(v) = subcortical_lut {
                p.subcortical_atlas_lut = v
            }

            if let Some(v) = cnn_weights {
                p.cnn_weights_path = Some(v)
            }

            if force {
                p.force = true
            }

            feature_extraction::run(&p)
        }
        Command::SplitData {
            subject_filter_dir,
            bold_ts_dir,
            training_subjects,
            test_subjects,
            validation_subjects,
            force,
        } => {
            let mut p = cfg.data_splitting;

            if let Some(v) = subject_filter_dir {
                p.subject_filter_dir = v
            }

            if let Some(v) = bold_ts_dir {
                p.bold_ts_dir = v
            }

            if let Some(v) = training_subjects {
                p.training_subjects_path = v
            }

            if let Some(v) = test_subjects {
                p.test_subjects_path = v
            }

            if let Some(v) = validation_subjects {
                p.validation_subjects_path = v
            }

            if force {
                p.force = true
            }

            data_splitting::run(&p)
        } // Command::TcpFmriProcess {
          //     bold_ts_dir,
          //     output_dir,
          //     cortical_lut,
          //     subcortical_lut,
          //     subject_file,
          //     force,
          // } => {
          //     let mut p = cfg.fmri_process;

          //     if let Some(v) = bold_ts_dir {
          //         p.bold_ts_dir = v
          //     };
          //     if let Some(v) = output_dir {
          //         p.output_dir = v
          //     };
          //     if let Some(v) = cortical_lut {
          //         p.cortical_atlas_lut = v
          //     };
          //     if let Some(v) = subcortical_lut {
          //         p.subcortical_atlas_lut = v
          //     };
          //     if let Some(v) = subject_file {
          //         p.subject_file = Some(v)
          //     };
          //     if force {
          //         p.force = true
          //     };

          //     fmri_process::run(&p)
          // }
    }
}
