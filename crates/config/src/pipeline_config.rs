mod cwt_config;
mod data_splitting;
mod fc_config;
mod feature_extraction;
mod hilbert_config;
mod legacy_processing_config;
mod mvmd_config;
mod parcellation_config;
mod segmentation_config;
mod subject_selection_config;

pub use {
    cwt_config::CwtConfig, data_splitting::DataSplitConfig, fc_config::FcConfig,
    feature_extraction::FeatureExtractionConfig, hilbert_config::HilbertHuangConfig,
    mvmd_config::MvmdConfig, parcellation_config::FmriParcellationConfig,
    segmentation_config::TrialSegmentationConfig,
    subject_selection_config::TcpSubjectSelectionConfig,
};

pub use legacy_processing_config::FmriProcessConfig;
