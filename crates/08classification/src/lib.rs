pub mod analyses;
pub mod calibration;
pub mod classifiers;
pub mod dataset;
pub mod eval;
pub mod metrics;
pub mod normalizer;
pub mod splits;

use crate::analyses::{
    baseline, baseline_averaging, baseline_chunked_feature_mean, baseline_img_resize,
    face_block_averaging, face_block_averaging_img_resize, face_block_concatenation,
    face_block_single, face_block_single_img_resize,
    baseline_subject_stratified, baseline_averaging_subject_stratified,
    baseline_chunked_feature_mean_subject_stratified, baseline_img_resize_subject_stratified,
    face_block_concatenation_subject_stratified, face_block_single_subject_stratified,
    face_block_single_img_resize_subject_stratified, face_block_averaging_subject_stratified,
    face_block_averaging_img_resize_subject_stratified,
};

use anyhow::Result;
use utils::config::AppConfig;

pub fn run(cfg: &AppConfig) -> Result<()> {
    baseline::run(cfg)?;
    baseline_subject_stratified::run(cfg)?;
    baseline_averaging::run(cfg)?;
    baseline_averaging_subject_stratified::run(cfg)?;
    baseline_chunked_feature_mean::run(cfg)?;
    baseline_chunked_feature_mean_subject_stratified::run(cfg)?;
    baseline_img_resize::run(cfg)?;
    baseline_img_resize_subject_stratified::run(cfg)?;
    face_block_concatenation::run(cfg)?;
    face_block_concatenation_subject_stratified::run(cfg)?;
    face_block_single::run(cfg)?;
    face_block_single_subject_stratified::run(cfg)?;
    face_block_single_img_resize::run(cfg)?;
    face_block_single_img_resize_subject_stratified::run(cfg)?;
    face_block_averaging::run(cfg)?;
    face_block_averaging_subject_stratified::run(cfg)?;
    face_block_averaging_img_resize::run(cfg)?;
    face_block_averaging_img_resize_subject_stratified::run(cfg)?;
    Ok(())
}
