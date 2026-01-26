use anyhow::{Result, bail};
use nalgebra::Matrix4;
use ndarray::{Array2, Array3, Array4, ShapeBuilder};
use nifti::{IntoNdArray, NiftiHeader, NiftiObject, NiftiVolume, ReaderOptions};
use std::{collections::HashSet, path::PathBuf};
use tracing::{debug, trace};

pub struct LabelsMasker {
    /// The atlas with integer labels for each ROI
    labels_volume: Array3<f32>,
    /// Affine transformation matrix of the atlas (voxel to world)
    labels_affine: Matrix4<f64>,
    /// Unique labels in the atlas (excluding 0/background)
    unique_labels: Vec<i32>,
    /// Shape of the atlas volume
    atlas_shape: [usize; 3],
}

impl LabelsMasker {
    /// Load and prepare the atlas
    pub fn new(atlas_path: &PathBuf) -> Result<Self> {
        debug!(
            atlas_path = %atlas_path.display(),
            "loading atlas"
        );

        let atlas_obj = ReaderOptions::new().read_file(atlas_path)?;
        let header = atlas_obj.header();

        let affine_source = if header.sform_code > 0 {
            "sform"
        } else if header.qform_code > 0 {
            "qform"
        } else {
            "pixdim"
        };

        let labels_affine = Self::get_affine_from_header(header);
        let labels_volume = Self::volume_to_array3(atlas_obj.into_volume())?;
        let atlas_shape = [
            labels_volume.shape()[0],
            labels_volume.shape()[1],
            labels_volume.shape()[2],
        ];

        // Find unique labels (excluding 0 which is background)
        let mut unique_labels: Vec<i32> = labels_volume
            .iter()
            .map(|&v| v.round() as i32)
            .collect::<HashSet<_>>()
            .into_iter()
            .filter(|&l| l != 0)
            .collect();
        unique_labels.sort();

        debug!(
            atlas_path = %atlas_path.display(),
            atlas_shape_x = atlas_shape[0],
            atlas_shape_y = atlas_shape[1],
            atlas_shape_z = atlas_shape[2],
            n_labels = unique_labels.len(),
            affine_source = affine_source,
            label_min = unique_labels.first().copied().unwrap_or(0),
            label_max = unique_labels.last().copied().unwrap_or(0),
            "atlas loaded"
        );

        Ok(Self {
            labels_volume,
            labels_affine,
            unique_labels,
            atlas_shape,
        })
    }

    /// Get the number of unique labels (ROIs) in the atlas
    pub fn n_labels(&self) -> usize {
        self.unique_labels.len()
    }

    /// Extract time series for each labeled region from BOLD data
    /// Returns Array2 of shape (n_labels, n_timepoints)
    pub fn fit_transform(&self, bold_path: &PathBuf) -> Result<Array2<f32>> {
        debug!(
            bold_path = %bold_path.display(),
            n_atlas_labels = self.unique_labels.len(),
            "starting fit_transform"
        );

        let bold_obj = ReaderOptions::new().read_file(bold_path)?;
        let bold_header = bold_obj.header();
        let bold_affine = Self::get_affine_from_header(bold_header);
        let bold_data = Self::volume_to_array4(bold_obj.into_volume())?;

        let bold_shape = bold_data.shape();
        let n_timepoints = bold_shape[3];

        debug!(
            bold_shape_x = bold_shape[0],
            bold_shape_y = bold_shape[1],
            bold_shape_z = bold_shape[2],
            n_timepoints = n_timepoints,
            bold_path = %bold_path.display(),
            "BOLD data loaded"
        );

        // Resample atlas to BOLD space
        let needs_resampling =
            self.labels_volume.shape() != &[bold_shape[0], bold_shape[1], bold_shape[2]];

        debug!(
            needs_resampling = needs_resampling,
            atlas_shape = ?self.atlas_shape,
            bold_shape = ?[bold_shape[0], bold_shape[1], bold_shape[2]],
            "checking resampling requirement"
        );

        let resampled_labels = self.resample_to_target(&bold_data, &bold_affine)?;

        // Extract time series for each label
        let n_labels = self.unique_labels.len();
        let mut result = Array2::<f32>::zeros((n_labels, n_timepoints));
        let mut empty_rois = 0usize;

        for (label_idx, &label) in self.unique_labels.iter().enumerate() {
            let mask: Vec<(usize, usize, usize)> = resampled_labels
                .indexed_iter()
                .filter_map(|((x, y, z), &val)| {
                    if val.round() as i32 == label {
                        Some((x, y, z))
                    } else {
                        None
                    }
                })
                .collect();

            if mask.is_empty() {
                empty_rois += 1;
                trace!(
                    label = label,
                    label_idx = label_idx,
                    "ROI has no voxels after resampling"
                );
                continue;
            }

            trace!(
                label = label,
                label_idx = label_idx,
                n_voxels = mask.len(),
                "extracting timeseries for ROI"
            );

            // Compute mean time series across all voxels in this ROI
            for t in 0..n_timepoints {
                let sum: f32 = mask.iter().map(|&(x, y, z)| bold_data[[x, y, z, t]]).sum();
                result[[label_idx, t]] = sum / mask.len() as f32;
            }
        }

        debug!(
            n_labels = n_labels,
            n_timepoints = n_timepoints,
            empty_rois = empty_rois,
            output_shape = ?[n_labels, n_timepoints],
            bold_path = %bold_path.display(),
            "fit_transform completed"
        );

        Ok(result)
    }

    /// Resample labels to match target BOLD data dimensions using nearest-neighbor interpolation
    fn resample_to_target(
        &self,
        bold_data: &Array4<f32>,
        bold_affine: &Matrix4<f64>,
    ) -> Result<Array3<f32>> {
        let target_shape = (
            bold_data.shape()[0],
            bold_data.shape()[1],
            bold_data.shape()[2],
        );

        // If dimensions already match, return as is
        if self.labels_volume.shape() == &[target_shape.0, target_shape.1, target_shape.2] {
            trace!(
                target_shape = ?target_shape,
                "dimensions match, skipping resampling"
            );
            return Ok(self.labels_volume.clone());
        }

        debug!(
            source_shape = ?self.atlas_shape,
            target_shape = ?target_shape,
            "resampling atlas to BOLD space"
        );

        let labels_affine_inv = self
            .labels_affine
            .try_inverse()
            .ok_or_else(|| anyhow::anyhow!("Failed to invert atlas affine matrix"))?;
        let transform = labels_affine_inv * bold_affine;

        let mut resampled = Array3::<f32>::zeros(target_shape);
        let src_shape = self.labels_volume.shape();

        let mut in_bounds_count = 0usize;
        let mut out_of_bounds_count = 0usize;

        for x in 0..target_shape.0 {
            for y in 0..target_shape.1 {
                for z in 0..target_shape.2 {
                    let bold_voxel = nalgebra::Vector4::new(x as f64, y as f64, z as f64, 1.0);
                    let atlas_voxel = transform * bold_voxel;

                    let sx = atlas_voxel[0].round() as i64;
                    let sy = atlas_voxel[1].round() as i64;
                    let sz = atlas_voxel[2].round() as i64;

                    if sx >= 0
                        && sx < src_shape[0] as i64
                        && sy >= 0
                        && sy < src_shape[1] as i64
                        && sz >= 0
                        && sz < src_shape[2] as i64
                    {
                        resampled[[x, y, z]] =
                            self.labels_volume[[sx as usize, sy as usize, sz as usize]];
                        in_bounds_count += 1;
                    } else {
                        out_of_bounds_count += 1;
                    }
                }
            }
        }

        let total_voxels = target_shape.0 * target_shape.1 * target_shape.2;
        let coverage_pct = (in_bounds_count as f64 / total_voxels as f64) * 100.0;

        debug!(
            total_voxels = total_voxels,
            in_bounds = in_bounds_count,
            out_of_bounds = out_of_bounds_count,
            coverage_pct = format!("{:.1}", coverage_pct),
            "resampling completed"
        );

        Ok(resampled)
    }

    /// Extract affine matrix from NIfTI header (sform or qform)
    fn get_affine_from_header(header: &NiftiHeader) -> Matrix4<f64> {
        if header.sform_code > 0 {
            trace!(sform_code = header.sform_code, "using sform affine");
            Matrix4::new(
                header.srow_x[0] as f64,
                header.srow_x[1] as f64,
                header.srow_x[2] as f64,
                header.srow_x[3] as f64,
                header.srow_y[0] as f64,
                header.srow_y[1] as f64,
                header.srow_y[2] as f64,
                header.srow_y[3] as f64,
                header.srow_z[0] as f64,
                header.srow_z[1] as f64,
                header.srow_z[2] as f64,
                header.srow_z[3] as f64,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        } else if header.qform_code > 0 {
            trace!(qform_code = header.qform_code, "using qform affine");
            Self::qform_to_affine(header)
        } else {
            trace!(
                pixdim = ?[header.pixdim[1], header.pixdim[2], header.pixdim[3]],
                "using pixdim fallback for affine"
            );
            Matrix4::new(
                header.pixdim[1] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                header.pixdim[2] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                header.pixdim[3] as f64,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
            )
        }
    }

    /// Convert qform quaternion parameters to affine matrix
    fn qform_to_affine(header: &NiftiHeader) -> Matrix4<f64> {
        let b = header.quatern_b as f64;
        let c = header.quatern_c as f64;
        let d = header.quatern_d as f64;
        let a = (1.0 - b * b - c * c - d * d).max(0.0).sqrt();

        let r11 = a * a + b * b - c * c - d * d;
        let r12 = 2.0 * (b * c - a * d);
        let r13 = 2.0 * (b * d + a * c);
        let r21 = 2.0 * (b * c + a * d);
        let r22 = a * a + c * c - b * b - d * d;
        let r23 = 2.0 * (c * d - a * b);
        let r31 = 2.0 * (b * d - a * c);
        let r32 = 2.0 * (c * d + a * b);
        let r33 = a * a + d * d - b * b - c * c;

        let pi = header.pixdim[1] as f64;
        let pj = header.pixdim[2] as f64;
        let pk = header.pixdim[3] as f64;

        let qfac = if header.pixdim[0] < 0.0 { -1.0 } else { 1.0 };

        let qx = header.quatern_x as f64;
        let qy = header.quatern_y as f64;
        let qz = header.quatern_z as f64;

        trace!(
            quatern_b = b,
            quatern_c = c,
            quatern_d = d,
            quatern_a = a,
            qfac = qfac,
            offset = ?[qx, qy, qz],
            "qform parameters"
        );

        Matrix4::new(
            r11 * pi,
            r12 * pj,
            r13 * pk * qfac,
            qx,
            r21 * pi,
            r22 * pj,
            r23 * pk * qfac,
            qy,
            r31 * pi,
            r32 * pj,
            r33 * pk * qfac,
            qz,
            0.0,
            0.0,
            0.0,
            1.0,
        )
    }

    /// Helper: Convert NiftiVolume to Array3
    ///
    /// Note: NIfTI data is stored in Fortran (column-major) order, so we must
    /// preserve this memory layout when converting to ndarray.
    fn volume_to_array3<V>(volume: V) -> Result<Array3<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        trace!(
            ndim = ndim,
            shape = ?shape,
            "converting volume to Array3"
        );

        match ndim {
            3 => {
                // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
                let (data, _offset) = array.into_raw_vec_and_offset();
                Ok(Array3::from_shape_vec(
                    (shape[0], shape[1], shape[2]).f(),
                    data,
                )?)
            }
            4 => {
                if shape[3] == 1 {
                    // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
                    let (data, _offset) = array.into_raw_vec_and_offset();
                    Ok(Array3::from_shape_vec(
                        (shape[0], shape[1], shape[2]).f(),
                        data,
                    )?)
                } else {
                    bail!("Expected 3D volume for atlas, got 4D with multiple timepoints")
                }
            }
            _ => bail!("Unexpected number of dimensions: {}", ndim),
        }
    }

    /// Helper: Convert NiftiVolume to Array4 (for 4D BOLD data)
    ///
    /// Note: NIfTI data is stored in Fortran (column-major) order, so we must
    /// preserve this memory layout when converting to ndarray.
    fn volume_to_array4<V>(volume: V) -> Result<Array4<f32>>
    where
        V: NiftiVolume + IntoNdArray,
    {
        let array = volume.into_ndarray::<f32>()?;
        let ndim = array.ndim();
        let shape: Vec<usize> = array.shape().to_vec();

        trace!(
            ndim = ndim,
            shape = ?shape,
            "converting volume to Array4"
        );

        if ndim == 4 {
            // nifti-rs returns data in Fortran order, so we must use .f() to preserve it
            let (data, _offset) = array.into_raw_vec_and_offset();
            Ok(Array4::from_shape_vec(
                (shape[0], shape[1], shape[2], shape[3]).f(),
                data,
            )?)
        } else {
            bail!("Expected 4D BOLD volume, got {}D", ndim)
        }
    }
}
