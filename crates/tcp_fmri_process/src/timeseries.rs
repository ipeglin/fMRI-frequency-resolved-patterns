use ndarray::Array2;
use polars::prelude::*;

/// BOLD timeseries data for HDF5 output
pub struct TimeseriesData {
    /// Timeseries matrix: (T time-points x N_ROIs)
    pub data: DataFrame,
    // timeseries: Array2<f64>,
    /// ROI labels (column headers)
    pub labels: Vec<String>,
}

impl TimeseriesData {
    pub fn new(df: DataFrame) -> PolarsResult<Self> {
        let labels: Vec<String> = df
            .get_column_names()
            .iter()
            .map(|s| s.to_string())
            .collect();

        Ok(Self { data: df, labels })
    }

    pub fn to_ndarray(&self) -> PolarsResult<Array2<f64>> {
        let num_channels = self.data.width();
        let num_timepoints = self.data.height();

        // Convert DataFrame to Array2<f64> (T x N_ROIs)
        let mut timeseries_array = Array2::<f64>::zeros((num_timepoints, num_channels));
        for (col_idx, col_name) in self.labels.iter().enumerate() {
            if let Ok(col) = self.data.column(col_name.as_str()) {
                if let Ok(values) = col.f32() {
                    for (row_idx, val) in values.iter().enumerate() {
                        timeseries_array[[row_idx, col_idx]] = val.unwrap_or(f32::NAN) as f64;
                    }
                }
            }
        }

        Ok(timeseries_array)
    }

    pub fn get_channel_count(&self) -> usize {
        self.data.width()
    }

    pub fn get_timepoint_count(&self) -> usize {
        self.data.height()
    }
}
