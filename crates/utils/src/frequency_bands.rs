/// fMRI slow-band (Buzsáki) frequency ranges in Hz.
/// Intervals are [low, high) — a mode/frequency `f` falls in a band iff low <= f < high.
///
/// Used as the project-wide reference for the analysed BOLD frequency range.
/// CWT scale grids, MVMD initialisation/grid bounds, and HHT spectrum binning
/// all derive `f_min` / `f_max` from this table so every spectral representation
/// shares one consistent frequency window.
pub const SLOW_BANDS: &[(&str, f64, f64)] = &[
    ("slow_5_trunc", 0.005, 0.010),
    ("slow_5", 0.010, 0.027),
    ("slow_4", 0.027, 0.073),
    ("slow_3", 0.073, 0.198),
    ("slow_2_trunc", 0.198, 0.250),
    // ("slow_2", 0.198, 0.500),
];

/// Lowest frequency covered by `SLOW_BANDS` (inclusive lower bound of the lowest band).
pub fn f_min() -> f64 {
    SLOW_BANDS
        .iter()
        .map(|(_, lo, _)| *lo)
        .fold(f64::INFINITY, f64::min)
}

/// Highest frequency covered by `SLOW_BANDS` (exclusive upper bound of the highest band).
pub fn f_max() -> f64 {
    SLOW_BANDS
        .iter()
        .map(|(_, _, hi)| *hi)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Minimum number of Δt samples required to define an instantaneous frequency stably
/// via differentiation (Huang, 1998: "absolute minimum … is five for a whole sine wave").
///
/// Also satisfies: `1 / (HILBERT_MIN_SAMPLES_PER_OSCILLATION * TR)` = `f_max()` at TR = 0.8 s,
/// so the SLOW_BANDS ceiling equals the Huang-theoretic maximum extractable frequency.
pub const HILBERT_MIN_SAMPLES_PER_OSCILLATION: usize = 5;

/// Optimal number of Hilbert-spectrum frequency cells for a segment of `n_timepoints` samples
/// (Huang, 1998: `N = T / (n·Δt) = n_timepoints / n`).
///
/// Returns at least 1.
pub fn hilbert_native_cells(n_timepoints: usize) -> usize {
    ((n_timepoints as f64 / HILBERT_MIN_SAMPLES_PER_OSCILLATION as f64).round() as usize).max(1)
}

/// Lowest frequency resolvable from a segment of `n_timepoints` samples at `sampling_rate` Hz
/// (`1/T = sampling_rate / n_timepoints`).
pub fn hilbert_lowest_resolvable_hz(n_timepoints: usize, sampling_rate: f64) -> f64 {
    sampling_rate / n_timepoints as f64
}
