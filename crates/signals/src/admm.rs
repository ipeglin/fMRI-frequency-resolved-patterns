/// Configuration for ADMM (Alternating Direction Method of Multipliers) dual ascent.
///
/// These parameters are general to any ADMM-based optimization and can be reused
/// across different algorithms (MVMD, VMD, sparse coding, etc.).
#[derive(Debug, Clone)]
pub struct ADMMConfig {
    /// Stopping criterion for dual ascent convergence.
    /// The algorithm terminates when the residual falls below this threshold.
    pub tolerance: f64,

    /// Time-step of the dual ascent (Lagrangian multiplier update step size).
    /// Use 0 for noise-slack (no dual update).
    pub tau: f64,

    /// Maximum number of iterations before forced termination.
    pub max_iterations: u32,
}

impl Default for ADMMConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-3,
            tau: 1e-2,
            max_iterations: 600,
        }
    }
}

impl ADMMConfig {
    pub fn new(tolerance: f64, tau: f64, max_iterations: u32) -> Self {
        Self {
            tolerance,
            tau,
            max_iterations,
        }
    }

    /// Builder-style method to set tolerance
    pub fn with_tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Builder-style method to set tau
    pub fn with_tau(mut self, tau: f64) -> Self {
        self.tau = tau;
        self
    }

    /// Builder-style method to set max iterations
    pub fn with_max_iterations(mut self, max_iterations: u32) -> Self {
        self.max_iterations = max_iterations;
        self
    }
}

/// Trait for algorithms that use ADMM-based optimization.
///
/// Implementors should store the ADMMConfig and use it during their
/// optimization procedure.
pub trait ADMMOptimizer {
    /// Returns a reference to the ADMM configuration
    fn admm_config(&self) -> &ADMMConfig;

    /// Returns a mutable reference to the ADMM configuration
    fn admm_config_mut(&mut self) -> &mut ADMMConfig;
}
