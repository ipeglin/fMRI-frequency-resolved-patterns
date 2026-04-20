use anyhow::Result;

pub struct KNN {
    pub config: KnnConfig,
}

pub struct KnnConfig {
    pub num_neighbors: usize,
}

impl Default for KnnConfig {
    fn default() -> Self {
        Self { num_neighbors: 3 }
    }
}

impl KNN {
    pub fn from_training_data() -> Result<Self> {
        Ok(Self {
            config: KnnConfig::default(),
        })
    }

    pub fn with_config(mut self, config: KnnConfig) -> Self {
        self.config = config;
        self
    }

    pub fn classify() -> Result<()> {
        Ok(())
    }
}
