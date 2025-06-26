//! # Neuro-Divergent: High-Performance Neural Forecasting Library
//!
//! Neuro-Divergent provides 100% compatibility with the NeuralForecast Python library
//! while delivering the performance and safety benefits of Rust. Built on the ruv-FANN
//! neural network foundation, it offers high-performance neural forecasting capabilities
//! with a user-friendly API.
//!
//! ## Features
//!
//! - **100% NeuralForecast API Compatibility**: Drop-in replacement for Python users
//! - **High Performance**: Rust performance with SIMD optimization
//! - **Memory Safety**: Zero-cost abstractions with compile-time guarantees
//! - **Async Support**: Asynchronous training and prediction
//! - **Multiple Model Support**: LSTM, NBEATS, DeepAR, Transformers, and more
//! - **Extensible Architecture**: Easy to add custom models and components
//!
//! ## Quick Start
//!
//! ```rust
//! use neuro_divergent::{NeuralForecast, models::LSTM, data::TimeSeriesDataFrame, Frequency};
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create LSTM model
//! let lstm = LSTM::builder()
//!     .hidden_size(128)
//!     .num_layers(2)
//!     .dropout(0.1)
//!     .horizon(12)
//!     .input_size(24)
//!     .build()?;
//!
//! // Create NeuralForecast instance
//! let mut nf = NeuralForecast::builder()
//!     .with_model(Box::new(lstm))
//!     .with_frequency(Frequency::Monthly)
//!     .build()?;
//!
//! // Load your time series data
//! let data = TimeSeriesDataFrame::from_csv("data.csv")?;
//!
//! // Fit the model
//! nf.fit(data.clone())?;
//!
//! // Generate forecasts
//! let forecasts = nf.predict()?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Architecture
//!
//! The library is structured in several layers:
//!
//! - **API Layer**: User-facing interface with NeuralForecast compatibility
//! - **Model Layer**: Neural network model implementations
//! - **Core Layer**: Base traits and abstractions
//! - **Data Layer**: Time series data handling and preprocessing
//! - **Foundation Layer**: Integration with ruv-FANN neural networks

#![cfg_attr(docsrs, feature(doc_cfg))]
#![warn(missing_docs)]
#![warn(rust_2018_idioms)]
#![allow(clippy::new_without_default)]

// Re-export essential types at the crate root
pub use neural_forecast::{NeuralForecast, NeuralForecastBuilder};
pub use config::{Frequency, ScalerType, PredictionIntervals};
pub use results::{ForecastDataFrame, CrossValidationDataFrame, TimeSeriesDataFrame};
pub use errors::{NeuroDivergentError, NeuroDivergentResult};

// Core modules
pub mod neural_forecast;
pub mod builders;
pub mod config;
pub mod results;
pub mod utils;
pub mod errors;

// Core abstractions
pub mod core {
    //! Core abstractions and traits for the forecasting system
    pub mod base_model;
    pub mod training;
    pub mod forecasting;
    pub mod registry;
    
    pub use base_model::{BaseModel, ModelConfig, ModelState, ModelMetadata};
    pub use training::{TrainingConfig, TrainingEngine};
    pub use forecasting::{ForecastingEngine, ForecastResult};
    pub use registry::{ModelRegistry, ModelFactory};
}

// Model implementations
pub mod models {
    //! Neural network model implementations
    pub mod lstm;
    pub mod nbeats;
    pub mod deepar;
    pub mod rnn;
    pub mod transformer;
    
    pub use lstm::{LSTM, LSTMConfig, LSTMBuilder};
    pub use nbeats::{NBEATS, NBEATSConfig, NBEATSBuilder};
    pub use deepar::{DeepAR, DeepARConfig, DeepARBuilder};
    pub use rnn::{RNN, RNNConfig, RNNBuilder};
    pub use transformer::{Transformer, TransformerConfig, TransformerBuilder};
}

// Data handling
pub mod data {
    //! Time series data handling and preprocessing
    pub mod dataset;
    pub mod preprocessing;
    pub mod scaling;
    pub mod validation;
    pub mod transforms;
    
    pub use dataset::{TimeSeriesDataset, TimeSeriesSchema};
    pub use preprocessing::{DataPreprocessor, PreprocessingConfig};
    pub use scaling::{Scaler, StandardScaler, MinMaxScaler, RobustScaler};
    pub use validation::{DataValidator, ValidationReport};
    pub use transforms::{DataTransform, WindowTransform, LagsTransform};
}

// Training infrastructure
pub mod training {
    //! Training algorithms and optimization
    pub mod optimizers;
    pub mod losses;
    pub mod schedulers;
    pub mod callbacks;
    pub mod metrics;
    
    pub use optimizers::{Optimizer, Adam, SGD, AdamW};
    pub use losses::{LossFunction, MSE, MAE, MAPE, SMAPE, Huber};
    pub use schedulers::{LearningRateScheduler, StepLR, ExponentialLR};
    pub use callbacks::{TrainingCallback, EarlyStopping, ModelCheckpoint};
    pub use metrics::{AccuracyMetrics, ForecastingMetrics};
}

// Integration with ruv-FANN
pub mod integration {
    //! Integration with ruv-FANN neural networks
    pub mod network_adapter;
    pub mod activation_bridge;
    pub mod training_bridge;
    
    pub use network_adapter::{NetworkAdapter, FannNetworkBridge};
    pub use activation_bridge::{ActivationMapper, ActivationBridge};
    pub use training_bridge::{TrainingBridge, FannTrainingAdapter};
}

// Prelude for common imports
pub mod prelude {
    //! Common imports for everyday use
    
    pub use crate::{
        NeuralForecast, NeuralForecastBuilder,
        ForecastDataFrame, CrossValidationDataFrame, TimeSeriesDataFrame,
        Frequency, ScalerType, PredictionIntervals,
        NeuroDivergentError, NeuroDivergentResult,
    };
    
    pub use crate::core::{BaseModel, ModelConfig};
    pub use crate::models::{LSTM, NBEATS, DeepAR, RNN, Transformer};
    pub use crate::data::{TimeSeriesDataset, TimeSeriesSchema};
    pub use crate::training::{TrainingConfig, AccuracyMetrics};
}

// Version information
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
pub const PACKAGE_NAME: &str = env!("CARGO_PKG_NAME");
pub const DESCRIPTION: &str = env!("CARGO_PKG_DESCRIPTION");

/// Library information
pub mod info {
    //! Library metadata and version information
    
    /// Get the library version
    pub fn version() -> &'static str {
        super::VERSION
    }
    
    /// Get the package name
    pub fn name() -> &'static str {
        super::PACKAGE_NAME
    }
    
    /// Get the library description
    pub fn description() -> &'static str {
        super::DESCRIPTION
    }
    
    /// Check if GPU support is available
    #[cfg(feature = "gpu")]
    pub fn has_gpu_support() -> bool {
        true
    }
    
    #[cfg(not(feature = "gpu"))]
    pub fn has_gpu_support() -> bool {
        false
    }
    
    /// Check if async support is available
    #[cfg(feature = "async")]
    pub fn has_async_support() -> bool {
        true
    }
    
    #[cfg(not(feature = "async"))]
    pub fn has_async_support() -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_version_info() {
        assert!(!info::version().is_empty());
        assert!(!info::name().is_empty());
        assert!(!info::description().is_empty());
    }
    
    #[test]
    fn test_feature_detection() {
        // These should not panic
        let _ = info::has_gpu_support();
        let _ = info::has_async_support();
    }
}