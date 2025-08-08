//! # NNL - Rust Neural Network Library
//!
//! A high-performance neural network library for Rust with comprehensive GPU and CPU support.
//!
//! ## Features
//!
//! - **Multi-backend Support**: NVIDIA CUDA, AMD ROCm/Vulkan, and optimized CPU execution
//! - **Automatic Hardware Detection**: Seamlessly selects the best available compute backend
//! - **Multiple Training Methods**: Backpropagation, Newton's method, and advanced optimizers
//! - **Flexible Architecture**: Support for both linear and convolutional networks
//! - **Model Persistence**: Import/export trained models to disk
//! - **Production Ready**: Zero-copy operations, SIMD optimizations, and batched processing
//!
//! ## Quick Start
//!
//! ```rust
//! use nnl::prelude::*;
//!
//! // Create a simple network
//! let mut network = NetworkBuilder::new()
//!     .add_layer(LayerConfig::Dense {
//!         input_size: 2,
//!         output_size: 4,
//!         activation: Activation::ReLU,
//!     })
//!     .add_layer(LayerConfig::Dense {
//!         input_size: 4,
//!         output_size: 1,
//!         activation: Activation::Sigmoid,
//!     })
//!     .loss(LossFunction::MeanSquaredError)
//!     .optimizer(OptimizerConfig::Adam { learning_rate: 0.001 })
//!     .build()?;
//!
//! // Train the network
//! let inputs = Tensor::from_slice(&[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], &[4, 2])?;
//! let targets = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4, 1])?;
//!
//! network.train(&inputs, &targets, 1000)?;
//!
//! // Make predictions
//! let prediction = network.forward(&Tensor::from_slice(&[1.0, 0.0], &[1, 2])?)?;
//! println!("Prediction: {}", prediction);
//! ```

#![warn(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]

// Core modules
pub mod activations;
pub mod device;
pub mod io;
pub mod layers;
pub mod losses;
pub mod network;
pub mod optimizers;
pub mod tensor;

// Utilities and error handling
pub mod error;
pub mod utils;

// Re-exports for convenience
pub use error::{NnlError, Result};

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::activations::Activation;
    pub use crate::device::{Backend, Device, DeviceType};
    pub use crate::error::{NnlError, Result};
    pub use crate::io::{
        DatasetInfo, ModelFormat, ModelMetadata, TrainingInfo, load_model, load_network,
        load_network_auto, save_model,
    };
    pub use crate::layers::{Layer, LayerConfig, WeightInit};
    pub use crate::losses::LossFunction;
    pub use crate::network::{
        LearningRateSchedule, Network, NetworkBuilder, TrainingConfig, TrainingHistory,
        TrainingMetrics,
    };
    pub use crate::optimizers::{Optimizer, OptimizerConfig};
    pub use crate::tensor::{Shape, Tensor, TensorView};
    pub use crate::utils;

    // External dependencies commonly used
    pub use anyhow::Result as AnyhowResult;
    pub use chrono;
    pub use env_logger;
    pub use std::collections::HashMap;
}

#[cfg(test)]
mod tests {
    use super::prelude::*;

    #[test]
    fn test_device_detection() {
        let device = Device::auto_select();
        assert!(device.is_ok());

        let device = device.unwrap();
        println!("Auto-selected device: {:?}", device.device_type());
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::zeros(&[2, 3]);
        assert!(tensor.is_ok());

        let tensor = tensor.unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_simple_network() -> Result<()> {
        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 1,
                activation: Activation::Sigmoid,
                use_bias: true,
                weight_init: crate::layers::WeightInit::Xavier,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::SGD {
                learning_rate: 0.1,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            })
            .build()?;

        let inputs = Tensor::from_slice(&[1.0, 0.0], &[1, 2])?;
        let output = network.forward(&inputs)?;

        assert_eq!(output.shape(), &[1, 1]);
        Ok(())
    }
}
