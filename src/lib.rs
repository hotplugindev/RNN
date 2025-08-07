//! # RNN - Rust Neural Network Library
//!
//! A high-performance neural network library for Rust with support for both CPU and GPU computation.
//!
//! ## Features
//!
//! - Multiple activation functions (ReLU, Sigmoid, Tanh, etc.)
//! - Various training algorithms (Backpropagation, Newton's method, Adam, etc.)
//! - CPU and GPU support
//! - Network serialization and deserialization
//! - Flexible architecture for different network types
//! - Optimized for performance with BLAS/LAPACK support
//!
//! ## Example
//!
//! ```rust
//! use rnn::{Network, LayerBuilder, ActivationFunction, TrainingMethod};
//!
//! // Create a simple feedforward network
//! let mut network = Network::builder()
//!     .input_size(784)
//!     .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
//!     .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
//!     .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
//!     .build()?;
//!
//! // Train the network
//! network.train(&training_data, &labels, TrainingMethod::Backpropagation)?;
//!
//! // Make predictions
//! let predictions = network.predict(&test_data)?;
//! ```

pub mod activation;
pub mod error;
pub mod layer;
pub mod loss;
pub mod network;
pub mod optimizer;
pub mod training;
pub mod utils;

pub mod gpu;

pub mod io;

// Re-export main types for convenience
pub use activation::ActivationFunction;
pub use error::{NetworkError, Result};
pub use layer::{Layer, LayerBuilder, LayerType};
pub use loss::LossFunction;
pub use network::{Network, NetworkBuilder};
pub use optimizer::{Optimizer, OptimizerType};
pub use training::{TrainingConfig, TrainingMethod};

// Re-export GPU types for convenience
pub use gpu::{GpuDataType, GpuDevice, GpuDeviceType, GpuManager, GpuTensor};

// Re-export ndarray for users
pub use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Common type aliases for convenience
pub type Matrix = ndarray::Array2<f64>;
pub type Vector = ndarray::Array1<f64>;
pub type MatrixView<'a> = ndarray::ArrayView2<'a, f64>;
pub type VectorView<'a> = ndarray::ArrayView1<'a, f64>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_network_creation() -> Result<()> {
        let network = Network::with_input_size(2)?
            .add_layer(LayerBuilder::dense(3).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .build()?;

        assert_eq!(network.input_dim, 2);
        assert_eq!(network.output_dim, 1);
        Ok(())
    }
}
