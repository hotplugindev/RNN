//! Error types and handling for the RNN library.

use std::fmt;
use thiserror::Error;

/// The main error type for the RNN library.
#[derive(Error, Debug)]
pub enum NetworkError {
    /// Errors related to network architecture and configuration
    #[error("Network architecture error: {message}")]
    Architecture { message: String },

    /// Errors during training process
    #[error("Training error: {message}")]
    Training { message: String },

    /// Errors during forward/backward propagation
    #[error("Propagation error: {message}")]
    Propagation { message: String },

    /// Errors related to input/output operations
    #[error("IO error: {source}")]
    Io {
        #[from]
        source: std::io::Error,
    },

    /// Errors during serialization/deserialization
    #[error("Serialization error: {source}")]
    Serialization {
        #[from]
        source: serde_json::Error,
    },

    /// Binary serialization errors
    #[error("Binary serialization error: {source}")]
    BinarySerialization {
        #[from]
        source: bincode::Error,
    },

    /// Dimension mismatch errors
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Invalid parameter errors
    #[error("Invalid parameter: {parameter} = {value}, reason: {reason}")]
    InvalidParameter {
        parameter: String,
        value: String,
        reason: String,
    },

    /// Numerical computation errors
    #[error("Numerical error: {message}")]
    Numerical { message: String },

    /// GPU-related errors
    #[error("GPU error: {message}")]
    Gpu { message: String },

    /// CUDA-specific errors
    #[error("CUDA error: {message}")]
    Cuda { message: String },

    /// Linear algebra errors
    #[cfg(feature = "blas")]
    #[error("Linear algebra error: {source}")]
    LinearAlgebra {
        #[from]
        source: ndarray_linalg::error::LinalgError,
    },

    /// Shape errors from ndarray
    #[error("Shape error: {source}")]
    Shape {
        #[from]
        source: ndarray::ShapeError,
    },

    /// Convergence errors during training
    #[error("Convergence error: {message} after {iterations} iterations")]
    Convergence { message: String, iterations: usize },

    /// Data loading and preprocessing errors
    #[error("Data error: {message}")]
    Data { message: String },

    /// Optimizer-specific errors
    #[error("Optimizer error: {message}")]
    Optimizer { message: String },

    /// Loss function errors
    #[error("Loss function error: {message}")]
    LossFunction { message: String },

    /// Activation function errors
    #[error("Activation function error: {message}")]
    ActivationFunction { message: String },

    /// Generic computation errors
    #[error("Computation error: {message}")]
    Computation { message: String },

    /// Memory allocation errors
    #[error("Memory allocation error: {message}")]
    Memory { message: String },

    /// Thread safety and concurrency errors
    #[error("Concurrency error: {message}")]
    Concurrency { message: String },

    /// Configuration errors
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Validation errors
    #[error("Validation error: {message}")]
    Validation { message: String },
}

/// Result type alias for the RNN library.
pub type Result<T> = std::result::Result<T, NetworkError>;

impl NetworkError {
    /// Create a new architecture error
    pub fn architecture<S: Into<String>>(message: S) -> Self {
        Self::Architecture {
            message: message.into(),
        }
    }

    /// Create a new training error
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self::Training {
            message: message.into(),
        }
    }

    /// Create a new propagation error
    pub fn propagation<S: Into<String>>(message: S) -> Self {
        Self::Propagation {
            message: message.into(),
        }
    }

    /// Create a new dimension mismatch error
    pub fn dimension_mismatch<S: Into<String>>(expected: S, actual: S) -> Self {
        Self::DimensionMismatch {
            expected: expected.into(),
            actual: actual.into(),
        }
    }

    /// Create a new invalid parameter error
    pub fn invalid_parameter<S: Into<String>>(parameter: S, value: S, reason: S) -> Self {
        Self::InvalidParameter {
            parameter: parameter.into(),
            value: value.into(),
            reason: reason.into(),
        }
    }

    /// Create a new numerical error
    pub fn numerical<S: Into<String>>(message: S) -> Self {
        Self::Numerical {
            message: message.into(),
        }
    }

    /// Create a new GPU error
    pub fn gpu<S: Into<String>>(message: S) -> Self {
        Self::Gpu {
            message: message.into(),
        }
    }

    /// Create a new CUDA error
    pub fn cuda<S: Into<String>>(message: S) -> Self {
        Self::Cuda {
            message: message.into(),
        }
    }

    /// Create a new convergence error
    pub fn convergence<S: Into<String>>(message: S, iterations: usize) -> Self {
        Self::Convergence {
            message: message.into(),
            iterations,
        }
    }

    /// Create a new data error
    pub fn data<S: Into<String>>(message: S) -> Self {
        Self::Data {
            message: message.into(),
        }
    }

    /// Create a new optimizer error
    pub fn optimizer<S: Into<String>>(message: S) -> Self {
        Self::Optimizer {
            message: message.into(),
        }
    }

    /// Create a new loss function error
    pub fn loss_function<S: Into<String>>(message: S) -> Self {
        Self::LossFunction {
            message: message.into(),
        }
    }

    /// Create a new activation function error
    pub fn activation_function<S: Into<String>>(message: S) -> Self {
        Self::ActivationFunction {
            message: message.into(),
        }
    }

    /// Create a new computation error
    pub fn computation<S: Into<String>>(message: S) -> Self {
        Self::Computation {
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::Memory {
            message: message.into(),
        }
    }

    /// Create a new concurrency error
    pub fn concurrency<S: Into<String>>(message: S) -> Self {
        Self::Concurrency {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn configuration<S: Into<String>>(message: S) -> Self {
        Self::Configuration {
            message: message.into(),
        }
    }

    /// Create a new validation error
    pub fn validation<S: Into<String>>(message: S) -> Self {
        Self::Validation {
            message: message.into(),
        }
    }

    /// Check if this error is recoverable
    pub fn is_recoverable(&self) -> bool {
        match self {
            Self::Convergence { .. } => true,
            Self::Training { .. } => true,
            Self::Numerical { .. } => true,
            _ => false,
        }
    }

    /// Get the error category
    pub fn category(&self) -> &'static str {
        match self {
            Self::Architecture { .. } => "Architecture",
            Self::Training { .. } => "Training",
            Self::Propagation { .. } => "Propagation",
            Self::Io { .. } => "IO",
            Self::Serialization { .. } => "Serialization",
            Self::BinarySerialization { .. } => "BinarySerialization",
            Self::DimensionMismatch { .. } => "DimensionMismatch",
            Self::InvalidParameter { .. } => "InvalidParameter",
            Self::Numerical { .. } => "Numerical",
            Self::Gpu { .. } => "GPU",
            Self::Cuda { .. } => "CUDA",
            #[cfg(feature = "blas")]
            Self::LinearAlgebra { .. } => "LinearAlgebra",
            Self::Shape { .. } => "Shape",
            Self::Convergence { .. } => "Convergence",
            Self::Data { .. } => "Data",
            Self::Optimizer { .. } => "Optimizer",
            Self::LossFunction { .. } => "LossFunction",
            Self::ActivationFunction { .. } => "ActivationFunction",
            Self::Computation { .. } => "Computation",
            Self::Memory { .. } => "Memory",
            Self::Concurrency { .. } => "Concurrency",
            Self::Configuration { .. } => "Configuration",
            Self::Validation { .. } => "Validation",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = NetworkError::architecture("Invalid layer configuration");
        assert_eq!(err.category(), "Architecture");
        assert!(!err.is_recoverable());
    }

    #[test]
    fn test_dimension_mismatch() {
        let err = NetworkError::dimension_mismatch("(10, 20)", "(15, 25)");
        match err {
            NetworkError::DimensionMismatch { expected, actual } => {
                assert_eq!(expected, "(10, 20)");
                assert_eq!(actual, "(15, 25)");
            }
            _ => panic!("Wrong error type"),
        }
    }

    #[test]
    fn test_convergence_error() {
        let err = NetworkError::convergence("Failed to converge", 1000);
        assert!(err.is_recoverable());
        assert_eq!(err.category(), "Convergence");
    }
}
