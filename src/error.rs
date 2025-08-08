//! Error handling for the NNL library
//!
//! This module provides comprehensive error types and utilities for handling
//! various failure modes in neural network operations.

use thiserror::Error;

/// Result type alias for NNL operations
pub type Result<T> = std::result::Result<T, NnlError>;

/// Comprehensive error type for all NNL operations
#[derive(Error, Debug)]
pub enum NnlError {
    /// Tensor operation errors
    #[error("Tensor error: {message}")]
    TensorError {
        /// Error message describing the tensor operation failure
        message: String,
    },

    /// Shape mismatch errors
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        /// Expected tensor shape
        expected: Vec<usize>,
        /// Actual tensor shape that was provided
        actual: Vec<usize>,
    },

    /// Device/backend errors
    #[error("Device error: {message}")]
    DeviceError {
        /// Error message describing the device operation failure
        message: String,
    },

    /// Network architecture errors
    #[error("Network error: {message}")]
    NetworkError {
        /// Error message describing the network operation failure
        message: String,
    },

    /// Training errors
    #[error("Training error: {message}")]
    TrainingError {
        /// Error message describing the training operation failure
        message: String,
    },

    /// I/O errors for model loading/saving
    #[error("I/O error: {0}")]
    IoError(#[from] std::io::Error),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    /// JSON serialization errors
    #[error("JSON error: {0}")]
    JsonError(#[from] serde_json::Error),

    /// CUDA-specific errors
    #[cfg(feature = "cuda")]
    #[error("CUDA error: {0}")]
    CudaError(String),

    /// Vulkan/WGPU errors
    #[error("GPU compute error: {0}")]
    GpuError(String),

    /// Invalid configuration errors
    #[error("Invalid configuration: {message}")]
    ConfigError {
        /// Error message describing the invalid configuration
        message: String,
    },

    /// Mathematical operation errors (e.g., division by zero, NaN)
    #[error("Math error: {message}")]
    MathError {
        /// Error message describing the mathematical operation failure
        message: String,
    },

    /// Memory allocation errors
    #[error("Memory error: {message}")]
    MemoryError {
        /// Error message describing the memory allocation failure
        message: String,
    },

    /// Unsupported operation errors
    #[error("Unsupported operation: {message}")]
    UnsupportedError {
        /// Error message describing the unsupported operation
        message: String,
    },

    /// Invalid input errors
    #[error("Invalid input: {message}")]
    InvalidInputError {
        /// Error message describing the invalid input
        message: String,
    },
}

impl NnlError {
    /// Create a new tensor error
    pub fn tensor<S: Into<String>>(message: S) -> Self {
        Self::TensorError {
            message: message.into(),
        }
    }

    /// Create a new shape mismatch error
    pub fn shape_mismatch(expected: &[usize], actual: &[usize]) -> Self {
        Self::ShapeMismatch {
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Create a new device error
    pub fn device<S: Into<String>>(message: S) -> Self {
        Self::DeviceError {
            message: message.into(),
        }
    }

    /// Create a new network error
    pub fn network<S: Into<String>>(message: S) -> Self {
        Self::NetworkError {
            message: message.into(),
        }
    }

    /// Create a new training error
    pub fn training<S: Into<String>>(message: S) -> Self {
        Self::TrainingError {
            message: message.into(),
        }
    }

    /// Create a new configuration error
    pub fn config<S: Into<String>>(message: S) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Create a new math error
    pub fn math<S: Into<String>>(message: S) -> Self {
        Self::MathError {
            message: message.into(),
        }
    }

    /// Create a new memory error
    pub fn memory<S: Into<String>>(message: S) -> Self {
        Self::MemoryError {
            message: message.into(),
        }
    }

    /// Create a new unsupported operation error
    pub fn unsupported<S: Into<String>>(message: S) -> Self {
        Self::UnsupportedError {
            message: message.into(),
        }
    }

    /// Create a new invalid input error
    pub fn invalid_input<S: Into<String>>(message: S) -> Self {
        Self::InvalidInputError {
            message: message.into(),
        }
    }

    /// Create a GPU error
    pub fn gpu<S: Into<String>>(message: S) -> Self {
        Self::GpuError(message.into())
    }

    /// Create an I/O error
    pub fn io(error: std::io::Error) -> Self {
        Self::IoError(error)
    }

    #[cfg(feature = "cuda")]
    /// Create a CUDA error
    pub fn cuda<S: Into<String>>(message: S) -> Self {
        Self::CudaError(message.into())
    }
}

/// Utility trait for converting common error types
pub trait IntoNnlError<T> {
    /// Convert the error into an NnlError
    fn into_nnl_error(self) -> Result<T>;
    /// Add context to the error before converting to NnlError
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> IntoNnlError<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn into_nnl_error(self) -> Result<T> {
        self.map_err(|e| NnlError::device(e.to_string()))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| NnlError::device(format!("{}: {}", f(), e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = NnlError::tensor("test message");
        assert!(matches!(err, NnlError::TensorError { .. }));
        assert_eq!(err.to_string(), "Tensor error: test message");
    }

    #[test]
    fn test_shape_mismatch() {
        let err = NnlError::shape_mismatch(&[2, 3], &[4, 5]);
        assert!(matches!(err, NnlError::ShapeMismatch { .. }));
        assert!(err.to_string().contains("expected [2, 3], got [4, 5]"));
    }

    #[test]
    fn test_error_chaining() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let nnl_err: NnlError = io_err.into();
        assert!(matches!(nnl_err, NnlError::IoError(_)));
    }
}
