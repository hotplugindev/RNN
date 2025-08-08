//! Error handling for the RNN library
//!
//! This module provides comprehensive error types and utilities for handling
//! various failure modes in neural network operations.

use thiserror::Error;

/// Result type alias for RNN operations
pub type Result<T> = std::result::Result<T, RnnError>;

/// Comprehensive error type for all RNN operations
#[derive(Error, Debug)]
pub enum RnnError {
    /// Tensor operation errors
    #[error("Tensor error: {message}")]
    TensorError { message: String },

    /// Shape mismatch errors
    #[error("Shape mismatch: expected {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    /// Device/backend errors
    #[error("Device error: {message}")]
    DeviceError { message: String },

    /// Network architecture errors
    #[error("Network error: {message}")]
    NetworkError { message: String },

    /// Training errors
    #[error("Training error: {message}")]
    TrainingError { message: String },

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
    ConfigError { message: String },

    /// Mathematical operation errors (e.g., division by zero, NaN)
    #[error("Math error: {message}")]
    MathError { message: String },

    /// Memory allocation errors
    #[error("Memory error: {message}")]
    MemoryError { message: String },

    /// Unsupported operation errors
    #[error("Unsupported operation: {message}")]
    UnsupportedError { message: String },
}

impl RnnError {
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
pub trait IntoRnnError<T> {
    fn into_rnn_error(self) -> Result<T>;
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> IntoRnnError<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn into_rnn_error(self) -> Result<T> {
        self.map_err(|e| RnnError::device(e.to_string()))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| RnnError::device(format!("{}: {}", f(), e)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = RnnError::tensor("test message");
        assert!(matches!(err, RnnError::TensorError { .. }));
        assert_eq!(err.to_string(), "Tensor error: test message");
    }

    #[test]
    fn test_shape_mismatch() {
        let err = RnnError::shape_mismatch(&[2, 3], &[4, 5]);
        assert!(matches!(err, RnnError::ShapeMismatch { .. }));
        assert!(err.to_string().contains("expected [2, 3], got [4, 5]"));
    }

    #[test]
    fn test_error_chaining() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let rnn_err: RnnError = io_err.into();
        assert!(matches!(rnn_err, RnnError::IoError(_)));
    }
}
