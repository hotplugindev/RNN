//! Neural network layers module
//!
//! This module provides implementations of various neural network layers
//! including dense (fully connected) layers, convolutional layers, and
//! other common layer types with forward and backward pass implementations.

use crate::activations::Activation;
use crate::device::Device;
use crate::error::{NnlError, Result};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::fmt;

pub mod conv;
pub mod dense;
pub mod dropout;
pub mod normalization;
pub mod pooling;

pub use conv::Conv2DLayer;
pub use dense::DenseLayer;
pub use dropout::DropoutLayer;
pub use normalization::{BatchNormLayer, LayerNormLayer};
pub use pooling::{AvgPool2DLayer, FlattenLayer, MaxPool2DLayer, ReshapeLayer};

/// Configuration for different layer types
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerConfig {
    /// Dense (fully connected) layer
    Dense {
        /// Number of input features
        input_size: usize,
        /// Number of output features
        output_size: usize,
        /// Activation function to apply
        activation: Activation,
        /// Whether to use bias parameters
        use_bias: bool,
        /// Weight initialization strategy
        weight_init: WeightInit,
    },
    /// 2D Convolution layer
    Conv2D {
        /// Number of input channels
        in_channels: usize,
        /// Number of output channels
        out_channels: usize,
        /// Size of convolution kernel (height, width)
        kernel_size: (usize, usize),
        /// Stride of convolution (height, width)
        stride: (usize, usize),
        /// Padding applied to input (height, width)
        padding: (usize, usize),
        /// Dilation of convolution kernel (height, width)
        dilation: (usize, usize),
        /// Activation function to apply
        activation: Activation,
        /// Whether to use bias parameters
        use_bias: bool,
        /// Weight initialization strategy
        weight_init: WeightInit,
    },
    /// Dropout layer for regularization
    Dropout {
        /// Probability of dropping each element
        dropout_rate: f32,
    },
    /// Batch normalization layer
    BatchNorm {
        /// Number of features to normalize
        num_features: usize,
        /// Small constant for numerical stability
        eps: f32,
        /// Momentum for running statistics
        momentum: f32,
        /// Whether to use learnable affine parameters
        affine: bool,
    },
    /// Layer normalization
    LayerNorm {
        /// Shape of the normalized tensor dimensions
        normalized_shape: Vec<usize>,
        /// Small constant for numerical stability
        eps: f32,
        /// Whether to use learnable affine parameters
        elementwise_affine: bool,
    },
    /// 2D Max pooling layer
    MaxPool2D {
        /// Size of pooling window (height, width)
        kernel_size: (usize, usize),
        /// Stride of pooling operation (height, width)
        stride: Option<(usize, usize)>,
        /// Padding applied to input (height, width)
        padding: (usize, usize),
    },
    /// 2D Average pooling layer
    AvgPool2D {
        /// Size of pooling window (height, width)
        kernel_size: (usize, usize),
        /// Stride of pooling operation (height, width)
        stride: Option<(usize, usize)>,
        /// Padding applied to input (height, width)
        padding: (usize, usize),
    },
    /// Flatten tensor into 1D
    Flatten {
        /// First dimension to flatten
        start_dim: usize,
        /// Last dimension to flatten (None means all remaining)
        end_dim: Option<usize>,
    },
    /// Reshape tensor to target shape
    Reshape {
        /// Target tensor shape
        target_shape: Vec<usize>,
    },
}

/// Weight initialization strategies
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum WeightInit {
    /// Xavier/Glorot uniform initialization
    Xavier,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He uniform initialization (for ReLU)
    He,
    /// He normal initialization (for ReLU)
    HeNormal,
    /// Random uniform in range [-bound, bound]
    Uniform(f32),
    /// Random normal with mean=0, std=std
    Normal(f32),
    /// All zeros
    Zeros,
    /// All ones
    Ones,
    /// Constant value
    Constant(f32),
}

/// Training mode for layers that behave differently during training vs inference
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMode {
    /// Training mode with gradient computation
    Training,
    /// Inference mode without gradient computation
    Inference,
}

/// Base trait for all neural network layers
pub trait Layer: Send + Sync + std::fmt::Debug {
    /// Forward pass through the layer
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor>;

    /// Backward pass through the layer
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor>;

    /// Get layer parameters (weights, biases, etc.)
    fn parameters(&self) -> Vec<&Tensor>;

    /// Get mutable layer parameters for optimization
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;

    /// Get layer gradients
    fn gradients(&self) -> Vec<&Tensor>;

    /// Get mutable layer gradients
    fn gradients_mut(&mut self) -> Vec<&mut Tensor>;

    /// Zero gradients
    fn zero_grad(&mut self);

    /// Get layer name/type
    fn name(&self) -> &str;

    /// Get output shape for given input shape
    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>>;

    /// Set training mode
    fn set_training(&mut self, training: bool);

    /// Check if layer is in training mode
    fn training(&self) -> bool;

    /// Get number of parameters
    fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.size()).sum()
    }

    /// Move layer to device
    fn to_device(&mut self, device: crate::device::Device) -> Result<()>;

    /// Clone layer (for model copying)
    fn clone_layer(&self) -> Result<Box<dyn Layer>>;
}

impl WeightInit {
    /// Initialize a tensor with the specified initialization strategy
    pub fn initialize(&self, tensor: &mut Tensor, fan_in: usize, fan_out: usize) -> Result<()> {
        match self {
            WeightInit::Xavier => {
                let bound = (6.0 / (fan_in + fan_out) as f32).sqrt();
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        let mut rng = thread_rng();
                        rng.gen_range(-bound..bound)
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::XavierNormal => {
                let std = (2.0 / (fan_in + fan_out) as f32).sqrt();
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        use rand_distr::StandardNormal;
                        let mut rng = thread_rng();
                        rng.sample::<f32, _>(StandardNormal) * std
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::He => {
                let bound = (6.0 / fan_in as f32).sqrt();
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        let mut rng = thread_rng();
                        rng.gen_range(-bound..bound)
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::HeNormal => {
                let std = (2.0 / fan_in as f32).sqrt();
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        use rand_distr::StandardNormal;
                        let mut rng = thread_rng();
                        rng.sample::<f32, _>(StandardNormal) * std
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::Uniform(bound) => {
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        let mut rng = thread_rng();
                        rng.gen_range(-*bound..*bound)
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::Normal(std) => {
                let data = tensor.to_vec()?;
                let new_data: Vec<f32> = data
                    .iter()
                    .map(|_| {
                        use rand::prelude::*;
                        use rand_distr::StandardNormal;
                        let mut rng = thread_rng();
                        rng.sample::<f32, _>(StandardNormal) * std
                    })
                    .collect();
                tensor.copy_from_slice(&new_data)?;
                Ok(())
            }
            WeightInit::Zeros => tensor.fill(0.0),
            WeightInit::Ones => tensor.fill(1.0),
            WeightInit::Constant(value) => tensor.fill(*value),
        }
    }

    /// Get a reasonable default initialization for a given activation
    pub fn default_for_activation(activation: &Activation) -> Self {
        match activation {
            Activation::ReLU | Activation::LeakyReLU(_) => WeightInit::HeNormal,
            Activation::Sigmoid | Activation::Tanh => WeightInit::XavierNormal,
            _ => WeightInit::XavierNormal,
        }
    }
}

impl fmt::Display for WeightInit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            WeightInit::Xavier => write!(f, "Xavier"),
            WeightInit::XavierNormal => write!(f, "XavierNormal"),
            WeightInit::He => write!(f, "He"),
            WeightInit::HeNormal => write!(f, "HeNormal"),
            WeightInit::Uniform(bound) => write!(f, "Uniform(±{})", bound),
            WeightInit::Normal(std) => write!(f, "Normal(σ={})", std),
            WeightInit::Zeros => write!(f, "Zeros"),
            WeightInit::Ones => write!(f, "Ones"),
            WeightInit::Constant(value) => write!(f, "Constant({})", value),
        }
    }
}

impl Default for WeightInit {
    fn default() -> Self {
        WeightInit::XavierNormal
    }
}

/// Factory function to create layers from configuration
pub fn create_layer(config: LayerConfig, device: Device) -> Result<Box<dyn Layer>> {
    match config {
        LayerConfig::Dense {
            input_size,
            output_size,
            activation,
            use_bias,
            weight_init,
        } => Ok(Box::new(DenseLayer::new_on_device(
            input_size,
            output_size,
            activation,
            use_bias,
            weight_init,
            device,
        )?)),
        LayerConfig::Conv2D {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            weight_init,
        } => Ok(Box::new(Conv2DLayer::new_on_device(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            weight_init,
            device,
        )?)),
        LayerConfig::Dropout { dropout_rate } => Ok(Box::new(DropoutLayer::new(dropout_rate)?)),
        LayerConfig::BatchNorm {
            num_features,
            eps,
            momentum,
            affine,
        } => Ok(Box::new(BatchNormLayer::new_on_device(
            num_features,
            eps,
            momentum,
            affine,
            device,
        )?)),
        LayerConfig::LayerNorm {
            normalized_shape,
            eps,
            elementwise_affine,
        } => Ok(Box::new(LayerNormLayer::new_on_device(
            normalized_shape,
            eps,
            elementwise_affine,
            device,
        )?)),
        LayerConfig::MaxPool2D {
            kernel_size,
            stride,
            padding,
        } => Ok(Box::new(MaxPool2DLayer::new(kernel_size, stride, padding)?)),
        LayerConfig::AvgPool2D {
            kernel_size,
            stride,
            padding,
        } => Ok(Box::new(AvgPool2DLayer::new(kernel_size, stride, padding)?)),
        LayerConfig::Flatten { start_dim, end_dim } => {
            Ok(Box::new(FlattenLayer::new(start_dim, end_dim)?))
        }
        LayerConfig::Reshape { target_shape } => {
            Ok(Box::new(ReshapeLayer::new(target_shape.clone())?))
        }
    }
}

impl fmt::Display for LayerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerConfig::Dense {
                input_size,
                output_size,
                activation,
                ..
            } => write!(f, "Dense({} → {}, {})", input_size, output_size, activation),
            LayerConfig::Conv2D {
                in_channels,
                out_channels,
                kernel_size,
                ..
            } => write!(
                f,
                "Conv2D({} → {}, kernel={}×{})",
                in_channels, out_channels, kernel_size.0, kernel_size.1
            ),
            LayerConfig::Dropout { dropout_rate } => write!(f, "Dropout(p={})", dropout_rate),
            LayerConfig::BatchNorm { num_features, .. } => {
                write!(f, "BatchNorm({})", num_features)
            }
            LayerConfig::LayerNorm {
                normalized_shape, ..
            } => write!(f, "LayerNorm({:?})", normalized_shape),
            LayerConfig::MaxPool2D { kernel_size, .. } => {
                write!(f, "MaxPool2D({}×{})", kernel_size.0, kernel_size.1)
            }
            LayerConfig::AvgPool2D { kernel_size, .. } => {
                write!(f, "AvgPool2D({}×{})", kernel_size.0, kernel_size.1)
            }
            LayerConfig::Flatten { .. } => write!(f, "Flatten"),
            LayerConfig::Reshape { target_shape } => write!(f, "Reshape({:?})", target_shape),
        }
    }
}

/// Common layer configuration presets
impl LayerConfig {
    /// Dense layer with ReLU activation
    pub fn dense_relu(input_size: usize, output_size: usize) -> Self {
        LayerConfig::Dense {
            input_size,
            output_size,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        }
    }

    /// Dense layer with sigmoid activation
    pub fn dense_sigmoid(input_size: usize, output_size: usize) -> Self {
        LayerConfig::Dense {
            input_size,
            output_size,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::XavierNormal,
        }
    }

    /// Dense layer with linear activation (no activation)
    pub fn dense_linear(input_size: usize, output_size: usize) -> Self {
        LayerConfig::Dense {
            input_size,
            output_size,
            activation: Activation::Linear,
            use_bias: true,
            weight_init: WeightInit::XavierNormal,
        }
    }

    /// 3x3 Conv2D layer with ReLU
    pub fn conv2d_3x3(in_channels: usize, out_channels: usize) -> Self {
        LayerConfig::Conv2D {
            in_channels,
            out_channels,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        }
    }

    /// 5x5 Conv2D layer with ReLU
    pub fn conv2d_5x5(in_channels: usize, out_channels: usize) -> Self {
        LayerConfig::Conv2D {
            in_channels,
            out_channels,
            kernel_size: (5, 5),
            stride: (1, 1),
            padding: (2, 2),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        }
    }

    /// Dropout with standard rate
    pub fn dropout(dropout_rate: f32) -> Self {
        LayerConfig::Dropout { dropout_rate }
    }

    /// Batch normalization with default parameters
    pub fn batch_norm(num_features: usize) -> Self {
        LayerConfig::BatchNorm {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            affine: true,
        }
    }

    /// Layer normalization with default parameters
    pub fn layer_norm(normalized_shape: Vec<usize>) -> Self {
        LayerConfig::LayerNorm {
            normalized_shape,
            eps: 1e-5,
            elementwise_affine: true,
        }
    }

    /// 2x2 max pooling
    pub fn max_pool2d() -> Self {
        LayerConfig::MaxPool2D {
            kernel_size: (2, 2),
            stride: None, // Defaults to kernel_size
            padding: (0, 0),
        }
    }

    /// Flatten layer starting from dimension 1
    pub fn flatten() -> Self {
        LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        }
    }
}

/// Utility functions for layers
pub mod utils {
    use super::*;

    /// Calculate output size for convolution operation
    pub fn conv_output_size(
        input_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> usize {
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    }

    /// Calculate output size for pooling operation
    pub fn pool_output_size(
        input_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> usize {
        (input_size + 2 * padding - kernel_size) / stride + 1
    }

    /// Calculate padding needed for "same" convolution
    pub fn same_padding(kernel_size: usize, dilation: usize) -> usize {
        (dilation * (kernel_size - 1)) / 2
    }

    /// Validate tensor shapes are compatible
    pub fn validate_shapes(expected: &[usize], actual: &[usize]) -> Result<()> {
        if expected != actual {
            return Err(NnlError::shape_mismatch(expected, actual));
        }
        Ok(())
    }

    /// Calculate the number of parameters for a layer
    pub fn count_parameters(weights_shape: &[usize], has_bias: bool, bias_size: usize) -> usize {
        let weight_params: usize = weights_shape.iter().product();
        let bias_params = if has_bias { bias_size } else { 0 };
        weight_params + bias_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_weight_initialization() {
        let mut tensor = Tensor::zeros(&[10, 10]).unwrap();
        let init = WeightInit::Xavier;
        init.initialize(&mut tensor, 10, 10).unwrap();

        let data = tensor.to_vec().unwrap();
        // Check that not all values are zero
        assert!(data.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn test_layer_config_display() {
        let dense = LayerConfig::dense_relu(784, 128);
        let display = format!("{}", dense);
        assert!(display.contains("Dense"));
        assert!(display.contains("784"));
        assert!(display.contains("128"));

        let conv = LayerConfig::conv2d_3x3(3, 64);
        let display = format!("{}", conv);
        assert!(display.contains("Conv2D"));
        assert!(display.contains("3"));
        assert!(display.contains("64"));
    }

    #[test]
    fn test_weight_init_display() {
        assert_eq!(format!("{}", WeightInit::Xavier), "Xavier");
        assert_eq!(format!("{}", WeightInit::HeNormal), "HeNormal");
        assert_eq!(format!("{}", WeightInit::Uniform(0.5)), "Uniform(±0.5)");
        assert_eq!(format!("{}", WeightInit::Normal(0.1)), "Normal(σ=0.1)");
    }

    #[test]
    fn test_conv_output_size_calculation() {
        // Standard convolution: input=32, kernel=3, stride=1, padding=1
        let output_size = utils::conv_output_size(32, 3, 1, 1, 1);
        assert_eq!(output_size, 32); // Same size with padding

        // Stride 2 convolution: input=32, kernel=3, stride=2, padding=1
        let output_size = utils::conv_output_size(32, 3, 2, 1, 1);
        assert_eq!(output_size, 16); // Half size
    }

    #[test]
    fn test_pool_output_size_calculation() {
        // 2x2 max pooling with stride 2
        let output_size = utils::pool_output_size(32, 2, 2, 0);
        assert_eq!(output_size, 16);

        // 3x3 pooling with stride 1
        let output_size = utils::pool_output_size(32, 3, 1, 1);
        assert_eq!(output_size, 32);
    }

    #[test]
    fn test_same_padding_calculation() {
        assert_eq!(utils::same_padding(3, 1), 1);
        assert_eq!(utils::same_padding(5, 1), 2);
        assert_eq!(utils::same_padding(3, 2), 2);
    }

    #[test]
    fn test_weight_init_defaults() {
        let relu_init = WeightInit::default_for_activation(&Activation::ReLU);
        assert_eq!(relu_init, WeightInit::HeNormal);

        let sigmoid_init = WeightInit::default_for_activation(&Activation::Sigmoid);
        assert_eq!(sigmoid_init, WeightInit::XavierNormal);
    }

    #[test]
    fn test_parameter_counting() {
        // Dense layer: 784 -> 128 + bias
        let weight_params = utils::count_parameters(&[784, 128], true, 128);
        assert_eq!(weight_params, 784 * 128 + 128);

        // Conv layer: 3 input channels, 64 output channels, 3x3 kernel + bias
        let conv_params = utils::count_parameters(&[64, 3, 3, 3], true, 64);
        assert_eq!(conv_params, 64 * 3 * 3 * 3 + 64);
    }

    #[test]
    fn test_shape_validation() {
        let result = utils::validate_shapes(&[2, 3, 4], &[2, 3, 4]);
        assert!(result.is_ok());

        let result = utils::validate_shapes(&[2, 3, 4], &[2, 3, 5]);
        assert!(result.is_err());
    }
}
