//! Layer definitions and implementations for neural networks.
//!
//! This module provides various types of layers that can be used to build neural networks,
//! including dense (fully connected) layers, convolutional layers, and more specialized layers.

use crate::activation::ActivationFunction;
use crate::error::{NetworkError, Result};
use ndarray::{Array1, Array2, Zip};
use rand::Rng;
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};

/// Types of layers available in the neural network.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    /// Fully connected (dense) layer
    Dense,
    /// Convolutional layer (2D)
    Convolutional2D,
    /// Max pooling layer
    MaxPooling2D,
    /// Average pooling layer
    AveragePooling2D,
    /// Dropout layer for regularization
    Dropout,
    /// Batch normalization layer
    BatchNormalization,
    /// LSTM layer for recurrent networks
    LSTM,
    /// GRU layer for recurrent networks
    GRU,
    /// Embedding layer
    Embedding,
    /// Flatten layer
    Flatten,
    /// Reshape layer
    Reshape,
}

/// Weight initialization methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WeightInitialization {
    /// Zero initialization
    Zeros,
    /// Random uniform initialization
    Uniform { min: i32, max: i32 }, // Using i32 for serialization
    /// Random normal initialization
    Normal { mean: i32, std: i32 }, // Using i32 for serialization (will be converted to f64)
    /// Xavier/Glorot uniform initialization
    XavierUniform,
    /// Xavier/Glorot normal initialization
    XavierNormal,
    /// He uniform initialization (good for ReLU)
    HeUniform,
    /// He normal initialization (good for ReLU)
    HeNormal,
    /// LeCun uniform initialization
    LeCunUniform,
    /// LeCun normal initialization
    LeCunNormal,
}

impl WeightInitialization {
    /// Initialize weights for a layer with given input and output dimensions.
    pub fn initialize_weights(&self, input_dim: usize, output_dim: usize) -> Result<Array2<f64>> {
        let mut rng = rand::thread_rng();
        let mut weights = Array2::zeros((input_dim, output_dim));

        match self {
            WeightInitialization::Zeros => {
                // Weights are already zeros
            }
            WeightInitialization::Uniform { min, max } => {
                let min_f = *min as f64;
                let max_f = *max as f64;
                let uniform = Uniform::new(min_f, max_f);
                Zip::from(&mut weights).for_each(|w| {
                    *w = uniform.sample(&mut rng);
                });
            }
            WeightInitialization::Normal { mean, std } => {
                let mean_f = *mean as f64;
                let std_f = *std as f64;
                let normal = Normal::new(mean_f, std_f).map_err(|e| {
                    NetworkError::configuration(format!(
                        "Invalid normal distribution parameters: {}",
                        e
                    ))
                })?;
                Zip::from(&mut weights).for_each(|w| {
                    *w = normal.sample(&mut rng);
                });
            }
            WeightInitialization::XavierUniform => {
                let limit = (6.0 / (input_dim + output_dim) as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                Zip::from(&mut weights).for_each(|w| {
                    *w = uniform.sample(&mut rng);
                });
            }
            WeightInitialization::XavierNormal => {
                let std = (2.0 / (input_dim + output_dim) as f64).sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    NetworkError::configuration(format!(
                        "Xavier normal initialization failed: {}",
                        e
                    ))
                })?;
                Zip::from(&mut weights).for_each(|w| {
                    *w = normal.sample(&mut rng);
                });
            }
            WeightInitialization::HeUniform => {
                let limit = (6.0 / input_dim as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                Zip::from(&mut weights).for_each(|w| {
                    *w = uniform.sample(&mut rng);
                });
            }
            WeightInitialization::HeNormal => {
                let std = (2.0 / input_dim as f64).sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    NetworkError::configuration(format!("He normal initialization failed: {}", e))
                })?;
                Zip::from(&mut weights).for_each(|w| {
                    *w = normal.sample(&mut rng);
                });
            }
            WeightInitialization::LeCunUniform => {
                let limit = (3.0 / input_dim as f64).sqrt();
                let uniform = Uniform::new(-limit, limit);
                Zip::from(&mut weights).for_each(|w| {
                    *w = uniform.sample(&mut rng);
                });
            }
            WeightInitialization::LeCunNormal => {
                let std = (1.0 / input_dim as f64).sqrt();
                let normal = Normal::new(0.0, std).map_err(|e| {
                    NetworkError::configuration(format!(
                        "LeCun normal initialization failed: {}",
                        e
                    ))
                })?;
                Zip::from(&mut weights).for_each(|w| {
                    *w = normal.sample(&mut rng);
                });
            }
        }

        Ok(weights)
    }

    /// Initialize bias vector.
    pub fn initialize_bias(&self, size: usize) -> Array1<f64> {
        match self {
            WeightInitialization::Zeros => Array1::zeros(size),
            _ => Array1::zeros(size), // Typically biases are initialized to zero
        }
    }
}

impl Default for WeightInitialization {
    fn default() -> Self {
        WeightInitialization::XavierUniform
    }
}

/// Configuration for dropout regularization.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct DropoutConfig {
    /// Dropout rate (probability of setting a unit to 0)
    pub rate: f64,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl DropoutConfig {
    pub fn new(rate: f64) -> Result<Self> {
        if rate < 0.0 || rate >= 1.0 {
            return Err(NetworkError::invalid_parameter(
                "dropout_rate",
                &rate.to_string(),
                "must be in range [0, 1)",
            ));
        }
        Ok(Self { rate, seed: None })
    }

    pub fn with_seed(rate: f64, seed: u64) -> Result<Self> {
        let mut config = Self::new(rate)?;
        config.seed = Some(seed);
        Ok(config)
    }
}

/// A neural network layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Layer {
    /// Type of the layer
    pub layer_type: LayerType,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Weights matrix (input_dim x output_dim)
    pub weights: Array2<f64>,
    /// Bias vector (output_dim)
    pub bias: Array1<f64>,
    /// Activation function
    pub activation: ActivationFunction,
    /// Weight initialization method
    pub weight_init: WeightInitialization,
    /// Dropout configuration (if applicable)
    pub dropout: Option<DropoutConfig>,
    /// Whether to use bias
    pub use_bias: bool,
    /// Layer name (optional)
    pub name: Option<String>,
    /// Whether the layer is trainable
    pub trainable: bool,
    /// Cached forward pass output for backpropagation
    #[serde(skip)]
    pub last_output: Option<Array2<f64>>,
    /// Cached input for backpropagation
    #[serde(skip)]
    pub last_input: Option<Array2<f64>>,
}

impl Layer {
    /// Create a new dense layer.
    pub fn dense(
        input_dim: usize,
        output_dim: usize,
        activation: ActivationFunction,
        weight_init: WeightInitialization,
        use_bias: bool,
    ) -> Result<Self> {
        if input_dim == 0 || output_dim == 0 {
            return Err(NetworkError::architecture(
                "Layer dimensions must be greater than 0",
            ));
        }

        let weights = weight_init.initialize_weights(input_dim, output_dim)?;
        let bias = if use_bias {
            weight_init.initialize_bias(output_dim)
        } else {
            Array1::zeros(output_dim)
        };

        Ok(Self {
            layer_type: LayerType::Dense,
            input_dim,
            output_dim,
            weights,
            bias,
            activation,
            weight_init,
            dropout: None,
            use_bias,
            name: None,
            trainable: true,
            last_output: None,
            last_input: None,
        })
    }

    /// Create a new dropout layer.
    pub fn dropout(input_dim: usize, dropout_config: DropoutConfig) -> Result<Self> {
        Ok(Self {
            layer_type: LayerType::Dropout,
            input_dim,
            output_dim: input_dim, // Dropout doesn't change dimensions
            weights: Array2::zeros((input_dim, input_dim)),
            bias: Array1::zeros(input_dim),
            activation: ActivationFunction::Linear,
            weight_init: WeightInitialization::Zeros,
            dropout: Some(dropout_config),
            use_bias: false,
            name: None,
            trainable: false, // Dropout layers don't have trainable parameters
            last_output: None,
            last_input: None,
        })
    }

    /// Set the layer name.
    pub fn with_name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set whether the layer is trainable.
    pub fn set_trainable(mut self, trainable: bool) -> Self {
        self.trainable = trainable;
        self
    }

    /// Forward pass through the layer.
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Result<Array2<f64>> {
        if input.ncols() != self.input_dim {
            return Err(NetworkError::dimension_mismatch(
                format!("input columns: {}", self.input_dim),
                format!("actual input columns: {}", input.ncols()),
            ));
        }

        // Cache input for backpropagation
        self.last_input = Some(input.clone());

        let output = match self.layer_type {
            LayerType::Dense => self.forward_dense(input)?,
            LayerType::Dropout => self.forward_dropout(input, training)?,
            _ => {
                return Err(NetworkError::architecture(format!(
                    "Forward pass not implemented for layer type: {:?}",
                    self.layer_type
                )));
            }
        };

        // Cache output for backpropagation
        self.last_output = Some(output.clone());

        Ok(output)
    }

    /// Dense layer forward pass.
    fn forward_dense(&self, input: &Array2<f64>) -> Result<Array2<f64>> {
        // Compute z = input * weights + bias
        let z = input.dot(&self.weights);
        let z = if self.use_bias { z + &self.bias } else { z };

        // Apply activation function
        self.activation.apply_matrix(&z)
    }

    /// Dropout layer forward pass.
    fn forward_dropout(&self, input: &Array2<f64>, training: bool) -> Result<Array2<f64>> {
        if !training {
            // During inference, simply return the input
            return Ok(input.clone());
        }

        let dropout_config = self.dropout.ok_or_else(|| {
            NetworkError::configuration("Dropout layer missing dropout configuration")
        })?;

        let mut rng = rand::thread_rng();
        let mut mask = Array2::zeros(input.raw_dim());

        // Generate dropout mask
        let keep_prob = 1.0 - dropout_config.rate;
        Zip::from(&mut mask).for_each(|m| {
            *m = if rng.gen::<f64>() < keep_prob {
                1.0 / keep_prob // Scale to maintain expected value
            } else {
                0.0
            };
        });

        Ok(input * mask)
    }

    /// Backward pass through the layer.
    pub fn backward(&mut self, grad_output: &Array2<f64>) -> Result<BackwardResult> {
        match self.layer_type {
            LayerType::Dense => self.backward_dense(grad_output),
            LayerType::Dropout => self.backward_dropout(grad_output),
            _ => Err(NetworkError::architecture(format!(
                "Backward pass not implemented for layer type: {:?}",
                self.layer_type
            ))),
        }
    }

    /// Dense layer backward pass.
    fn backward_dense(&mut self, grad_output: &Array2<f64>) -> Result<BackwardResult> {
        let last_input = self
            .last_input
            .as_ref()
            .ok_or_else(|| NetworkError::propagation("No cached input found for backward pass"))?;

        let last_output = self
            .last_output
            .as_ref()
            .ok_or_else(|| NetworkError::propagation("No cached output found for backward pass"))?;

        // Compute activation derivative
        let activation_grad = self.activation.derivative_matrix(last_output)?;
        let grad_z = grad_output * activation_grad;

        // Compute gradients
        let grad_weights = last_input.t().dot(&grad_z);
        let grad_bias = if self.use_bias {
            grad_z.sum_axis(ndarray::Axis(0))
        } else {
            Array1::zeros(self.output_dim)
        };
        let grad_input = grad_z.dot(&self.weights.t());

        Ok(BackwardResult {
            grad_input,
            grad_weights: Some(grad_weights),
            grad_bias: Some(grad_bias),
        })
    }

    /// Dropout layer backward pass.
    fn backward_dropout(&mut self, grad_output: &Array2<f64>) -> Result<BackwardResult> {
        // For simplicity, we'll pass through the gradient unchanged
        // In a full implementation, you'd apply the same mask used in forward pass
        Ok(BackwardResult {
            grad_input: grad_output.clone(),
            grad_weights: None,
            grad_bias: None,
        })
    }

    /// Update weights and biases using gradients.
    pub fn update_weights(
        &mut self,
        grad_weights: &Array2<f64>,
        grad_bias: &Array1<f64>,
        learning_rate: f64,
    ) -> Result<()> {
        if !self.trainable {
            return Ok(());
        }

        // Update weights
        self.weights = &self.weights - learning_rate * grad_weights;

        // Update bias if used
        if self.use_bias {
            self.bias = &self.bias - learning_rate * grad_bias;
        }

        Ok(())
    }

    /// Get the number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        if !self.trainable {
            return 0;
        }

        let weight_params = self.weights.len();
        let bias_params = if self.use_bias { self.bias.len() } else { 0 };
        weight_params + bias_params
    }

    /// Reset the layer's weights and biases.
    pub fn reset_parameters(&mut self) -> Result<()> {
        self.weights = self
            .weight_init
            .initialize_weights(self.input_dim, self.output_dim)?;
        self.bias = self.weight_init.initialize_bias(self.output_dim);
        Ok(())
    }

    /// Get layer summary information.
    pub fn summary(&self) -> LayerSummary {
        LayerSummary {
            name: self
                .name
                .clone()
                .unwrap_or_else(|| format!("{:?}", self.layer_type)),
            layer_type: self.layer_type.clone(),
            input_shape: vec![self.input_dim],
            output_shape: vec![self.output_dim],
            parameter_count: self.parameter_count(),
            activation: self.activation.name().to_string(),
            trainable: self.trainable,
        }
    }
}

/// Result of a backward pass through a layer.
#[derive(Debug, Clone)]
pub struct BackwardResult {
    /// Gradient with respect to the input
    pub grad_input: Array2<f64>,
    /// Gradient with respect to the weights (None for layers without weights)
    pub grad_weights: Option<Array2<f64>>,
    /// Gradient with respect to the bias (None for layers without bias)
    pub grad_bias: Option<Array1<f64>>,
}

/// Summary information about a layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerSummary {
    pub name: String,
    pub layer_type: LayerType,
    pub input_shape: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub parameter_count: usize,
    pub activation: String,
    pub trainable: bool,
}

/// Builder for creating layers with a fluent interface.
#[derive(Debug, Clone)]
pub struct LayerBuilder {
    layer_type: LayerType,
    output_dim: usize,
    activation: ActivationFunction,
    weight_init: WeightInitialization,
    use_bias: bool,
    dropout: Option<DropoutConfig>,
    name: Option<String>,
    trainable: bool,
}

impl LayerBuilder {
    /// Create a new dense layer builder.
    pub fn dense(output_dim: usize) -> Self {
        Self {
            layer_type: LayerType::Dense,
            output_dim,
            activation: ActivationFunction::ReLU,
            weight_init: WeightInitialization::XavierUniform,
            use_bias: true,
            dropout: None,
            name: None,
            trainable: true,
        }
    }

    /// Create a new dropout layer builder.
    pub fn dropout(rate: f64) -> Result<Self> {
        let dropout_config = DropoutConfig::new(rate)?;
        Ok(Self {
            layer_type: LayerType::Dropout,
            output_dim: 0, // Will be set when building
            activation: ActivationFunction::Linear,
            weight_init: WeightInitialization::Zeros,
            use_bias: false,
            dropout: Some(dropout_config),
            name: None,
            trainable: false,
        })
    }

    /// Set the activation function.
    pub fn activation(mut self, activation: ActivationFunction) -> Self {
        self.activation = activation;
        self
    }

    /// Set the weight initialization method.
    pub fn weight_init(mut self, weight_init: WeightInitialization) -> Self {
        self.weight_init = weight_init;
        self
    }

    /// Set whether to use bias.
    pub fn use_bias(mut self, use_bias: bool) -> Self {
        self.use_bias = use_bias;
        self
    }

    /// Set the layer name.
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set whether the layer is trainable.
    pub fn trainable(mut self, trainable: bool) -> Self {
        self.trainable = trainable;
        self
    }

    /// Build the layer with the specified input dimension.
    pub fn build(self, input_dim: usize) -> Result<Layer> {
        let output_dim = if self.layer_type == LayerType::Dropout {
            input_dim
        } else {
            self.output_dim
        };

        let mut layer = match self.layer_type {
            LayerType::Dense => Layer::dense(
                input_dim,
                output_dim,
                self.activation,
                self.weight_init,
                self.use_bias,
            )?,
            LayerType::Dropout => {
                let dropout_config = self.dropout.ok_or_else(|| {
                    NetworkError::configuration("Dropout layer missing configuration")
                })?;
                Layer::dropout(input_dim, dropout_config)?
            }
            _ => {
                return Err(NetworkError::architecture(format!(
                    "Layer type {:?} not yet implemented",
                    self.layer_type
                )));
            }
        };

        if let Some(name) = self.name {
            layer.name = Some(name);
        }
        layer.trainable = self.trainable;

        Ok(layer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_layer_creation() {
        let layer = Layer::dense(
            10,
            5,
            ActivationFunction::ReLU,
            WeightInitialization::XavierUniform,
            true,
        )
        .unwrap();

        assert_eq!(layer.input_dim, 10);
        assert_eq!(layer.output_dim, 5);
        assert_eq!(layer.weights.shape(), &[10, 5]);
        assert_eq!(layer.bias.len(), 5);
        assert!(layer.use_bias);
        assert!(layer.trainable);
    }

    #[test]
    fn test_layer_builder() {
        let layer = LayerBuilder::dense(128)
            .activation(ActivationFunction::Sigmoid)
            .weight_init(WeightInitialization::HeNormal)
            .name("hidden_layer")
            .build(784)
            .unwrap();

        assert_eq!(layer.input_dim, 784);
        assert_eq!(layer.output_dim, 128);
        assert_eq!(layer.activation, ActivationFunction::Sigmoid);
        assert_eq!(layer.weight_init, WeightInitialization::HeNormal);
        assert_eq!(layer.name, Some("hidden_layer".to_string()));
    }

    #[test]
    fn test_forward_pass() {
        let mut layer = Layer::dense(
            2,
            1,
            ActivationFunction::Linear,
            WeightInitialization::Zeros,
            false,
        )
        .unwrap();

        // Set specific weights for testing
        layer.weights[[0, 0]] = 1.0;
        layer.weights[[1, 0]] = 2.0;

        let input = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).unwrap();
        let output = layer.forward(&input, false).unwrap();

        // Expected: 3*1 + 4*2 = 11
        assert_abs_diff_eq!(output[[0, 0]], 11.0, epsilon = 1e-10);
    }

    #[test]
    fn test_dropout_layer() {
        let dropout_config = DropoutConfig::new(0.5).unwrap();
        let mut layer = Layer::dropout(10, dropout_config).unwrap();

        let input = Array2::ones((5, 10));

        // Test inference mode (should pass through unchanged)
        let output_inference = layer.forward(&input, false).unwrap();
        assert_eq!(input, output_inference);

        // Test training mode (should apply dropout)
        let output_training = layer.forward(&input, true).unwrap();
        // We can't test exact values due to randomness, but we can check dimensions
        assert_eq!(output_training.shape(), input.shape());
    }

    #[test]
    fn test_weight_initialization() {
        let xavier = WeightInitialization::XavierUniform;
        let weights = xavier.initialize_weights(10, 5).unwrap();

        // Check dimensions
        assert_eq!(weights.shape(), &[10, 5]);

        // Check that weights are not all zeros (with high probability)
        assert!(weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_parameter_count() {
        let layer_with_bias = Layer::dense(
            10,
            5,
            ActivationFunction::ReLU,
            WeightInitialization::XavierUniform,
            true,
        )
        .unwrap();

        assert_eq!(layer_with_bias.parameter_count(), 10 * 5 + 5);

        let layer_without_bias = Layer::dense(
            10,
            5,
            ActivationFunction::ReLU,
            WeightInitialization::XavierUniform,
            false,
        )
        .unwrap();

        assert_eq!(layer_without_bias.parameter_count(), 10 * 5);
    }

    #[test]
    fn test_layer_summary() {
        let layer = LayerBuilder::dense(64)
            .activation(ActivationFunction::ReLU)
            .name("test_layer")
            .build(32)
            .unwrap();

        let summary = layer.summary();
        assert_eq!(summary.name, "test_layer");
        assert_eq!(summary.layer_type, LayerType::Dense);
        assert_eq!(summary.input_shape, vec![32]);
        assert_eq!(summary.output_shape, vec![64]);
        assert_eq!(summary.activation, "ReLU");
        assert!(summary.trainable);
    }

    #[test]
    fn test_invalid_parameters() {
        // Test invalid dropout rate
        assert!(DropoutConfig::new(-0.1).is_err());
        assert!(DropoutConfig::new(1.0).is_err());
        assert!(DropoutConfig::new(1.5).is_err());

        // Test invalid layer dimensions
        assert!(Layer::dense(
            0,
            5,
            ActivationFunction::ReLU,
            WeightInitialization::XavierUniform,
            true,
        )
        .is_err());
    }
}
