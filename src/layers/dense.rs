//! Dense (fully connected) layer implementation
//!
//! This module provides a complete implementation of dense/fully connected layers
//! with forward and backward passes, weight initialization, and gradient computation.

use crate::activations::Activation;
use crate::device::Device;
use crate::error::{NnlError, Result};
use crate::layers::{Layer, TrainingMode, WeightInit};
use crate::tensor::Tensor;
use std::fmt;

/// Dense/Fully connected layer
#[derive(Debug)]
pub struct DenseLayer {
    /// Weight matrix [input_size, output_size]
    weights: Tensor,
    /// Bias vector [output_size] (optional)
    bias: Option<Tensor>,
    /// Weight gradients
    weight_grad: Tensor,
    /// Bias gradients (optional)
    bias_grad: Option<Tensor>,
    /// Activation function
    activation: Activation,
    /// Whether to use bias
    use_bias: bool,
    /// Input size
    input_size: usize,
    /// Output size
    output_size: usize,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached pre-activation for backward pass
    cached_pre_activation: Option<Tensor>,
    /// Training mode
    training: bool,
}

impl DenseLayer {
    /// Create a new dense layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
    ) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::new_on_device(
            input_size,
            output_size,
            activation,
            use_bias,
            weight_init,
            device,
        )
    }

    /// Create a new dense layer on specific device
    pub fn new_on_device(
        input_size: usize,
        output_size: usize,
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
        device: Device,
    ) -> Result<Self> {
        if input_size == 0 || output_size == 0 {
            return Err(NnlError::config("Input and output sizes must be positive"));
        }

        // Initialize weights
        let mut weights = Tensor::zeros_on_device(&[input_size, output_size], device.clone())?;
        weight_init.initialize(&mut weights, input_size, output_size)?;

        // Initialize weight gradients
        let weight_grad = Tensor::zeros_on_device(&[input_size, output_size], device.clone())?;

        // Initialize bias if needed
        let (bias, bias_grad) = if use_bias {
            let mut bias_tensor = Tensor::zeros_on_device(&[output_size], device.clone())?;
            // Initialize bias to small values
            WeightInit::Zeros.initialize(&mut bias_tensor, 1, output_size)?;
            let bias_grad_tensor = Tensor::zeros_on_device(&[output_size], device)?;
            (Some(bias_tensor), Some(bias_grad_tensor))
        } else {
            (None, None)
        };

        Ok(Self {
            weights,
            bias,
            weight_grad,
            bias_grad,
            activation,
            use_bias,
            input_size,
            output_size,
            cached_input: None,
            cached_pre_activation: None,
            training: true,
        })
    }

    /// Get input size
    pub fn input_size(&self) -> usize {
        self.input_size
    }

    /// Get output size
    pub fn output_size(&self) -> usize {
        self.output_size
    }

    /// Get activation function
    pub fn activation(&self) -> &Activation {
        &self.activation
    }

    /// Set activation function
    pub fn set_activation(&mut self, activation: Activation) {
        self.activation = activation;
    }

    /// Get weights tensor
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Get bias tensor (if exists)
    pub fn bias(&self) -> Option<&Tensor> {
        self.bias.as_ref()
    }

    /// Get weight gradients
    pub fn weight_gradients(&self) -> &Tensor {
        &self.weight_grad
    }

    /// Get bias gradients (if exists)
    pub fn bias_gradients(&self) -> Option<&Tensor> {
        self.bias_grad.as_ref()
    }

    /// Compute linear transformation: input @ weights + bias
    fn linear_forward(&self, input: &Tensor) -> Result<Tensor> {
        // Validate input shape
        if input.shape().len() < 2 {
            return Err(NnlError::tensor("Input must be at least 2D"));
        }

        let input_features = input.shape()[input.shape().len() - 1];
        if input_features != self.input_size {
            return Err(NnlError::shape_mismatch(
                &[self.input_size],
                &[input_features],
            ));
        }

        // Handle batch dimensions
        let batch_shape = &input.shape()[..input.shape().len() - 1];
        let batch_size: usize = batch_shape.iter().product();

        // Reshape input to [batch_size, input_size]
        let input_2d = if batch_size == 1 && input.shape().len() == 2 {
            input.clone_data()?
        } else {
            input.reshape(&[batch_size, self.input_size])?
        };

        // Matrix multiplication: [batch_size, input_size] @ [input_size, output_size]
        let output = input_2d.matmul(&self.weights)?;

        // Add bias if present
        let output_with_bias = if let Some(ref bias) = self.bias {
            // Broadcast bias across batch dimension
            let bias_expanded = bias.reshape(&[1, self.output_size])?;
            output.add(&bias_expanded)?
        } else {
            output
        };

        // Reshape back to original batch dimensions + output size
        let mut output_shape = batch_shape.to_vec();
        output_shape.push(self.output_size);

        if output_shape == output_with_bias.shape() {
            Ok(output_with_bias)
        } else {
            output_with_bias.reshape(&output_shape)
        }
    }

    /// Compute gradients for weights and bias
    fn compute_gradients(&mut self, input: &Tensor, grad_output: &Tensor) -> Result<()> {
        // Validate shapes
        if input.shape().len() != grad_output.shape().len() {
            return Err(NnlError::tensor(
                "Input and grad_output must have same number of dimensions",
            ));
        }

        let batch_shape = &input.shape()[..input.shape().len() - 1];
        let batch_size: usize = batch_shape.iter().product();

        // Reshape tensors for batch processing
        let input_2d = input.reshape(&[batch_size, self.input_size])?;
        let grad_output_2d = grad_output.reshape(&[batch_size, self.output_size])?;

        // Compute weight gradients: input^T @ grad_output
        let input_transposed = input_2d.transpose()?;
        let weight_grad = input_transposed.matmul(&grad_output_2d)?;

        // Accumulate weight gradients
        self.weight_grad = self.weight_grad.add(&weight_grad)?;

        // Compute bias gradients if needed
        if let Some(ref mut bias_grad) = self.bias_grad {
            // Sum gradients across batch dimension
            let bias_grad_sum = grad_output_2d.sum_axis(0)?;
            *bias_grad = bias_grad.add(&bias_grad_sum)?;
        }

        Ok(())
    }
}

impl Layer for DenseLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        // Check for NaN inputs
        let input_data = input.to_vec()?;
        if input_data.iter().any(|x| x.is_nan() || !x.is_finite()) {
            return Err(NnlError::tensor("Input contains NaN or infinite values"));
        }

        // Cache input for backward pass
        if self.training {
            self.cached_input = Some(input.clone_data()?);
        }

        // Linear transformation
        let linear_output = self.linear_forward(input)?;

        // Check for NaN in linear output
        let linear_data = linear_output.to_vec()?;
        if linear_data.iter().any(|x| x.is_nan() || !x.is_finite()) {
            return Err(NnlError::tensor(
                "Linear transformation produced NaN or infinite values",
            ));
        }

        // Cache pre-activation for backward pass
        if self.training {
            self.cached_pre_activation = Some(linear_output.clone_data()?);
        }

        // Apply activation function
        let result = linear_output.activation(self.activation)?;

        // Check for NaN in final output
        let result_data = result.to_vec()?;
        if result_data.iter().any(|x| x.is_nan() || !x.is_finite()) {
            return Err(NnlError::tensor(
                "Activation function produced NaN or infinite values",
            ));
        }

        Ok(result)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let input = self
            .cached_input
            .take()
            .ok_or_else(|| NnlError::training("No cached input for backward pass"))?;

        let pre_activation = self
            .cached_pre_activation
            .take()
            .ok_or_else(|| NnlError::training("No cached pre-activation for backward pass"))?;

        // Compute activation gradient
        let activation_grad = self.compute_activation_gradient(&pre_activation, grad_output)?;

        // Compute parameter gradients
        self.compute_gradients(&input, &activation_grad)?;

        // Compute input gradient for previous layer
        let batch_shape = &input.shape()[..input.shape().len() - 1];
        let batch_size: usize = batch_shape.iter().product();

        let activation_grad_2d = activation_grad.reshape(&[batch_size, self.output_size])?;
        let weights_transposed = self.weights.transpose()?;
        let input_grad_2d = activation_grad_2d.matmul(&weights_transposed)?;

        // Reshape back to original input shape
        let input_grad = input_grad_2d.reshape(input.shape())?;

        Ok(input_grad)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weights];
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = vec![&mut self.weights];
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn gradients(&self) -> Vec<&Tensor> {
        let mut grads = vec![&self.weight_grad];
        if let Some(ref bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        let mut grads = vec![&mut self.weight_grad];
        if let Some(ref mut bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn zero_grad(&mut self) {
        if let Err(e) = self.weight_grad.fill(0.0) {
            eprintln!("Warning: Failed to zero weight gradients: {}", e);
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            if let Err(e) = bias_grad.fill(0.0) {
                eprintln!("Warning: Failed to zero bias gradients: {}", e);
            }
        }
    }

    fn name(&self) -> &str {
        "Dense"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.is_empty() {
            return Err(NnlError::tensor("Input shape cannot be empty"));
        }

        let input_features = input_shape[input_shape.len() - 1];
        if input_features != self.input_size {
            return Err(NnlError::shape_mismatch(
                &[self.input_size],
                &[input_features],
            ));
        }

        let mut output_shape = input_shape.to_vec();
        let len = output_shape.len();
        output_shape[len - 1] = self.output_size;
        Ok(output_shape)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        self.weights = self.weights.to_device(device.clone())?;
        self.weight_grad = self.weight_grad.to_device(device.clone())?;

        if let Some(ref bias) = self.bias {
            self.bias = Some(bias.to_device(device.clone())?);
        }
        if let Some(ref bias_grad) = self.bias_grad {
            self.bias_grad = Some(bias_grad.to_device(device)?);
        }

        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        let mut cloned = DenseLayer::new_on_device(
            self.input_size,
            self.output_size,
            self.activation,
            self.use_bias,
            WeightInit::Zeros, // Will be overwritten
            self.weights.device().clone(),
        )?;

        // Copy weights and biases
        cloned.weights = self.weights.clone_data()?;
        if let Some(ref bias) = self.bias {
            cloned.bias = Some(bias.clone_data()?);
        }

        cloned.training = self.training;
        Ok(Box::new(cloned))
    }
}

impl DenseLayer {
    /// Compute gradient through activation function
    fn compute_activation_gradient(
        &self,
        pre_activation: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor> {
        match self.activation {
            Activation::Linear => {
                // Linear activation: gradient passes through unchanged
                grad_output.clone_data()
            }
            Activation::ReLU => {
                // ReLU gradient: 1 if x > 0, 0 otherwise
                let pre_act_data = pre_activation.to_vec()?;
                let grad_data = grad_output.to_vec()?;

                let result_data: Vec<f32> = pre_act_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&pre, &grad)| if pre > 0.0 { grad } else { 0.0 })
                    .collect();

                Tensor::from_slice_on_device(
                    &result_data,
                    grad_output.shape(),
                    grad_output.device().clone(),
                )
            }
            Activation::Sigmoid => {
                // Sigmoid gradient: sigmoid(x) * (1 - sigmoid(x)) * grad_output
                let sigmoid_output = pre_activation.activation(Activation::Sigmoid)?;
                let sigmoid_data = sigmoid_output.to_vec()?;
                let grad_data = grad_output.to_vec()?;

                let result_data: Vec<f32> = sigmoid_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&sig, &grad)| sig * (1.0 - sig) * grad)
                    .collect();

                Tensor::from_slice_on_device(
                    &result_data,
                    grad_output.shape(),
                    grad_output.device().clone(),
                )
            }
            Activation::Tanh => {
                // Tanh gradient: (1 - tanh²(x)) * grad_output
                let tanh_output = pre_activation.activation(Activation::Tanh)?;
                let tanh_data = tanh_output.to_vec()?;
                let grad_data = grad_output.to_vec()?;

                let result_data: Vec<f32> = tanh_data
                    .iter()
                    .zip(grad_data.iter())
                    .map(|(&tanh_val, &grad)| (1.0 - tanh_val * tanh_val) * grad)
                    .collect();

                Tensor::from_slice_on_device(
                    &result_data,
                    grad_output.shape(),
                    grad_output.device().clone(),
                )
            }
            _ => {
                // For other activations, use numerical differentiation as fallback
                self.numerical_activation_gradient(pre_activation, grad_output)
            }
        }
    }

    /// Numerical gradient computation for complex activations
    fn numerical_activation_gradient(
        &self,
        pre_activation: &Tensor,
        grad_output: &Tensor,
    ) -> Result<Tensor> {
        let h = 1e-5; // Small step for numerical differentiation
        let pre_act_data = pre_activation.to_vec()?;
        let grad_data = grad_output.to_vec()?;

        let mut result_data = Vec::with_capacity(pre_act_data.len());

        for (&x, &grad) in pre_act_data.iter().zip(grad_data.iter()) {
            let f_plus = self.activation.forward(x + h);
            let f_minus = self.activation.forward(x - h);
            let derivative = (f_plus - f_minus) / (2.0 * h);
            result_data.push(derivative * grad);
        }

        Tensor::from_slice_on_device(
            &result_data,
            grad_output.shape(),
            grad_output.device().clone(),
        )
    }
}

impl fmt::Display for DenseLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dense({} → {}, {}, bias={})",
            self.input_size, self.output_size, self.activation, self.use_bias
        )
    }
}

// Extension to Tensor for additional operations needed by DenseLayer
trait TensorExt {
    fn sum_axis(&self, axis: usize) -> Result<Tensor>;
}

impl TensorExt for Tensor {
    fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        if axis >= self.ndim() {
            return Err(NnlError::tensor("Axis out of bounds"));
        }

        let data = self.to_vec()?;
        let shape = self.shape();

        if axis == 0 && shape.len() == 2 {
            // Sum along batch dimension for 2D tensor
            let batch_size = shape[0];
            let feature_size = shape[1];
            let mut result = vec![0.0; feature_size];

            for i in 0..batch_size {
                for j in 0..feature_size {
                    result[j] += data[i * feature_size + j];
                }
            }

            Tensor::from_slice_on_device(&result, &[feature_size], self.device().clone())
        } else {
            Err(NnlError::unsupported(
                "Only axis=0 sum for 2D tensors is currently supported",
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::layers::WeightInit;
    use approx::assert_relative_eq;

    #[test]
    fn test_dense_layer_creation() {
        let layer = DenseLayer::new(784, 128, Activation::ReLU, true, WeightInit::Xavier).unwrap();

        assert_eq!(layer.input_size(), 784);
        assert_eq!(layer.output_size(), 128);
        assert_eq!(*layer.activation(), Activation::ReLU);
        assert!(layer.bias().is_some());
    }

    #[test]
    fn test_dense_layer_forward() {
        let mut layer = DenseLayer::new(3, 2, Activation::Linear, true, WeightInit::Ones).unwrap();

        let input = Tensor::from_slice(&[1.0, 2.0, 3.0], &[1, 3]).unwrap();
        let output = layer.forward(&input, TrainingMode::Inference).unwrap();

        assert_eq!(output.shape(), &[1, 2]);
        // With all weights = 1 and bias = 0, output should be [6, 6]
        let output_data = output.to_vec().unwrap();
        assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 6.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dense_layer_backward() {
        let mut layer = DenseLayer::new(2, 1, Activation::Linear, false, WeightInit::Ones).unwrap();

        let input = Tensor::from_slice(&[1.0, 2.0], &[1, 2]).unwrap();
        let _output = layer.forward(&input, TrainingMode::Training).unwrap();

        let grad_output = Tensor::from_slice(&[1.0], &[1, 1]).unwrap();
        let grad_input = layer.backward(&grad_output).unwrap();

        assert_eq!(grad_input.shape(), &[1, 2]);
        // With weights = [1, 1] and grad_output = [1], grad_input should be [1, 1]
        let grad_data = grad_input.to_vec().unwrap();
        assert_relative_eq!(grad_data[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(grad_data[1], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dense_layer_output_shape() {
        let layer =
            DenseLayer::new(784, 10, Activation::Softmax, true, WeightInit::Xavier).unwrap();

        let output_shape = layer.output_shape(&[32, 784]).unwrap();
        assert_eq!(output_shape, vec![32, 10]);

        let output_shape = layer.output_shape(&[784]).unwrap();
        assert_eq!(output_shape, vec![10]);
    }

    #[test]
    fn test_dense_layer_parameters() {
        let layer = DenseLayer::new(10, 5, Activation::ReLU, true, WeightInit::Zeros).unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weights + bias
        assert_eq!(params[0].shape(), &[10, 5]); // weights
        assert_eq!(params[1].shape(), &[5]); // bias

        assert_eq!(layer.num_parameters(), 10 * 5 + 5);
    }

    #[test]
    fn test_dense_layer_without_bias() {
        let layer = DenseLayer::new(5, 3, Activation::ReLU, false, WeightInit::Zeros).unwrap();

        assert!(layer.bias().is_none());
        assert_eq!(layer.parameters().len(), 1); // only weights
        assert_eq!(layer.num_parameters(), 5 * 3);
    }

    #[test]
    fn test_dense_layer_gradients() {
        let mut layer = DenseLayer::new(2, 1, Activation::Linear, true, WeightInit::Ones).unwrap();

        // Forward pass
        let input = Tensor::from_slice(&[1.0, 2.0], &[1, 2]).unwrap();
        let _output = layer.forward(&input, TrainingMode::Training).unwrap();

        // Backward pass
        let grad_output = Tensor::from_slice(&[1.0], &[1, 1]).unwrap();
        let _grad_input = layer.backward(&grad_output).unwrap();

        // Check gradients
        let weight_grad = layer.weight_gradients();
        let bias_grad = layer.bias_gradients().unwrap();

        assert_eq!(weight_grad.shape(), &[2, 1]);
        assert_eq!(bias_grad.shape(), &[1]);

        // Weight gradients should be input values
        let weight_grad_data = weight_grad.to_vec().unwrap();
        assert_relative_eq!(weight_grad_data[0], 1.0, epsilon = 1e-5);
        assert_relative_eq!(weight_grad_data[1], 2.0, epsilon = 1e-5);

        // Bias gradient should be grad_output
        let bias_grad_data = bias_grad.to_vec().unwrap();
        assert_relative_eq!(bias_grad_data[0], 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dense_layer_zero_grad() {
        let mut layer = DenseLayer::new(2, 1, Activation::Linear, true, WeightInit::Ones).unwrap();

        // Forward and backward to populate gradients
        let input = Tensor::from_slice(&[1.0, 2.0], &[1, 2]).unwrap();
        let _output = layer.forward(&input, TrainingMode::Training).unwrap();
        let grad_output = Tensor::from_slice(&[1.0], &[1, 1]).unwrap();
        let _grad_input = layer.backward(&grad_output).unwrap();

        // Zero gradients
        layer.zero_grad();

        // Check that gradients are zero
        let weight_grad_data = layer.weight_gradients().to_vec().unwrap();
        let bias_grad_data = layer.bias_gradients().unwrap().to_vec().unwrap();

        assert!(weight_grad_data.iter().all(|&x| x == 0.0));
        assert!(bias_grad_data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_dense_layer_activation_gradients() {
        let mut layer =
            DenseLayer::new(1, 1, Activation::ReLU, false, WeightInit::Constant(1.0)).unwrap();

        // Test positive input (should pass gradient through)
        let input_pos = Tensor::from_slice(&[2.0], &[1, 1]).unwrap();
        let _output_pos = layer.forward(&input_pos, TrainingMode::Training).unwrap();
        let grad_output = Tensor::from_slice(&[1.0], &[1, 1]).unwrap();
        let grad_input_pos = layer.backward(&grad_output).unwrap();

        let grad_data_pos = grad_input_pos.to_vec().unwrap();
        assert_relative_eq!(grad_data_pos[0], 1.0, epsilon = 1e-5);

        // Test negative input (should block gradient)
        let input_neg = Tensor::from_slice(&[-2.0], &[1, 1]).unwrap();
        let _output_neg = layer.forward(&input_neg, TrainingMode::Training).unwrap();
        let grad_input_neg = layer.backward(&grad_output).unwrap();

        let grad_data_neg = grad_input_neg.to_vec().unwrap();
        assert_relative_eq!(grad_data_neg[0], 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_dense_layer_batch_processing() {
        let mut layer = DenseLayer::new(3, 2, Activation::Linear, false, WeightInit::Ones).unwrap();

        // Batch of 2 samples
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let output = layer.forward(&input, TrainingMode::Inference).unwrap();

        assert_eq!(output.shape(), &[2, 2]);
        let output_data = output.to_vec().unwrap();

        // First sample: [1, 2, 3] * [[1, 1], [1, 1], [1, 1]] = [6, 6]
        assert_relative_eq!(output_data[0], 6.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[1], 6.0, epsilon = 1e-5);

        // Second sample: [4, 5, 6] * [[1, 1], [1, 1], [1, 1]] = [15, 15]
        assert_relative_eq!(output_data[2], 15.0, epsilon = 1e-5);
        assert_relative_eq!(output_data[3], 15.0, epsilon = 1e-5);
    }
}
