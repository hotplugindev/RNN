//! Normalization layer implementations
//!
//! This module provides batch normalization and layer normalization
//! implementations for neural networks.

use crate::device::Device;
use crate::error::{NnlError, Result};
use crate::layers::{Layer, TrainingMode};
use crate::tensor::Tensor;
use std::fmt;

/// Batch Normalization layer
#[derive(Debug)]
pub struct BatchNormLayer {
    /// Number of features
    num_features: usize,
    /// Small constant for numerical stability
    eps: f32,
    /// Momentum for running statistics
    momentum: f32,
    /// Whether to use learnable affine parameters
    affine: bool,
    /// Learnable scale parameter (gamma)
    weight: Option<Tensor>,
    /// Learnable shift parameter (beta)
    bias: Option<Tensor>,
    /// Weight gradients
    weight_grad: Option<Tensor>,
    /// Bias gradients
    bias_grad: Option<Tensor>,
    /// Running mean
    running_mean: Tensor,
    /// Running variance
    running_var: Tensor,
    /// Training mode flag
    training: bool,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached normalized input
    cached_normalized: Option<Tensor>,
}

impl BatchNormLayer {
    /// Create a new batch normalization layer
    pub fn new(num_features: usize, eps: f32, momentum: f32, affine: bool) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::new_on_device(num_features, eps, momentum, affine, device)
    }

    /// Create a new batch normalization layer on specific device
    pub fn new_on_device(
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
        device: Device,
    ) -> Result<Self> {
        if num_features == 0 {
            return Err(NnlError::config("Number of features must be positive"));
        }
        if eps <= 0.0 {
            return Err(NnlError::config("Epsilon must be positive"));
        }
        if !(0.0..=1.0).contains(&momentum) {
            return Err(NnlError::config("Momentum must be between 0 and 1"));
        }

        // Initialize running statistics
        let running_mean = Tensor::zeros_on_device(&[num_features], device.clone())?;
        let running_var = Tensor::ones_on_device(&[num_features], device.clone())?;

        // Initialize learnable parameters if affine
        let (weight, bias, weight_grad, bias_grad) = if affine {
            let weight = Tensor::ones_on_device(&[num_features], device.clone())?;
            let bias = Tensor::zeros_on_device(&[num_features], device.clone())?;
            let weight_grad = Tensor::zeros_on_device(&[num_features], device.clone())?;
            let bias_grad = Tensor::zeros_on_device(&[num_features], device)?;
            (Some(weight), Some(bias), Some(weight_grad), Some(bias_grad))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            weight,
            bias,
            weight_grad,
            bias_grad,
            running_mean,
            running_var,
            training: true,
            cached_input: None,
            cached_normalized: None,
        })
    }

    /// Batch normalize the input
    fn batch_norm_forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let input_shape = input.shape();
        if input_shape.len() < 2 {
            return Err(NnlError::tensor("Input must have at least 2 dimensions"));
        }

        let batch_size = input_shape[0];
        let features = input_shape[1];
        if features != self.num_features {
            return Err(NnlError::shape_mismatch(&[self.num_features], &[features]));
        }

        if self.training {
            // Training mode: compute batch statistics
            let input_data = input.to_vec()?;
            let spatial_size = if input_shape.len() > 2 {
                input_shape[2..].iter().product::<usize>()
            } else {
                1
            };

            // Compute mean and variance per feature
            let mut batch_mean = vec![0.0; features];
            let mut batch_var = vec![0.0; features];
            let total_elements = batch_size * spatial_size;

            // Calculate mean
            for f in 0..features {
                let mut sum = 0.0;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * features * spatial_size + f * spatial_size + s;
                        sum += input_data[idx];
                    }
                }
                batch_mean[f] = sum / total_elements as f32;
            }

            // Calculate variance
            for f in 0..features {
                let mut sum_sq_diff = 0.0;
                for b in 0..batch_size {
                    for s in 0..spatial_size {
                        let idx = b * features * spatial_size + f * spatial_size + s;
                        let diff = input_data[idx] - batch_mean[f];
                        sum_sq_diff += diff * diff;
                    }
                }
                batch_var[f] = sum_sq_diff / total_elements as f32;
            }

            // Update running statistics
            let mut running_mean_data = self.running_mean.to_vec()?;
            let mut running_var_data = self.running_var.to_vec()?;
            for f in 0..features {
                running_mean_data[f] =
                    (1.0 - self.momentum) * running_mean_data[f] + self.momentum * batch_mean[f];
                running_var_data[f] =
                    (1.0 - self.momentum) * running_var_data[f] + self.momentum * batch_var[f];
            }
            self.running_mean.copy_from_slice(&running_mean_data)?;
            self.running_var.copy_from_slice(&running_var_data)?;

            // Normalize using batch statistics
            self.normalize_with_stats(input, &batch_mean, &batch_var)
        } else {
            // Inference mode: use running statistics
            let running_mean_data = self.running_mean.to_vec()?;
            let running_var_data = self.running_var.to_vec()?;
            self.normalize_with_stats(input, &running_mean_data, &running_var_data)
        }
    }

    /// Normalize input with given statistics
    fn normalize_with_stats(
        &mut self,
        input: &Tensor,
        mean: &[f32],
        variance: &[f32],
    ) -> Result<Tensor> {
        let input_data = input.to_vec()?;
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let features = input_shape[1];
        let spatial_size = if input_shape.len() > 2 {
            input_shape[2..].iter().product::<usize>()
        } else {
            1
        };

        let mut normalized_data = vec![0.0; input_data.len()];

        // Normalize
        for b in 0..batch_size {
            for f in 0..features {
                let std_dev = (variance[f] + self.eps).sqrt();
                for s in 0..spatial_size {
                    let idx = b * features * spatial_size + f * spatial_size + s;
                    normalized_data[idx] = (input_data[idx] - mean[f]) / std_dev;
                }
            }
        }

        let normalized =
            Tensor::from_slice_on_device(&normalized_data, input_shape, input.device().clone())?;

        // Cache normalized input for backward pass
        if self.training {
            self.cached_normalized = Some(normalized.clone_data()?);
        }

        // Apply affine transformation if enabled
        if let (Some(weight), Some(bias)) = (&self.weight, &self.bias) {
            let weight_data = weight.to_vec()?;
            let bias_data = bias.to_vec()?;
            let mut output_data = normalized_data;

            for b in 0..batch_size {
                for f in 0..features {
                    for s in 0..spatial_size {
                        let idx = b * features * spatial_size + f * spatial_size + s;
                        output_data[idx] = output_data[idx] * weight_data[f] + bias_data[f];
                    }
                }
            }

            Tensor::from_slice_on_device(&output_data, input_shape, input.device().clone())
        } else {
            Ok(normalized)
        }
    }
}

impl Layer for BatchNormLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        // Cache input for backward pass
        if self.training {
            self.cached_input = Some(input.clone_data()?);
        }

        self.batch_norm_forward(input)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        // Simplified backward pass - real implementation would compute proper gradients
        let _input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| NnlError::training("No cached input for backward pass"))?;

        // Return gradient with same shape as input
        Ok(grad_output.clone_data()?)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn gradients(&self) -> Vec<&Tensor> {
        let mut grads = Vec::new();
        if let Some(ref weight_grad) = self.weight_grad {
            grads.push(weight_grad);
        }
        if let Some(ref bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        let mut grads = Vec::new();
        if let Some(ref mut weight_grad) = self.weight_grad {
            grads.push(weight_grad);
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut weight_grad) = self.weight_grad {
            let _ = weight_grad.fill(0.0);
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            let _ = bias_grad.fill(0.0);
        }
    }

    fn name(&self) -> &str {
        "BatchNorm"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() < 2 {
            return Err(NnlError::tensor("Input must have at least 2 dimensions"));
        }
        if input_shape[1] != self.num_features {
            return Err(NnlError::shape_mismatch(
                &[self.num_features],
                &[input_shape[1]],
            ));
        }
        Ok(input_shape.to_vec())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        self.running_mean = self.running_mean.to_device(device.clone())?;
        self.running_var = self.running_var.to_device(device.clone())?;

        if let Some(ref weight) = self.weight {
            self.weight = Some(weight.to_device(device.clone())?);
        }
        if let Some(ref bias) = self.bias {
            self.bias = Some(bias.to_device(device.clone())?);
        }
        if let Some(ref weight_grad) = self.weight_grad {
            self.weight_grad = Some(weight_grad.to_device(device.clone())?);
        }
        if let Some(ref bias_grad) = self.bias_grad {
            self.bias_grad = Some(bias_grad.to_device(device)?);
        }

        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        let mut cloned =
            BatchNormLayer::new(self.num_features, self.eps, self.momentum, self.affine)?;

        // Copy parameters and running statistics
        cloned.running_mean = self.running_mean.clone_data()?;
        cloned.running_var = self.running_var.clone_data()?;

        if let Some(ref weight) = self.weight {
            cloned.weight = Some(weight.clone_data()?);
        }
        if let Some(ref bias) = self.bias {
            cloned.bias = Some(bias.clone_data()?);
        }

        cloned.training = self.training;
        Ok(Box::new(cloned))
    }
}

impl fmt::Display for BatchNormLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "BatchNorm({}, eps={}, momentum={}, affine={})",
            self.num_features, self.eps, self.momentum, self.affine
        )
    }
}

/// Layer Normalization layer
#[derive(Debug)]
pub struct LayerNormLayer {
    /// Normalized shape
    normalized_shape: Vec<usize>,
    /// Small constant for numerical stability
    eps: f32,
    /// Whether to use learnable affine parameters
    elementwise_affine: bool,
    /// Learnable scale parameter (gamma)
    weight: Option<Tensor>,
    /// Learnable shift parameter (beta)
    bias: Option<Tensor>,
    /// Weight gradients
    weight_grad: Option<Tensor>,
    /// Bias gradients
    bias_grad: Option<Tensor>,
    /// Training mode flag
    training: bool,
}

impl LayerNormLayer {
    /// Create a new layer normalization layer
    pub fn new(normalized_shape: Vec<usize>, eps: f32, elementwise_affine: bool) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::new_on_device(normalized_shape, eps, elementwise_affine, device)
    }

    /// Create a new layer normalization layer on specific device
    pub fn new_on_device(
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
        device: Device,
    ) -> Result<Self> {
        if normalized_shape.is_empty() {
            return Err(NnlError::config("Normalized shape cannot be empty"));
        }
        if eps <= 0.0 {
            return Err(NnlError::config("Epsilon must be positive"));
        }

        // Initialize learnable parameters if affine
        let (weight, bias, weight_grad, bias_grad) = if elementwise_affine {
            let weight = Tensor::ones_on_device(&normalized_shape, device.clone())?;
            let bias = Tensor::zeros_on_device(&normalized_shape, device.clone())?;
            let weight_grad = Tensor::zeros_on_device(&normalized_shape, device.clone())?;
            let bias_grad = Tensor::zeros_on_device(&normalized_shape, device)?;
            (Some(weight), Some(bias), Some(weight_grad), Some(bias_grad))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
            weight_grad,
            bias_grad,
            training: true,
        })
    }
}

impl Layer for LayerNormLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        // Simplified layer normalization - normalize last dimensions
        let _input_data = input.to_vec()?;
        let _input_shape = input.shape();

        // For now, just return input unchanged as placeholder
        // Real implementation would normalize over the specified dimensions
        Ok(input.clone_data()?)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        // Simplified backward pass
        Ok(grad_output.clone_data()?)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = Vec::new();
        if let Some(ref weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut params = Vec::new();
        if let Some(ref mut weight) = self.weight {
            params.push(weight);
        }
        if let Some(ref mut bias) = self.bias {
            params.push(bias);
        }
        params
    }

    fn gradients(&self) -> Vec<&Tensor> {
        let mut grads = Vec::new();
        if let Some(ref weight_grad) = self.weight_grad {
            grads.push(weight_grad);
        }
        if let Some(ref bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        let mut grads = Vec::new();
        if let Some(ref mut weight_grad) = self.weight_grad {
            grads.push(weight_grad);
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            grads.push(bias_grad);
        }
        grads
    }

    fn zero_grad(&mut self) {
        if let Some(ref mut weight_grad) = self.weight_grad {
            let _ = weight_grad.fill(0.0);
        }
        if let Some(ref mut bias_grad) = self.bias_grad {
            let _ = bias_grad.fill(0.0);
        }
    }

    fn name(&self) -> &str {
        "LayerNorm"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        Ok(input_shape.to_vec())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn to_device(&mut self, device: Device) -> Result<()> {
        if let Some(ref weight) = self.weight {
            self.weight = Some(weight.to_device(device.clone())?);
        }
        if let Some(ref bias) = self.bias {
            self.bias = Some(bias.to_device(device.clone())?);
        }
        if let Some(ref weight_grad) = self.weight_grad {
            self.weight_grad = Some(weight_grad.to_device(device.clone())?);
        }
        if let Some(ref bias_grad) = self.bias_grad {
            self.bias_grad = Some(bias_grad.to_device(device)?);
        }

        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        let mut cloned = LayerNormLayer::new(
            self.normalized_shape.clone(),
            self.eps,
            self.elementwise_affine,
        )?;

        if let Some(ref weight) = self.weight {
            cloned.weight = Some(weight.clone_data()?);
        }
        if let Some(ref bias) = self.bias {
            cloned.bias = Some(bias.clone_data()?);
        }

        cloned.training = self.training;
        Ok(Box::new(cloned))
    }
}

impl fmt::Display for LayerNormLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LayerNorm({:?}, eps={}, elementwise_affine={})",
            self.normalized_shape, self.eps, self.elementwise_affine
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_norm_creation() {
        let layer = BatchNormLayer::new(64, 1e-5, 0.1, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.num_parameters(), 128); // 64 weights + 64 biases
    }

    #[test]
    fn test_batch_norm_forward() {
        let mut layer = BatchNormLayer::new(3, 1e-5, 0.1, true).unwrap();
        let input = Tensor::randn(&[2, 3, 4, 4]).unwrap(); // Batch=2, Channels=3, 4x4 spatial

        let output = layer.forward(&input, TrainingMode::Training);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), input.shape());
    }

    #[test]
    fn test_layer_norm_creation() {
        let layer = LayerNormLayer::new(vec![128], 1e-5, true);
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.num_parameters(), 256); // 128 weights + 128 biases
    }

    #[test]
    fn test_layer_norm_forward() {
        let mut layer = LayerNormLayer::new(vec![10], 1e-5, true).unwrap();
        let input = Tensor::randn(&[32, 10]).unwrap();

        let output = layer.forward(&input, TrainingMode::Inference);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), input.shape());
    }
}
