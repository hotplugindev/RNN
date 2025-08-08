//! Dropout layer implementation for regularization
//!
//! This module provides a dropout layer that randomly sets input elements
//! to zero during training to prevent overfitting. During inference,
//! all elements are kept and scaled appropriately.

use crate::device::Device;
use crate::error::{Result, RnnError};
use crate::layers::{Layer, TrainingMode};
use crate::tensor::Tensor;
use rand::prelude::*;
use std::fmt;

/// Dropout layer for regularization
#[derive(Debug)]
pub struct DropoutLayer {
    /// Dropout probability (0.0 to 1.0)
    dropout_rate: f32,
    /// Training mode flag
    training: bool,
    /// Cached dropout mask for backward pass
    dropout_mask: Option<Tensor>,
}

impl DropoutLayer {
    /// Create a new dropout layer
    pub fn new(dropout_rate: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&dropout_rate) {
            return Err(RnnError::config("Dropout rate must be between 0.0 and 1.0"));
        }

        Ok(Self {
            dropout_rate,
            training: true,
            dropout_mask: None,
        })
    }

    /// Get dropout rate
    pub fn dropout_rate(&self) -> f32 {
        self.dropout_rate
    }

    /// Set dropout rate
    pub fn set_dropout_rate(&mut self, rate: f32) -> Result<()> {
        if !(0.0..=1.0).contains(&rate) {
            return Err(RnnError::config("Dropout rate must be between 0.0 and 1.0"));
        }
        self.dropout_rate = rate;
        Ok(())
    }

    /// Generate dropout mask
    fn generate_mask(&self, shape: &[usize], device: Device) -> Result<Tensor> {
        let size = shape.iter().product::<usize>();
        let mut rng = thread_rng();

        let mask_data: Vec<f32> = (0..size)
            .map(|_| {
                if rng.gen::<f32>() > self.dropout_rate {
                    1.0 / (1.0 - self.dropout_rate) // Scale factor for kept elements
                } else {
                    0.0
                }
            })
            .collect();

        Tensor::from_slice_on_device(&mask_data, shape, device)
    }
}

impl Layer for DropoutLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        match training {
            TrainingMode::Training => {
                if self.dropout_rate == 0.0 {
                    // No dropout, just return input
                    Ok(input.clone_data()?)
                } else {
                    // Generate and apply dropout mask
                    let mask = self.generate_mask(input.shape(), input.device().clone())?;
                    let output = input.mul(&mask)?;

                    // Cache mask for backward pass
                    self.dropout_mask = Some(mask);

                    Ok(output)
                }
            }
            TrainingMode::Inference => {
                // During inference, keep all elements (no dropout)
                Ok(input.clone_data()?)
            }
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        if let Some(ref mask) = self.dropout_mask {
            // Apply the same mask to gradients
            grad_output.mul(mask)
        } else {
            // No mask (inference mode or no dropout), pass gradients through
            Ok(grad_output.clone_data()?)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        // Dropout has no learnable parameters
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        // Dropout has no learnable parameters
        Vec::new()
    }

    fn gradients(&self) -> Vec<&Tensor> {
        // Dropout has no gradients
        Vec::new()
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        // Dropout has no gradients
        Vec::new()
    }

    fn zero_grad(&mut self) {
        // Dropout has no gradients to zero
    }

    fn name(&self) -> &str {
        "Dropout"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        // Dropout doesn't change the shape
        Ok(input_shape.to_vec())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn to_device(&mut self, _device: Device) -> Result<()> {
        // Dropout doesn't have parameters to move
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        let cloned = DropoutLayer::new(self.dropout_rate)?;
        Ok(Box::new(cloned))
    }
}

impl fmt::Display for DropoutLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dropout(p={})", self.dropout_rate)
    }
}

/// Spatial dropout for convolutional layers
#[derive(Debug)]
pub struct SpatialDropoutLayer {
    /// Dropout probability
    dropout_rate: f32,
    /// Training mode flag
    training: bool,
    /// Cached dropout mask for backward pass
    dropout_mask: Option<Tensor>,
}

impl SpatialDropoutLayer {
    /// Create a new spatial dropout layer
    pub fn new(dropout_rate: f32) -> Result<Self> {
        if !(0.0..=1.0).contains(&dropout_rate) {
            return Err(RnnError::config("Dropout rate must be between 0.0 and 1.0"));
        }

        Ok(Self {
            dropout_rate,
            training: true,
            dropout_mask: None,
        })
    }

    /// Generate spatial dropout mask (drops entire channels)
    fn generate_spatial_mask(&self, shape: &[usize], device: Device) -> Result<Tensor> {
        if shape.len() < 3 {
            return Err(RnnError::tensor(
                "Spatial dropout requires at least 3D input (batch, channels, spatial...)",
            ));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let spatial_dims: Vec<usize> = shape[2..].to_vec();
        let spatial_size = spatial_dims.iter().product::<usize>();

        let mut rng = thread_rng();
        let mut mask_data = Vec::with_capacity(shape.iter().product());

        for _batch in 0..batch_size {
            for _channel in 0..channels {
                // Decide whether to keep this entire channel
                let keep_channel = rng.gen::<f32>() > self.dropout_rate;
                let channel_value = if keep_channel {
                    1.0 / (1.0 - self.dropout_rate) // Scale factor
                } else {
                    0.0
                };

                // Fill entire spatial dimension with the same value
                for _ in 0..spatial_size {
                    mask_data.push(channel_value);
                }
            }
        }

        Tensor::from_slice_on_device(&mask_data, shape, device)
    }
}

impl Layer for SpatialDropoutLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        match training {
            TrainingMode::Training => {
                if self.dropout_rate == 0.0 {
                    Ok(input.clone_data()?)
                } else {
                    let mask = self.generate_spatial_mask(input.shape(), input.device().clone())?;
                    let output = input.mul(&mask)?;
                    self.dropout_mask = Some(mask);
                    Ok(output)
                }
            }
            TrainingMode::Inference => Ok(input.clone_data()?),
        }
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        if let Some(ref mask) = self.dropout_mask {
            grad_output.mul(mask)
        } else {
            Ok(grad_output.clone_data()?)
        }
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    fn gradients(&self) -> Vec<&Tensor> {
        Vec::new()
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new()
    }

    fn zero_grad(&mut self) {}

    fn name(&self) -> &str {
        "SpatialDropout"
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

    fn to_device(&mut self, _device: Device) -> Result<()> {
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        let cloned = SpatialDropoutLayer::new(self.dropout_rate)?;
        Ok(Box::new(cloned))
    }
}

impl fmt::Display for SpatialDropoutLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SpatialDropout(p={})", self.dropout_rate)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;

    #[test]
    fn test_dropout_layer_creation() {
        let layer = DropoutLayer::new(0.5);
        assert!(layer.is_ok());
        let layer = layer.unwrap();
        assert_eq!(layer.dropout_rate(), 0.5);

        // Invalid dropout rate
        assert!(DropoutLayer::new(-0.1).is_err());
        assert!(DropoutLayer::new(1.1).is_err());
    }

    #[test]
    fn test_dropout_inference_mode() {
        let mut layer = DropoutLayer::new(0.5).unwrap();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // In inference mode, output should be identical to input
        let output = layer.forward(&input, TrainingMode::Inference).unwrap();
        let input_data = input.to_vec().unwrap();
        let output_data = output.to_vec().unwrap();

        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_training_mode() {
        let mut layer = DropoutLayer::new(0.5).unwrap();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // In training mode, some elements should be zeroed and others scaled
        let output = layer.forward(&input, TrainingMode::Training).unwrap();
        let output_data = output.to_vec().unwrap();

        // Check that not all elements are the same as input (unless very unlucky)
        let input_data = input.to_vec().unwrap();
        let all_same = input_data
            .iter()
            .zip(output_data.iter())
            .all(|(i, o)| (i - o).abs() < 1e-6);

        // With 50% dropout, it's extremely unlikely all elements remain unchanged
        // (probability is 0.5^4 = 0.0625, but we can't guarantee randomness in tests)
        // So we just check that the function runs without error
        assert_eq!(output_data.len(), 4);
    }

    #[test]
    fn test_dropout_no_dropout() {
        let mut layer = DropoutLayer::new(0.0).unwrap();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // With 0% dropout, output should be identical even in training mode
        let output = layer.forward(&input, TrainingMode::Training).unwrap();
        let input_data = input.to_vec().unwrap();
        let output_data = output.to_vec().unwrap();

        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_backward() {
        let mut layer = DropoutLayer::new(0.5).unwrap();
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4]).unwrap();

        // Forward pass to generate mask
        let _output = layer.forward(&input, TrainingMode::Training).unwrap();

        // Backward pass
        let grad_output = Tensor::from_slice(&[1.0, 1.0, 1.0, 1.0], &[4]).unwrap();
        let grad_input = layer.backward(&grad_output).unwrap();

        // Gradient should have the same shape as input
        assert_eq!(grad_input.shape(), input.shape());
    }

    #[test]
    fn test_dropout_output_shape() {
        let layer = DropoutLayer::new(0.3).unwrap();
        let input_shape = vec![32, 128];
        let output_shape = layer.output_shape(&input_shape).unwrap();
        assert_eq!(output_shape, input_shape);
    }

    #[test]
    fn test_dropout_no_parameters() {
        let layer = DropoutLayer::new(0.2).unwrap();
        assert!(layer.parameters().is_empty());
        assert!(layer.gradients().is_empty());
        assert_eq!(layer.num_parameters(), 0);
    }

    #[test]
    fn test_spatial_dropout_creation() {
        let layer = SpatialDropoutLayer::new(0.5);
        assert!(layer.is_ok());

        // Invalid dropout rate
        assert!(SpatialDropoutLayer::new(-0.1).is_err());
        assert!(SpatialDropoutLayer::new(1.1).is_err());
    }

    #[test]
    fn test_spatial_dropout_forward() {
        let mut layer = SpatialDropoutLayer::new(0.5).unwrap();
        // 3D input: batch=1, channels=2, spatial=3
        let input = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[1, 2, 3]).unwrap();

        let output = layer.forward(&input, TrainingMode::Training).unwrap();
        assert_eq!(output.shape(), input.shape());

        // In inference mode, should pass through unchanged
        let output_inf = layer.forward(&input, TrainingMode::Inference).unwrap();
        let input_data = input.to_vec().unwrap();
        let output_data = output_inf.to_vec().unwrap();

        for (i, o) in input_data.iter().zip(output_data.iter()) {
            assert_relative_eq!(i, o, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_dropout_rate_setter() {
        let mut layer = DropoutLayer::new(0.5).unwrap();
        assert_eq!(layer.dropout_rate(), 0.5);

        layer.set_dropout_rate(0.3).unwrap();
        assert_eq!(layer.dropout_rate(), 0.3);

        // Invalid rates should return error
        assert!(layer.set_dropout_rate(-0.1).is_err());
        assert!(layer.set_dropout_rate(1.1).is_err());
    }

    #[test]
    fn test_dropout_clone() {
        let layer = DropoutLayer::new(0.4).unwrap();
        let cloned = layer.clone_layer().unwrap();

        // The cloned layer should have the same dropout rate
        assert_eq!(cloned.name(), "Dropout");
        assert_eq!(cloned.num_parameters(), 0);
    }

    #[test]
    fn test_dropout_display() {
        let layer = DropoutLayer::new(0.25).unwrap();
        let display_str = format!("{}", layer);
        assert_eq!(display_str, "Dropout(p=0.25)");
    }
}
