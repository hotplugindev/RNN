//! Convolutional layer implementations
//!
//! This module provides 2D convolutional layers for building CNNs with
//! forward and backward pass implementations.

use crate::activations::Activation;
use crate::device::Device;
use crate::error::{Result, RnnError};
use crate::layers::{Layer, TrainingMode, WeightInit};
use crate::tensor::Tensor;
use std::fmt;

/// 2D Convolutional layer
#[derive(Debug)]
pub struct Conv2DLayer {
    /// Convolution weights [out_channels, in_channels, kernel_height, kernel_width]
    weights: Tensor,
    /// Bias vector [out_channels] (optional)
    bias: Option<Tensor>,
    /// Weight gradients
    weight_grad: Tensor,
    /// Bias gradients (optional)
    bias_grad: Option<Tensor>,
    /// Input channels
    in_channels: usize,
    /// Output channels
    out_channels: usize,
    /// Kernel size (height, width)
    kernel_size: (usize, usize),
    /// Stride (height, width)
    stride: (usize, usize),
    /// Padding (height, width)
    padding: (usize, usize),
    /// Dilation (height, width)
    dilation: (usize, usize),
    /// Activation function
    activation: Activation,
    /// Whether to use bias
    use_bias: bool,
    /// Cached input for backward pass
    cached_input: Option<Tensor>,
    /// Cached pre-activation for backward pass
    cached_pre_activation: Option<Tensor>,
    /// Training mode
    training: bool,
}

impl Conv2DLayer {
    /// Create a new Conv2D layer
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
    ) -> Result<Self> {
        let device = Device::auto_select()?;
        Self::new_on_device(
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
        )
    }

    /// Create a new Conv2D layer on specific device
    pub fn new_on_device(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
        device: Device,
    ) -> Result<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(RnnError::config("Channel counts must be positive"));
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(RnnError::config("Kernel size must be positive"));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(RnnError::config("Stride must be positive"));
        }
        if dilation.0 == 0 || dilation.1 == 0 {
            return Err(RnnError::config("Dilation must be positive"));
        }

        // Initialize weights [out_channels, in_channels, kernel_height, kernel_width]
        let weight_shape = [out_channels, in_channels, kernel_size.0, kernel_size.1];
        let mut weights = Tensor::zeros_on_device(&weight_shape, device.clone())?;
        let fan_in = in_channels * kernel_size.0 * kernel_size.1;
        let fan_out = out_channels * kernel_size.0 * kernel_size.1;
        weight_init.initialize(&mut weights, fan_in, fan_out)?;

        // Initialize weight gradients
        let weight_grad = Tensor::zeros_on_device(&weight_shape, device.clone())?;

        // Initialize bias if needed
        let (bias, bias_grad) = if use_bias {
            let mut bias_tensor = Tensor::zeros_on_device(&[out_channels], device.clone())?;
            WeightInit::Zeros.initialize(&mut bias_tensor, 1, out_channels)?;
            let bias_grad_tensor = Tensor::zeros_on_device(&[out_channels], device)?;
            (Some(bias_tensor), Some(bias_grad_tensor))
        } else {
            (None, None)
        };

        Ok(Self {
            weights,
            bias,
            weight_grad,
            bias_grad,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            activation,
            use_bias,
            cached_input: None,
            cached_pre_activation: None,
            training: true,
        })
    }

    /// Get input channels
    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    /// Get output channels
    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    /// Get kernel size
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Get stride
    pub fn stride(&self) -> (usize, usize) {
        self.stride
    }

    /// Get padding
    pub fn padding(&self) -> (usize, usize) {
        self.padding
    }

    /// Calculate output dimensions
    fn calculate_output_size(&self, input_height: usize, input_width: usize) -> (usize, usize) {
        let output_height =
            (input_height + 2 * self.padding.0 - self.dilation.0 * (self.kernel_size.0 - 1) - 1)
                / self.stride.0
                + 1;
        let output_width =
            (input_width + 2 * self.padding.1 - self.dilation.1 * (self.kernel_size.1 - 1) - 1)
                / self.stride.1
                + 1;
        (output_height, output_width)
    }

    /// Perform 2D convolution (simplified implementation)
    fn conv2d_forward(&self, input: &Tensor) -> Result<Tensor> {
        // This is a simplified placeholder implementation
        // In a real implementation, this would use optimized convolution algorithms
        // such as im2col + GEMM or direct convolution with proper parallelization

        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(RnnError::tensor(
                "Expected 4D input [batch, channels, height, width]",
            ));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        if in_channels != self.in_channels {
            return Err(RnnError::shape_mismatch(
                &[self.in_channels],
                &[in_channels],
            ));
        }

        let (output_height, output_width) = self.calculate_output_size(input_height, input_width);
        let output_shape = [batch_size, self.out_channels, output_height, output_width];

        // For now, return a tensor filled with ones as a placeholder
        // Real implementation would perform actual convolution
        let output_size = output_shape.iter().product::<usize>();
        let output_data = vec![1.0; output_size];
        let mut output =
            Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())?;

        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias across spatial dimensions
            // This is also simplified - real implementation would be more efficient
            let bias_data = bias.to_vec()?;
            let mut output_data = output.to_vec()?;

            for batch in 0..batch_size {
                for out_ch in 0..self.out_channels {
                    let bias_val = bias_data[out_ch];
                    for h in 0..output_height {
                        for w in 0..output_width {
                            let idx = batch * (self.out_channels * output_height * output_width)
                                + out_ch * (output_height * output_width)
                                + h * output_width
                                + w;
                            output_data[idx] += bias_val;
                        }
                    }
                }
            }

            output =
                Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())?;
        }

        Ok(output)
    }
}

impl Layer for Conv2DLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.training = matches!(training, TrainingMode::Training);

        // Cache input for backward pass
        if self.training {
            self.cached_input = Some(input.clone_data()?);
        }

        // Convolution
        let conv_output = self.conv2d_forward(input)?;

        // Cache pre-activation for backward pass
        if self.training {
            self.cached_pre_activation = Some(conv_output.clone_data()?);
        }

        // Apply activation
        conv_output.activation(self.activation)
    }

    fn backward(&mut self, _grad_output: &Tensor) -> Result<Tensor> {
        // This is a simplified placeholder implementation
        // Real implementation would compute proper gradients for convolution

        let input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| RnnError::training("No cached input for backward pass"))?;

        // For now, just return gradients with same shape as input
        let grad_input = Tensor::zeros_on_device(input.shape(), input.device().clone())?;
        Ok(grad_input)
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
        "Conv2D"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(RnnError::tensor(
                "Expected 4D input [batch, channels, height, width]",
            ));
        }

        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_height = input_shape[2];
        let input_width = input_shape[3];

        if in_channels != self.in_channels {
            return Err(RnnError::shape_mismatch(
                &[self.in_channels],
                &[in_channels],
            ));
        }

        let (output_height, output_width) = self.calculate_output_size(input_height, input_width);
        Ok(vec![
            batch_size,
            self.out_channels,
            output_height,
            output_width,
        ])
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
        let mut cloned = Conv2DLayer::new(
            self.in_channels,
            self.out_channels,
            self.kernel_size,
            self.stride,
            self.padding,
            self.dilation,
            self.activation,
            self.use_bias,
            WeightInit::Zeros, // Will be overwritten
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

impl fmt::Display for Conv2DLayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Conv2D({} → {}, kernel={}×{}, stride={}×{}, padding={}×{}, {})",
            self.in_channels,
            self.out_channels,
            self.kernel_size.0,
            self.kernel_size.1,
            self.stride.0,
            self.stride.1,
            self.padding.0,
            self.padding.1,
            self.activation
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::layers::WeightInit;

    #[test]
    fn test_conv2d_creation() {
        let layer = Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier,
        );
        assert!(layer.is_ok());

        let layer = layer.unwrap();
        assert_eq!(layer.in_channels(), 3);
        assert_eq!(layer.out_channels(), 64);
        assert_eq!(layer.kernel_size(), (3, 3));
        assert_eq!(layer.stride(), (1, 1));
        assert_eq!(layer.padding(), (1, 1));
    }

    #[test]
    fn test_conv2d_output_shape() {
        let layer = Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier,
        )
        .unwrap();

        let input_shape = vec![1, 3, 32, 32]; // Batch=1, Channels=3, 32x32 image
        let output_shape = layer.output_shape(&input_shape).unwrap();
        assert_eq!(output_shape, vec![1, 64, 32, 32]); // Same size due to padding
    }

    #[test]
    fn test_conv2d_forward() {
        let mut layer = Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier,
        )
        .unwrap();

        let input = Tensor::randn(&[1, 3, 32, 32]).unwrap();
        let output = layer.forward(&input, TrainingMode::Inference);
        assert!(output.is_ok());

        let output = output.unwrap();
        assert_eq!(output.shape(), &[1, 64, 32, 32]);
    }

    #[test]
    fn test_conv2d_parameters() {
        let layer = Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier,
        )
        .unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 2); // weights + bias

        // Check weight shape: [out_channels, in_channels, kernel_h, kernel_w]
        assert_eq!(params[0].shape(), &[64, 3, 3, 3]);
        // Check bias shape: [out_channels]
        assert_eq!(params[1].shape(), &[64]);

        // Total parameters: 64 * 3 * 3 * 3 + 64 = 1728 + 64 = 1792
        assert_eq!(layer.num_parameters(), 64 * 3 * 3 * 3 + 64);
    }

    #[test]
    fn test_conv2d_without_bias() {
        let layer = Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            false,
            WeightInit::Xavier,
        )
        .unwrap();

        let params = layer.parameters();
        assert_eq!(params.len(), 1); // only weights
        assert_eq!(layer.num_parameters(), 64 * 3 * 3 * 3);
    }

    #[test]
    fn test_conv2d_invalid_parameters() {
        // Zero channels
        assert!(Conv2DLayer::new(
            0,
            64,
            (3, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier
        )
        .is_err());

        // Zero kernel size
        assert!(Conv2DLayer::new(
            3,
            64,
            (0, 3),
            (1, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier
        )
        .is_err());

        // Zero stride
        assert!(Conv2DLayer::new(
            3,
            64,
            (3, 3),
            (0, 1),
            (1, 1),
            (1, 1),
            Activation::ReLU,
            true,
            WeightInit::Xavier
        )
        .is_err());
    }
}
