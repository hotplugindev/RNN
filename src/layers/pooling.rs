//! Pooling layers for neural networks
//!
//! This module provides implementations for various pooling operations
//! commonly used in convolutional neural networks.

use crate::error::{NnlError, Result};
use crate::layers::{Layer, TrainingMode};
use crate::tensor::Tensor;
// Note: Serialization is not implemented for layers containing Tensor fields

/// 2D Max Pooling Layer
#[derive(Debug)]
pub struct MaxPool2DLayer {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    training: bool,
    cached_input: Option<Tensor>,
}

impl MaxPool2DLayer {
    /// Create a new MaxPool2D layer
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Result<Self> {
        Ok(Self {
            kernel_size,
            stride,
            padding,
            training: true,
            cached_input: None,
        })
    }

    /// Get the effective stride
    fn effective_stride(&self) -> (usize, usize) {
        self.stride.unwrap_or(self.kernel_size)
    }
}

impl Layer for MaxPool2DLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        if input.shape().len() != 4 {
            return Err(NnlError::tensor("MaxPool2D expects 4D input [N, C, H, W]"));
        }

        self.set_training(matches!(training, TrainingMode::Training));

        // Cache input for backward pass
        if self.training {
            self.cached_input = Some(input.clone_data()?);
        }

        // Proper max pooling implementation
        let output_shape = self.compute_output_shape(input.shape())?;
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![f32::NEG_INFINITY; output_size];

        let input_data = input.to_vec()?;
        let input_shape = input.shape();
        let (stride_h, stride_w) = self.effective_stride();
        let (kernel_h, kernel_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;

        for batch in 0..input_shape[0] {
            for channel in 0..input_shape[1] {
                for out_h in 0..output_shape[2] {
                    for out_w in 0..output_shape[3] {
                        let mut max_val = f32::NEG_INFINITY;

                        // Iterate over the pooling window
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let in_h = (out_h * stride_h + kh) as i32 - pad_h as i32;
                                let in_w = (out_w * stride_w + kw) as i32 - pad_w as i32;

                                // Check bounds
                                if in_h >= 0
                                    && in_w >= 0
                                    && (in_h as usize) < input_shape[2]
                                    && (in_w as usize) < input_shape[3]
                                {
                                    let input_idx =
                                        batch * input_shape[1] * input_shape[2] * input_shape[3]
                                            + channel * input_shape[2] * input_shape[3]
                                            + (in_h as usize) * input_shape[3]
                                            + (in_w as usize);

                                    if input_idx < input_data.len() {
                                        max_val = max_val.max(input_data[input_idx]);
                                    }
                                }
                            }
                        }

                        let output_idx =
                            batch * output_shape[1] * output_shape[2] * output_shape[3]
                                + channel * output_shape[2] * output_shape[3]
                                + out_h * output_shape[3]
                                + out_w;

                        if output_idx < output_data.len() {
                            output_data[output_idx] = if max_val == f32::NEG_INFINITY {
                                0.0
                            } else {
                                max_val
                            };
                        }
                    }
                }
            }
        }

        // Create tensor from the computed data on the same device as input
        let output =
            Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())?;
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| NnlError::training("No cached input for backward pass"))?;

        // Proper MaxPool2D backward pass
        let input_data = input.to_vec()?;
        let input_shape = input.shape();
        let grad_output_data = grad_output.to_vec()?;
        let output_shape = grad_output.shape();

        let mut grad_input_data = vec![0.0; input_data.len()];
        let (stride_h, stride_w) = self.effective_stride();
        let (kernel_h, kernel_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;

        for batch in 0..input_shape[0] {
            for channel in 0..input_shape[1] {
                for out_h in 0..output_shape[2] {
                    for out_w in 0..output_shape[3] {
                        let mut max_val = f32::NEG_INFINITY;
                        let mut max_h = 0;
                        let mut max_w = 0;

                        // Find the position of the maximum value in the pooling window
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let in_h = (out_h * stride_h + kh) as i32 - pad_h as i32;
                                let in_w = (out_w * stride_w + kw) as i32 - pad_w as i32;

                                if in_h >= 0
                                    && in_w >= 0
                                    && (in_h as usize) < input_shape[2]
                                    && (in_w as usize) < input_shape[3]
                                {
                                    let input_idx =
                                        batch * input_shape[1] * input_shape[2] * input_shape[3]
                                            + channel * input_shape[2] * input_shape[3]
                                            + (in_h as usize) * input_shape[3]
                                            + (in_w as usize);

                                    if input_idx < input_data.len()
                                        && input_data[input_idx] > max_val
                                    {
                                        max_val = input_data[input_idx];
                                        max_h = in_h as usize;
                                        max_w = in_w as usize;
                                    }
                                }
                            }
                        }

                        // Propagate gradient only to the max position
                        if max_val != f32::NEG_INFINITY {
                            let grad_output_idx =
                                batch * output_shape[1] * output_shape[2] * output_shape[3]
                                    + channel * output_shape[2] * output_shape[3]
                                    + out_h * output_shape[3]
                                    + out_w;

                            let grad_input_idx =
                                batch * input_shape[1] * input_shape[2] * input_shape[3]
                                    + channel * input_shape[2] * input_shape[3]
                                    + max_h * input_shape[3]
                                    + max_w;

                            if grad_output_idx < grad_output_data.len()
                                && grad_input_idx < grad_input_data.len()
                            {
                                grad_input_data[grad_input_idx] +=
                                    grad_output_data[grad_output_idx];
                            }
                        }
                    }
                }
            }
        }

        let grad_input =
            Tensor::from_slice_on_device(&grad_input_data, input_shape, input.device().clone())?;
        Ok(grad_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new() // Pooling layers have no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Pooling layers have no parameters
    }

    fn gradients(&self) -> Vec<&Tensor> {
        Vec::new() // Pooling layers have no gradients
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Pooling layers have no gradients
    }

    fn zero_grad(&mut self) {
        // No gradients to zero for pooling layers
    }

    fn name(&self) -> &str {
        "MaxPool2D"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        self.compute_output_shape(input_shape)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn num_parameters(&self) -> usize {
        0 // Pooling layers have no parameters
    }

    fn to_device(&mut self, _device: crate::device::Device) -> Result<()> {
        // Pooling layers don't have parameters to move
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        Ok(Box::new(Self {
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            training: self.training,
            cached_input: None, // Don't clone cached data
        }))
    }
}

impl MaxPool2DLayer {
    /// Compute output shape after pooling
    fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(NnlError::tensor("Expected 4D input shape"));
        }

        let [batch_size, channels, height, width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];

        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.effective_stride();
        let (pad_h, pad_w) = self.padding;

        let output_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let output_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        Ok(vec![batch_size, channels, output_height, output_width])
    }
}

/// 2D Average Pooling Layer
#[derive(Debug)]
pub struct AvgPool2DLayer {
    kernel_size: (usize, usize),
    stride: Option<(usize, usize)>,
    padding: (usize, usize),
    training: bool,
    cached_input: Option<Tensor>,
}

impl AvgPool2DLayer {
    /// Create a new AvgPool2D layer
    pub fn new(
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    ) -> Result<Self> {
        Ok(Self {
            kernel_size,
            stride,
            padding,
            training: true,
            cached_input: None,
        })
    }

    /// Get the effective stride
    fn effective_stride(&self) -> (usize, usize) {
        self.stride.unwrap_or(self.kernel_size)
    }
}

impl Layer for AvgPool2DLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        if input.shape().len() != 4 {
            return Err(NnlError::tensor("AvgPool2D expects 4D input [N, C, H, W]"));
        }

        self.set_training(matches!(training, TrainingMode::Training));

        // Cache input for backward pass
        if self.training {
            self.cached_input = Some(input.clone_data()?);
        }

        // Proper average pooling implementation
        let output_shape = self.compute_output_shape(input.shape())?;
        let output_size: usize = output_shape.iter().product();
        let mut output_data = vec![0.0; output_size];

        let input_data = input.to_vec()?;
        let input_shape = input.shape();
        let (stride_h, stride_w) = self.effective_stride();
        let (kernel_h, kernel_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;

        for batch in 0..input_shape[0] {
            for channel in 0..input_shape[1] {
                for out_h in 0..output_shape[2] {
                    for out_w in 0..output_shape[3] {
                        let mut sum = 0.0;
                        let mut count = 0;

                        // Iterate over the pooling window
                        for kh in 0..kernel_h {
                            for kw in 0..kernel_w {
                                let in_h = (out_h * stride_h + kh) as i32 - pad_h as i32;
                                let in_w = (out_w * stride_w + kw) as i32 - pad_w as i32;

                                // Check bounds
                                if in_h >= 0
                                    && in_w >= 0
                                    && (in_h as usize) < input_shape[2]
                                    && (in_w as usize) < input_shape[3]
                                {
                                    let input_idx =
                                        batch * input_shape[1] * input_shape[2] * input_shape[3]
                                            + channel * input_shape[2] * input_shape[3]
                                            + (in_h as usize) * input_shape[3]
                                            + (in_w as usize);

                                    if input_idx < input_data.len() {
                                        sum += input_data[input_idx];
                                        count += 1;
                                    }
                                }
                            }
                        }

                        let output_idx =
                            batch * output_shape[1] * output_shape[2] * output_shape[3]
                                + channel * output_shape[2] * output_shape[3]
                                + out_h * output_shape[3]
                                + out_w;

                        if output_idx < output_data.len() {
                            output_data[output_idx] =
                                if count > 0 { sum / count as f32 } else { 0.0 };
                        }
                    }
                }
            }
        }

        // Create tensor from the computed data on the same device as input
        let output =
            Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())?;
        Ok(output)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let input = self
            .cached_input
            .as_ref()
            .ok_or_else(|| NnlError::training("No cached input for backward pass"))?;

        // Proper AvgPool2D backward pass
        let input_shape = input.shape();
        let grad_output_data = grad_output.to_vec()?;
        let output_shape = grad_output.shape();

        let mut grad_input_data = vec![0.0; input_shape.iter().product()];
        let (stride_h, stride_w) = self.effective_stride();
        let (kernel_h, kernel_w) = self.kernel_size;
        let (pad_h, pad_w) = self.padding;

        for batch in 0..input_shape[0] {
            for channel in 0..input_shape[1] {
                for out_h in 0..output_shape[2] {
                    for out_w in 0..output_shape[3] {
                        let grad_output_idx =
                            batch * output_shape[1] * output_shape[2] * output_shape[3]
                                + channel * output_shape[2] * output_shape[3]
                                + out_h * output_shape[3]
                                + out_w;

                        if grad_output_idx < grad_output_data.len() {
                            let grad_value = grad_output_data[grad_output_idx];
                            let mut count = 0;

                            // Count valid positions in the pooling window
                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let in_h = (out_h * stride_h + kh) as i32 - pad_h as i32;
                                    let in_w = (out_w * stride_w + kw) as i32 - pad_w as i32;

                                    if in_h >= 0
                                        && in_w >= 0
                                        && (in_h as usize) < input_shape[2]
                                        && (in_w as usize) < input_shape[3]
                                    {
                                        count += 1;
                                    }
                                }
                            }

                            // Distribute gradient equally among all positions in the pooling window
                            let distributed_grad = if count > 0 {
                                grad_value / count as f32
                            } else {
                                0.0
                            };

                            for kh in 0..kernel_h {
                                for kw in 0..kernel_w {
                                    let in_h = (out_h * stride_h + kh) as i32 - pad_h as i32;
                                    let in_w = (out_w * stride_w + kw) as i32 - pad_w as i32;

                                    if in_h >= 0
                                        && in_w >= 0
                                        && (in_h as usize) < input_shape[2]
                                        && (in_w as usize) < input_shape[3]
                                    {
                                        let grad_input_idx = batch
                                            * input_shape[1]
                                            * input_shape[2]
                                            * input_shape[3]
                                            + channel * input_shape[2] * input_shape[3]
                                            + (in_h as usize) * input_shape[3]
                                            + (in_w as usize);

                                        if grad_input_idx < grad_input_data.len() {
                                            grad_input_data[grad_input_idx] += distributed_grad;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_input =
            Tensor::from_slice_on_device(&grad_input_data, input_shape, input.device().clone())?;
        Ok(grad_input)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new() // Pooling layers have no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Pooling layers have no parameters
    }

    fn gradients(&self) -> Vec<&Tensor> {
        Vec::new() // Pooling layers have no gradients
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Pooling layers have no gradients
    }

    fn zero_grad(&mut self) {
        // No gradients to zero for pooling layers
    }

    fn name(&self) -> &str {
        "AvgPool2D"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        self.compute_output_shape(input_shape)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn num_parameters(&self) -> usize {
        0 // Pooling layers have no parameters
    }

    fn to_device(&mut self, _device: crate::device::Device) -> Result<()> {
        // Pooling layers don't have parameters to move
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        Ok(Box::new(Self {
            kernel_size: self.kernel_size,
            stride: self.stride,
            padding: self.padding,
            training: self.training,
            cached_input: None, // Don't clone cached data
        }))
    }
}

impl AvgPool2DLayer {
    /// Compute output shape after pooling
    fn compute_output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        if input_shape.len() != 4 {
            return Err(NnlError::tensor("Expected 4D input shape"));
        }

        let [batch_size, channels, height, width] = [
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[3],
        ];

        let (kernel_h, kernel_w) = self.kernel_size;
        let (stride_h, stride_w) = self.effective_stride();
        let (pad_h, pad_w) = self.padding;

        let output_height = (height + 2 * pad_h - kernel_h) / stride_h + 1;
        let output_width = (width + 2 * pad_w - kernel_w) / stride_w + 1;

        Ok(vec![batch_size, channels, output_height, output_width])
    }
}

/// Flatten Layer - reshapes input to 1D while preserving batch dimension
#[derive(Debug)]
pub struct FlattenLayer {
    start_dim: usize,
    end_dim: Option<usize>,
    training: bool,
    cached_shape: Option<Vec<usize>>,
}

impl FlattenLayer {
    /// Create a new Flatten layer
    pub fn new(start_dim: usize, end_dim: Option<usize>) -> Result<Self> {
        Ok(Self {
            start_dim,
            end_dim,
            training: true,
            cached_shape: None,
        })
    }
}

impl Layer for FlattenLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        let input_shape = input.shape();

        self.set_training(matches!(training, TrainingMode::Training));

        // Cache original shape for backward pass
        if self.training {
            self.cached_shape = Some(input_shape.to_vec());
        }

        let end_dim = self.end_dim.unwrap_or(input_shape.len() - 1);

        if self.start_dim > end_dim || end_dim >= input_shape.len() {
            return Err(NnlError::tensor("Invalid flatten dimensions"));
        }

        // Calculate flattened size
        let mut output_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..self.start_dim {
            output_shape.push(input_shape[i]);
        }

        // Flatten dimensions from start_dim to end_dim
        let flattened_size: usize = input_shape[self.start_dim..=end_dim].iter().product();
        output_shape.push(flattened_size);

        // Keep dimensions after end_dim
        for i in (end_dim + 1)..input_shape.len() {
            output_shape.push(input_shape[i]);
        }

        input.reshape(&output_shape)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let original_shape = self
            .cached_shape
            .as_ref()
            .ok_or_else(|| NnlError::training("No cached shape for backward pass"))?;

        grad_output.reshape(original_shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new() // Flatten layer has no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Flatten layer has no parameters
    }

    fn gradients(&self) -> Vec<&Tensor> {
        Vec::new() // Flatten layer has no gradients
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Flatten layer has no gradients
    }

    fn zero_grad(&mut self) {
        // No gradients to zero for flatten layer
    }

    fn name(&self) -> &str {
        "Flatten"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let end_dim = self.end_dim.unwrap_or(input_shape.len() - 1);

        if self.start_dim > end_dim || end_dim >= input_shape.len() {
            return Err(NnlError::tensor("Invalid flatten dimensions"));
        }

        let mut output_shape = Vec::new();

        // Keep dimensions before start_dim
        for i in 0..self.start_dim {
            output_shape.push(input_shape[i]);
        }

        // Flatten dimensions from start_dim to end_dim
        let flattened_size: usize = input_shape[self.start_dim..=end_dim].iter().product();
        output_shape.push(flattened_size);

        // Keep dimensions after end_dim
        for i in (end_dim + 1)..input_shape.len() {
            output_shape.push(input_shape[i]);
        }

        Ok(output_shape)
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn num_parameters(&self) -> usize {
        0 // Flatten layer has no parameters
    }

    fn to_device(&mut self, _device: crate::device::Device) -> Result<()> {
        // Flatten layer doesn't have parameters to move
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        Ok(Box::new(Self {
            start_dim: self.start_dim,
            end_dim: self.end_dim,
            training: self.training,
            cached_shape: None, // Don't clone cached data
        }))
    }
}

/// Reshape Layer - reshapes input to specified target shape
#[derive(Debug)]
pub struct ReshapeLayer {
    target_shape: Vec<usize>,
    training: bool,
    cached_shape: Option<Vec<usize>>,
}

impl ReshapeLayer {
    /// Create a new Reshape layer
    pub fn new(target_shape: Vec<usize>) -> Result<Self> {
        Ok(Self {
            target_shape,
            training: true,
            cached_shape: None,
        })
    }
}

impl Layer for ReshapeLayer {
    fn forward(&mut self, input: &Tensor, training: TrainingMode) -> Result<Tensor> {
        self.set_training(matches!(training, TrainingMode::Training));

        // Cache original shape for backward pass
        if self.training {
            self.cached_shape = Some(input.shape().to_vec());
        }

        // Check that total number of elements matches
        let input_size: usize = input.shape().iter().product();
        let target_size: usize = self.target_shape.iter().product();

        if input_size != target_size {
            return Err(NnlError::tensor(&format!(
                "Cannot reshape tensor: input size {} does not match target size {}",
                input_size, target_size
            )));
        }

        input.reshape(&self.target_shape)
    }

    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor> {
        let original_shape = self
            .cached_shape
            .as_ref()
            .ok_or_else(|| NnlError::training("No cached shape for backward pass"))?;

        grad_output.reshape(original_shape)
    }

    fn parameters(&self) -> Vec<&Tensor> {
        Vec::new() // Reshape layer has no parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Reshape layer has no parameters
    }

    fn gradients(&self) -> Vec<&Tensor> {
        Vec::new() // Reshape layer has no gradients
    }

    fn gradients_mut(&mut self) -> Vec<&mut Tensor> {
        Vec::new() // Reshape layer has no gradients
    }

    fn zero_grad(&mut self) {
        // No gradients to zero for reshape layer
    }

    fn name(&self) -> &str {
        "Reshape"
    }

    fn output_shape(&self, input_shape: &[usize]) -> Result<Vec<usize>> {
        let input_size: usize = input_shape.iter().product();
        let target_size: usize = self.target_shape.iter().product();

        if input_size != target_size {
            return Err(NnlError::tensor(&format!(
                "Cannot reshape tensor: input size {} does not match target size {}",
                input_size, target_size
            )));
        }

        Ok(self.target_shape.clone())
    }

    fn set_training(&mut self, training: bool) {
        self.training = training;
    }

    fn training(&self) -> bool {
        self.training
    }

    fn num_parameters(&self) -> usize {
        0 // Reshape layer has no parameters
    }

    fn to_device(&mut self, _device: crate::device::Device) -> Result<()> {
        // Reshape layer doesn't have parameters to move
        Ok(())
    }

    fn clone_layer(&self) -> Result<Box<dyn Layer>> {
        Ok(Box::new(Self {
            target_shape: self.target_shape.clone(),
            training: self.training,
            cached_shape: None, // Don't clone cached data
        }))
    }
}
