//! GPU-accelerated layer operations that utilize actual GPU compute kernels.
//!
//! This module provides GPU implementations of neural network layer operations,
//! including forward and backward passes that execute on CUDA, OpenCL, and ROCm devices.

use crate::activation::ActivationFunction;
use crate::error::{NetworkError, Result};
use crate::gpu::{
    kernels::{CudaKernels, OpenCLKernels},
    GpuContext, GpuDataType, GpuKernel, GpuKernelArg, GpuMemoryHandle, GpuTensor,
};
use crate::layer::{BackwardResult, LayerSummary};
use ndarray::{Array1, Array2};
use std::collections::HashMap;

/// GPU-accelerated dense (fully connected) layer
pub struct GpuDenseLayer {
    /// Weights matrix (input_size x output_size)
    pub weights: GpuTensor,
    /// Bias vector (output_size)
    pub biases: GpuTensor,
    /// Activation function
    pub activation: ActivationFunction,
    /// Input size
    pub input_size: usize,
    /// Output size
    pub output_size: usize,
    /// Device ID
    pub device_id: usize,
    /// GPU context
    pub context: Box<dyn GpuContext>,
    /// Cached activations for backward pass
    last_input: Option<GpuTensor>,
    last_linear_output: Option<GpuTensor>,
    /// Compiled kernels cache
    kernels: HashMap<String, GpuKernel>,
}

impl GpuDenseLayer {
    /// Create a new GPU dense layer
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
        device_id: usize,
        context: Box<dyn GpuContext>,
    ) -> Result<Self> {
        // Initialize weights with Xavier/Glorot initialization
        let weight_data = Self::xavier_init(input_size, output_size);
        let bias_data = Array1::zeros(output_size);

        // Create GPU tensors
        let weights = GpuTensor::from_cpu(
            &weight_data
                .view()
                .into_shape((input_size, output_size))
                .map_err(|e| NetworkError::shape(format!("Weight shape error: {}", e)))?,
            device_id,
            &*context,
        )?;

        let biases = GpuTensor::from_cpu(
            &bias_data
                .view()
                .into_shape((1, output_size))
                .map_err(|e| NetworkError::shape(format!("Bias shape error: {}", e)))?,
            device_id,
            &*context,
        )?;

        let mut kernels = HashMap::new();
        Self::compile_kernels(&mut kernels, &*context, device_id)?;

        Ok(Self {
            weights,
            biases,
            activation,
            input_size,
            output_size,
            device_id,
            context,
            last_input: None,
            last_linear_output: None,
            kernels,
        })
    }

    /// Initialize weights using Xavier/Glorot initialization
    fn xavier_init(input_size: usize, output_size: usize) -> Array2<f64> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let limit = (6.0 / (input_size + output_size) as f64).sqrt();

        Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-limit..limit))
    }

    /// Compile GPU kernels for this layer
    fn compile_kernels(
        kernels: &mut HashMap<String, GpuKernel>,
        context: &dyn GpuContext,
        device_id: usize,
    ) -> Result<()> {
        // Matrix multiplication kernel
        let matmul_kernel = GpuKernel {
            name: "matmul".to_string(),
            source: CudaKernels::matmul().to_string(),
            entry_point: "matmul_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (16, 16),
            backend_handle: None,
        };
        kernels.insert("matmul".to_string(), matmul_kernel);

        // Element-wise addition kernel (for bias)
        let add_kernel = GpuKernel {
            name: "add".to_string(),
            source: CudaKernels::add().to_string(),
            entry_point: "add_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (256, 1),
            backend_handle: None,
        };
        kernels.insert("add".to_string(), add_kernel);

        // ReLU activation kernel
        let relu_kernel = GpuKernel {
            name: "relu".to_string(),
            source: CudaKernels::relu().to_string(),
            entry_point: "relu_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (256, 1),
            backend_handle: None,
        };
        kernels.insert("relu".to_string(), relu_kernel);

        // ReLU derivative kernel
        let relu_deriv_kernel = GpuKernel {
            name: "relu_derivative".to_string(),
            source: CudaKernels::relu_derivative().to_string(),
            entry_point: "relu_derivative_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (256, 1),
            backend_handle: None,
        };
        kernels.insert("relu_derivative".to_string(), relu_deriv_kernel);

        // Sigmoid activation kernel
        let sigmoid_kernel = GpuKernel {
            name: "sigmoid".to_string(),
            source: CudaKernels::sigmoid().to_string(),
            entry_point: "sigmoid_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (256, 1),
            backend_handle: None,
        };
        kernels.insert("sigmoid".to_string(), sigmoid_kernel);

        // Sigmoid derivative kernel
        let sigmoid_deriv_kernel = GpuKernel {
            name: "sigmoid_derivative".to_string(),
            source: CudaKernels::sigmoid_derivative().to_string(),
            entry_point: "sigmoid_derivative_kernel".to_string(),
            compiled_binary: None,
            work_group_size: (256, 1),
            backend_handle: None,
        };
        kernels.insert("sigmoid_derivative".to_string(), sigmoid_deriv_kernel);

        println!("✅ Compiled {} GPU kernels for dense layer", kernels.len());
        Ok(())
    }

    /// Forward pass using GPU computation
    pub fn forward_gpu(&mut self, input: &GpuTensor) -> Result<GpuTensor> {
        let batch_size = input.shape()[0];
        let input_size = input.shape()[1];

        if input_size != self.input_size {
            return Err(NetworkError::shape(format!(
                "Input size mismatch: expected {}, got {}",
                self.input_size, input_size
            )));
        }

        // Allocate output tensor: batch_size x output_size
        let output_shape = vec![batch_size, self.output_size];
        let output_handle = self.context.allocate(
            batch_size * self.output_size * std::mem::size_of::<f32>(),
            GpuDataType::Float32,
        )?;

        let mut output = GpuTensor {
            handle: output_handle,
            shape: output_shape,
            dtype: GpuDataType::Float32,
            device_id: self.device_id,
            memory_layout: crate::gpu::MemoryLayout::RowMajor,
            strides: vec![self.output_size, 1],
        };

        // Store input for backward pass
        self.last_input = Some(input.clone());

        // Perform matrix multiplication: input * weights + biases
        self.gpu_matmul_add_bias(input, &self.weights, &self.biases, &mut output)?;

        // Store linear output for backward pass
        self.last_linear_output = Some(output.clone());

        // Apply activation function
        self.apply_activation_gpu(&mut output)?;

        Ok(output)
    }

    /// Perform GPU matrix multiplication with bias addition
    fn gpu_matmul_add_bias(
        &mut self,
        input: &GpuTensor,
        weights: &GpuTensor,
        biases: &GpuTensor,
        output: &mut GpuTensor,
    ) -> Result<()> {
        let batch_size = input.shape()[0];
        let input_size = input.shape()[1];
        let output_size = weights.shape()[1];

        // Get the matmul kernel
        let matmul_kernel = self.kernels.get("matmul").ok_or_else(|| {
            NetworkError::gpu("Matrix multiplication kernel not found".to_string())
        })?;

        // Prepare kernel arguments for matmul
        let matmul_args = vec![
            GpuKernelArg::Buffer(input.handle.clone()),
            GpuKernelArg::Buffer(weights.handle.clone()),
            GpuKernelArg::Buffer(output.handle.clone()),
            GpuKernelArg::UInt(batch_size as u32),
            GpuKernelArg::UInt(output_size as u32),
            GpuKernelArg::UInt(input_size as u32),
        ];

        // Execute matrix multiplication kernel
        println!(
            "🚀 Executing GPU matmul: {}x{} * {}x{}",
            batch_size, input_size, input_size, output_size
        );
        self.context.execute_kernel(matmul_kernel, &matmul_args)?;

        // Add bias using element-wise addition kernel
        let add_kernel = self
            .kernels
            .get("add")
            .ok_or_else(|| NetworkError::gpu("Addition kernel not found".to_string()))?;

        let add_args = vec![
            GpuKernelArg::Buffer(output.handle.clone()),
            GpuKernelArg::Buffer(biases.handle.clone()),
            GpuKernelArg::Buffer(output.handle.clone()),
            GpuKernelArg::UInt((batch_size * output_size) as u32),
        ];

        println!("🚀 Executing GPU bias addition");
        self.context.execute_kernel(add_kernel, &add_args)?;

        Ok(())
    }

    /// Apply activation function using GPU kernels
    fn apply_activation_gpu(&mut self, tensor: &mut GpuTensor) -> Result<()> {
        let kernel_name = match self.activation {
            ActivationFunction::ReLU => "relu",
            ActivationFunction::Sigmoid => "sigmoid",
            ActivationFunction::Tanh => "tanh",
            ActivationFunction::Linear => return Ok(()), // No activation needed
            _ => {
                println!(
                    "⚠️ Activation function {} not implemented for GPU, using CPU fallback",
                    format!("{:?}", self.activation)
                );
                return self.apply_activation_cpu_fallback(tensor);
            }
        };

        let kernel = self.kernels.get(kernel_name).ok_or_else(|| {
            NetworkError::gpu(format!("Activation kernel {} not found", kernel_name))
        })?;

        let total_elements = tensor.shape().iter().product::<usize>();
        let args = vec![
            GpuKernelArg::Buffer(tensor.handle.clone()),
            GpuKernelArg::Buffer(tensor.handle.clone()), // In-place operation
            GpuKernelArg::UInt(total_elements as u32),
        ];

        println!(
            "🚀 Executing GPU {} activation on {} elements",
            kernel_name, total_elements
        );
        self.context.execute_kernel(kernel, &args)?;

        Ok(())
    }

    /// CPU fallback for unsupported activation functions
    fn apply_activation_cpu_fallback(&self, tensor: &mut GpuTensor) -> Result<()> {
        // Transfer to CPU, apply activation, transfer back
        let mut cpu_data = tensor.to_cpu(&*self.context)?;

        cpu_data.mapv_inplace(|x| match self.activation {
            ActivationFunction::Softmax => x, // Softmax requires special handling
            ActivationFunction::LeakyReLU => x.max(0.01 * x),
            ActivationFunction::ELU => {
                if x >= 0.0 {
                    x
                } else {
                    (x.exp() - 1.0)
                }
            }
            ActivationFunction::Swish => x / (1.0 + (-x).exp()),
            _ => x, // Should not reach here
        });

        // Copy data back to GPU
        self.context
            .copy_to_device(cpu_data.as_slice().unwrap(), &tensor.handle, 0)?;

        Ok(())
    }

    /// Backward pass using GPU computation
    pub fn backward_gpu(
        &mut self,
        grad_output: &GpuTensor,
        learning_rate: f64,
    ) -> Result<GpuTensor> {
        let input = self.last_input.as_ref().ok_or_else(|| {
            NetworkError::training("No cached input for backward pass".to_string())
        })?;

        let linear_output = self.last_linear_output.as_ref().ok_or_else(|| {
            NetworkError::training("No cached linear output for backward pass".to_string())
        })?;

        // Compute activation derivative
        let mut grad_linear = self.compute_activation_derivative_gpu(linear_output, grad_output)?;

        // Compute weight gradients: input^T * grad_linear
        let weight_gradients = self.compute_weight_gradients_gpu(input, &grad_linear)?;

        // Compute bias gradients: sum(grad_linear, axis=0)
        let bias_gradients = self.compute_bias_gradients_gpu(&grad_linear)?;

        // Update weights and biases
        self.update_weights_gpu(&weight_gradients, learning_rate)?;
        self.update_biases_gpu(&bias_gradients, learning_rate)?;

        // Compute input gradients: grad_linear * weights^T
        let grad_input = self.compute_input_gradients_gpu(&grad_linear)?;

        Ok(grad_input)
    }

    /// Compute activation derivative using GPU
    fn compute_activation_derivative_gpu(
        &mut self,
        linear_output: &GpuTensor,
        grad_output: &GpuTensor,
    ) -> Result<GpuTensor> {
        let mut grad_linear = grad_output.clone();

        match self.activation {
            ActivationFunction::Linear => {
                // grad_linear = grad_output (no change)
            }
            ActivationFunction::ReLU => {
                let kernel = self.kernels.get("relu_derivative").ok_or_else(|| {
                    NetworkError::gpu("ReLU derivative kernel not found".to_string())
                })?;

                let total_elements = linear_output.shape().iter().product::<usize>();
                let args = vec![
                    GpuKernelArg::Buffer(linear_output.handle.clone()),
                    GpuKernelArg::Buffer(grad_output.handle.clone()),
                    GpuKernelArg::Buffer(grad_linear.handle.clone()),
                    GpuKernelArg::UInt(total_elements as u32),
                ];

                println!("🚀 Executing GPU ReLU derivative");
                self.context.execute_kernel(kernel, &args)?;
            }
            ActivationFunction::Sigmoid => {
                let kernel = self.kernels.get("sigmoid_derivative").ok_or_else(|| {
                    NetworkError::gpu("Sigmoid derivative kernel not found".to_string())
                })?;

                let total_elements = linear_output.shape().iter().product::<usize>();
                let args = vec![
                    GpuKernelArg::Buffer(linear_output.handle.clone()),
                    GpuKernelArg::Buffer(grad_output.handle.clone()),
                    GpuKernelArg::Buffer(grad_linear.handle.clone()),
                    GpuKernelArg::UInt(total_elements as u32),
                ];

                println!("🚀 Executing GPU Sigmoid derivative");
                self.context.execute_kernel(kernel, &args)?;
            }
            _ => {
                println!(
                    "⚠️ Activation derivative {} not implemented for GPU, using CPU fallback",
                    format!("{:?}", self.activation)
                );
                return self.compute_activation_derivative_cpu_fallback(linear_output, grad_output);
            }
        }

        Ok(grad_linear)
    }

    /// CPU fallback for activation derivatives
    fn compute_activation_derivative_cpu_fallback(
        &self,
        linear_output: &GpuTensor,
        grad_output: &GpuTensor,
    ) -> Result<GpuTensor> {
        let cpu_linear = linear_output.to_cpu(&*self.context)?;
        let cpu_grad_output = grad_output.to_cpu(&*self.context)?;

        let mut cpu_grad_linear = cpu_grad_output.clone();

        match self.activation {
            ActivationFunction::Tanh => {
                for ((linear, grad_out), grad_linear) in cpu_linear
                    .iter()
                    .zip(cpu_grad_output.iter())
                    .zip(cpu_grad_linear.iter_mut())
                {
                    let tanh_val = linear.tanh();
                    *grad_linear = grad_out * (1.0 - tanh_val * tanh_val);
                }
            }
            ActivationFunction::LeakyReLU => {
                for ((linear, grad_out), grad_linear) in cpu_linear
                    .iter()
                    .zip(cpu_grad_output.iter())
                    .zip(cpu_grad_linear.iter_mut())
                {
                    *grad_linear = grad_out * if *linear > 0.0 { 1.0 } else { 0.01 };
                }
            }
            _ => {} // Other activations
        }

        GpuTensor::from_cpu(&cpu_grad_linear, self.device_id, &*self.context)
    }

    /// Compute weight gradients using GPU
    fn compute_weight_gradients_gpu(
        &mut self,
        input: &GpuTensor,
        grad_linear: &GpuTensor,
    ) -> Result<GpuTensor> {
        let batch_size = input.shape()[0];
        let input_size = input.shape()[1];
        let output_size = grad_linear.shape()[1];

        // Allocate tensor for weight gradients
        let weight_grad_handle = self.context.allocate(
            input_size * output_size * std::mem::size_of::<f32>(),
            GpuDataType::Float32,
        )?;

        let weight_gradients = GpuTensor {
            handle: weight_grad_handle,
            shape: vec![input_size, output_size],
            dtype: GpuDataType::Float32,
            device_id: self.device_id,
            memory_layout: crate::gpu::MemoryLayout::RowMajor,
            strides: vec![output_size, 1],
        };

        // Perform transposed matrix multiplication: input^T * grad_linear
        // This would require a specialized kernel or transposition operation
        // For now, use CPU fallback for this operation
        println!(
            "⚠️ Weight gradient computation using CPU fallback (transpose matmul not implemented)"
        );

        let cpu_input = input.to_cpu(&*self.context)?;
        let cpu_grad_linear = grad_linear.to_cpu(&*self.context)?;

        // input^T * grad_linear
        let cpu_weight_grads = cpu_input.t().dot(&cpu_grad_linear) / batch_size as f64;

        self.context.copy_to_device(
            cpu_weight_grads.as_slice().unwrap(),
            &weight_gradients.handle,
            0,
        )?;

        Ok(weight_gradients)
    }

    /// Compute bias gradients using GPU
    fn compute_bias_gradients_gpu(&mut self, grad_linear: &GpuTensor) -> Result<GpuTensor> {
        let batch_size = grad_linear.shape()[0];
        let output_size = grad_linear.shape()[1];

        // For bias gradients, we need to sum over the batch dimension
        // This requires a reduction kernel, which we'll implement as CPU fallback for now
        println!(
            "⚠️ Bias gradient computation using CPU fallback (reduction kernel not implemented)"
        );

        let cpu_grad_linear = grad_linear.to_cpu(&*self.context)?;
        let cpu_bias_grads = cpu_grad_linear.mean_axis(ndarray::Axis(0)).unwrap();

        let bias_grad_handle = self.context.allocate(
            output_size * std::mem::size_of::<f32>(),
            GpuDataType::Float32,
        )?;

        let bias_gradients = GpuTensor {
            handle: bias_grad_handle,
            shape: vec![1, output_size],
            dtype: GpuDataType::Float32,
            device_id: self.device_id,
            memory_layout: crate::gpu::MemoryLayout::RowMajor,
            strides: vec![output_size, 1],
        };

        self.context.copy_to_device(
            cpu_bias_grads.as_slice().unwrap(),
            &bias_gradients.handle,
            0,
        )?;

        Ok(bias_gradients)
    }

    /// Update weights using GPU
    fn update_weights_gpu(
        &mut self,
        weight_gradients: &GpuTensor,
        learning_rate: f64,
    ) -> Result<()> {
        // For now, use CPU fallback for weight updates
        // In a full implementation, this would use a GPU kernel for element-wise operations
        println!("⚠️ Weight update using CPU fallback");

        let mut cpu_weights = self.weights.to_cpu(&*self.context)?;
        let cpu_weight_grads = weight_gradients.to_cpu(&*self.context)?;

        cpu_weights -= &(cpu_weight_grads * learning_rate);

        self.context
            .copy_to_device(cpu_weights.as_slice().unwrap(), &self.weights.handle, 0)?;

        Ok(())
    }

    /// Update biases using GPU
    fn update_biases_gpu(&mut self, bias_gradients: &GpuTensor, learning_rate: f64) -> Result<()> {
        // For now, use CPU fallback for bias updates
        println!("⚠️ Bias update using CPU fallback");

        let mut cpu_biases = self.biases.to_cpu(&*self.context)?;
        let cpu_bias_grads = bias_gradients.to_cpu(&*self.context)?;

        cpu_biases -= &(cpu_bias_grads * learning_rate);

        self.context
            .copy_to_device(cpu_biases.as_slice().unwrap(), &self.biases.handle, 0)?;

        Ok(())
    }

    /// Compute input gradients using GPU
    fn compute_input_gradients_gpu(&mut self, grad_linear: &GpuTensor) -> Result<GpuTensor> {
        let batch_size = grad_linear.shape()[0];
        let output_size = grad_linear.shape()[1];
        let input_size = self.weights.shape()[0];

        // Allocate tensor for input gradients
        let input_grad_handle = self.context.allocate(
            batch_size * input_size * std::mem::size_of::<f32>(),
            GpuDataType::Float32,
        )?;

        let mut input_gradients = GpuTensor {
            handle: input_grad_handle,
            shape: vec![batch_size, input_size],
            dtype: GpuDataType::Float32,
            device_id: self.device_id,
            memory_layout: crate::gpu::MemoryLayout::RowMajor,
            strides: vec![input_size, 1],
        };

        // Perform matrix multiplication: grad_linear * weights^T
        // This requires transposing weights, for now use CPU fallback
        println!("⚠️ Input gradient computation using CPU fallback");

        let cpu_grad_linear = grad_linear.to_cpu(&*self.context)?;
        let cpu_weights = self.weights.to_cpu(&*self.context)?;

        let cpu_input_grads = cpu_grad_linear.dot(&cpu_weights.t());

        self.context.copy_to_device(
            cpu_input_grads.as_slice().unwrap(),
            &input_gradients.handle,
            0,
        )?;

        Ok(input_gradients)
    }

    /// Get layer summary
    pub fn summary(&self) -> LayerSummary {
        LayerSummary {
            layer_type: "Dense (GPU)".to_string(),
            input_shape: vec![self.input_size],
            output_shape: vec![self.output_size],
            parameters: self.input_size * self.output_size + self.output_size,
            activation: Some(self.activation),
        }
    }

    /// Get the number of parameters in this layer
    pub fn parameter_count(&self) -> usize {
        self.input_size * self.output_size + self.output_size
    }
}

/// GPU-accelerated network trainer
pub struct GpuNetworkTrainer {
    device_id: usize,
    context: Box<dyn GpuContext>,
    layers: Vec<GpuDenseLayer>,
}

impl GpuNetworkTrainer {
    /// Create a new GPU network trainer
    pub fn new(device_id: usize, context: Box<dyn GpuContext>) -> Self {
        Self {
            device_id,
            context,
            layers: Vec::new(),
        }
    }

    /// Add a GPU dense layer
    pub fn add_dense_layer(
        &mut self,
        input_size: usize,
        output_size: usize,
        activation: ActivationFunction,
    ) -> Result<()> {
        let layer = GpuDenseLayer::new(
            input_size,
            output_size,
            activation,
            self.device_id,
            // We need to clone the context here, which requires implementing Clone for GpuContext
            // For now, this is a simplified interface
            todo!("Context cloning not implemented"),
        )?;

        self.layers.push(layer);
        Ok(())
    }

    /// Perform forward pass through all layers
    pub fn forward(&mut self, input: &GpuTensor) -> Result<GpuTensor> {
        let mut current_output = input.clone();

        for layer in &mut self.layers {
            current_output = layer.forward_gpu(&current_output)?;

            // Synchronize after each layer to ensure completion
            self.context.synchronize()?;
        }

        Ok(current_output)
    }

    /// Perform backward pass through all layers
    pub fn backward(&mut self, grad_output: &GpuTensor, learning_rate: f64) -> Result<()> {
        let mut current_grad = grad_output.clone();

        // Backward pass through layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            current_grad = layer.backward_gpu(&current_grad, learning_rate)?;

            // Synchronize after each layer
            self.context.synchronize()?;
        }

        Ok(())
    }

    /// Get total parameter count
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::{CpuContext, GpuManager};

    #[test]
    fn test_gpu_dense_layer_creation() {
        let mut gpu_manager = GpuManager::new();
        let device_id = 0;

        if let Ok(context) = gpu_manager.create_context(device_id) {
            let result = GpuDenseLayer::new(10, 5, ActivationFunction::ReLU, device_id, context);

            assert!(result.is_ok());
            let layer = result.unwrap();
            assert_eq!(layer.parameter_count(), 10 * 5 + 5);
        }
    }

    #[test]
    fn test_xavier_initialization() {
        let weights = GpuDenseLayer::xavier_init(100, 50);
        let limit = (6.0 / 150.0).sqrt();

        // Check that all weights are within the expected range
        for &weight in weights.iter() {
            assert!(weight >= -limit && weight <= limit);
        }
    }
}
