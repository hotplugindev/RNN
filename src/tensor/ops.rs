//! Tensor operations module
//!
//! This module provides optimized implementations of tensor operations
//! that work across different device backends.

use crate::activations::Activation;
use crate::device::DeviceType;
use crate::error::{Result, RnnError};
use crate::tensor::Tensor;
use rayon::prelude::*;

/// Enumeration of tensor operations
#[derive(Debug, Clone, Copy)]
pub enum TensorOp {
    Add,
    Sub,
    Mul,
    Div,
    AddScalar,
    MulScalar,
    MatMul,
}

/// Perform binary operation between two tensors
pub fn binary_op(a: &Tensor, b: &Tensor, op: TensorOp) -> Result<Tensor> {
    if a.shape() != b.shape() {
        return Err(RnnError::shape_mismatch(a.shape(), b.shape()));
    }

    match (a.device().device_type(), b.device().device_type()) {
        (DeviceType::Cpu, DeviceType::Cpu) => cpu_binary_op(a, b, op),
        _ => {
            // For GPU operations, we'd dispatch to appropriate kernels
            // For now, fall back to CPU implementation
            let a_cpu = a.to_host()?;
            let b_cpu = b.to_host()?;
            let result = cpu_binary_op(&a_cpu, &b_cpu, op)?;

            // Convert back to original device if needed
            if a.device().device_type() != DeviceType::Cpu {
                result.to_device(a.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU implementation of binary operations
fn cpu_binary_op(a: &Tensor, b: &Tensor, op: TensorOp) -> Result<Tensor> {
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    let result_data: Vec<f32> = match op {
        TensorOp::Add => a_data
            .par_iter()
            .zip(b_data.par_iter())
            .map(|(&x, &y)| x + y)
            .collect(),
        TensorOp::Sub => a_data
            .par_iter()
            .zip(b_data.par_iter())
            .map(|(&x, &y)| x - y)
            .collect(),
        TensorOp::Mul => a_data
            .par_iter()
            .zip(b_data.par_iter())
            .map(|(&x, &y)| x * y)
            .collect(),
        TensorOp::Div => a_data
            .par_iter()
            .zip(b_data.par_iter())
            .map(|(&x, &y)| x / y)
            .collect(),
        _ => {
            return Err(RnnError::unsupported(
                "Operation not supported for binary tensors",
            ))
        }
    };

    Tensor::from_slice_on_device(&result_data, a.shape(), a.device().clone())
}

/// Perform scalar operation on tensor
pub fn scalar_op(tensor: &Tensor, scalar: f32, op: TensorOp) -> Result<Tensor> {
    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_scalar_op(tensor, scalar, op),
        _ => {
            // For GPU operations, we'd dispatch to appropriate kernels
            let cpu_tensor = tensor.to_host()?;
            let result = cpu_scalar_op(&cpu_tensor, scalar, op)?;

            if tensor.device().device_type() != DeviceType::Cpu {
                result.to_device(tensor.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU implementation of scalar operations
fn cpu_scalar_op(tensor: &Tensor, scalar: f32, op: TensorOp) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    let result_data: Vec<f32> = match op {
        TensorOp::AddScalar => data.par_iter().map(|&x| x + scalar).collect(),
        TensorOp::MulScalar => data.par_iter().map(|&x| x * scalar).collect(),
        _ => {
            return Err(RnnError::unsupported(
                "Operation not supported for scalar operations",
            ))
        }
    };

    Tensor::from_slice_on_device(&result_data, tensor.shape(), tensor.device().clone())
}

/// Matrix multiplication
pub fn matmul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(RnnError::tensor(
            "Matrix multiplication requires 2D tensors",
        ));
    }

    let a_shape = a.shape();
    let b_shape = b.shape();

    if a_shape[1] != b_shape[0] {
        return Err(RnnError::shape_mismatch(
            &[a_shape[0], a_shape[1], b_shape[1]],
            &[a_shape[0], a_shape[1], b_shape[0]],
        ));
    }

    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];

    match (a.device().device_type(), b.device().device_type()) {
        (DeviceType::Cpu, DeviceType::Cpu) => cpu_matmul(a, b, m, k, n),
        _ => {
            // For GPU, we'd use optimized GEMM kernels
            let a_cpu = a.to_host()?;
            let b_cpu = b.to_host()?;
            let result = cpu_matmul(&a_cpu, &b_cpu, m, k, n)?;

            if a.device().device_type() != DeviceType::Cpu {
                result.to_device(a.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU matrix multiplication implementation
fn cpu_matmul(a: &Tensor, b: &Tensor, m: usize, k: usize, n: usize) -> Result<Tensor> {
    let a_data = a.to_vec()?;
    let b_data = b.to_vec()?;

    let mut result_data = vec![0.0; m * n];

    // Parallel matrix multiplication
    result_data
        .par_chunks_mut(n)
        .enumerate()
        .for_each(|(i, row)| {
            for (j, cell) in row.iter_mut().enumerate() {
                let mut sum = 0.0;
                for l in 0..k {
                    sum += a_data[i * k + l] * b_data[l * n + j];
                }
                *cell = sum;
            }
        });

    Tensor::from_slice_on_device(&result_data, &[m, n], a.device().clone())
}

/// Apply activation function to tensor
pub fn activation(tensor: &Tensor, activation: Activation) -> Result<Tensor> {
    match tensor.device().device_type() {
        DeviceType::Cpu => cpu_activation(tensor, activation),
        _ => {
            let cpu_tensor = tensor.to_host()?;
            let result = cpu_activation(&cpu_tensor, activation)?;

            if tensor.device().device_type() != DeviceType::Cpu {
                result.to_device(tensor.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU activation function implementation
fn cpu_activation(tensor: &Tensor, activation: Activation) -> Result<Tensor> {
    let data = tensor.to_vec()?;

    let result_data: Vec<f32> = match activation {
        Activation::ReLU => data.par_iter().map(|&x| x.max(0.0)).collect(),
        Activation::Sigmoid => data.par_iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect(),
        Activation::Tanh => data.par_iter().map(|&x| x.tanh()).collect(),
        Activation::LeakyReLU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect(),
        Activation::Softmax => cpu_softmax(&data)?,
        Activation::Linear => data,
        Activation::ELU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * (x.exp() - 1.0) })
            .collect(),
        Activation::Swish => data.par_iter().map(|&x| x / (1.0 + (-x).exp())).collect(),
        Activation::GELU => data
            .par_iter()
            .map(|&x| 0.5 * x * (1.0 + (x * 0.7978845608 * (1.0 + 0.044715 * x * x)).tanh()))
            .collect(),
        Activation::Mish => data
            .par_iter()
            .map(|&x| x * (1.0 + x.exp()).ln().tanh())
            .collect(),
        Activation::PReLU(alpha) => data
            .par_iter()
            .map(|&x| if x > 0.0 { x } else { alpha * x })
            .collect(),
        Activation::SELU => data
            .par_iter()
            .map(|&x| {
                if x > 0.0 {
                    1.0507009873554804934193349852946 * x
                } else {
                    1.0507009873554804934193349852946
                        * 1.6732632423543772848170429916717
                        * (x.exp() - 1.0)
                }
            })
            .collect(),
    };

    Tensor::from_slice_on_device(&result_data, tensor.shape(), tensor.device().clone())
}

/// CPU softmax implementation
fn cpu_softmax(data: &[f32]) -> Result<Vec<f32>> {
    // Find maximum for numerical stability
    let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exponentials
    let exp_data: Vec<f32> = data.iter().map(|&x| (x - max_val).exp()).collect();

    // Compute sum
    let sum: f32 = exp_data.iter().sum();

    if sum == 0.0 {
        return Err(RnnError::math("Softmax sum is zero"));
    }

    // Normalize
    let result: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
    Ok(result)
}

/// Reduce tensor to sum
pub fn reduce_sum(tensor: &Tensor) -> Result<f32> {
    let data = tensor.to_vec()?;
    Ok(data.par_iter().sum())
}

/// Reduce tensor to maximum
pub fn reduce_max(tensor: &Tensor) -> Result<f32> {
    let data = tensor.to_vec()?;
    let max_val = data
        .par_iter()
        .fold(|| f32::NEG_INFINITY, |a, &b| a.max(b))
        .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));
    Ok(max_val)
}

/// Reduce tensor to minimum
pub fn reduce_min(tensor: &Tensor) -> Result<f32> {
    let data = tensor.to_vec()?;
    let min_val = data
        .par_iter()
        .fold(|| f32::INFINITY, |a, &b| a.min(b))
        .reduce(|| f32::INFINITY, |a, b| a.min(b));
    Ok(min_val)
}

/// Check if two shapes are broadcastable
pub fn is_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
    let max_ndim = shape1.len().max(shape2.len());

    for i in 0..max_ndim {
        let dim1 = shape1.get(shape1.len().saturating_sub(i + 1)).unwrap_or(&1);
        let dim2 = shape2.get(shape2.len().saturating_sub(i + 1)).unwrap_or(&1);

        if *dim1 != *dim2 && *dim1 != 1 && *dim2 != 1 {
            return false;
        }
    }

    true
}

/// Broadcast tensor to target shape
pub fn broadcast_to(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    if !is_broadcastable(tensor.shape(), target_shape) {
        return Err(RnnError::shape_mismatch(tensor.shape(), target_shape));
    }

    if tensor.shape() == target_shape {
        return Ok(tensor.clone_data()?);
    }

    // For simplicity, implement broadcasting on CPU
    let cpu_tensor = tensor.to_host()?;
    let broadcasted = cpu_broadcast(&cpu_tensor, target_shape)?;

    if tensor.device().device_type() != DeviceType::Cpu {
        broadcasted.to_device(tensor.device().clone())
    } else {
        Ok(broadcasted)
    }
}

/// CPU broadcasting implementation
fn cpu_broadcast(tensor: &Tensor, target_shape: &[usize]) -> Result<Tensor> {
    let source_data = tensor.to_vec()?;
    let source_shape = tensor.shape();
    let target_size = target_shape.iter().product::<usize>();

    let mut result_data = vec![0.0; target_size];

    // Simple broadcasting implementation
    for i in 0..target_size {
        let mut source_idx = 0;
        let mut temp_i = i;
        let mut stride = 1;

        // Calculate source index based on broadcasting rules
        for dim_idx in (0..target_shape.len()).rev() {
            let target_dim = target_shape[dim_idx];
            let coord = temp_i % target_dim;
            temp_i /= target_dim;

            if dim_idx < source_shape.len() {
                let source_dim_idx = source_shape.len() - target_shape.len() + dim_idx;
                if source_dim_idx < source_shape.len() {
                    let source_dim = source_shape[source_dim_idx];
                    if source_dim > 1 {
                        source_idx += (coord % source_dim) * stride;
                        stride *= source_dim;
                    }
                }
            }
        }

        result_data[i] = source_data[source_idx % source_data.len()];
    }

    Tensor::from_slice_on_device(&result_data, target_shape, tensor.device().clone())
}

/// Convolution 2D operation
pub fn conv2d(
    input: &Tensor,
    kernel: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Tensor> {
    if input.ndim() != 4 || kernel.ndim() != 4 {
        return Err(RnnError::tensor("Conv2D requires 4D tensors (NCHW format)"));
    }

    let input_shape = input.shape();
    let kernel_shape = kernel.shape();

    let batch_size = input_shape[0];
    let in_channels = input_shape[1];
    let in_height = input_shape[2];
    let in_width = input_shape[3];

    let out_channels = kernel_shape[0];
    let kernel_height = kernel_shape[2];
    let kernel_width = kernel_shape[3];

    if kernel_shape[1] != in_channels {
        return Err(RnnError::shape_mismatch(&[in_channels], &[kernel_shape[1]]));
    }

    let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
    let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;

    match input.device().device_type() {
        DeviceType::Cpu => cpu_conv2d(
            input,
            kernel,
            stride,
            padding,
            batch_size,
            in_channels,
            in_height,
            in_width,
            out_channels,
            out_height,
            out_width,
            kernel_height,
            kernel_width,
        ),
        _ => {
            let input_cpu = input.to_host()?;
            let kernel_cpu = kernel.to_host()?;
            let result = cpu_conv2d(
                &input_cpu,
                &kernel_cpu,
                stride,
                padding,
                batch_size,
                in_channels,
                in_height,
                in_width,
                out_channels,
                out_height,
                out_width,
                kernel_height,
                kernel_width,
            )?;

            if input.device().device_type() != DeviceType::Cpu {
                result.to_device(input.device().clone())
            } else {
                Ok(result)
            }
        }
    }
}

/// CPU convolution 2D implementation
#[allow(clippy::too_many_arguments)]
fn cpu_conv2d(
    input: &Tensor,
    kernel: &Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    batch_size: usize,
    in_channels: usize,
    in_height: usize,
    in_width: usize,
    out_channels: usize,
    out_height: usize,
    out_width: usize,
    kernel_height: usize,
    kernel_width: usize,
) -> Result<Tensor> {
    let input_data = input.to_vec()?;
    let kernel_data = kernel.to_vec()?;

    let output_size = batch_size * out_channels * out_height * out_width;
    let mut output_data = vec![0.0; output_size];

    // Parallel convolution implementation
    output_data
        .par_chunks_mut(out_channels * out_height * out_width)
        .enumerate()
        .for_each(|(batch, batch_output)| {
            for out_c in 0..out_channels {
                for out_y in 0..out_height {
                    for out_x in 0..out_width {
                        let mut sum = 0.0;

                        for in_c in 0..in_channels {
                            for ky in 0..kernel_height {
                                for kx in 0..kernel_width {
                                    let in_y = out_y * stride.0 + ky;
                                    let in_x = out_x * stride.1 + kx;

                                    // Apply padding
                                    if in_y >= padding.0 && in_x >= padding.1 {
                                        let in_y = in_y - padding.0;
                                        let in_x = in_x - padding.1;

                                        if in_y < in_height && in_x < in_width {
                                            let input_idx = batch
                                                * (in_channels * in_height * in_width)
                                                + in_c * (in_height * in_width)
                                                + in_y * in_width
                                                + in_x;

                                            let kernel_idx = out_c
                                                * (in_channels * kernel_height * kernel_width)
                                                + in_c * (kernel_height * kernel_width)
                                                + ky * kernel_width
                                                + kx;

                                            sum += input_data[input_idx] * kernel_data[kernel_idx];
                                        }
                                    }
                                }
                            }
                        }

                        let output_idx =
                            out_c * (out_height * out_width) + out_y * out_width + out_x;
                        batch_output[output_idx] = sum;
                    }
                }
            }
        });

    let output_shape = [batch_size, out_channels, out_height, out_width];
    Tensor::from_slice_on_device(&output_data, &output_shape, input.device().clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_binary_operations() {
        let a = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = Tensor::from_slice(&[2.0, 2.0, 2.0, 2.0], &[2, 2]).unwrap();

        let sum = binary_op(&a, &b, TensorOp::Add).unwrap();
        assert_eq!(sum.to_vec().unwrap(), vec![3.0, 4.0, 5.0, 6.0]);

        let product = binary_op(&a, &b, TensorOp::Mul).unwrap();
        assert_eq!(product.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_scalar_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let add_result = scalar_op(&tensor, 5.0, TensorOp::AddScalar).unwrap();
        assert_eq!(add_result.to_vec().unwrap(), vec![6.0, 7.0, 8.0, 9.0]);

        let mul_result = scalar_op(&tensor, 2.0, TensorOp::MulScalar).unwrap();
        assert_eq!(mul_result.to_vec().unwrap(), vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_matrix_multiplication() {
        let a = Tensor::from_array_2d(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        let b = Tensor::from_array_2d(&[&[2.0, 0.0], &[1.0, 3.0]]).unwrap();

        let result = matmul(&a, &b).unwrap();
        let expected = vec![4.0, 6.0, 10.0, 12.0]; // [[4, 6], [10, 12]]
        assert_eq!(result.to_vec().unwrap(), expected);
    }

    #[test]
    fn test_activation_functions() {
        let tensor = Tensor::from_slice(&[-1.0, 0.0, 1.0, 2.0], &[4]).unwrap();

        // Test ReLU
        let relu_result = activation(&tensor, Activation::ReLU).unwrap();
        assert_eq!(relu_result.to_vec().unwrap(), vec![0.0, 0.0, 1.0, 2.0]);

        // Test Sigmoid
        let sigmoid_result = activation(&tensor, Activation::Sigmoid).unwrap();
        let sigmoid_expected: Vec<f32> = tensor
            .to_vec()
            .unwrap()
            .iter()
            .map(|&x| 1.0 / (1.0 + (-x).exp()))
            .collect();

        let sigmoid_actual = sigmoid_result.to_vec().unwrap();
        for (actual, expected) in sigmoid_actual.iter().zip(sigmoid_expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_reduction_operations() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        assert_eq!(reduce_sum(&tensor).unwrap(), 10.0);
        assert_eq!(reduce_max(&tensor).unwrap(), 4.0);
        assert_eq!(reduce_min(&tensor).unwrap(), 1.0);
    }

    #[test]
    fn test_broadcasting() {
        let tensor = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();

        assert!(is_broadcastable(&[2], &[3, 2]));
        assert!(is_broadcastable(&[1, 2], &[3, 2]));
        assert!(!is_broadcastable(&[3], &[2]));

        let broadcasted = broadcast_to(&tensor, &[2, 2]).unwrap();
        assert_eq!(broadcasted.shape(), &[2, 2]);
    }

    #[test]
    fn test_softmax() {
        let data = vec![1.0, 2.0, 3.0];
        let result = cpu_softmax(&data).unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Check that all probabilities are positive
        assert!(result.iter().all(|&x| x > 0.0));
    }
}
