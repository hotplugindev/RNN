//! GPU backend implementation with simplified Vulkan support
//!
//! This module provides a basic GPU compute backend using Vulkan.
//! For now, we use a simpler approach to get the GPU backend working.

use crate::device::{Backend, DeviceInfo, DeviceMemory, DeviceType, Kernel};
use crate::error::{Result, RnnError};

use std::sync::{Arc, Mutex};

/// Vulkan compute backend (simplified implementation)
pub struct VulkanBackend {
    device_info: DeviceInfo,
}

impl VulkanBackend {
    /// Create a new Vulkan backend
    pub fn new() -> Result<Self> {
        // For now, we'll create a mock Vulkan backend that delegates to CPU
        // This allows the API to work while we develop the full Vulkan implementation

        let device_info = DeviceInfo {
            name: "Mock Vulkan Device".to_string(),
            device_type: DeviceType::Vulkan,
            memory_size: Some(4_000_000_000), // 4GB mock
            compute_units: Some(32),
            supports_f16: false,
            supports_f64: false,
        };

        Ok(Self { device_info })
    }

    /// Execute a compute operation (simplified CPU fallback for now)
    pub fn execute_compute_operation(
        &self,
        operation: &str,
        input_buffers: &[Arc<VulkanBuffer>],
        output_buffers: &[Arc<VulkanBuffer>],
        uniform_data: Option<&[u32]>,
    ) -> Result<()> {
        // For now, we'll perform the operations on CPU and copy back
        // This ensures the API works while we develop proper Vulkan shaders

        match operation {
            "elementwise_add" => {
                if input_buffers.len() != 2 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for elementwise_add"));
                }

                let a_data = input_buffers[0].read_data()?;
                let b_data = input_buffers[1].read_data()?;

                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| a + b)
                    .collect();

                output_buffers[0].write_data(&result)?;
            }
            "elementwise_sub" => {
                if input_buffers.len() != 2 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for elementwise_sub"));
                }

                let a_data = input_buffers[0].read_data()?;
                let b_data = input_buffers[1].read_data()?;

                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| a - b)
                    .collect();

                output_buffers[0].write_data(&result)?;
            }
            "elementwise_mul" => {
                if input_buffers.len() != 2 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for elementwise_mul"));
                }

                let a_data = input_buffers[0].read_data()?;
                let b_data = input_buffers[1].read_data()?;

                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| a * b)
                    .collect();

                output_buffers[0].write_data(&result)?;
            }
            "elementwise_div" => {
                if input_buffers.len() != 2 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for elementwise_div"));
                }

                let a_data = input_buffers[0].read_data()?;
                let b_data = input_buffers[1].read_data()?;

                let result: Vec<f32> = a_data
                    .iter()
                    .zip(b_data.iter())
                    .map(|(&a, &b)| a / b)
                    .collect();

                output_buffers[0].write_data(&result)?;
            }
            "scalar_add" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for scalar_add"));
                }

                let scalar = if let Some(uniform) = uniform_data {
                    f32::from_bits(uniform[0])
                } else {
                    return Err(RnnError::device("Scalar operation requires uniform data"));
                };

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data.iter().map(|&x| x + scalar).collect();
                output_buffers[0].write_data(&result)?;
            }
            "scalar_mul" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for scalar_mul"));
                }

                let scalar = if let Some(uniform) = uniform_data {
                    f32::from_bits(uniform[0])
                } else {
                    return Err(RnnError::device("Scalar operation requires uniform data"));
                };

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data.iter().map(|&x| x * scalar).collect();
                output_buffers[0].write_data(&result)?;
            }
            "matrix_mul" => {
                if input_buffers.len() != 2 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for matrix_mul"));
                }

                let [m, n, k] = if let Some(uniform) = uniform_data {
                    [
                        uniform[0] as usize,
                        uniform[1] as usize,
                        uniform[2] as usize,
                    ]
                } else {
                    return Err(RnnError::device(
                        "Matrix multiplication requires dimensions",
                    ));
                };

                let a_data = input_buffers[0].read_data()?;
                let b_data = input_buffers[1].read_data()?;

                let mut result = vec![0.0; m * n];

                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0;
                        for l in 0..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        result[i * n + j] = sum;
                    }
                }

                output_buffers[0].write_data(&result)?;
            }
            "relu" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for relu"));
                }

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data.iter().map(|&x| x.max(0.0)).collect();
                output_buffers[0].write_data(&result)?;
            }
            "sigmoid" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for sigmoid"));
                }

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data
                    .iter()
                    .map(|&x| 1.0 / (1.0 + (-x).exp()))
                    .collect();
                output_buffers[0].write_data(&result)?;
            }
            "tanh" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for tanh"));
                }

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data.iter().map(|&x| x.tanh()).collect();
                output_buffers[0].write_data(&result)?;
            }
            "softmax" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for softmax"));
                }

                let input_data = input_buffers[0].read_data()?;
                let max_val = input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                let exp_data: Vec<f32> = input_data.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exp_data.iter().sum();
                let result: Vec<f32> = exp_data.iter().map(|&x| x / sum).collect();
                output_buffers[0].write_data(&result)?;
            }
            "transpose" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for transpose"));
                }

                let [rows, cols] = if let Some(uniform) = uniform_data {
                    [uniform[0] as usize, uniform[1] as usize]
                } else {
                    return Err(RnnError::device("Transpose requires dimensions"));
                };

                let input_data = input_buffers[0].read_data()?;
                let mut result = vec![0.0; rows * cols];

                for i in 0..rows {
                    for j in 0..cols {
                        result[j * rows + i] = input_data[i * cols + j];
                    }
                }

                output_buffers[0].write_data(&result)?;
            }
            "copy" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for copy"));
                }

                let input_data = input_buffers[0].read_data()?;
                output_buffers[0].write_data(&input_data)?;
            }
            "sqrt" => {
                if input_buffers.len() != 1 || output_buffers.len() != 1 {
                    return Err(RnnError::device("Invalid buffer count for sqrt"));
                }

                let input_data = input_buffers[0].read_data()?;
                let result: Vec<f32> = input_data.iter().map(|&x| x.sqrt()).collect();
                output_buffers[0].write_data(&result)?;
            }
            _ => {
                return Err(RnnError::device(&format!(
                    "Unknown operation: {}",
                    operation
                )));
            }
        }

        Ok(())
    }
}

impl Backend for VulkanBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        Ok(self.device_info.clone())
    }

    fn allocate(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        let buffer = VulkanBuffer::new(size * std::mem::size_of::<f32>())?;
        Ok(Arc::new(buffer) as Arc<dyn DeviceMemory>)
    }

    fn allocate_uniform(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        let buffer = VulkanBuffer::new(size * std::mem::size_of::<u32>())?;
        Ok(Arc::new(buffer) as Arc<dyn DeviceMemory>)
    }

    fn copy_to_device(&self, data: &[f32], memory: &dyn DeviceMemory) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| RnnError::device("Invalid memory type for Vulkan backend"))?;

        vulkan_buffer.write_data(data)
    }

    fn copy_u32_to_device(&self, data: &[u32], memory: &dyn DeviceMemory) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| RnnError::device("Invalid memory type for Vulkan backend"))?;

        vulkan_buffer.write_u32_data(data)
    }

    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()> {
        let vulkan_buffer = memory
            .as_any()
            .downcast_ref::<VulkanBuffer>()
            .ok_or_else(|| RnnError::device("Invalid memory type for Vulkan backend"))?;

        let buffer_data = vulkan_buffer.read_data()?;
        if data.len() != buffer_data.len() {
            return Err(RnnError::device("Data size mismatch"));
        }
        data.copy_from_slice(&buffer_data);
        Ok(())
    }

    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        self.execute_kernel_with_uniform(kernel, inputs, outputs, None)
    }

    fn execute_kernel_with_uniform(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        uniform: Option<&dyn DeviceMemory>,
    ) -> Result<()> {
        let vulkan_kernel = kernel
            .as_any()
            .downcast_ref::<VulkanKernel>()
            .ok_or_else(|| RnnError::device("Invalid kernel type for Vulkan backend"))?;

        // Convert memory references to VulkanBuffer
        let input_buffers: Result<Vec<_>> = inputs
            .iter()
            .map(|mem| {
                mem.as_any()
                    .downcast_ref::<VulkanBuffer>()
                    .ok_or_else(|| RnnError::device("Invalid input buffer type"))
                    .map(|buf| Arc::new(buf.clone()))
            })
            .collect();
        let input_buffers = input_buffers?;

        let output_buffers: Result<Vec<_>> = outputs
            .iter()
            .map(|mem| {
                mem.as_any()
                    .downcast_ref::<VulkanBuffer>()
                    .ok_or_else(|| RnnError::device("Invalid output buffer type"))
                    .map(|buf| Arc::new(buf.clone()))
            })
            .collect();
        let output_buffers = output_buffers?;

        // Get uniform data if provided
        let uniform_data = if let Some(uniform_mem) = uniform {
            let uniform_buffer = uniform_mem
                .as_any()
                .downcast_ref::<VulkanBuffer>()
                .ok_or_else(|| RnnError::device("Invalid uniform buffer type"))?;
            Some(uniform_buffer.read_u32_data()?)
        } else {
            None
        };

        self.execute_compute_operation(
            vulkan_kernel.name(),
            &input_buffers,
            &output_buffers,
            uniform_data.as_deref(),
        )
    }

    fn synchronize(&self) -> Result<()> {
        // No-op for simplified implementation
        Ok(())
    }

    fn is_available(&self) -> bool {
        true
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Simplified Vulkan buffer wrapper
#[derive(Debug, Clone)]
pub struct VulkanBuffer {
    data: Arc<Mutex<Vec<f32>>>,
    size_in_bytes: usize,
}

impl VulkanBuffer {
    /// Create a new Vulkan buffer
    pub fn new(size_in_bytes: usize) -> Result<Self> {
        let size_in_elements = size_in_bytes / std::mem::size_of::<f32>();
        Ok(Self {
            data: Arc::new(Mutex::new(vec![0.0; size_in_elements])),
            size_in_bytes,
        })
    }

    /// Write f32 data to buffer
    pub fn write_data(&self, data: &[f32]) -> Result<()> {
        let mut buffer = self.data.lock().unwrap();
        if data.len() != buffer.len() {
            return Err(RnnError::device("Data size mismatch"));
        }
        buffer.copy_from_slice(data);
        Ok(())
    }

    /// Write u32 data to buffer (for uniform buffers)
    pub fn write_u32_data(&self, data: &[u32]) -> Result<()> {
        let mut buffer = self.data.lock().unwrap();
        if data.len() * std::mem::size_of::<u32>() != self.size_in_bytes {
            return Err(RnnError::device("Data size mismatch for u32 data"));
        }

        // Convert u32 to f32 for storage (bit representation)
        for (i, &val) in data.iter().enumerate() {
            buffer[i] = f32::from_bits(val);
        }
        Ok(())
    }

    /// Read f32 data from buffer
    pub fn read_data(&self) -> Result<Vec<f32>> {
        let buffer = self.data.lock().unwrap();
        Ok(buffer.clone())
    }

    /// Read u32 data from buffer
    pub fn read_u32_data(&self) -> Result<Vec<u32>> {
        let buffer = self.data.lock().unwrap();
        let u32_data: Vec<u32> = buffer.iter().map(|&x| x.to_bits()).collect();
        Ok(u32_data)
    }
}

impl DeviceMemory for VulkanBuffer {
    fn size(&self) -> usize {
        self.size_in_bytes
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Vulkan
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// Vulkan compute kernel
#[derive(Debug)]
pub struct VulkanKernel {
    name: String,
    dispatch_size: [u32; 3],
}

impl VulkanKernel {
    /// Create a new Vulkan kernel
    pub fn new(name: String, dispatch_size: [u32; 3]) -> Self {
        Self {
            name,
            dispatch_size,
        }
    }

    /// Create kernel for element-wise operations
    pub fn elementwise(name: String, size: u32) -> Self {
        Self::new(name, [size.div_ceil(64), 1, 1])
    }

    /// Create kernel for matrix operations
    pub fn matrix(name: String, rows: u32, cols: u32) -> Self {
        Self::new(name, [cols.div_ceil(16), rows.div_ceil(16), 1])
    }

    /// Create kernel for reduction operations
    pub fn reduction(name: String, size: u32) -> Self {
        Self::new(name, [size.div_ceil(256), 1, 1])
    }
}

impl Kernel for VulkanKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_size(&self) -> Option<[u32; 3]> {
        Some(self.dispatch_size)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vulkan_backend_creation() {
        let backend = VulkanBackend::new().unwrap();
        let info = backend.device_info().unwrap();
        assert_eq!(info.device_type, DeviceType::Vulkan);
        println!("Vulkan device: {}", info.name);
    }

    #[test]
    fn test_vulkan_buffer_operations() {
        let backend = VulkanBackend::new().unwrap();
        let memory = backend.allocate(1024).unwrap();
        assert_eq!(memory.size(), 1024 * std::mem::size_of::<f32>());
        assert_eq!(memory.device_type(), DeviceType::Vulkan);

        let test_data = vec![1.0, 2.0, 3.0, 4.0];
        let memory = backend.allocate(4).unwrap(); // 4 elements
        backend.copy_to_device(&test_data, memory.as_ref()).unwrap();

        let mut result = vec![0.0; 4];
        backend.copy_to_host(memory.as_ref(), &mut result).unwrap();
        assert_eq!(result, test_data);
    }

    #[test]
    fn test_elementwise_operations() {
        let backend = VulkanBackend::new().unwrap();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let mem_a = backend.allocate(4).unwrap();
        let mem_b = backend.allocate(4).unwrap();
        let mem_c = backend.allocate(4).unwrap();

        backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
        backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

        let kernel = VulkanKernel::elementwise("elementwise_add".to_string(), 4);
        backend
            .execute_kernel(
                &kernel,
                &[mem_a.as_ref(), mem_b.as_ref()],
                &[mem_c.as_ref()],
            )
            .unwrap();

        let mut result = vec![0.0; 4];
        backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

        let expected = vec![3.0, 5.0, 7.0, 9.0];
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matrix_multiplication() {
        let backend = VulkanBackend::new().unwrap();

        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]; // 3x2

        let mem_a = backend.allocate(6).unwrap(); // 6 elements
        let mem_b = backend.allocate(6).unwrap(); // 6 elements
        let mem_c = backend.allocate(4).unwrap(); // 4 elements

        backend.copy_to_device(&a, mem_a.as_ref()).unwrap();
        backend.copy_to_device(&b, mem_b.as_ref()).unwrap();

        // Create uniform buffer for dimensions
        let dims = [2u32, 2u32, 3u32]; // M, N, K
        let uniform_mem = backend.allocate_uniform(3).unwrap();
        backend
            .copy_u32_to_device(&dims, uniform_mem.as_ref())
            .unwrap();

        let kernel = VulkanKernel::matrix("matrix_mul".to_string(), 2, 2);
        backend
            .execute_kernel_with_uniform(
                &kernel,
                &[mem_a.as_ref(), mem_b.as_ref()],
                &[mem_c.as_ref()],
                Some(uniform_mem.as_ref()),
            )
            .unwrap();

        let mut result = vec![0.0; 4];
        backend.copy_to_host(mem_c.as_ref(), &mut result).unwrap();

        // Expected: [58, 64, 139, 154]
        let expected = vec![58.0, 64.0, 139.0, 154.0];
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}
