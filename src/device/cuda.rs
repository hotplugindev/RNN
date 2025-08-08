//! CUDA backend implementation for NVIDIA GPUs
//!
//! This module provides high-performance CUDA support for neural network operations
//! using the cudarc crate for CUDA runtime bindings.

use crate::device::{Backend, DeviceInfo, DeviceMemory, DeviceType, Kernel};
use crate::error::{NnlError, Result};
use cudarc::driver::{CudaDevice, CudaFunction, CudaSlice, DevicePtr, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;
use std::collections::HashMap;
use std::sync::Arc;

/// CUDA backend implementation
pub struct CudaBackend {
    device: Arc<CudaDevice>,
    device_info: CudaDeviceInfo,
    kernels: HashMap<String, CudaFunction>,
}

/// CUDA device information
#[derive(Debug, Clone)]
pub struct CudaDeviceInfo {
    pub name: String,
    pub compute_capability: (i32, i32),
    pub total_memory: usize,
    pub multiprocessor_count: i32,
    pub max_threads_per_block: i32,
    pub warp_size: i32,
}

impl CudaBackend {
    /// Create a new CUDA backend
    pub fn new() -> Result<Self> {
        Self::new_with_device(0)
    }

    /// Create a new CUDA backend with specific device ID
    pub fn new_with_device(device_id: usize) -> Result<Self> {
        let device = CudaDevice::new(device_id).map_err(|e| {
            NnlError::cuda(format!(
                "Failed to initialize CUDA device {}: {}",
                device_id, e
            ))
        })?;

        let device = Arc::new(device);
        let device_info = Self::get_device_info(&device)?;

        Ok(Self {
            device,
            device_info,
            kernels: HashMap::new(),
        })
    }

    /// Get CUDA device information
    fn get_device_info(device: &CudaDevice) -> Result<CudaDeviceInfo> {
        let name = device
            .name()
            .map_err(|e| NnlError::cuda(format!("Failed to get device name: {}", e)))?;

        let (major, minor) = device
            .compute_capability()
            .map_err(|e| NnlError::cuda(format!("Failed to get compute capability: {}", e)))?;

        let total_memory = device
            .total_memory()
            .map_err(|e| NnlError::cuda(format!("Failed to get total memory: {}", e)))?;

        let multiprocessor_count = device
            .multiprocessor_count()
            .map_err(|e| NnlError::cuda(format!("Failed to get multiprocessor count: {}", e)))?;

        let max_threads_per_block = device
            .max_threads_per_block()
            .map_err(|e| NnlError::cuda(format!("Failed to get max threads per block: {}", e)))?;

        let warp_size = device
            .warp_size()
            .map_err(|e| NnlError::cuda(format!("Failed to get warp size: {}", e)))?;

        Ok(CudaDeviceInfo {
            name,
            compute_capability: (major, minor),
            total_memory,
            multiprocessor_count,
            max_threads_per_block,
            warp_size,
        })
    }

    /// Load and compile a CUDA kernel
    pub fn load_kernel(&mut self, name: &str, source: &str, function_name: &str) -> Result<()> {
        let ptx = Ptx::compile(source).map_err(|e| {
            NnlError::cuda(format!("Failed to compile CUDA kernel '{}': {}", name, e))
        })?;

        self.device
            .load_ptx(ptx, "neural_network_module", &[function_name])
            .map_err(|e| {
                NnlError::cuda(format!("Failed to load PTX for kernel '{}': {}", name, e))
            })?;

        let function = self
            .device
            .get_func("neural_network_module", function_name)
            .map_err(|e| {
                NnlError::cuda(format!("Failed to get function '{}': {}", function_name, e))
            })?;

        self.kernels.insert(name.to_string(), function);
        Ok(())
    }

    /// Get CUDA device reference
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get device information
    pub fn device_info(&self) -> &CudaDeviceInfo {
        &self.device_info
    }
}

impl Backend for CudaBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: format!("CUDA - {}", self.device_info.name),
            device_type: DeviceType::Cuda,
            memory_size: Some(self.device_info.total_memory as u64),
            compute_units: Some(self.device_info.multiprocessor_count as u32),
            supports_f16: self.device_info.compute_capability.0 >= 5, // FP16 support from compute 5.3+
            supports_f64: true,
        })
    }

    fn allocate(&self, size: usize) -> Result<Box<dyn DeviceMemory>> {
        CudaMemory::new(self.device.clone(), size).map(|m| Box::new(m) as Box<dyn DeviceMemory>)
    }

    fn copy_to_device(&self, data: &[f32], memory: &mut dyn DeviceMemory) -> Result<()> {
        let cuda_memory = memory
            .as_any_mut()
            .downcast_mut::<CudaMemory>()
            .ok_or_else(|| NnlError::device("Invalid memory type for CUDA backend"))?;

        cuda_memory.copy_from_host(data)
    }

    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()> {
        let cuda_memory = memory
            .as_any()
            .downcast_ref::<CudaMemory>()
            .ok_or_else(|| NnlError::device("Invalid memory type for CUDA backend"))?;

        cuda_memory.copy_to_host(data)
    }

    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&mut dyn DeviceMemory],
    ) -> Result<()> {
        let cuda_kernel = kernel
            .as_any()
            .downcast_ref::<CudaKernel>()
            .ok_or_else(|| NnlError::device("Invalid kernel type for CUDA backend"))?;

        cuda_kernel.execute(inputs, outputs, &self.device)
    }

    fn synchronize(&self) -> Result<()> {
        self.device
            .synchronize()
            .map_err(|e| NnlError::cuda(format!("Failed to synchronize CUDA device: {}", e)))
    }

    fn is_available(&self) -> bool {
        CudaDevice::new(0).is_ok()
    }
}

/// CUDA memory implementation
pub struct CudaMemory {
    buffer: CudaSlice<f32>,
    device: Arc<CudaDevice>,
    size: usize,
}

impl CudaMemory {
    /// Create new CUDA memory buffer
    pub fn new(device: Arc<CudaDevice>, size: usize) -> Result<Self> {
        if size == 0 {
            return Err(NnlError::memory("Cannot allocate zero-sized CUDA memory"));
        }

        let buffer = device
            .alloc_zeros::<f32>(size)
            .map_err(|e| NnlError::cuda(format!("Failed to allocate CUDA memory: {}", e)))?;

        Ok(Self {
            buffer,
            device,
            size,
        })
    }

    /// Copy data from host to device
    pub fn copy_from_host(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.size {
            return Err(NnlError::shape_mismatch(&[self.size], &[data.len()]));
        }

        self.device
            .htod_copy_into(data, &mut self.buffer)
            .map_err(|e| NnlError::cuda(format!("Failed to copy data to CUDA device: {}", e)))
    }

    /// Copy data from device to host
    pub fn copy_to_host(&self, data: &mut [f32]) -> Result<()> {
        if data.len() != self.size {
            return Err(NnlError::shape_mismatch(&[self.size], &[data.len()]));
        }

        self.device
            .dtoh_sync_copy_into(&self.buffer, data)
            .map_err(|e| NnlError::cuda(format!("Failed to copy data from CUDA device: {}", e)))
    }

    /// Get device pointer
    pub fn device_ptr(&self) -> DevicePtr<f32> {
        *self.buffer.device_ptr()
    }

    /// Get the CUDA slice
    pub fn slice(&self) -> &CudaSlice<f32> {
        &self.buffer
    }

    /// Get mutable CUDA slice
    pub fn slice_mut(&mut self) -> &mut CudaSlice<f32> {
        &mut self.buffer
    }
}

impl DeviceMemory for CudaMemory {
    fn size(&self) -> usize {
        self.size
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cuda
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

/// CUDA kernel implementation
pub struct CudaKernel {
    name: String,
    function: CudaFunction,
    block_size: (u32, u32, u32),
    shared_memory: u32,
}

impl CudaKernel {
    /// Create a new CUDA kernel
    pub fn new(
        name: String,
        function: CudaFunction,
        block_size: (u32, u32, u32),
        shared_memory: u32,
    ) -> Self {
        Self {
            name,
            function,
            block_size,
            shared_memory,
        }
    }

    /// Execute the kernel
    pub fn execute(
        &self,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&mut dyn DeviceMemory],
        device: &CudaDevice,
    ) -> Result<()> {
        if inputs.is_empty() || outputs.is_empty() {
            return Err(NnlError::device(
                "CUDA kernel requires at least one input and output",
            ));
        }

        // For simplicity, assume the kernel operates on the first input/output pair
        let input_memory = inputs[0]
            .as_any()
            .downcast_ref::<CudaMemory>()
            .ok_or_else(|| NnlError::device("Invalid input memory type for CUDA kernel"))?;

        let output_memory = outputs[0]
            .as_any()
            .downcast_ref::<CudaMemory>()
            .ok_or_else(|| NnlError::device("Invalid output memory type for CUDA kernel"))?;

        let num_elements = input_memory.size() as u32;
        let grid_size = (
            (num_elements + self.block_size.0 - 1) / self.block_size.0,
            1,
            1,
        );

        let config = LaunchConfig {
            grid_dim: grid_size,
            block_dim: self.block_size,
            shared_mem_bytes: self.shared_memory,
        };

        // Launch kernel with input and output pointers
        unsafe {
            self.function.launch(
                config,
                (
                    &input_memory.device_ptr(),
                    &output_memory.device_ptr(),
                    &num_elements,
                ),
            )
        }
        .map_err(|e| {
            NnlError::cuda(format!(
                "Failed to launch CUDA kernel '{}': {}",
                self.name, e
            ))
        })
    }
}

impl Kernel for CudaKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_size(&self) -> Option<[u32; 3]> {
        Some([self.block_size.0, self.block_size.1, self.block_size.2])
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// CUDA kernel sources for common operations
pub mod kernels {
    /// Element-wise addition kernel
    pub const ELEMENTWISE_ADD: &str = r#"
        extern "C" __global__ void elementwise_add(
            const float* input_a,
            const float* input_b,
            float* output,
            unsigned int n
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input_a[idx] + input_b[idx];
            }
        }
    "#;

    /// Element-wise multiplication kernel
    pub const ELEMENTWISE_MUL: &str = r#"
        extern "C" __global__ void elementwise_mul(
            const float* input_a,
            const float* input_b,
            float* output,
            unsigned int n
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = input_a[idx] * input_b[idx];
            }
        }
    "#;

    /// ReLU activation kernel
    pub const RELU: &str = r#"
        extern "C" __global__ void relu(
            const float* input,
            float* output,
            unsigned int n
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = fmaxf(0.0f, input[idx]);
            }
        }
    "#;

    /// Sigmoid activation kernel
    pub const SIGMOID: &str = r#"
        extern "C" __global__ void sigmoid(
            const float* input,
            float* output,
            unsigned int n
        ) {
            unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
            if (idx < n) {
                output[idx] = 1.0f / (1.0f + expf(-input[idx]));
            }
        }
    "#;

    /// Matrix multiplication kernel (naive implementation)
    pub const MATRIX_MUL: &str = r#"
        extern "C" __global__ void matrix_mul(
            const float* matrix_a,
            const float* matrix_b,
            float* matrix_c,
            unsigned int M,
            unsigned int N,
            unsigned int K
        ) {
            unsigned int row = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int col = blockIdx.x * blockDim.x + threadIdx.x;

            if (row < M && col < N) {
                float sum = 0.0f;
                for (unsigned int k = 0; k < K; k++) {
                    sum += matrix_a[row * K + k] * matrix_b[k * N + col];
                }
                matrix_c[row * N + col] = sum;
            }
        }
    "#;

    /// Optimized matrix multiplication with shared memory
    pub const MATRIX_MUL_SHARED: &str = r#"
        #define TILE_SIZE 16

        extern "C" __global__ void matrix_mul_shared(
            const float* matrix_a,
            const float* matrix_b,
            float* matrix_c,
            unsigned int M,
            unsigned int N,
            unsigned int K
        ) {
            __shared__ float tile_a[TILE_SIZE][TILE_SIZE];
            __shared__ float tile_b[TILE_SIZE][TILE_SIZE];

            unsigned int row = blockIdx.y * TILE_SIZE + threadIdx.y;
            unsigned int col = blockIdx.x * TILE_SIZE + threadIdx.x;

            float sum = 0.0f;

            for (unsigned int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
                // Load tiles into shared memory
                if (row < M && tile * TILE_SIZE + threadIdx.x < K) {
                    tile_a[threadIdx.y][threadIdx.x] = matrix_a[row * K + tile * TILE_SIZE + threadIdx.x];
                } else {
                    tile_a[threadIdx.y][threadIdx.x] = 0.0f;
                }

                if (col < N && tile * TILE_SIZE + threadIdx.y < K) {
                    tile_b[threadIdx.y][threadIdx.x] = matrix_b[(tile * TILE_SIZE + threadIdx.y) * N + col];
                } else {
                    tile_b[threadIdx.y][threadIdx.x] = 0.0f;
                }

                __syncthreads();

                // Compute partial dot product
                for (unsigned int k = 0; k < TILE_SIZE; k++) {
                    sum += tile_a[threadIdx.y][k] * tile_b[k][threadIdx.x];
                }

                __syncthreads();
            }

            if (row < M && col < N) {
                matrix_c[row * N + col] = sum;
            }
        }
    "#;

    /// Convolution 2D kernel
    pub const CONV2D: &str = r#"
        extern "C" __global__ void conv2d(
            const float* input,
            const float* kernel,
            float* output,
            unsigned int input_height,
            unsigned int input_width,
            unsigned int kernel_height,
            unsigned int kernel_width,
            unsigned int output_height,
            unsigned int output_width,
            unsigned int stride_y,
            unsigned int stride_x,
            unsigned int pad_y,
            unsigned int pad_x
        ) {
            unsigned int out_y = blockIdx.y * blockDim.y + threadIdx.y;
            unsigned int out_x = blockIdx.x * blockDim.x + threadIdx.x;

            if (out_y < output_height && out_x < output_width) {
                float sum = 0.0f;

                for (unsigned int ky = 0; ky < kernel_height; ky++) {
                    for (unsigned int kx = 0; kx < kernel_width; kx++) {
                        int in_y = out_y * stride_y - pad_y + ky;
                        int in_x = out_x * stride_x - pad_x + kx;

                        if (in_y >= 0 && in_y < input_height && in_x >= 0 && in_x < input_width) {
                            unsigned int input_idx = in_y * input_width + in_x;
                            unsigned int kernel_idx = ky * kernel_width + kx;
                            sum += input[input_idx] * kernel[kernel_idx];
                        }
                    }
                }

                unsigned int output_idx = out_y * output_width + out_x;
                output[output_idx] = sum;
            }
        }
    "#;
}

/// Utility functions for CUDA operations
impl CudaBackend {
    /// Initialize common neural network kernels
    pub fn init_nn_kernels(&mut self) -> Result<()> {
        self.load_kernel(
            "elementwise_add",
            kernels::ELEMENTWISE_ADD,
            "elementwise_add",
        )?;
        self.load_kernel(
            "elementwise_mul",
            kernels::ELEMENTWISE_MUL,
            "elementwise_mul",
        )?;
        self.load_kernel("relu", kernels::RELU, "relu")?;
        self.load_kernel("sigmoid", kernels::SIGMOID, "sigmoid")?;
        self.load_kernel("matrix_mul", kernels::MATRIX_MUL, "matrix_mul")?;
        self.load_kernel(
            "matrix_mul_shared",
            kernels::MATRIX_MUL_SHARED,
            "matrix_mul_shared",
        )?;
        self.load_kernel("conv2d", kernels::CONV2D, "conv2d")?;
        Ok(())
    }

    /// Get optimal block size for given problem size
    pub fn get_optimal_block_size(&self, problem_size: u32) -> (u32, u32, u32) {
        let max_threads = self.device_info.max_threads_per_block as u32;
        let warp_size = self.device_info.warp_size as u32;

        // Use multiple of warp size, up to max_threads
        let block_size = if problem_size >= max_threads {
            max_threads
        } else {
            ((problem_size + warp_size - 1) / warp_size) * warp_size
        };

        (block_size, 1, 1)
    }

    /// Get optimal grid size for given problem and block size
    pub fn get_optimal_grid_size(&self, problem_size: u32, block_size: u32) -> (u32, u32, u32) {
        let grid_size = (problem_size + block_size - 1) / block_size;
        (grid_size, 1, 1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cuda_availability() {
        // This test will pass if CUDA is available
        match CudaBackend::new() {
            Ok(backend) => {
                assert!(backend.is_available());
                println!("CUDA backend created successfully");
                println!("Device: {}", backend.device_info().name);
                println!(
                    "Compute capability: {:?}",
                    backend.device_info().compute_capability
                );
            }
            Err(e) => {
                println!(
                    "CUDA backend creation failed (expected on systems without CUDA): {}",
                    e
                );
            }
        }
    }

    #[test]
    fn test_cuda_memory_allocation() {
        if let Ok(backend) = CudaBackend::new() {
            let memory = backend.allocate(1024);
            assert!(memory.is_ok());

            let memory = memory.unwrap();
            assert_eq!(memory.size(), 1024);
            assert_eq!(memory.device_type(), DeviceType::Cuda);
        }
    }

    #[test]
    fn test_cuda_memory_copy() {
        if let Ok(backend) = CudaBackend::new() {
            let mut memory = backend.allocate(4).unwrap();

            let input_data = vec![1.0, 2.0, 3.0, 4.0];
            let copy_result = backend.copy_to_device(&input_data, memory.as_mut());
            assert!(copy_result.is_ok());

            let mut output_data = vec![0.0; 4];
            let copy_result = backend.copy_to_host(memory.as_ref(), &mut output_data);
            assert!(copy_result.is_ok());

            assert_eq!(input_data, output_data);
        }
    }

    #[test]
    fn test_kernel_compilation() {
        if let Ok(mut backend) = CudaBackend::new() {
            let result = backend.load_kernel("test_relu", kernels::RELU, "relu");
            assert!(result.is_ok());
            assert!(backend.kernels.contains_key("test_relu"));
        }
    }
}
