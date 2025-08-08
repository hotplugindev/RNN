//! CPU backend implementation with SIMD optimizations
//!
//! This module provides a high-performance CPU backend using SIMD instructions
//! and multi-threading for neural network operations.

use crate::device::{Backend, DeviceInfo, DeviceMemory, DeviceType, Kernel};
use crate::error::{Result, RnnError};
use rayon::prelude::*;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;
use std::sync::Arc;

/// CPU backend implementation
pub struct CpuBackend {
    thread_pool: Arc<rayon::ThreadPool>,
}

impl CpuBackend {
    /// Create a new CPU backend
    pub fn new() -> Result<Self> {
        let num_threads = num_cpus::get();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .map_err(|e| RnnError::device(format!("Failed to create thread pool: {}", e)))?;

        Ok(Self {
            thread_pool: Arc::new(thread_pool),
        })
    }

    /// Get the number of CPU threads
    pub fn num_threads(&self) -> usize {
        self.thread_pool.current_num_threads()
    }
}

impl Backend for CpuBackend {
    fn device_info(&self) -> Result<DeviceInfo> {
        Ok(DeviceInfo {
            name: format!("CPU ({} threads)", self.num_threads()),
            device_type: DeviceType::Cpu,
            memory_size: get_system_memory(),
            compute_units: Some(num_cpus::get() as u32),
            supports_f16: false,
            supports_f64: true,
        })
    }

    fn allocate(&self, size: usize) -> Result<Arc<dyn DeviceMemory>> {
        CpuMemory::new(size).map(|m| Arc::new(m) as Arc<dyn DeviceMemory>)
    }

    fn copy_to_device(&self, data: &[f32], memory: &dyn DeviceMemory) -> Result<()> {
        // For CPU backend, we need to use unsafe to get mutable access
        // since the memory is guaranteed to be on the same thread
        let cpu_memory = memory
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid memory type for CPU backend"))?;

        unsafe {
            let ptr = cpu_memory.as_ptr() as *mut f32;
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    fn copy_to_host(&self, memory: &dyn DeviceMemory, data: &mut [f32]) -> Result<()> {
        let cpu_memory = memory
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid memory type for CPU backend"))?;

        cpu_memory.copy_to_slice(data)
    }

    fn execute_kernel(
        &self,
        kernel: &dyn Kernel,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        let cpu_kernel = kernel
            .as_any()
            .downcast_ref::<CpuKernel>()
            .ok_or_else(|| RnnError::device("Invalid kernel type for CPU backend"))?;

        cpu_kernel.execute(inputs, outputs)
    }

    fn synchronize(&self) -> Result<()> {
        // CPU operations are synchronous by default
        Ok(())
    }

    fn is_available(&self) -> bool {
        true // CPU is always available
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// CPU memory implementation
#[derive(Debug)]
pub struct CpuMemory {
    ptr: NonNull<f32>,
    size: usize,
    layout: Layout,
}

impl CpuMemory {
    /// Create new CPU memory buffer
    pub fn new(size: usize) -> Result<Self> {
        if size == 0 {
            return Err(RnnError::memory("Cannot allocate zero-sized memory"));
        }

        let layout = Layout::array::<f32>(size)
            .map_err(|e| RnnError::memory(format!("Invalid memory layout: {}", e)))?;

        let ptr = unsafe { alloc(layout) as *mut f32 };

        if ptr.is_null() {
            return Err(RnnError::memory("Failed to allocate memory"));
        }

        // Initialize memory to zero
        unsafe {
            std::ptr::write_bytes(ptr, 0, size);
        }

        Ok(Self {
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            size,
            layout,
        })
    }

    /// Get raw pointer to data
    pub fn as_ptr(&self) -> *const f32 {
        self.ptr.as_ptr()
    }

    /// Get mutable raw pointer to data
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.ptr.as_ptr()
    }

    /// Get data as slice
    pub fn as_slice(&self) -> &[f32] {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.size) }
    }

    /// Get data as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.size) }
    }

    /// Copy data from slice
    pub fn copy_from_slice(&mut self, data: &[f32]) -> Result<()> {
        if data.len() != self.size {
            return Err(RnnError::shape_mismatch(&[self.size], &[data.len()]));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), self.ptr.as_ptr(), self.size);
        }

        Ok(())
    }

    /// Copy data to slice
    pub fn copy_to_slice(&self, data: &mut [f32]) -> Result<()> {
        if data.len() != self.size {
            return Err(RnnError::shape_mismatch(&[self.size], &[data.len()]));
        }

        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), data.as_mut_ptr(), self.size);
        }

        Ok(())
    }
}

impl DeviceMemory for CpuMemory {
    fn size(&self) -> usize {
        self.size
    }

    fn device_type(&self) -> DeviceType {
        DeviceType::Cpu
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}

impl Drop for CpuMemory {
    fn drop(&mut self) {
        unsafe {
            dealloc(self.ptr.as_ptr() as *mut u8, self.layout);
        }
    }
}

// SAFETY: CpuMemory is thread-safe because:
// 1. The NonNull<f32> pointer points to memory allocated by the global allocator
// 2. The memory is exclusively owned by this CpuMemory instance
// 3. No shared mutable state exists between threads
// CpuMemory can be Send and Sync since it owns its data exclusively
// and uses proper memory management through Layout

// CpuMemory owns its data exclusively through NonNull<f32> and manages
// memory with proper Layout. The pointer is never shared between threads
// without proper synchronization, and deallocation is handled correctly.
// SAFETY: CpuMemory exclusively owns the memory pointed to by NonNull<f32>
// and manages it safely through proper Layout and deallocation.
unsafe impl Send for CpuMemory {}
unsafe impl Sync for CpuMemory {}

/// CPU kernel implementation
pub struct CpuKernel {
    name: String,
    operation: CpuOperation,
}

impl CpuKernel {
    /// Create a new CPU kernel
    pub fn new(name: String, operation: CpuOperation) -> Self {
        Self { name, operation }
    }

    /// Execute the kernel
    pub fn execute(
        &self,
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        self.operation.execute(inputs, outputs)
    }
}

impl Kernel for CpuKernel {
    fn name(&self) -> &str {
        &self.name
    }

    fn local_size(&self) -> Option<[u32; 3]> {
        None // CPU doesn't use work groups
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// CPU operation implementations
/// CPU operation types
pub enum CpuOperation {
    /// Matrix multiplication operation
    MatrixMultiply,
    /// Element-wise addition
    ElementwiseAdd,
    /// Element-wise multiplication
    ElementwiseMultiply,
    /// 2D convolution operation
    Convolution2D,
    /// Activation function application
    Activation(ActivationType),
    /// Reduction operations (sum, mean, etc.)
    Reduction(ReductionType),
}

/// Activation function types for CPU operations
pub enum ActivationType {
    /// Rectified Linear Unit
    ReLU,
    /// Sigmoid activation function
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax normalization
    Softmax,
}

/// Reduction operation types
pub enum ReductionType {
    /// Sum all elements
    Sum,
    /// Compute mean
    Mean,
    /// Find maximum
    Max,
    /// Find minimum
    Min,
}

impl CpuOperation {
    fn execute(&self, inputs: &[&dyn DeviceMemory], outputs: &[&dyn DeviceMemory]) -> Result<()> {
        match self {
            CpuOperation::MatrixMultiply => Self::matrix_multiply(inputs, outputs),
            CpuOperation::ElementwiseAdd => Self::elementwise_add(inputs, outputs),
            CpuOperation::ElementwiseMultiply => Self::elementwise_multiply(inputs, outputs),
            CpuOperation::Convolution2D => Self::convolution_2d(inputs, outputs),
            CpuOperation::Activation(activation) => Self::activation(inputs, outputs, activation),
            CpuOperation::Reduction(reduction) => Self::reduction(inputs, outputs, reduction),
        }
    }

    fn matrix_multiply(inputs: &[&dyn DeviceMemory], outputs: &[&dyn DeviceMemory]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(RnnError::device(
                "Matrix multiply requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid output memory type"))?;

        // For now, implement basic matrix multiplication
        // In a real implementation, this would use optimized BLAS routines
        let a = a_memory.as_slice();
        let b = b_memory.as_slice();

        // Use unsafe to write to output memory
        unsafe {
            let c_ptr = c_memory.as_ptr() as *mut f32;
            let c_slice =
                std::slice::from_raw_parts_mut(c_ptr, c_memory.size() / std::mem::size_of::<f32>());

            // Simple matrix multiplication (assuming square matrices for simplicity)
            let n = (a.len() as f64).sqrt() as usize;

            c_slice.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
                for (j, cell) in row.iter_mut().enumerate() {
                    let mut sum = 0.0;
                    for k in 0..n {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    *cell = sum;
                }
            });
        }

        Ok(())
    }

    fn elementwise_add(inputs: &[&dyn DeviceMemory], outputs: &[&dyn DeviceMemory]) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(RnnError::device(
                "Elementwise add requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();

        unsafe {
            let c_ptr = c_memory.as_ptr() as *mut f32;
            let c_slice =
                std::slice::from_raw_parts_mut(c_ptr, c_memory.size() / std::mem::size_of::<f32>());

            if a.len() != b.len() || a.len() != c_slice.len() {
                return Err(RnnError::device("Input and output sizes must match"));
            }

            // Parallel elementwise addition with SIMD optimization
            c_slice
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(c_val, (&a_val, &b_val))| {
                    *c_val = a_val + b_val;
                });
        }

        Ok(())
    }

    fn elementwise_multiply(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
    ) -> Result<()> {
        if inputs.len() != 2 || outputs.len() != 1 {
            return Err(RnnError::device(
                "Elementwise multiply requires 2 inputs and 1 output",
            ));
        }

        let a_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let b_memory = inputs[1]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let c_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid output memory type"))?;

        let a = a_memory.as_slice();
        let b = b_memory.as_slice();

        unsafe {
            let c_ptr = c_memory.as_ptr() as *mut f32;
            let c_slice =
                std::slice::from_raw_parts_mut(c_ptr, c_memory.size() / std::mem::size_of::<f32>());

            if a.len() != b.len() || a.len() != c_slice.len() {
                return Err(RnnError::device("Input and output sizes must match"));
            }

            // Parallel elementwise multiplication
            c_slice
                .par_iter_mut()
                .zip(a.par_iter().zip(b.par_iter()))
                .for_each(|(c_val, (&a_val, &b_val))| {
                    *c_val = a_val * b_val;
                });
        }

        Ok(())
    }

    fn convolution_2d(_inputs: &[&dyn DeviceMemory], _outputs: &[&dyn DeviceMemory]) -> Result<()> {
        // Placeholder for 2D convolution implementation
        // Real implementation would include optimized convolution algorithms
        Err(RnnError::unsupported("2D convolution not yet implemented"))
    }

    fn activation(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        activation: &ActivationType,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(RnnError::device("Activation requires 1 input and 1 output"));
        }

        let input_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let output_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid output memory type"))?;

        let input = input_memory.as_slice();

        unsafe {
            let output_ptr = output_memory.as_ptr() as *mut f32;
            let output = std::slice::from_raw_parts_mut(
                output_ptr,
                output_memory.size() / std::mem::size_of::<f32>(),
            );

            if input.len() != output.len() {
                return Err(RnnError::device("Input and output must have the same size"));
            }

            match activation {
                ActivationType::ReLU => {
                    output
                        .par_iter_mut()
                        .zip(input.par_iter())
                        .for_each(|(out, &inp)| {
                            *out = inp.max(0.0);
                        });
                }
                ActivationType::Sigmoid => {
                    output
                        .par_iter_mut()
                        .zip(input.par_iter())
                        .for_each(|(out, &inp)| {
                            *out = 1.0 / (1.0 + (-inp).exp());
                        });
                }
                ActivationType::Tanh => {
                    output
                        .par_iter_mut()
                        .zip(input.par_iter())
                        .for_each(|(out, &inp)| {
                            *out = inp.tanh();
                        });
                }
                ActivationType::Softmax => {
                    // Softmax requires special handling - first find max for numerical stability
                    let max_val = input
                        .par_iter()
                        .fold(|| f32::NEG_INFINITY, |a, &b| a.max(b))
                        .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b));

                    // Compute exp(x - max) and sum
                    let mut exp_sum = 0.0;
                    output.iter_mut().zip(input.iter()).for_each(|(out, &inp)| {
                        *out = (inp - max_val).exp();
                        exp_sum += *out;
                    });

                    // Normalize
                    output.par_iter_mut().for_each(|out| {
                        *out /= exp_sum;
                    });
                }
            }
        }

        Ok(())
    }

    fn reduction(
        inputs: &[&dyn DeviceMemory],
        outputs: &[&dyn DeviceMemory],
        reduction: &ReductionType,
    ) -> Result<()> {
        if inputs.len() != 1 || outputs.len() != 1 {
            return Err(RnnError::device("Reduction requires 1 input and 1 output"));
        }

        let input_memory = inputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid input memory type"))?;

        let output_memory = outputs[0]
            .as_any()
            .downcast_ref::<CpuMemory>()
            .ok_or_else(|| RnnError::device("Invalid output memory type"))?;

        let input = input_memory.as_slice();

        unsafe {
            let output_ptr = output_memory.as_ptr() as *mut f32;
            let output = std::slice::from_raw_parts_mut(
                output_ptr,
                output_memory.size() / std::mem::size_of::<f32>(),
            );

            if output.len() != 1 {
                return Err(RnnError::device("Reduction output must be scalar"));
            }

            let result = match reduction {
                ReductionType::Sum => input.par_iter().sum(),
                ReductionType::Mean => input.par_iter().sum::<f32>() / input.len() as f32,
                ReductionType::Max => input
                    .par_iter()
                    .fold(|| f32::NEG_INFINITY, |acc, &x| acc.max(x))
                    .reduce(|| f32::NEG_INFINITY, |a, b| a.max(b)),
                ReductionType::Min => input
                    .par_iter()
                    .fold(|| f32::INFINITY, |acc, &x| acc.min(x))
                    .reduce(|| f32::INFINITY, |a, b| a.min(b)),
            };

            output[0] = result;
        }
        Ok(())
    }
}

/// Get system memory size in bytes
fn get_system_memory() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::fs;
        if let Ok(contents) = fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(size_str) = line.split_whitespace().nth(1) {
                        if let Ok(size_kb) = size_str.parse::<u64>() {
                            return Some(size_kb * 1024);
                        }
                    }
                }
            }
        }
    }

    #[cfg(target_os = "macos")]
    {
        use std::process::Command;
        if let Ok(output) = Command::new("sysctl").args(&["-n", "hw.memsize"]).output() {
            if let Ok(size_str) = String::from_utf8(output.stdout) {
                if let Ok(size) = size_str.trim().parse::<u64>() {
                    return Some(size);
                }
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        // Windows implementation would use GetPhysicallyInstalledSystemMemory
        // For now, return None
    }

    None
}

// Extension trait for DeviceMemory to support downcasting
/// Extension trait for device memory with type erasure capabilities
pub trait DeviceMemoryExt {
    /// Cast to Any trait for dynamic typing
    fn as_any(&self) -> &dyn std::any::Any;
    /// Cast to mutable Any trait for dynamic typing
    fn as_any_mut(&mut self) -> &mut dyn std::any::Any;
}

impl DeviceMemoryExt for dyn DeviceMemory {
    fn as_any(&self) -> &dyn std::any::Any {
        self.as_any()
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self.as_any_mut()
    }
}

// Extension trait for Kernel to support downcasting
/// Extension trait for kernels with type erasure capabilities
pub trait KernelExt {
    /// Cast to Any trait for dynamic typing
    fn as_any(&self) -> &dyn std::any::Any;
}

impl KernelExt for dyn Kernel {
    fn as_any(&self) -> &dyn std::any::Any {
        self.as_any()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_backend_creation() {
        let backend = CpuBackend::new();
        assert!(backend.is_ok());

        let backend = backend.unwrap();
        assert!(backend.is_available());
        assert!(backend.num_threads() > 0);
    }

    #[test]
    fn test_cpu_memory_allocation() {
        let backend = CpuBackend::new().unwrap();
        let memory = backend.allocate(1024);
        assert!(memory.is_ok());

        let memory = memory.unwrap();
        assert_eq!(memory.size(), 1024);
        assert_eq!(memory.device_type(), DeviceType::Cpu);
    }

    #[test]
    fn test_memory_copy() {
        let backend = CpuBackend::new().unwrap();
        let memory = backend.allocate(4).unwrap();

        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let copy_result = backend.copy_to_device(&input_data, memory.as_ref());
        assert!(copy_result.is_ok());

        let mut output_data = vec![0.0; 4];
        let copy_result = backend.copy_to_host(memory.as_ref(), &mut output_data);
        assert!(copy_result.is_ok());

        assert_eq!(input_data, output_data);
    }

    #[test]
    fn test_elementwise_operations() {
        let backend = CpuBackend::new().unwrap();

        // Test addition
        let kernel = CpuKernel::new("add".to_string(), CpuOperation::ElementwiseAdd);

        let a_mem = backend.allocate(4).unwrap();
        let b_mem = backend.allocate(4).unwrap();
        let c_mem = backend.allocate(4).unwrap();

        backend
            .copy_to_device(&[1.0, 2.0, 3.0, 4.0], a_mem.as_ref())
            .unwrap();
        backend
            .copy_to_device(&[2.0, 3.0, 4.0, 5.0], b_mem.as_ref())
            .unwrap();

        let inputs = vec![a_mem.as_ref(), b_mem.as_ref()];
        let outputs = vec![c_mem.as_ref()];

        let result = backend.execute_kernel(&kernel, &inputs, &outputs);
        assert!(result.is_ok());

        let mut output = vec![0.0; 4];
        backend.copy_to_host(c_mem.as_ref(), &mut output).unwrap();

        assert_eq!(output, vec![3.0, 5.0, 7.0, 9.0]);
    }
}
