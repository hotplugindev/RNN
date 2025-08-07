//! Real GPU Neural Network Training Example
//!
//! This example demonstrates ACTUAL GPU computation for neural network training.
//! Unlike simulation examples, this uses real GPU kernels (CUDA, OpenCL, ROCm)
//! for matrix operations, activation functions, and gradient computations.
//!
//! GPU utilization should be visible in nvidia-smi when running this example.

use ndarray::{Array1, Array2};
use rnn::{
    ActivationFunction, GpuDeviceType, GpuManager, LayerBuilder, LossFunction, Network, Result,
    TrainingConfig,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("🚀 Real GPU Neural Network Training");
    println!("===================================");
    println!("This example uses ACTUAL GPU kernels for computation");
    println!("Monitor with: nvidia-smi -l 1");
    println!("===================================\n");

    // Set up graceful shutdown
    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();
    ctrlc::set_handler(move || {
        println!("\n🛑 Graceful shutdown initiated...");
        r.store(false, Ordering::SeqCst);
    })
    .expect("Error setting Ctrl-C handler");

    // Initialize GPU and verify real compute capability
    let gpu_info = setup_gpu_environment()?;

    // Create a substantial dataset for meaningful GPU work
    let (train_data, train_labels) = create_training_dataset(5000, 256, 10);

    // Test 1: GPU Matrix Operations
    println!("\n🧮 Test 1: Direct GPU Matrix Operations");
    test_gpu_matrix_operations(&gpu_info, &train_data, running.clone())?;

    if !running.load(Ordering::SeqCst) {
        return Ok(());
    }

    // Test 2: GPU Neural Network Training
    println!("\n🧠 Test 2: GPU Neural Network Training");
    test_gpu_neural_network(&train_data, &train_labels, running.clone())?;

    if !running.load(Ordering::SeqCst) {
        return Ok(());
    }

    // Test 3: Performance Comparison
    println!("\n⚡ Test 3: CPU vs GPU Performance");
    performance_comparison(&train_data, &train_labels)?;

    println!("\n✅ All GPU compute tests completed!");
    println!("Check nvidia-smi to verify GPU utilization occurred");

    Ok(())
}

struct GpuInfo {
    device_id: usize,
    device_name: String,
    device_type: GpuDeviceType,
    total_memory: usize,
    context: Box<dyn rnn::gpu::GpuContext>,
}

fn setup_gpu_environment() -> Result<GpuInfo> {
    println!("🔧 Initializing GPU environment...");

    let mut gpu_manager = GpuManager::new();

    // List all available devices
    let devices = gpu_manager.devices();
    println!("Available devices:");
    for device in &devices {
        println!(
            "  {}: {} ({:?}) - {:.1} GB",
            device.id,
            device.name,
            device.device_type,
            device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
        );
    }

    // Select best GPU device
    let gpu_device = devices
        .iter()
        .find(|d| d.device_type != GpuDeviceType::Generic)
        .cloned()
        .ok_or_else(|| {
            rnn::error::NetworkError::gpu("No GPU device available for real compute".to_string())
        })?;

    println!(
        "✅ Selected GPU: {} ({:?})",
        gpu_device.name, gpu_device.device_type
    );

    // Create GPU context for real compute
    let context = gpu_manager.create_context(gpu_device.id)?;

    // Verify GPU context can allocate memory
    let test_allocation = context.allocate(1024 * 1024, rnn::gpu::GpuDataType::Float32)?;
    println!("✅ GPU memory allocation test passed");
    context.deallocate(&test_allocation)?;

    Ok(GpuInfo {
        device_id: gpu_device.id,
        device_name: gpu_device.name,
        device_type: gpu_device.device_type,
        total_memory: gpu_device.total_memory,
        context,
    })
}

fn test_gpu_matrix_operations(
    gpu_info: &GpuInfo,
    train_data: &Array2<f64>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    println!("Creating large matrices for GPU computation...");

    let matrix_size = 1024;
    let matrix_a = create_test_matrix(matrix_size, matrix_size);
    let matrix_b = create_test_matrix(matrix_size, matrix_size);

    println!(
        "Matrix size: {}x{} ({:.1} MB each)",
        matrix_size,
        matrix_size,
        (matrix_size * matrix_size * 8) as f64 / (1024.0 * 1024.0)
    );

    // Transfer matrices to GPU
    println!("📦 Transferring matrices to GPU...");
    let gpu_a = rnn::gpu::GpuTensor::from_cpu(&matrix_a, gpu_info.device_id, &*gpu_info.context)?;
    let gpu_b = rnn::gpu::GpuTensor::from_cpu(&matrix_b, gpu_info.device_id, &*gpu_info.context)?;

    // Check memory usage
    if let Ok(stats) = gpu_info.context.memory_stats() {
        println!(
            "GPU Memory: {:.1} MB allocated, {:.1} MB available",
            stats.allocated as f64 / (1024.0 * 1024.0),
            stats.available as f64 / (1024.0 * 1024.0)
        );
    }

    let mut operation_count = 0;
    let start_time = Instant::now();

    println!("🔥 Starting continuous GPU matrix operations...");
    println!("Press Ctrl+C to stop and proceed to next test");

    while running.load(Ordering::SeqCst) && operation_count < 100 {
        // Perform GPU matrix multiplication using kernels
        let result = gpu_matrix_multiply(&gpu_a, &gpu_b, &*gpu_info.context, gpu_info.device_id)?;

        // Perform additional operations to keep GPU busy
        apply_gpu_relu(&result, &*gpu_info.context)?;

        // Synchronize to ensure operations complete
        gpu_info.context.synchronize()?;

        operation_count += 1;

        if operation_count % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let ops_per_sec = operation_count as f64 / elapsed;
            println!(
                "🧮 Completed {} GPU operations ({:.1} ops/sec) - GPU utilization should be visible",
                operation_count, ops_per_sec
            );
        }

        // Brief pause to allow monitoring
        std::thread::sleep(Duration::from_millis(100));
    }

    let total_time = start_time.elapsed();
    println!(
        "📊 GPU operations completed: {} ops in {:.1}s ({:.1} ops/sec)",
        operation_count,
        total_time.as_secs_f64(),
        operation_count as f64 / total_time.as_secs_f64()
    );

    Ok(())
}

fn gpu_matrix_multiply(
    a: &rnn::gpu::GpuTensor,
    b: &rnn::gpu::GpuTensor,
    context: &dyn rnn::gpu::GpuContext,
    device_id: usize,
) -> Result<rnn::gpu::GpuTensor> {
    use rnn::gpu::{GpuDataType, GpuKernel, GpuKernelArg, MemoryLayout};

    let m = a.shape()[0];
    let n = a.shape()[1];
    let k = b.shape()[1];

    // Allocate result matrix
    let result_handle =
        context.allocate(m * k * std::mem::size_of::<f32>(), GpuDataType::Float32)?;

    let result = rnn::gpu::GpuTensor {
        handle: result_handle,
        shape: vec![m, k],
        dtype: GpuDataType::Float32,
        device_id,
        memory_layout: MemoryLayout::RowMajor,
        strides: vec![k, 1],
    };

    // Create matrix multiplication kernel
    let matmul_kernel = GpuKernel {
        name: "gpu_matmul".to_string(),
        source: rnn::gpu::kernels::CudaKernels::matmul().to_string(),
        entry_point: "matmul_kernel".to_string(),
        compiled_binary: None,
        work_group_size: (16, 16),
        backend_handle: None,
    };

    // Prepare kernel arguments
    let args = vec![
        GpuKernelArg::Buffer(a.handle.clone()),
        GpuKernelArg::Buffer(b.handle.clone()),
        GpuKernelArg::Buffer(result.handle.clone()),
        GpuKernelArg::UInt(m as u32),
        GpuKernelArg::UInt(k as u32),
        GpuKernelArg::UInt(n as u32),
    ];

    // Execute kernel on GPU
    println!("🚀 Executing GPU matrix multiplication kernel");
    context.execute_kernel(&matmul_kernel, &args)?;

    Ok(result)
}

fn apply_gpu_relu(tensor: &rnn::gpu::GpuTensor, context: &dyn rnn::gpu::GpuContext) -> Result<()> {
    use rnn::gpu::{GpuKernel, GpuKernelArg};

    let relu_kernel = GpuKernel {
        name: "gpu_relu".to_string(),
        source: rnn::gpu::kernels::CudaKernels::relu().to_string(),
        entry_point: "relu_kernel".to_string(),
        compiled_binary: None,
        work_group_size: (256, 1),
        backend_handle: None,
    };

    let total_elements = tensor.shape().iter().product::<usize>();
    let args = vec![
        GpuKernelArg::Buffer(tensor.handle.clone()),
        GpuKernelArg::Buffer(tensor.handle.clone()),
        GpuKernelArg::UInt(total_elements as u32),
    ];

    println!("🚀 Executing GPU ReLU activation kernel");
    context.execute_kernel(&relu_kernel, &args)?;

    Ok(())
}

fn test_gpu_neural_network(
    train_data: &Array2<f64>,
    train_labels: &Array2<f64>,
    running: Arc<AtomicBool>,
) -> Result<()> {
    if !running.load(Ordering::SeqCst) {
        return Ok(());
    }

    println!("🧠 Creating neural network for GPU training...");

    // Create a substantial network
    let mut network = Network::with_input_size(train_data.ncols())?
        .add_layer(LayerBuilder::dense(512).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(256).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(
            LayerBuilder::dense(train_labels.ncols()).activation(ActivationFunction::Softmax),
        )
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("Real GPU Network")
        .build()?;

    println!(
        "Network created: {} parameters, {} layers",
        network.parameter_count(),
        network.layer_count()
    );

    // Configure for GPU training
    let mut config = TrainingConfig::default();
    config.max_epochs = 20;
    config.batch_size = 128; // Large batches for GPU efficiency
    config.verbose = true;
    config.use_gpu = true;
    config.prefer_gpu = true;

    println!("🔥 Starting REAL GPU neural network training...");
    println!("All forward/backward passes will use GPU kernels");
    println!("Monitor GPU utilization with nvidia-smi");

    let training_start = Instant::now();

    // Train on GPU with real kernel execution
    let history = network.train(train_data, train_labels, &config)?;

    let training_time = training_start.elapsed();

    println!("\n✅ GPU neural network training completed!");
    println!("Training time: {:.1}s", training_time.as_secs_f64());
    println!(
        "Average epoch time: {:.1}s",
        training_time.as_secs_f64() / config.max_epochs as f64
    );

    if let Some(final_loss) = history.train_loss.last() {
        println!("Final training loss: {:.6}", final_loss);
    }

    // Test the trained network
    println!("\n🔍 Testing trained network...");
    let test_batch = train_data.slice(ndarray::s![0..10, ..]).to_owned();
    let predictions = network.predict(&test_batch)?;

    println!("Prediction shape: {:?}", predictions.shape());
    println!(
        "Sample predictions: {:?}",
        predictions.row(0).as_slice().unwrap()[0..5].to_vec()
    );

    Ok(())
}

fn performance_comparison(train_data: &Array2<f64>, train_labels: &Array2<f64>) -> Result<()> {
    println!("⚡ Comparing CPU vs GPU performance...");

    let n_samples = 1000;
    let subset_data = train_data.slice(ndarray::s![0..n_samples, ..]).to_owned();
    let subset_labels = train_labels.slice(ndarray::s![0..n_samples, ..]).to_owned();

    // Create identical networks
    let mut cpu_network = Network::with_input_size(subset_data.ncols())?
        .add_layer(LayerBuilder::dense(256).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(
            LayerBuilder::dense(subset_labels.ncols()).activation(ActivationFunction::Softmax),
        )
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("CPU Network")
        .build()?;

    let mut gpu_network = Network::with_input_size(subset_data.ncols())?
        .add_layer(LayerBuilder::dense(256).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(
            LayerBuilder::dense(subset_labels.ncols()).activation(ActivationFunction::Softmax),
        )
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("GPU Network")
        .build()?;

    // CPU training configuration
    let mut cpu_config = TrainingConfig::default();
    cpu_config.max_epochs = 5;
    cpu_config.batch_size = 64;
    cpu_config.verbose = false;
    cpu_config.use_gpu = false;

    // GPU training configuration
    let mut gpu_config = TrainingConfig::default();
    gpu_config.max_epochs = 5;
    gpu_config.batch_size = 64;
    gpu_config.verbose = false;
    gpu_config.use_gpu = true;
    gpu_config.prefer_gpu = true;

    // CPU training
    println!("🔄 Training on CPU...");
    let cpu_start = Instant::now();
    let _cpu_history = cpu_network.train(&subset_data, &subset_labels, &cpu_config)?;
    let cpu_time = cpu_start.elapsed();

    // GPU training
    println!("🔄 Training on GPU...");
    let gpu_start = Instant::now();
    let _gpu_history = gpu_network.train(&subset_data, &subset_labels, &gpu_config)?;
    let gpu_time = gpu_start.elapsed();

    // Performance comparison
    println!("\n📊 Performance Results:");
    println!("CPU training time: {:.2}s", cpu_time.as_secs_f64());
    println!("GPU training time: {:.2}s", gpu_time.as_secs_f64());

    if gpu_time.as_secs_f64() > 0.0 {
        let speedup = cpu_time.as_secs_f64() / gpu_time.as_secs_f64();
        if speedup > 1.0 {
            println!("🚀 GPU speedup: {:.1}x faster than CPU", speedup);
        } else {
            println!(
                "⚠️ GPU overhead: {:.1}x slower than CPU (small dataset)",
                1.0 / speedup
            );
            println!("   (GPU becomes beneficial with larger datasets/networks)");
        }
    }

    Ok(())
}

fn create_training_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array2<f64>) {
    println!("📊 Generating training dataset...");
    println!(
        "Samples: {}, Features: {}, Classes: {}",
        n_samples, n_features, n_classes
    );

    use rand::Rng;
    let mut rng = rand::thread_rng();

    // Create feature data with patterns
    let mut data = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let pattern1 = ((i as f64 / 100.0) + (j as f64 / 50.0)).sin();
            let pattern2 = ((i as f64 / 200.0) * (j as f64 / 100.0)).cos();
            let noise = rng.gen_range(-0.1..0.1);
            data[[i, j]] = pattern1 + 0.5 * pattern2 + noise;
        }
    }

    // Create labels based on data patterns
    let mut labels = Array2::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        let feature_sum: f64 = data.row(i).sum();
        let feature_mean = feature_sum / n_features as f64;
        let class_id = ((feature_mean + 2.0) * n_classes as f64 / 4.0) as usize % n_classes;
        labels[[i, class_id]] = 1.0;
    }

    println!("✅ Dataset generated successfully");

    (data, labels)
}

fn create_test_matrix(rows: usize, cols: usize) -> Array2<f64> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let base = (i * 7 + j * 13) as f64 / (rows * cols) as f64;
        let pattern = (base * 2.0 * std::f64::consts::PI).sin();
        let noise = rng.gen_range(-0.1..0.1);
        pattern + noise
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataset_generation() {
        let (data, labels) = create_training_dataset(100, 50, 5);

        assert_eq!(data.shape(), &[100, 50]);
        assert_eq!(labels.shape(), &[100, 5]);

        // Each sample should have exactly one label
        for i in 0..100 {
            let label_sum: f64 = labels.row(i).sum();
            assert!((label_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_matrix_creation() {
        let matrix = create_test_matrix(10, 10);
        assert_eq!(matrix.shape(), &[10, 10]);
        assert!(matrix.iter().all(|&x| x.is_finite()));
    }
}
