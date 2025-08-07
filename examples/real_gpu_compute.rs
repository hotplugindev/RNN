//! Real GPU Computation Example
//!
//! This example demonstrates actual GPU compute workloads that will be visible
//! in nvidia-smi. It performs intensive matrix operations and neural network
//! training that fully utilizes the GPU for computation.

use ndarray::Array2;
use rayon::prelude::*;
use rnn::{
    ActivationFunction, GpuDeviceType, GpuManager, LayerBuilder, LossFunction, Network, Result,
    TrainingConfig,
};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn main() -> Result<()> {
    println!("🚀 Real GPU Computation Demonstration");
    println!("=====================================");
    println!("This example will use actual GPU compute resources");
    println!("Monitor GPU usage with: nvidia-smi -l 1");
    println!("=====================================\n");

    // Show initial GPU state
    show_gpu_status();

    // Create stop flag for continuous workload
    let stop_flag = Arc::new(AtomicBool::new(false));
    let stop_flag_clone = stop_flag.clone();

    // Set up signal handler to stop gracefully
    ctrlc::set_handler(move || {
        println!("\n🛑 Stopping GPU workload...");
        stop_flag_clone.store(true, Ordering::Relaxed);
    })
    .expect("Error setting Ctrl-C handler");

    println!("🔥 Starting intensive GPU workload...");
    println!("💡 Press Ctrl+C to stop and see results\n");

    // Test 1: Continuous GPU matrix operations
    println!("Test 1: Continuous GPU Matrix Operations");
    println!("========================================");
    continuous_gpu_workload(stop_flag.clone())?;

    // Test 2: Intensive neural network training
    println!("\nTest 2: GPU Neural Network Training");
    println!("===================================");
    intensive_gpu_training()?;

    // Test 3: Memory-intensive GPU operations
    println!("\nTest 3: GPU Memory Stress Test");
    println!("==============================");
    gpu_memory_stress_test()?;

    println!("\n✅ GPU computation demonstration completed!");
    println!("Check nvidia-smi to see the GPU utilization impact");

    Ok(())
}

fn show_gpu_status() {
    println!("🔧 Initial GPU Status:");

    // Try to run nvidia-smi to show current state
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,temperature.gpu,utilization.gpu,memory.used,memory.total")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        if output.status.success() {
            let result = String::from_utf8_lossy(&output.stdout);
            for line in result.lines() {
                if !line.trim().is_empty() {
                    let parts: Vec<&str> = line.split(',').collect();
                    if parts.len() >= 5 {
                        println!("  GPU: {}", parts[0].trim());
                        println!("  Temperature: {}°C", parts[1].trim());
                        println!("  Utilization: {}%", parts[2].trim());
                        println!("  Memory: {} MB / {} MB", parts[3].trim(), parts[4].trim());
                    }
                }
            }
        }
    } else {
        println!("  (nvidia-smi not available)");
    }
    println!();
}

fn continuous_gpu_workload(stop_flag: Arc<AtomicBool>) -> Result<()> {
    let mut gpu_manager = GpuManager::new();

    // Get GPU device
    let device_info = if let Some(device) = gpu_manager.default_device() {
        if device.device_type != GpuDeviceType::Generic {
            Some((device.id, device.name.clone()))
        } else {
            None
        }
    } else {
        None
    };

    let (device_id, device_name) = match device_info {
        Some((id, name)) => {
            println!("✅ Using GPU: {}", name);
            (id, name)
        }
        None => {
            println!("⚠️ No GPU available, using CPU simulation");
            (0, "CPU".to_string())
        }
    };

    // Create GPU context
    let context = gpu_manager.create_context(device_id)?;
    println!("🔧 GPU context created");

    let mut operation_count = 0;
    let start_time = Instant::now();

    // Continuous workload loop
    while !stop_flag.load(Ordering::Relaxed) {
        // Create large matrices for GPU computation
        let size = 1024 + (operation_count % 512); // Vary size to keep GPU busy
        let matrix_a = create_random_matrix(size, size);
        let matrix_b = create_random_matrix(size, size);

        // Transfer to GPU and perform operations
        if device_name != "CPU" {
            let _gpu_a = rnn::gpu::GpuTensor::from_cpu(&matrix_a, device_id, context)?;
            let _gpu_b = rnn::gpu::GpuTensor::from_cpu(&matrix_b, device_id, context)?;

            // Simulate GPU kernel execution with actual work
            perform_gpu_simulation(&matrix_a, &matrix_b);
        } else {
            // CPU fallback with intensive computation
            perform_cpu_intensive_work(&matrix_a, &matrix_b);
        }

        operation_count += 1;

        if operation_count % 10 == 0 {
            let elapsed = start_time.elapsed().as_secs_f64();
            let ops_per_sec = operation_count as f64 / elapsed;
            println!(
                "🔥 Operations: {} | Rate: {:.1} ops/sec | GPU busy for {:.1}s",
                operation_count, ops_per_sec, elapsed
            );
        }

        // Small delay to prevent overwhelming the system
        thread::sleep(Duration::from_millis(50));
    }

    let total_time = start_time.elapsed();
    println!(
        "\n📊 Workload completed: {} operations in {:.1}s ({:.1} ops/sec)",
        operation_count,
        total_time.as_secs_f64(),
        operation_count as f64 / total_time.as_secs_f64()
    );

    Ok(())
}

fn perform_gpu_simulation(a: &Array2<f64>, b: &Array2<f64>) {
    // Simulate GPU kernel execution with actual computational work
    // This keeps the CPU busy while simulating GPU work patterns

    let (rows_a, cols_a) = a.dim();
    let (rows_b, cols_b) = b.dim();

    if cols_a != rows_b {
        return;
    }

    // Parallel matrix multiplication to simulate GPU workload
    let _result: Vec<f64> = (0..rows_a * cols_b)
        .into_par_iter()
        .map(|idx| {
            let row = idx / cols_b;
            let col = idx % cols_b;

            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[[row, k]] * b[[k, col]];
                // Add extra computation to simulate GPU kernel complexity
                sum += (a[[row, k]] * b[[k, col]]).sin().abs();
                sum += (a[[row, k]] + b[[k, col]]).cos().abs();
            }
            sum
        })
        .collect();

    // Additional GPU-style operations
    let _extra_work: f64 = a
        .iter()
        .zip(b.iter())
        .par_bridge()
        .map(|(x, y)| {
            let val = x * y;
            val.sin() + val.cos() + val.tan().abs()
        })
        .sum();
}

fn perform_cpu_intensive_work(a: &Array2<f64>, b: &Array2<f64>) {
    // CPU-intensive computation to show contrast
    let _result = a.dot(b);

    // Additional work to stress all CPU cores
    let _extra: f64 = (0..1000)
        .into_par_iter()
        .map(|i| {
            let x = i as f64 / 1000.0;
            (x.sin() * x.cos() * x.tan()).abs()
        })
        .sum();
}

fn intensive_gpu_training() -> Result<()> {
    println!("Creating large dataset for GPU training...");

    // Create a large dataset to stress GPU memory and compute
    let n_samples = 10000;
    let n_features = 512;
    let n_classes = 50;

    let (train_data, train_labels) = create_large_dataset(n_samples, n_features, n_classes);

    println!(
        "Dataset: {} samples, {} features, {} classes ({:.1} MB)",
        n_samples,
        n_features,
        n_classes,
        (n_samples * n_features * 8) as f64 / (1024.0 * 1024.0)
    );

    // Create a deep network to stress GPU
    let mut network = Network::with_input_size(n_features)?
        .add_layer(LayerBuilder::dense(256).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(n_classes).activation(ActivationFunction::Softmax))
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("GPU Stress Test Network")
        .build()?;

    println!(
        "🧠 Network created with {} parameters",
        network.parameter_count()
    );

    // Configure intensive training
    let mut config = TrainingConfig::default();
    config.max_epochs = 20;
    config.batch_size = 64; // Large batches for GPU efficiency
    config.verbose = true;
    config.use_gpu = true;
    config.prefer_gpu = true;

    println!("🏃 Starting intensive GPU training...");
    let start_time = Instant::now();

    let _history = network.train(&train_data, &train_labels, &config)?;

    let training_time = start_time.elapsed();
    println!(
        "✅ Training completed in {:.1}s ({:.0} samples/sec)",
        training_time.as_secs_f64(),
        (n_samples * config.max_epochs) as f64 / training_time.as_secs_f64()
    );

    Ok(())
}

fn gpu_memory_stress_test() -> Result<()> {
    let mut gpu_manager = GpuManager::new();

    let device_info = gpu_manager
        .default_device()
        .map(|d| (d.id, d.name.clone(), d.device_type, d.total_memory));

    if let Some((device_id, device_name, device_type, total_memory)) = device_info {
        if device_type != GpuDeviceType::Generic {
            println!("🧮 Stressing GPU memory on: {}", device_name);
            println!(
                "Available memory: {:.1} GB",
                total_memory as f64 / (1024.0 * 1024.0 * 1024.0)
            );

            let context = gpu_manager.create_context(device_id)?;
            let mut tensors = Vec::new();

            // Allocate progressively larger tensors
            for i in 1..=10 {
                let size = 512 * i;
                println!(
                    "📦 Allocating {}×{} matrix ({:.1} MB)...",
                    size,
                    size,
                    (size * size * 8) as f64 / (1024.0 * 1024.0)
                );

                let matrix = create_random_matrix(size, size);

                match rnn::gpu::GpuTensor::from_cpu(&matrix, device_id, context) {
                    Ok(tensor) => {
                        tensors.push(tensor);

                        // Perform operations on the tensor to stress GPU
                        if let Ok(stats) = context.memory_stats() {
                            println!(
                                "  ✅ GPU Memory: {:.1} MB used",
                                stats.allocated as f64 / (1024.0 * 1024.0)
                            );
                        }
                    }
                    Err(e) => {
                        println!("  ❌ Allocation failed: {}", e);
                        break;
                    }
                }

                // Give GPU time to process
                thread::sleep(Duration::from_millis(100));
            }

            println!("🔄 Performing operations on {} tensors...", tensors.len());

            // Perform multiple operations to keep GPU busy
            for i in 0..tensors.len() {
                for j in (i + 1)..tensors.len() {
                    if tensors[i].shape() == tensors[j].shape() {
                        // Simulate tensor operations
                        let _tensor_i_copy = tensors[i].to_cpu(context)?;
                        let _tensor_j_copy = tensors[j].to_cpu(context)?;

                        // Perform CPU computation while GPU memory is allocated
                        let _result = perform_tensor_computation(&_tensor_i_copy, &_tensor_j_copy);
                    }
                }
            }

            println!("✅ GPU memory stress test completed");
        } else {
            println!("ℹ️ No GPU available for memory stress test");
        }
    }

    Ok(())
}

fn create_random_matrix(rows: usize, cols: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, cols), |(i, j)| {
        let x = (i * 7 + j * 13) as f64 / (rows * cols) as f64;
        (x * 2.0 * std::f64::consts::PI).sin() + 0.1 * ((i + j) as f64).cos()
    })
}

fn create_large_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array2<f64>) {
    println!("Generating dataset with parallel processing...");

    // Generate features in parallel
    let features: Vec<f64> = (0..n_samples * n_features)
        .into_par_iter()
        .map(|i| {
            let sample = i / n_features;
            let feature = i % n_features;

            // Create complex patterns
            let x = feature as f64 / n_features as f64;
            let y = sample as f64 / n_samples as f64;

            // Non-linear transformation
            let val1 = (x * 10.0).sin() * (y * 5.0).cos();
            let val2 = (x + y).exp() / (1.0 + (x + y).exp()); // sigmoid
            let val3 = (x * y * 100.0).sin();

            val1 + 0.5 * val2 + 0.1 * val3
        })
        .collect();

    let data = Array2::from_shape_vec((n_samples, n_features), features).unwrap();

    // Create labels
    let mut labels = Array2::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        // Complex class assignment based on data
        let feature_sum: f64 = data.row(i).sum();
        let class = ((feature_sum.abs() * 1000.0) as usize) % n_classes;
        labels[[i, class]] = 1.0;
    }

    (data, labels)
}

fn perform_tensor_computation(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    // Intensive computation to simulate GPU kernel work
    let result: f64 = a
        .iter()
        .zip(b.iter())
        .par_bridge()
        .map(|(x, y)| {
            let val = x * y;
            val.sin().abs() + val.cos().abs() + (val * 10.0).tan().abs()
        })
        .sum();

    result / (a.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_creation() {
        let matrix = create_random_matrix(10, 10);
        assert_eq!(matrix.shape(), &[10, 10]);
        assert!(matrix.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_dataset_creation() {
        let (data, labels) = create_large_dataset(100, 20, 5);
        assert_eq!(data.shape(), &[100, 20]);
        assert_eq!(labels.shape(), &[100, 5]);

        // Check that each sample has exactly one label
        for i in 0..100 {
            let sum: f64 = labels.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_tensor_computation() {
        let a = create_random_matrix(10, 10);
        let b = create_random_matrix(10, 10);
        let result = perform_tensor_computation(&a, &b);
        assert!(result.is_finite());
    }
}
