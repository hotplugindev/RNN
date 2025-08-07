//! CPU Utilization Demonstration
//!
//! This example shows how the RNN library effectively utilizes multiple CPU cores
//! for neural network training and matrix operations when GPU acceleration is not
//! available or falls back to CPU processing.

use ndarray::Array2;
use rayon::prelude::*;
use rnn::{
    ActivationFunction, GpuManager, LayerBuilder, LossFunction, Network, Result, TrainingConfig,
};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 CPU Utilization Demonstration");
    println!("=================================\n");

    // Show system capabilities
    show_system_info();

    // Test 1: Matrix operations with different approaches
    println!("📊 Matrix Operations Performance:");
    println!("=================================");
    test_matrix_operations()?;

    // Test 2: Neural network training with parallel processing
    println!("\n🧠 Neural Network Training:");
    println!("===========================");
    test_neural_network_training()?;

    // Test 3: Batch processing demonstration
    println!("\n📦 Batch Processing Test:");
    println!("=========================");
    test_batch_processing()?;

    println!("\n✅ CPU utilization demonstration completed!");
    println!("\n💡 Tips for maximum CPU utilization:");
    println!("   - Use appropriate batch sizes (32-128 typically optimal)");
    println!("   - Enable parallel processing in training config");
    println!("   - Ensure sufficient data to justify parallel overhead");
    println!("   - Monitor CPU usage with 'htop' or 'top' during training");

    Ok(())
}

fn show_system_info() {
    println!("🔧 System Information:");
    println!("CPU Cores (Total): {}", num_cpus::get());
    println!("CPU Cores (Physical): {}", num_cpus::get_physical());
    println!(
        "Rayon Thread Pool: {} threads",
        rayon::current_num_threads()
    );

    // Show GPU availability
    let gpu_manager = GpuManager::new();
    println!("GPU Devices: {}", gpu_manager.devices().len());

    if let Some(device) = gpu_manager.default_device() {
        println!("Default Device: {} ({:?})", device.name, device.device_type);
    }
    println!();
}

fn test_matrix_operations() -> Result<()> {
    let sizes = vec![256, 512, 1024];

    for size in sizes {
        println!("Testing {}×{} matrix multiplication:", size, size);

        // Create test matrices
        let a = Array2::from_shape_fn((size, size), |(i, j)| (i + j) as f64 / 1000.0);
        let b = Array2::from_shape_fn((size, size), |(i, j)| (i * j + 1) as f64 / 1000.0);

        // Sequential CPU
        let start = Instant::now();
        let _result_seq = a.dot(&b);
        let seq_time = start.elapsed();
        println!("  Sequential: {:.2}ms", seq_time.as_millis());

        // Parallel CPU using rayon
        let start = Instant::now();
        let _result_par = parallel_matrix_multiply(&a, &b);
        let par_time = start.elapsed();
        println!("  Parallel: {:.2}ms", par_time.as_millis());

        if par_time.as_millis() > 0 {
            let speedup = seq_time.as_millis() as f64 / par_time.as_millis() as f64;
            println!("  Speedup: {:.1}x", speedup);
        }
        println!();
    }

    Ok(())
}

fn parallel_matrix_multiply(a: &Array2<f64>, b: &Array2<f64>) -> Array2<f64> {
    let (rows_a, cols_a) = a.dim();
    let (_, cols_b) = b.dim();

    // Parallel computation using rayon
    let result_data: Vec<f64> = (0..rows_a * cols_b)
        .into_par_iter()
        .map(|linear_idx| {
            let row = linear_idx / cols_b;
            let col = linear_idx % cols_b;

            let mut sum = 0.0;
            for k in 0..cols_a {
                sum += a[[row, k]] * b[[k, col]];
            }
            sum
        })
        .collect();

    Array2::from_shape_vec((rows_a, cols_b), result_data).unwrap()
}

fn test_neural_network_training() -> Result<()> {
    // Create a moderately sized dataset to show CPU utilization
    let (train_data, train_labels) = create_training_dataset(5000, 100, 10);

    println!(
        "Dataset: {} samples, {} features, {} classes",
        train_data.nrows(),
        train_data.ncols(),
        train_labels.ncols()
    );

    // Test different training configurations

    // Configuration 1: Small batches (less parallelism)
    println!("\nTest 1: Small batch training (batch_size=16)");
    let start = Instant::now();
    let mut network1 = create_test_network(train_data.ncols(), train_labels.ncols())?;
    let mut config1 = TrainingConfig::default();
    config1.max_epochs = 3;
    config1.batch_size = 16;
    config1.verbose = false;
    config1.use_gpu = false;

    let _history1 = network1.train(&train_data, &train_labels, &config1)?;
    let time1 = start.elapsed();
    println!("  Time: {:.2}s", time1.as_secs_f64());

    // Configuration 2: Medium batches (better parallelism)
    println!("\nTest 2: Medium batch training (batch_size=64)");
    let start = Instant::now();
    let mut network2 = create_test_network(train_data.ncols(), train_labels.ncols())?;
    let mut config2 = TrainingConfig::default();
    config2.max_epochs = 3;
    config2.batch_size = 64;
    config2.verbose = false;
    config2.use_gpu = false;

    let _history2 = network2.train(&train_data, &train_labels, &config2)?;
    let time2 = start.elapsed();
    println!("  Time: {:.2}s", time2.as_secs_f64());

    // Configuration 3: Large batches (maximum parallelism)
    println!("\nTest 3: Large batch training (batch_size=128)");
    let start = Instant::now();
    let mut network3 = create_test_network(train_data.ncols(), train_labels.ncols())?;
    let mut config3 = TrainingConfig::default();
    config3.max_epochs = 3;
    config3.batch_size = 128;
    config3.verbose = false;
    config3.use_gpu = false;

    let _history3 = network3.train(&train_data, &train_labels, &config3)?;
    let time3 = start.elapsed();
    println!("  Time: {:.2}s", time3.as_secs_f64());

    // Show performance summary
    println!("\nPerformance Summary:");
    println!(
        "  Small batches: {:.0} samples/sec",
        5000.0 * 3.0 / time1.as_secs_f64()
    );
    println!(
        "  Medium batches: {:.0} samples/sec",
        5000.0 * 3.0 / time2.as_secs_f64()
    );
    println!(
        "  Large batches: {:.0} samples/sec",
        5000.0 * 3.0 / time3.as_secs_f64()
    );

    Ok(())
}

fn create_test_network(input_size: usize, output_size: usize) -> Result<Network> {
    Network::with_input_size(input_size)?
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(output_size).activation(ActivationFunction::Softmax))
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("CPU Utilization Test Network")
        .build()
}

fn create_training_dataset(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> (Array2<f64>, Array2<f64>) {
    println!("Creating synthetic dataset...");

    // Use parallel generation for large datasets
    let data_vec: Vec<f64> = (0..n_samples * n_features)
        .into_par_iter()
        .map(|i| {
            let sample = i / n_features;
            let feature = i % n_features;

            // Create some non-linear patterns
            let x = feature as f64 / n_features as f64;
            let y = sample as f64 / n_samples as f64;

            (x * 2.0 - 1.0) * (y * 2.0 - 1.0).cos() + 0.1 * ((sample + feature) as f64).sin()
        })
        .collect();

    let data = Array2::from_shape_vec((n_samples, n_features), data_vec).unwrap();

    // Create labels (one-hot encoded)
    let mut labels = Array2::zeros((n_samples, n_classes));
    for i in 0..n_samples {
        let class = i % n_classes;
        labels[[i, class]] = 1.0;
    }

    (data, labels)
}

fn test_batch_processing() -> Result<()> {
    println!("Testing parallel batch processing...");

    let n_batches = 100;
    let batch_size = 1000;
    let features = 50;

    // Create multiple batches to process
    let batches: Vec<Array2<f64>> = (0..n_batches)
        .map(|batch_id| {
            Array2::from_shape_fn((batch_size, features), |(i, j)| {
                (batch_id * batch_size + i + j) as f64 / 1000.0
            })
        })
        .collect();

    // Sequential processing
    println!("Sequential batch processing:");
    let start = Instant::now();
    let mut sequential_results = Vec::new();
    for batch in &batches {
        let result = process_batch(batch);
        sequential_results.push(result);
    }
    let seq_time = start.elapsed();
    println!("  Time: {:.2}ms", seq_time.as_millis());

    // Parallel processing
    println!("Parallel batch processing:");
    let start = Instant::now();
    let parallel_results: Vec<f64> = batches
        .par_iter()
        .map(|batch| process_batch(batch))
        .collect();
    let par_time = start.elapsed();
    println!("  Time: {:.2}ms", par_time.as_millis());

    if par_time.as_millis() > 0 {
        let speedup = seq_time.as_millis() as f64 / par_time.as_millis() as f64;
        println!("  Speedup: {:.1}x", speedup);
    }

    // Verify results are the same
    let results_match = sequential_results
        .iter()
        .zip(parallel_results.iter())
        .all(|(a, b)| (a - b).abs() < 1e-10);

    println!(
        "  Results match: {}",
        if results_match { "✅" } else { "❌" }
    );

    Ok(())
}

fn process_batch(batch: &Array2<f64>) -> f64 {
    // Simulate some computational work on a batch
    let mut sum = 0.0;
    for &value in batch.iter() {
        sum += value * value + value.sin() + value.cos();
    }
    sum / batch.len() as f64
}

/// Stress test to really show CPU utilization
#[allow(dead_code)]
fn stress_test_cpu() -> Result<()> {
    println!("🔥 CPU Stress Test (watch CPU usage with htop/top):");

    let iterations = 1000;
    let matrix_size = 512;

    println!("Running {} parallel matrix operations...", iterations);
    println!("(This will use all CPU cores - monitor with 'htop')");

    let start = Instant::now();

    let results: Vec<f64> = (0..iterations)
        .into_par_iter()
        .map(|i| {
            let a = Array2::from_shape_fn((matrix_size, matrix_size), |(row, col)| {
                (row + col + i) as f64 / 1000.0
            });
            let b = Array2::from_shape_fn((matrix_size, matrix_size), |(row, col)| {
                (row * col + i) as f64 / 1000.0
            });

            let result = parallel_matrix_multiply(&a, &b);
            result.sum()
        })
        .collect();

    let duration = start.elapsed();

    println!(
        "Completed {} operations in {:.2}s",
        iterations,
        duration.as_secs_f64()
    );
    println!(
        "Average result: {:.2}",
        results.iter().sum::<f64>() / results.len() as f64
    );
    println!(
        "Operations per second: {:.0}",
        iterations as f64 / duration.as_secs_f64()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_matrix_multiply() {
        let a = Array2::from_shape_fn((10, 10), |(i, j)| (i + j) as f64);
        let b = Array2::from_shape_fn((10, 10), |(i, j)| (i * j + 1) as f64);

        let result_parallel = parallel_matrix_multiply(&a, &b);
        let result_sequential = a.dot(&b);

        // Results should be approximately equal
        for i in 0..10 {
            for j in 0..10 {
                assert!((result_parallel[[i, j]] - result_sequential[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_batch_processing() {
        let batch = Array2::from_shape_fn((100, 10), |(i, j)| (i + j) as f64);
        let result = process_batch(&batch);
        assert!(result.is_finite());
        assert!(result > 0.0);
    }

    #[test]
    fn test_dataset_creation() {
        let (data, labels) = create_training_dataset(100, 10, 5);
        assert_eq!(data.shape(), &[100, 10]);
        assert_eq!(labels.shape(), &[100, 5]);

        // Each sample should have exactly one label
        for i in 0..100 {
            let sum: f64 = labels.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }
}
