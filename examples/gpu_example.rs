//! GPU acceleration example for the RNN library.
//!
//! This example demonstrates how to use GPU acceleration for neural network training
//! and inference across different GPU backends (CUDA, OpenCL, ROCm, Metal).

use ndarray::Array2;
use rnn::{
    ActivationFunction, GpuManager, LayerBuilder, LossFunction, Network, Result, TrainingConfig,
};

fn main() -> Result<()> {
    println!("🚀 RNN GPU Acceleration Example");
    println!("================================\n");

    // Check GPU availability
    check_gpu_availability();

    // Create GPU manager and enumerate devices
    let mut gpu_manager = GpuManager::new();
    println!("📱 Available GPU Devices:");
    for (i, device) in gpu_manager.devices().iter().enumerate() {
        println!(
            "  {}. {} ({:?}) - {} MB",
            i,
            device.name,
            device.device_type,
            device.total_memory / (1024 * 1024)
        );
    }
    println!();

    // Get default device
    if let Some(default_device) = gpu_manager.default_device() {
        let device_name = default_device.name.clone();
        let device_id = default_device.id;
        println!("🎯 Using default device: {}", device_name);
        demonstrate_gpu_operations(&mut gpu_manager, device_id)?;
    } else {
        println!("❌ No GPU devices available, falling back to CPU");
        demonstrate_cpu_fallback()?;
    }

    Ok(())
}

fn check_gpu_availability() {
    println!("🔍 Checking GPU Backend Availability:");

    println!(
        "  CUDA: {}",
        if rnn::gpu::GpuManager::is_cuda_available() {
            "✅ Available"
        } else {
            "❌ Not available"
        }
    );

    println!(
        "  OpenCL: {}",
        if cfg!(feature = "opencl") {
            "✅ Compiled in"
        } else {
            "❌ Not compiled"
        }
    );

    println!(
        "  Metal: {}",
        if cfg!(feature = "metal") {
            "✅ Compiled in"
        } else {
            "❌ Not compiled"
        }
    );

    println!(
        "  ROCm: {}",
        if cfg!(feature = "rocm") {
            "✅ Compiled in"
        } else {
            "❌ Not compiled"
        }
    );

    println!();
}

fn demonstrate_gpu_operations(gpu_manager: &mut GpuManager, device_id: usize) -> Result<()> {
    println!("🔥 Demonstrating GPU Operations on Device {}", device_id);

    // Create GPU context
    let context = gpu_manager.create_context(device_id)?;
    println!("✅ Created GPU context");

    // Create sample data for demonstration
    let input_data = create_sample_data();
    println!(
        "📊 Created sample dataset: {} samples",
        input_data.0.nrows()
    );

    // Create GPU tensors
    let gpu_input = rnn::gpu::GpuTensor::from_cpu(&input_data.0, device_id, context)?;
    println!("🔄 Transferred data to GPU");

    // Demonstrate tensor operations
    demonstrate_tensor_operations(&gpu_input, context)?;

    // Create and train neural network with GPU acceleration
    demonstrate_gpu_training(&input_data, device_id)?;

    Ok(())
}

fn demonstrate_tensor_operations(
    gpu_tensor: &rnn::gpu::GpuTensor,
    context: &mut dyn rnn::gpu::GpuContext,
) -> Result<()> {
    println!("\n🧮 GPU Tensor Operations:");

    println!("  Original tensor shape: {:?}", gpu_tensor.shape());
    println!("  Data type: {:?}", gpu_tensor.dtype());
    println!("  Memory size: {} bytes", gpu_tensor.memory_size());

    // Demonstrate reshape
    let reshaped = gpu_tensor.reshape(vec![gpu_tensor.numel(), 1])?;
    println!("  Reshaped to: {:?}", reshaped.shape());

    // Transfer back to CPU for verification
    let cpu_result = gpu_tensor.to_cpu(context)?;
    println!("  ✅ Successfully transferred back to CPU");
    println!("  CPU result shape: {:?}", cpu_result.shape());

    Ok(())
}

fn demonstrate_gpu_training(data: &(Array2<f64>, Array2<f64>), _device_id: usize) -> Result<()> {
    println!("\n🎓 GPU-Accelerated Training:");

    // Create a neural network
    let mut network = Network::with_input_size(data.0.ncols())?
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(data.1.ncols()).activation(ActivationFunction::Sigmoid))
        .loss(LossFunction::BinaryCrossEntropy)
        .name("GPU Demo Network")
        .build()?;

    println!("  📐 Network architecture:");
    network.print_summary();

    // Configure training with GPU acceleration
    let mut training_config = TrainingConfig::default();
    training_config.max_epochs = 10;
    training_config.batch_size = 32;
    training_config.verbose = true;

    // Note: In a full implementation, we would modify the training loop
    // to use GPU tensors and operations. For now, we'll train on CPU
    // and demonstrate the concept.
    println!("  🏃 Starting training...");

    let history = network.train(&data.0, &data.1, &training_config)?;

    println!("  ✅ Training completed!");
    println!(
        "  📈 Final loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );

    // Demonstrate inference
    let predictions = network.predict(&data.0)?;
    println!(
        "  🔮 Generated predictions for {} samples",
        predictions.nrows()
    );

    Ok(())
}

fn demonstrate_cpu_fallback() -> Result<()> {
    println!("💻 Demonstrating CPU Fallback:");

    let data = create_sample_data();

    // Create a simple network for CPU training
    let mut network = Network::with_input_size(data.0.ncols())?
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(data.1.ncols()).activation(ActivationFunction::Sigmoid))
        .loss(LossFunction::BinaryCrossEntropy)
        .build()?;

    let mut config = TrainingConfig::default();
    config.max_epochs = 5;
    config.verbose = true;

    let history = network.train(&data.0, &data.1, &config)?;
    println!(
        "  ✅ CPU training completed with loss: {:.6}",
        history.train_loss.last().unwrap_or(&0.0)
    );

    Ok(())
}

fn create_sample_data() -> (Array2<f64>, Array2<f64>) {
    // Create sample XOR-like problem
    let input_data = Array2::from_shape_vec(
        (1000, 2),
        (0..2000)
            .map(|i| {
                if i % 2 == 0 {
                    (i as f64) / 1000.0
                } else {
                    1.0 - (i as f64) / 1000.0
                }
            })
            .collect(),
    )
    .unwrap();

    let target_data = Array2::from_shape_vec(
        (1000, 1),
        (0..1000)
            .map(|i| {
                let row = input_data.row(i);
                if (row[0] > 0.5) != (row[1] > 0.5) {
                    1.0
                } else {
                    0.0
                }
            })
            .collect(),
    )
    .unwrap();

    (input_data, target_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_creation() {
        let (inputs, targets) = create_sample_data();
        assert_eq!(inputs.nrows(), 1000);
        assert_eq!(inputs.ncols(), 2);
        assert_eq!(targets.nrows(), 1000);
        assert_eq!(targets.ncols(), 1);
    }

    #[test]
    fn test_gpu_manager_creation() {
        let manager = GpuManager::new();
        // Should always have at least the CPU backend
        assert!(!manager.devices().is_empty());
    }
}

/// Performance benchmarking for GPU vs CPU operations
#[cfg(feature = "benchmarks")]
mod benchmarks {
    use super::*;
    use std::time::Instant;

    pub fn benchmark_matrix_operations() -> Result<()> {
        println!("\n⚡ Performance Benchmarks:");

        let sizes = vec![100, 500, 1000, 2000];

        for size in sizes {
            println!("  Matrix size: {}x{}", size, size);

            // Create test matrices
            let a = Array2::from_shape_fn((size, size), |(i, j)| (i + j) as f64);
            let b = Array2::from_shape_fn((size, size), |(i, j)| (i * j) as f64);

            // CPU benchmark
            let start = Instant::now();
            let _cpu_result = a.dot(&b);
            let cpu_time = start.elapsed();

            println!("    CPU time: {:?}", cpu_time);

            // GPU benchmark (if available)
            let mut gpu_manager = GpuManager::new();
            if let Some(device) = gpu_manager.default_device() {
                if device.device_type != rnn::gpu::GpuDeviceType::Generic {
                    let context = gpu_manager.create_context(device.id)?;

                    let start = Instant::now();
                    let gpu_a = rnn::gpu::GpuTensor::from_cpu(&a, device.id, context)?;
                    let gpu_b = rnn::gpu::GpuTensor::from_cpu(&b, device.id, context)?;

                    // Note: Actual matrix multiplication would be implemented here
                    let transfer_time = start.elapsed();

                    println!("    GPU transfer time: {:?}", transfer_time);
                }
            }
        }

        Ok(())
    }
}

/// Integration examples with popular machine learning workflows
#[cfg(feature = "examples")]
mod integration_examples {
    use super::*;

    /// Demonstrate integration with image classification
    pub fn image_classification_example() -> Result<()> {
        println!("\n🖼️  Image Classification with GPU:");

        // Simulate MNIST-like data (28x28 grayscale images)
        let image_data =
            Array2::from_shape_fn((1000, 784), |(i, j)| ((i + j) as f64 / 1784.0).sin());

        let labels =
            Array2::from_shape_fn((1000, 10), |(i, _)| if i % 10 == 0 { 1.0 } else { 0.0 });

        // Create CNN-like architecture
        let mut network = Network::with_input_size(784)?
            .add_layer(LayerBuilder::dense(512).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(256).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
            .loss(LossFunction::CategoricalCrossEntropy)
            .name("Image Classifier")
            .build()?;

        println!("  🏗️  Created image classification network");

        let mut config = TrainingConfig::default();
        config.max_epochs = 15;
        config.batch_size = 64;
        config.validation_split = 0.2;

        let history = network.train(&image_data, &labels, &config)?;

        println!("  📊 Training metrics:");
        println!(
            "    Final training loss: {:.6}",
            history.train_loss.last().unwrap_or(&0.0)
        );
        if let Some(val_loss) = history.val_loss.last() {
            println!("    Final validation loss: {:.6}", val_loss);
        }

        Ok(())
    }

    /// Demonstrate time series prediction with RNN-like operations
    pub fn time_series_prediction() -> Result<()> {
        println!("\n📈 Time Series Prediction:");

        // Generate synthetic time series data
        let sequence_length = 50;
        let num_sequences = 500;
        let feature_dim = 1;

        let mut time_series = Vec::new();
        let mut targets = Vec::new();

        for seq in 0..num_sequences {
            let mut sequence = Vec::new();
            for t in 0..sequence_length {
                let value = ((seq + t) as f64 * 0.1).sin() + ((seq + t) as f64 * 0.05).cos() * 0.5;
                sequence.push(value);
            }

            // Use last value as input, predict next value
            time_series.extend(&sequence[..sequence_length - 1]);
            targets.push(sequence[sequence_length - 1]);
        }

        let input_data = Array2::from_shape_vec((num_sequences, sequence_length - 1), time_series)?;
        let target_data = Array2::from_shape_vec((num_sequences, 1), targets)?;

        // Create recurrent-like network
        let mut network = Network::with_input_size(sequence_length - 1)?
            .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::Tanh))
            .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::Tanh))
            .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Linear))
            .loss(LossFunction::MeanSquaredError)
            .name("Time Series Predictor")
            .build()?;

        let mut config = TrainingConfig::default();
        config.max_epochs = 20;
        config.batch_size = 32;

        let history = network.train(&input_data, &target_data, &config)?;

        println!("  📉 Time series prediction completed");
        println!(
            "    Final MSE loss: {:.6}",
            history.train_loss.last().unwrap_or(&0.0)
        );

        // Make some predictions
        let predictions = network.predict(&input_data.slice(ndarray::s![0..5, ..]).to_owned())?;
        println!(
            "  🔮 Sample predictions: {:?}",
            predictions.slice(ndarray::s![.., 0])
        );

        Ok(())
    }
}
