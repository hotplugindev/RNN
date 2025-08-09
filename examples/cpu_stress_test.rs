//! CPU Stress Test - Verify parallel processing utilization
//!
//! This example creates a larger training workload to demonstrate that the neural network
//! library properly utilizes all available CPU cores during training.

use nnl::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Neural Network Library - CPU Stress Test");
    println!("========================================");

    // Use CPU device
    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

    // Get CPU info
    let num_cpus = num_cpus::get();
    println!("Available CPU cores: {}", num_cpus);

    // Create a larger synthetic dataset for stress testing
    let num_samples = 1000;
    let input_size = 100;
    let output_size = 10;
    let batch_size = 32;

    println!(
        "Creating synthetic dataset: {} samples, {} input features, {} output classes",
        num_samples, input_size, output_size
    );

    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();

    // Generate random training data
    for i in 0..num_samples {
        // Create random input
        let mut input_data = vec![0.0f32; input_size];
        for j in 0..input_size {
            input_data[j] = (i * input_size + j) as f32 / (num_samples * input_size) as f32;
            input_data[j] = input_data[j].sin(); // Add some non-linearity
        }

        // Create one-hot target
        let target_class = i % output_size;
        let mut target_data = vec![0.0f32; output_size];
        target_data[target_class] = 1.0;

        // Convert to tensors (individual samples with batch dimension 1)
        let input_tensor =
            Tensor::from_slice_on_device(&input_data, &[1, input_size], device.clone())?;
        let target_tensor =
            Tensor::from_slice_on_device(&target_data, &[1, output_size], device.clone())?;

        train_inputs.push(input_tensor);
        train_targets.push(target_tensor);
    }

    println!("Dataset created successfully!");

    // Build a moderately complex neural network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size,
            output_size: 256,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 256,
            output_size: 128,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 128,
            output_size: 64,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 64,
            output_size,
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::CrossEntropy)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "\nNetwork created with {} parameters",
        network.num_parameters()
    );
    println!("Network summary:");
    println!("{}", network);

    // Training configuration for stress testing
    let training_config = TrainingConfig {
        epochs: 10,
        batch_size,
        verbose: true,
        early_stopping_patience: 0, // Disable early stopping for full stress test
        early_stopping_threshold: 0.0,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("\nStarting CPU stress test training...");
    println!("Batch size: {}", batch_size);
    println!(
        "Total batches per epoch: {}",
        (num_samples + batch_size - 1) / batch_size
    );
    println!("Monitor your system's CPU usage during this training!");
    println!(
        "With proper parallelization, you should see high utilization across all {} cores.",
        num_cpus
    );

    let start_time = Instant::now();

    let history = network.train(&train_inputs, &train_targets, &training_config)?;

    let training_time = start_time.elapsed();

    println!("\nCPU stress test completed!");
    println!("Total training time: {:.2}s", training_time.as_secs_f64());
    println!("Final loss: {:.6}", history.final_loss());
    println!("Final accuracy: {:.2}%", history.final_accuracy() * 100.0);

    // Calculate throughput metrics
    let total_forward_passes = num_samples * training_config.epochs;
    let throughput = total_forward_passes as f64 / training_time.as_secs_f64();

    println!("\nPerformance metrics:");
    println!("Total forward passes: {}", total_forward_passes);
    println!("Throughput: {:.1} samples/second", throughput);
    println!("Average time per sample: {:.3}ms", 1000.0 / throughput);

    // Test inference speed
    println!("\nTesting inference performance...");
    let test_input = &train_inputs[0];
    let num_inference_tests = 1000;

    let start_time = Instant::now();
    for _ in 0..num_inference_tests {
        let _ = network.forward(test_input)?;
    }
    let inference_time = start_time.elapsed();

    let avg_inference_time = inference_time.as_secs_f64() * 1000.0 / num_inference_tests as f64;
    println!(
        "Average inference time: {:.3}ms ({} iterations)",
        avg_inference_time, num_inference_tests
    );

    println!("\nCPU stress test completed successfully!");
    println!("If you observed high CPU utilization across multiple cores during training,");
    println!("then the parallel processing optimization is working correctly!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_stress() -> Result<()> {
        let device = Device::cpu()?;

        // Create a small test dataset
        let input_data = vec![0.5f32; 10];
        let target_data = vec![0.0f32, 1.0f32, 0.0f32]; // 3 classes

        let input = Tensor::from_slice_on_device(&input_data, &[1, 10], device.clone())?;
        let target = Tensor::from_slice_on_device(&target_data, &[1, 3], device.clone())?;

        assert_eq!(input.shape(), &[1, 10]);
        assert_eq!(target.shape(), &[1, 3]);

        // Test network creation
        let network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 10,
                output_size: 5,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 5,
                output_size: 3,
                activation: Activation::Softmax,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::Adam {
                learning_rate: 0.001,
                beta1: 0.9,
                beta2: 0.999,
                epsilon: 1e-8,
                weight_decay: None,
                amsgrad: false,
            })
            .device(device)
            .build()?;

        // Test forward pass
        let output = network.forward(&input)?;
        assert_eq!(output.shape(), &[1, 3]);

        Ok(())
    }
}
