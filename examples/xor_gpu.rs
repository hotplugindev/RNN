//! XOR Neural Network Training Example - GPU Version
//!
//! This example demonstrates training a simple neural network to learn the XOR function
//! using GPU acceleration. The network consists of a single hidden layer and uses
//! backpropagation for training.

use rnn::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("XOR Neural Network Training - GPU Version");
    println!("=========================================");

    // Try to get a GPU device, fall back to CPU if not available
    let device = match Device::vulkan()
        .or_else(|_| Device::webgpu())
        .or_else(|_| Device::cpu())
    {
        Ok(device) => {
            println!(
                "Using device: {} ({})",
                device.info().name,
                device.device_type()
            );
            device
        }
        Err(e) => {
            println!("Failed to initialize GPU device: {}", e);
            println!("Falling back to CPU");
            Device::cpu()?
        }
    };

    // Create XOR training data
    let inputs = Tensor::from_slice_on_device(
        &[
            0.0, 0.0, // XOR(0, 0) = 0
            0.0, 1.0, // XOR(0, 1) = 1
            1.0, 0.0, // XOR(1, 0) = 1
            1.0, 1.0, // XOR(1, 1) = 0
        ],
        &[4, 2],
        device.clone(),
    )?;

    let targets = Tensor::from_slice_on_device(&[0.0, 1.0, 1.0, 0.0], &[4, 1], device.clone())?;

    println!("Training data created on device");
    println!("Input shape: {:?}", inputs.shape());
    println!("Target shape: {:?}", targets.shape());

    // Build the neural network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 4,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 4,
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network created with {} parameters",
        network.num_parameters()
    );

    // Training configuration
    let training_config = TrainingConfig {
        epochs: 20,
        batch_size: 4,
        verbose: true,
        early_stopping_patience: 0,
        early_stopping_threshold: 0.001,
        lr_schedule: None,
        validation_split: 0.2,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("\nStarting training...");
    let start_time = Instant::now();

    // Train the network - split batched data into individual samples
    let mut input_vec = Vec::new();
    let mut target_vec = Vec::new();

    let input_data = inputs.to_vec()?;
    let target_data = targets.to_vec()?;

    for i in 0..4 {
        let input_sample =
            Tensor::from_slice_on_device(&input_data[i * 2..(i + 1) * 2], &[1, 2], device.clone())?;
        let target_sample =
            Tensor::from_slice_on_device(&target_data[i..i + 1], &[1, 1], device.clone())?;
        input_vec.push(input_sample);
        target_vec.push(target_sample);
    }

    network.train(&input_vec, &target_vec, &training_config)?;

    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}s", training_time.as_secs_f64());

    // Test the trained network
    println!("\nTesting the trained network:");
    println!("============================");

    // Define test cases explicitly to ensure correct display
    let test_cases = [
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    for (input_vals, expected) in test_cases.iter() {
        let test_input = Tensor::from_slice_on_device(input_vals, &[1, 2], device.clone())?;

        let prediction = network.forward(&test_input)?;
        let predicted_value = prediction.to_vec()?[0];

        println!(
            "Input: [{:.0}, {:.0}] -> Predicted: {:.4}, Actual: {:.0}",
            input_vals[0], input_vals[1], predicted_value, expected
        );
    }

    // Benchmark inference speed
    println!("\nBenchmarking inference speed...");
    let benchmark_input = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?;
    let num_iterations = 1000;

    let start_time = Instant::now();
    for _ in 0..num_iterations {
        let _ = network.forward(&benchmark_input)?;
    }
    device.synchronize()?; // Ensure all GPU operations complete
    let inference_time = start_time.elapsed();

    println!(
        "Average inference time: {:.4}ms ({} iterations)",
        inference_time.as_secs_f64() * 1000.0 / num_iterations as f64,
        num_iterations
    );

    // Save the trained model
    let model_path = "xor_gpu_model.bin";
    rnn::io::save_model(&network, model_path, ModelFormat::Binary, None)?;
    println!("\nModel saved to: {}", model_path);

    // Demonstrate model loading
    println!("Testing model loading...");
    let mut loaded_network = rnn::io::load_network(model_path, ModelFormat::Binary)?;

    let test_prediction = loaded_network.forward(&benchmark_input)?;
    println!(
        "Loaded model prediction: {:.4}",
        test_prediction.to_vec()?[0]
    );

    println!("\nXOR GPU training example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xor_gpu_training() -> Result<()> {
        // Skip GPU test if no GPU is available
        let device = match Device::vulkan().or_else(|_| Device::webgpu()) {
            Ok(device) => device,
            Err(_) => {
                println!("Skipping GPU test - no GPU available");
                return Ok(());
            }
        };

        let inputs = Tensor::from_slice_on_device(
            &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            &[4, 2],
            device.clone(),
        )?;

        let targets = Tensor::from_slice_on_device(&[0.0, 1.0, 1.0, 0.0], &[4, 1], device.clone())?;

        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 4,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 4,
                output_size: 1,
                activation: Activation::Sigmoid,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::SGD {
                learning_rate: 0.1,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            })
            .device(device)
            .build()?;

        // Quick training for test
        let config = TrainingConfig {
            epochs: 100,
            batch_size: 4,
            verbose: false,
            early_stopping_patience: 0,
            early_stopping_threshold: 0.001,
            lr_schedule: None,
            validation_split: 0.0,
            shuffle: true,
            random_seed: Some(42),
        };

        let input_vec = vec![inputs];
        let target_vec = vec![targets];
        network.train(&input_vec, &target_vec, &config)?;

        // Test basic functionality
        let test_input = Tensor::from_slice(&[1.0, 0.0], &[1, 2])?;
        let output = network.forward(&test_input)?;
        assert_eq!(output.shape(), &[1, 1]);

        Ok(())
    }
}
