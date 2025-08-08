//! Basic Usage Example for RNN Neural Network Library
//!
//! This example demonstrates the basic functionality of the RNN library,
//! including creating networks, training, and making predictions.

use rnn::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("RNN Library - Basic Usage Example");
    println!("==================================");

    // Demonstrate device selection
    demonstrate_device_selection()?;

    // Demonstrate tensor operations
    demonstrate_tensor_operations()?;

    // Demonstrate network creation and training
    demonstrate_network_training()?;

    // Demonstrate model persistence
    demonstrate_model_persistence()?;

    println!("\nBasic usage example completed successfully!");
    Ok(())
}

/// Demonstrate automatic device selection and listing available devices
fn demonstrate_device_selection() -> Result<()> {
    println!("\n1. Device Selection");
    println!("-------------------");

    // List all available devices
    let devices = rnn::device::utils::list_devices();
    println!("Available devices:");
    for device in &devices {
        println!("  - {} ({})", device.name, device.device_type);
        if let Some(memory) = device.memory_size {
            println!("    Memory: {:.2} GB", memory as f64 / 1_000_000_000.0);
        }
        if let Some(compute_units) = device.compute_units {
            println!("    Compute units: {}", compute_units);
        }
    }

    // Auto-select the best device
    let device = Device::auto_select()?;
    println!(
        "\nAuto-selected device: {} ({})",
        device.info().name,
        device.device_type()
    );

    Ok(())
}

/// Demonstrate basic tensor operations
fn demonstrate_tensor_operations() -> Result<()> {
    println!("\n2. Tensor Operations");
    println!("--------------------");

    // Use CPU device for tensor operations
    let device = Device::cpu()?;

    // Create tensors
    let a = Tensor::from_slice_on_device(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone())?;
    let b = Tensor::from_slice_on_device(&[5.0, 6.0, 7.0, 8.0], &[2, 2], device.clone())?;

    println!("Tensor A:\n{}", a);
    println!("Tensor B:\n{}", b);

    // Basic arithmetic
    let sum = (&a + &b)?;
    println!("A + B:\n{}", sum);

    let product = (&a * &b)?;
    println!("A * B (element-wise):\n{}", product);

    // Matrix operations
    let matrix_a = Tensor::from_slice_on_device(&[1.0, 2.0, 3.0, 4.0], &[2, 2], device.clone())?;
    let matrix_b = Tensor::from_slice_on_device(&[5.0, 6.0, 7.0, 8.0], &[2, 2], device.clone())?;
    let matmul = matrix_a.matmul(&matrix_b)?;
    println!("Matrix multiplication A @ B:\n{}", matmul);

    // Reshape and transpose
    let reshaped = a.reshape(&[4, 1])?;
    println!("Reshaped A to [4, 1]:\n{}", reshaped);

    let transposed = a.transpose()?;
    println!("Transposed A:\n{}", transposed);

    // Reductions
    println!("Sum of A: {:.2}", a.sum()?);
    println!("Mean of A: {:.2}", a.mean()?);
    println!("Max of A: {:.2}", a.max()?);
    println!("Min of A: {:.2}", a.min()?);

    Ok(())
}

/// Demonstrate network creation, training, and evaluation
fn demonstrate_network_training() -> Result<()> {
    println!("\n3. Network Training");
    println!("-------------------");

    // Create a simple regression dataset
    let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y_data: Vec<f32> = x_data
        .iter()
        .map(|x| 2.0 * x + 1.0 + rand::random::<f32>() * 0.1 - 0.05)
        .collect();

    let device = Device::cpu()?;
    let inputs = Tensor::from_slice_on_device(&x_data, &[10, 1], device.clone())?;
    let targets = Tensor::from_slice_on_device(&y_data, &[10, 1], device.clone())?;

    println!("Training data:");
    for i in 0..5 {
        println!("  x: {:.1}, y: {:.3}", x_data[i], y_data[i]);
    }
    println!("  ... (showing first 5 of 10 samples)");

    // Build a simple network for linear regression
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 1,
            output_size: 8,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 8,
            output_size: 1,
            activation: Activation::Linear,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: Some(1e-4),
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network created with {} parameters",
        network.num_parameters()
    );

    // Training configuration
    let config = TrainingConfig {
        epochs: 500,
        batch_size: 10,
        verbose: true,
        early_stopping_patience: 50,
        early_stopping_threshold: 1e-6,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: false,
        random_seed: None,
    };

    // Train the network - split batched data into individual samples
    println!("\nTraining network...");
    let start_time = Instant::now();

    let mut input_vec = Vec::new();
    let mut target_vec = Vec::new();

    for i in 0..10 {
        let input_sample = Tensor::from_slice_on_device(&[x_data[i]], &[1, 1], device.clone())?;
        let target_sample = Tensor::from_slice_on_device(&[y_data[i]], &[1, 1], device.clone())?;
        input_vec.push(input_sample);
        target_vec.push(target_sample);
    }

    network.train(&input_vec, &target_vec, &config)?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2}s", training_time.as_secs_f64());

    // Test predictions
    println!("\nTesting predictions:");
    let test_inputs = vec![11.0, 12.0, 15.0, 20.0];
    let device = Device::cpu()?;
    for &x in &test_inputs {
        let input = Tensor::from_slice_on_device(&[x], &[1, 1], device.clone())?;
        let prediction = network.forward(&input)?;
        let pred_value = prediction.to_vec()?[0];
        let expected = 2.0 * x + 1.0;

        println!(
            "  x: {:.1} -> predicted: {:.3}, expected: {:.3}",
            x, pred_value, expected
        );
    }

    Ok(())
}

/// Demonstrate model saving and loading
fn demonstrate_model_persistence() -> Result<()> {
    println!("\n4. Model Persistence");
    println!("--------------------");

    // Create a simple network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 3,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 3,
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: Some(0.9),
            weight_decay: None,
            nesterov: false,
        })
        .device(Device::cpu()?)
        .build()?;

    // Make a prediction before saving
    let test_input = Tensor::from_slice_on_device(&[0.5, -0.3], &[1, 2], Device::cpu()?)?;
    let original_prediction = network.forward(&test_input)?;
    let original_value = original_prediction.to_vec()?[0];

    println!("Original prediction: {:.6}", original_value);

    // Save in different formats
    let formats = vec![
        ("binary", ModelFormat::Binary),
        ("json", ModelFormat::Json),
        ("msgpack", ModelFormat::MessagePack),
    ];

    for (name, format) in formats {
        let filename = format!("demo_model.{}", name);

        // Save model (note: loading functionality is still being implemented)
        save_model(&network, &filename, format, None)?;
        println!("Model saved as: {}", filename);

        // TODO: Model loading will be implemented when SerializableModel -> Network conversion is added
        println!("Model loading functionality is under development");
    }

    Ok(())
}

/// Demonstrate different activation functions
#[allow(dead_code)]
fn demonstrate_activations() -> Result<()> {
    println!("\n5. Activation Functions");
    println!("-----------------------");

    let input = Tensor::from_slice(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5, 1])?;
    println!("Input: {:?}", input.to_vec()?);

    let activations = vec![
        (Activation::ReLU, "ReLU"),
        (Activation::Sigmoid, "Sigmoid"),
        (Activation::Tanh, "Tanh"),
        (Activation::Linear, "Linear"),
    ];

    for (activation, name) in activations {
        let output = input.activation(activation)?;
        println!("{}: {:?}", name, output.to_vec()?);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_selection() {
        let result = demonstrate_device_selection();
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_operations() {
        let result = demonstrate_tensor_operations();
        assert!(result.is_ok());
    }

    #[test]
    fn test_basic_training() -> Result<()> {
        // Simple test with minimal training
        let device = Device::cpu()?;
        let inputs = Tensor::from_slice_on_device(&[1.0, 2.0, 3.0, 4.0], &[4, 1], device.clone())?;
        let targets = Tensor::from_slice_on_device(&[2.0, 4.0, 6.0, 8.0], &[4, 1], device.clone())?;

        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 1,
                output_size: 2,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 1,
                activation: Activation::Linear,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::SGD {
                learning_rate: 0.01,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            })
            .device(device.clone())
            .build()?;

        let config = TrainingConfig {
            epochs: 100,
            batch_size: 1,
            verbose: false,
            early_stopping_patience: 0,
            early_stopping_threshold: 0.0,
            lr_schedule: None,
            validation_split: 0.0,
            shuffle: false,
            random_seed: None,
        };

        network.train(&[inputs], &[targets], &config)?;

        let test_input = Tensor::from_slice_on_device(&[5.0], &[1, 1], device)?;
        let output = network.forward(&test_input)?;
        assert_eq!(output.shape(), &[1, 1]);

        Ok(())
    }
}
