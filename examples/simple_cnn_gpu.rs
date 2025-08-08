//! Simple CNN Test
//!
//! This example creates a minimal CNN to isolate training issues.

use nnl::prelude::*;

fn main() -> Result<()> {
    env_logger::init();

    println!("Simple CNN Test");
    println!("===============");

    let device = match Device::vulkan().or_else(|_| Device::cpu()) {
        Ok(device) => {
            println!("Using device: {:?}", device.device_type());
            device
        }
        Err(e) => {
            println!("Failed to initialize GPU device: {}", e);
            println!("Falling back to CPU");
            Device::cpu()?
        }
    };
    println!("Using device: {:?}", device.device_type());

    // Create very simple binary classification data
    // Two 3x3 patterns: cross (class 0) and square (class 1)
    let cross_pattern = vec![0.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0];

    let square_pattern = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0];

    // Create training data - 4 samples
    let mut train_inputs = Vec::new();
    let mut train_targets = Vec::new();

    // Add cross patterns (class 0)
    for i in 0..2 {
        let mut pattern = cross_pattern.clone();
        // Add small noise
        for p in pattern.iter_mut() {
            *p += (i as f32) * 0.1;
        }
        let input = Tensor::from_slice_on_device(&pattern, &[1, 1, 3, 3], device.clone())?;
        let target = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?; // One-hot for class 0
        train_inputs.push(input);
        train_targets.push(target);
    }

    // Add square patterns (class 1)
    for i in 0..2 {
        let mut pattern = square_pattern.clone();
        // Add small noise
        for p in pattern.iter_mut() {
            *p += (i as f32) * 0.1;
        }
        let input = Tensor::from_slice_on_device(&pattern, &[1, 1, 3, 3], device.clone())?;
        let target = Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?; // One-hot for class 1
        train_inputs.push(input);
        train_targets.push(target);
    }

    println!("Created {} training samples", train_inputs.len());

    // Test 1: Simple CNN with MSE loss (should work like basic examples)
    println!("\n1. Testing CNN with MSE loss...");

    let mut network1 = NetworkBuilder::new()
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (2, 2),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 2 * 2 * 2, // 2 channels, 2x2 after conv
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
        .device(device.clone())
        .build()?;

    println!(
        "Network 1 created with {} parameters",
        network1.num_parameters()
    );

    // Convert targets to single value for MSE
    let mse_targets: Vec<Tensor> = train_targets
        .iter()
        .map(|t| {
            let data = t.to_vec().unwrap();
            let value = if data[0] > 0.5 { 0.0 } else { 1.0 }; // Cross=0, Square=1
            Tensor::from_slice_on_device(&[value], &[1, 1], device.clone()).unwrap()
        })
        .collect();

    // Test forward pass
    println!("Testing forward pass...");
    let output1 = network1.forward(&train_inputs[0])?;
    println!(
        "Forward pass successful! Output shape: {:?}",
        output1.shape()
    );

    // Train for a few epochs
    let config = TrainingConfig {
        epochs: 5,
        batch_size: 1,
        verbose: true,
        early_stopping_patience: 0,
        early_stopping_threshold: 0.0,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: false,
        random_seed: Some(42),
    };

    println!("Training CNN with MSE loss...");
    let history1 = network1.train(&train_inputs, &mse_targets, &config)?;
    println!(
        "MSE Training completed! Final loss: {:.6}",
        history1.final_loss()
    );

    // Test predictions
    println!("\nTesting MSE predictions:");
    for (i, input) in train_inputs.iter().enumerate() {
        let prediction = network1.forward(input)?;
        let pred_value = prediction.to_vec()?[0];
        let target_value = mse_targets[i].to_vec()?[0];
        println!(
            "Sample {}: Predicted: {:.3}, Target: {:.3}",
            i + 1,
            pred_value,
            target_value
        );
    }

    // Test 2: CNN with CrossEntropy loss
    println!("\n2. Testing CNN with CrossEntropy loss...");

    let mut network2 = NetworkBuilder::new()
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (2, 2),
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 2 * 2 * 2, // 2 channels, 2x2 after conv
            output_size: 2,
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::CrossEntropy)
        .optimizer(OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network 2 created with {} parameters",
        network2.num_parameters()
    );

    // Test forward pass
    println!("Testing forward pass...");
    let output2 = network2.forward(&train_inputs[0])?;
    println!(
        "Forward pass successful! Output shape: {:?}",
        output2.shape()
    );

    println!("Training CNN with CrossEntropy loss...");
    let history2 = network2.train(&train_inputs, &train_targets, &config)?;
    println!(
        "CrossEntropy Training completed! Final loss: {:.6}",
        history2.final_loss()
    );

    // Test predictions
    println!("\nTesting CrossEntropy predictions:");
    for (i, input) in train_inputs.iter().enumerate() {
        println!("Debug: Processing sample {}", i + 1);

        // Debug: Check input tensor
        let input_data = input.to_vec()?;
        println!(
            "Debug: Input data contains NaN: {}",
            input_data.iter().any(|x| x.is_nan())
        );
        println!(
            "Debug: Input range: [{:.6}, {:.6}]",
            input_data.iter().fold(f32::INFINITY, |a, &b| a.min(b)),
            input_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        );

        let prediction = network2.forward(input)?;
        let pred_data = prediction.to_vec()?;

        // Debug: Check prediction tensor
        println!(
            "Debug: Prediction data contains NaN: {}",
            pred_data.iter().any(|x| x.is_nan())
        );
        println!(
            "Debug: Raw prediction values: [{:.10}, {:.10}]",
            pred_data[0], pred_data[1]
        );
        println!(
            "Debug: Sum of predictions: {:.10}",
            pred_data[0] + pred_data[1]
        );

        let target_data = train_targets[i].to_vec()?;
        let predicted_class = if pred_data[0] > pred_data[1] { 0 } else { 1 };
        let actual_class = if target_data[0] > 0.5 { 0 } else { 1 };
        println!(
            "Sample {}: Predicted class: {}, Actual class: {}, Probabilities: [{:.3}, {:.3}]",
            i + 1,
            predicted_class,
            actual_class,
            pred_data[0],
            pred_data[1]
        );
    }

    // Test 3: Dense-only with same data for comparison
    println!("\n3. Testing Dense-only with CrossEntropy for comparison...");

    let mut network3 = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 9, // Flattened 3x3
            output_size: 4,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 4,
            output_size: 2,
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::CrossEntropy)
        .optimizer(OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network 3 created with {} parameters",
        network3.num_parameters()
    );

    // Flatten inputs for dense network
    let dense_inputs: Result<Vec<Tensor>> = train_inputs
        .iter()
        .map(|input| {
            let data = input.to_vec()?;
            Tensor::from_slice_on_device(&data, &[1, 9], device.clone())
        })
        .collect();
    let dense_inputs = dense_inputs?;

    println!("Training Dense network with CrossEntropy loss...");
    let history3 = network3.train(&dense_inputs, &train_targets, &config)?;
    println!(
        "Dense Training completed! Final loss: {:.6}",
        history3.final_loss()
    );

    // Test predictions
    println!("\nTesting Dense predictions:");
    for (i, input) in dense_inputs.iter().enumerate() {
        println!("Debug Dense: Processing sample {}", i + 1);

        // Debug: Check input tensor
        let input_data = input.to_vec()?;
        println!(
            "Debug Dense: Input data contains NaN: {}",
            input_data.iter().any(|x| x.is_nan())
        );

        let prediction = network3.forward(input)?;
        let pred_data = prediction.to_vec()?;

        // Debug: Check prediction tensor
        println!(
            "Debug Dense: Prediction data contains NaN: {}",
            pred_data.iter().any(|x| x.is_nan())
        );
        println!(
            "Debug Dense: Raw prediction values: [{:.10}, {:.10}]",
            pred_data[0], pred_data[1]
        );
        println!(
            "Debug Dense: Sum of predictions: {:.10}",
            pred_data[0] + pred_data[1]
        );

        let target_data = train_targets[i].to_vec()?;
        let predicted_class = if pred_data[0] > pred_data[1] { 0 } else { 1 };
        let actual_class = if target_data[0] > 0.5 { 0 } else { 1 };
        println!(
            "Sample {}: Predicted class: {}, Actual class: {}, Probabilities: [{:.3}, {:.3}]",
            i + 1,
            predicted_class,
            actual_class,
            pred_data[0],
            pred_data[1]
        );
    }

    // Summary
    println!("\nSummary:");
    println!(
        "  CNN + MSE loss: Final loss = {:.6}",
        history1.final_loss()
    );
    println!(
        "  CNN + CrossEntropy loss: Final loss = {:.6}",
        history2.final_loss()
    );
    println!(
        "  Dense + CrossEntropy loss: Final loss = {:.6}",
        history3.final_loss()
    );

    if history2.final_loss().is_finite() && history2.final_loss() < 2.0 {
        println!("✅ CNN with CrossEntropy appears to be working!");
    } else {
        println!("❌ CNN with CrossEntropy still has issues");
    }

    Ok(())
}
