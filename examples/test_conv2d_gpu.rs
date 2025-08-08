//! Minimal Conv2D Test
//!
//! This example tests Conv2D layers in isolation to identify issues.

use rnn::prelude::*;

fn main() -> Result<()> {
    env_logger::init();

    println!("Minimal Conv2D Test");
    println!("===================");

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

    // Test 1: Create a simple Conv2D + Dense network (like working examples but with Conv2D)
    println!("\n1. Testing Conv2D + Dense with MSE loss (like working examples)...");

    let mut network1 = NetworkBuilder::new()
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
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
            input_size: 2 * 5 * 5, // 2 channels, 5x5 image
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError) // Use MSE like working examples
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

    // Create simple 5x5 image data
    let input1 = Tensor::from_slice_on_device(
        &vec![0.5f32; 25], // 5x5 = 25 pixels
        &[1, 1, 5, 5],
        device.clone(),
    )?;
    let target1 = Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?;

    println!("Testing forward pass...");
    let output1 = network1.forward(&input1)?;
    println!(
        "Forward pass successful! Output shape: {:?}",
        output1.shape()
    );

    // Test training for 1 epoch
    let config1 = TrainingConfig {
        epochs: 1,
        batch_size: 1,
        verbose: true,
        early_stopping_patience: 0,
        early_stopping_threshold: 0.0,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: false,
        random_seed: Some(42),
    };

    println!("Training for 1 epoch...");
    let _history1 = network1.train(&[input1.clone()], &[target1.clone()], &config1)?;
    println!("Training completed for network 1");

    // Test 2: Same architecture but with CrossEntropy loss
    println!("\n2. Testing Conv2D + Dense with CrossEntropy loss...");

    let mut network2 = NetworkBuilder::new()
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
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
            input_size: 2 * 5 * 5,
            output_size: 2, // 2 classes for CrossEntropy
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::CrossEntropy) // Use CrossEntropy like mnist_cnn
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

    // Create classification data
    let input2 = input1.clone();
    let target2 = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?; // One-hot

    println!("Testing forward pass...");
    let output2 = network2.forward(&input2)?;
    println!(
        "Forward pass successful! Output shape: {:?}",
        output2.shape()
    );

    println!("Training for 1 epoch...");
    let _history2 = network2.train(&[input2], &[target2], &config1)?;
    println!("Training completed for network 2");

    // Test 3: Dense-only network with CrossEntropy (should work based on working examples)
    println!("\n3. Testing Dense-only with CrossEntropy loss...");

    let mut network3 = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 25, // Flattened 5x5
            output_size: 8,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 8,
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

    // Use flattened input
    let input3 = Tensor::from_slice_on_device(&vec![0.5f32; 25], &[1, 25], device.clone())?;
    let target3 = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?;

    println!("Testing forward pass...");
    let output3 = network3.forward(&input3)?;
    println!(
        "Forward pass successful! Output shape: {:?}",
        output3.shape()
    );

    println!("Training for 1 epoch...");
    let _history3 = network3.train(&[input3], &[target3], &config1)?;
    println!("Training completed for network 3");

    // Test 4: Check weight initialization for Conv2D layer specifically
    println!("\n4. Testing Conv2D weight initialization...");

    let test_conv = NetworkBuilder::new()
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 4,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
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
        "Conv2D test network created with {} parameters",
        test_conv.num_parameters()
    );
    test_conv.check_weight_initialization()?;

    println!("\nAll tests completed!");
    Ok(())
}
