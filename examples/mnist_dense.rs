//! MNIST Dense Neural Network Example
//!
//! This example demonstrates training a simple dense neural network to classify
//! handwritten digits from the MNIST dataset. This serves as a baseline before
//! moving to more complex CNN architectures.

use rnn::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("MNIST Dense Neural Network Example");
    println!("==================================");

    // Auto-select the best available device
    let device = Device::auto_select()?;
    println!(
        "Using device: {} ({})",
        device.info().name,
        device.device_type()
    );

    // Load MNIST dataset (simulated for this example)
    let (train_images, train_labels, test_images, test_labels) = load_mnist_data(&device)?;

    println!("Dataset loaded:");
    println!("  Training samples: {}", train_images.len());
    println!("  Test samples: {}", test_images.len());
    println!("  Image dimensions: 28x28 (flattened to 784)");

    // Build simple dense neural network
    let mut network = NetworkBuilder::new()
        .name("MNIST Dense Network")
        .description("Simple dense network for MNIST classification")
        // Input layer (784 = 28*28 flattened pixels)
        .add_layer(LayerConfig::Dense {
            input_size: 784,
            output_size: 128,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
        // Hidden layer
        .add_layer(LayerConfig::Dense {
            input_size: 128,
            output_size: 64,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
        // Output layer (10 classes for digits 0-9)
        .add_layer(LayerConfig::Dense {
            input_size: 64,
            output_size: 10,
            activation: Activation::Softmax,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::CrossEntropy)
        .optimizer(OptimizerConfig::SGD {
            learning_rate: 0.01,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "Network created with {} parameters",
        network.num_parameters()
    );
    println!("Network architecture:");
    println!("{}", network.summary());

    // Test forward pass
    println!("\nTesting forward pass...");
    let test_input = &train_images[0];
    match network.forward(test_input) {
        Ok(output) => {
            println!(
                "Forward pass successful! Output shape: {:?}",
                output.shape()
            );
        }
        Err(e) => {
            println!("Forward pass failed: {}", e);
            return Err(e);
        }
    }

    // Training configuration - minimal like working test
    let training_config = TrainingConfig {
        epochs: 3,
        batch_size: 8,
        verbose: true,
        early_stopping_patience: 0,
        early_stopping_threshold: 1e-4,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: false,
        random_seed: Some(42),
    };

    println!("\nStarting training...");
    let start_time = Instant::now();

    // Train the network
    let history = network.train(&train_images, &train_labels, &training_config)?;

    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}s", training_time.as_secs_f64());

    // Print training summary
    println!("\nTraining Summary:");
    println!("{}", history.summary());

    // Evaluate on test set
    println!("\nEvaluating on test set...");
    network.set_training(false);

    let mut correct = 0;
    let total_test_samples = test_images.len();

    for i in 0..total_test_samples {
        let prediction = network.forward(&test_images[i])?;
        let predicted_class = argmax(&prediction.to_vec()?);
        let actual_class = argmax(&test_labels[i].to_vec()?);

        if predicted_class == actual_class {
            correct += 1;
        }

        // Print first few predictions
        if i < 5 {
            println!(
                "Sample {}: Predicted: {}, Actual: {}",
                i + 1,
                predicted_class,
                actual_class
            );
        }
    }

    let accuracy = correct as f64 / total_test_samples as f64;
    println!("\nTest Results:");
    println!(
        "  Accuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct,
        total_test_samples
    );

    // Save the trained model
    let model_path = "mnist_dense_model.bin";
    let metadata = rnn::io::ModelMetadata {
        name: "MNIST Dense Model".to_string(),
        description: "Dense neural network for MNIST digit classification".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        modified_at: chrono::Utc::now().to_rfc3339(),
        training_info: rnn::io::TrainingInfo {
            epochs_trained: history.epochs(),
            final_loss: history.final_loss(),
            best_accuracy: history.best_accuracy(),
            training_time_seconds: training_time.as_secs_f32(),
            dataset_info: Some(rnn::io::DatasetInfo {
                name: "MNIST".to_string(),
                train_samples: train_images.len(),
                val_samples: None,
                test_samples: Some(test_images.len()),
                num_classes: Some(10),
            }),
        },
        metrics: std::collections::HashMap::new(),
        custom: std::collections::HashMap::new(),
    };

    rnn::io::save_model(
        &network,
        model_path,
        rnn::io::ModelFormat::Binary,
        Some(metadata),
    )?;
    println!("\nModel saved to: {}", model_path);

    println!("\nMNIST Dense example completed successfully!");

    Ok(())
}

/// Load MNIST dataset - creates flattened 784-dimensional vectors for dense layers
fn load_mnist_data(
    device: &Device,
) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
    use rand::prelude::*;

    let mut rng = thread_rng();

    // Simulate training data (minimal size for testing)
    let train_size = 50;
    let test_size = 20;
    let img_size = 28;
    let flattened_size = img_size * img_size; // 784

    println!("Generating simulated MNIST data...");

    // Generate training data
    let mut train_images = Vec::new();
    let mut train_labels = Vec::new();

    for i in 0..train_size {
        // Create simple patterns for each digit
        let label = i % 10;

        // Create pattern-based image data
        let mut img_data = vec![0.1f32; flattened_size];

        // Add digit-specific patterns
        match label {
            0 => create_circle_pattern(&mut img_data, img_size),
            1 => create_vertical_line_pattern(&mut img_data, img_size),
            2 => create_horizontal_lines_pattern(&mut img_data, img_size),
            3 => create_curves_pattern(&mut img_data, img_size),
            4 => create_angle_pattern(&mut img_data, img_size),
            5 => create_s_pattern(&mut img_data, img_size),
            6 => create_loop_pattern(&mut img_data, img_size),
            7 => create_diagonal_pattern(&mut img_data, img_size),
            8 => create_double_circle_pattern(&mut img_data, img_size),
            9 => create_spiral_pattern(&mut img_data, img_size),
            _ => {}
        }

        // Add noise
        for pixel in img_data.iter_mut() {
            *pixel += rng.gen::<f32>() * 0.2 - 0.1;
            *pixel = pixel.clamp(0.0, 1.0);
        }

        // Create tensors
        let img_tensor = Tensor::from_slice_on_device(&img_data, &[1, 784], device.clone())?;

        // Create one-hot encoded label
        let mut one_hot = vec![0.0f32; 10];
        one_hot[label] = 1.0;
        let label_tensor = Tensor::from_slice_on_device(&one_hot, &[1, 10], device.clone())?;

        train_images.push(img_tensor);
        train_labels.push(label_tensor);
    }

    // Generate test data similarly
    let mut test_images = Vec::new();
    let mut test_labels = Vec::new();

    for i in 0..test_size {
        let label = i % 10;

        let mut img_data = vec![0.1f32; flattened_size];

        match label {
            0 => create_circle_pattern(&mut img_data, img_size),
            1 => create_vertical_line_pattern(&mut img_data, img_size),
            2 => create_horizontal_lines_pattern(&mut img_data, img_size),
            3 => create_curves_pattern(&mut img_data, img_size),
            4 => create_angle_pattern(&mut img_data, img_size),
            5 => create_s_pattern(&mut img_data, img_size),
            6 => create_loop_pattern(&mut img_data, img_size),
            7 => create_diagonal_pattern(&mut img_data, img_size),
            8 => create_double_circle_pattern(&mut img_data, img_size),
            9 => create_spiral_pattern(&mut img_data, img_size),
            _ => {}
        }

        // Add noise
        for pixel in img_data.iter_mut() {
            *pixel += rng.gen::<f32>() * 0.2 - 0.1;
            *pixel = pixel.clamp(0.0, 1.0);
        }

        let img_tensor = Tensor::from_slice_on_device(&img_data, &[1, 784], device.clone())?;

        let mut one_hot = vec![0.0f32; 10];
        one_hot[label] = 1.0;
        let label_tensor = Tensor::from_slice_on_device(&one_hot, &[1, 10], device.clone())?;

        test_images.push(img_tensor);
        test_labels.push(label_tensor);
    }

    println!(
        "Generated {} training samples and {} test samples",
        train_size, test_size
    );

    Ok((train_images, train_labels, test_images, test_labels))
}

// Pattern generation functions for different digits
fn create_circle_pattern(data: &mut [f32], size: usize) {
    let center = size / 2;
    let radius = size / 3;

    for y in 0..size {
        for x in 0..size {
            let dx = x as i32 - center as i32;
            let dy = y as i32 - center as i32;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();

            if (dist - radius as f32).abs() < 2.0 {
                data[y * size + x] = 0.8;
            }
        }
    }
}

fn create_vertical_line_pattern(data: &mut [f32], size: usize) {
    let center_x = size / 2;

    for y in 4..size - 4 {
        for x in (center_x - 1)..=(center_x + 1) {
            if x < size {
                data[y * size + x] = 0.8;
            }
        }
    }
}

fn create_horizontal_lines_pattern(data: &mut [f32], size: usize) {
    let y1 = size / 3;
    let y2 = 2 * size / 3;

    for y in [y1, y2] {
        for x in 4..size - 4 {
            data[y * size + x] = 0.8;
        }
    }
}

fn create_curves_pattern(data: &mut [f32], size: usize) {
    for y in 0..size {
        let x = (size as f32 * 0.3
            + size as f32 * 0.4 * (y as f32 / size as f32 * std::f32::consts::PI).sin())
            as usize;
        if x < size {
            data[y * size + x] = 0.8;
            if x + 1 < size {
                data[y * size + x + 1] = 0.8;
            }
        }
    }
}

fn create_angle_pattern(data: &mut [f32], size: usize) {
    let mid_y = size / 2;
    let mid_x = size / 2;

    // Vertical line
    for y in 0..mid_y {
        data[y * size + 4] = 0.8;
    }

    // Horizontal line
    for x in 4..size - 4 {
        data[mid_y * size + x] = 0.8;
    }
}

fn create_s_pattern(data: &mut [f32], size: usize) {
    for y in 0..size {
        let progress = y as f32 / size as f32;
        let x = (size as f32 * 0.5
            + size as f32 * 0.3 * (progress * 2.0 * std::f32::consts::PI).sin())
            as usize;
        if x < size {
            data[y * size + x] = 0.8;
        }
    }
}

fn create_loop_pattern(data: &mut [f32], size: usize) {
    create_circle_pattern(data, size);

    // Add a vertical line on the left
    for y in 8..size - 8 {
        data[y * size + 6] = 0.8;
    }
}

fn create_diagonal_pattern(data: &mut [f32], size: usize) {
    for i in 4..size - 4 {
        let x = i;
        let y = size - 1 - i;
        if x < size && y < size {
            data[y * size + x] = 0.8;
            if x + 1 < size {
                data[y * size + x + 1] = 0.8;
            }
        }
    }
}

fn create_double_circle_pattern(data: &mut [f32], size: usize) {
    let center = size / 2;
    let inner_radius = size / 5;
    let outer_radius = size / 3;

    for y in 0..size {
        for x in 0..size {
            let dx = x as i32 - center as i32;
            let dy = y as i32 - center as i32;
            let dist = ((dx * dx + dy * dy) as f32).sqrt();

            if (dist - inner_radius as f32).abs() < 1.5 || (dist - outer_radius as f32).abs() < 1.5
            {
                data[y * size + x] = 0.8;
            }
        }
    }
}

fn create_spiral_pattern(data: &mut [f32], size: usize) {
    let center_x = size / 2;
    let center_y = size / 2;

    for angle_steps in 0..100 {
        let angle = angle_steps as f32 * 0.3;
        let radius = angle * 1.5;

        let x = center_x as f32 + radius * angle.cos();
        let y = center_y as f32 + radius * angle.sin();

        if x >= 0.0 && x < size as f32 && y >= 0.0 && y < size as f32 {
            data[y as usize * size + x as usize] = 0.8;
        }
    }
}

/// Find the index of the maximum value in a slice
fn argmax(data: &[f32]) -> usize {
    data.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(index, _)| index)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_data_loading() -> Result<()> {
        let device = Device::cpu()?;
        let (train_images, train_labels, test_images, test_labels) = load_mnist_data(&device)?;

        assert_eq!(train_images.len(), 1000);
        assert_eq!(train_labels.len(), 1000);
        assert_eq!(test_images.len(), 200);
        assert_eq!(test_labels.len(), 200);

        // Check tensor shapes
        assert_eq!(train_images[0].shape(), &[1, 784]);
        assert_eq!(train_labels[0].shape(), &[1, 10]);

        Ok(())
    }

    #[test]
    fn test_simple_dense_training() -> Result<()> {
        let device = Device::cpu()?;

        // Create minimal network for testing
        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 784,
                output_size: 32,
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 32,
                output_size: 10,
                activation: Activation::Softmax,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::SGD {
                learning_rate: 0.01,
                momentum: None,
                weight_decay: None,
                nesterov: false,
            })
            .device(device.clone())
            .build()?;

        // Create minimal test data
        let test_input = Tensor::from_slice_on_device(&vec![0.5; 784], &[1, 784], device.clone())?;
        let test_label = Tensor::from_slice_on_device(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[1, 10],
            device.clone(),
        )?;

        // Test forward pass
        let output = network.forward(&test_input)?;
        assert_eq!(output.shape(), &[1, 10]);

        // Test training with minimal config
        let config = TrainingConfig {
            epochs: 1,
            batch_size: 1,
            verbose: false,
            early_stopping_patience: 0,
            early_stopping_threshold: 0.001,
            lr_schedule: None,
            validation_split: 0.0,
            shuffle: false,
            random_seed: Some(42),
        };

        network.train(&vec![test_input], &vec![test_label], &config)?;

        Ok(())
    }
}
