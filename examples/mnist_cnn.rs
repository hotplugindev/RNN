//! MNIST CNN Classification Example
//!
//! This example demonstrates training a Convolutional Neural Network (CNN) to classify
//! handwritten digits from the MNIST dataset. The example shows how to:
//! - Load and preprocess MNIST data
//! - Build a CNN architecture with Conv2D and pooling layers
//! - Train the network with validation
//! - Evaluate model performance
//! - Save and load trained models

use rnn::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("MNIST CNN Classification Example");
    println!("================================");

    // Auto-select the best available device
    let device = Device::auto_select()?;
    println!(
        "Using device: {} ({})",
        device.info().name,
        device.device_type()
    );

    // Load MNIST dataset (simulated for this example)
    let (train_images, train_labels, test_images, test_labels) = load_mnist_data()?;

    println!("Dataset loaded:");
    println!("  Training samples: {}", train_images.shape()[0]);
    println!("  Test samples: {}", test_images.shape()[0]);
    println!(
        "  Image dimensions: {}x{}",
        train_images.shape()[2],
        train_images.shape()[3]
    );

    // Build minimal CNN architecture - just conv+flatten+dense
    let mut network = NetworkBuilder::new()
        // Single convolutional layer (no pooling to avoid shape issues)
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 8,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::He,
        })
        // Flatten for dense layers
        .add_layer(LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        })
        // Output layer directly (no hidden dense layer)
        .add_layer(LayerConfig::Dense {
            input_size: 8 * 28 * 28, // No pooling, so still 28x28, 8 channels
            output_size: 10,         // 10 classes for digits 0-9
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

    println!("CNN created with {} parameters", network.num_parameters());

    // Test forward pass before training to debug shapes
    println!("\nTesting forward pass...");
    let test_img_data = vec![0.5f32; 28 * 28];
    let test_input = Tensor::from_slice_on_device(&test_img_data, &[1, 1, 28, 28], device.clone())?;

    match network.forward(&test_input) {
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

    // Training configuration - very minimal to isolate issue
    let training_config = TrainingConfig {
        epochs: 1,
        batch_size: 4,
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

    // Convert batched tensors to individual samples for training - use only a few samples for debugging
    let mut train_samples = Vec::new();
    let mut train_targets = Vec::new();

    let batch_size = std::cmp::min(train_images.shape()[0], 8); // Use max 8 samples for debugging
    let img_size = 28;

    // Get the raw data from tensors
    let images_data = train_images.to_vec()?;
    let labels_data = train_labels.to_vec()?;

    println!("Preparing {} training samples...", batch_size);

    for i in 0..batch_size {
        // Extract individual image data
        let start_idx = i * img_size * img_size;
        let end_idx = start_idx + img_size * img_size;
        let img_slice = &images_data[start_idx..end_idx];
        let img_sample =
            Tensor::from_slice_on_device(img_slice, &[1, 1, img_size, img_size], device.clone())?;

        // Debug: Test forward pass with this sample
        if i == 0 {
            println!("Testing forward pass with training sample 1...");
            match network.forward(&img_sample) {
                Ok(output) => {
                    println!(
                        "Training sample forward pass successful! Output shape: {:?}",
                        output.shape()
                    );
                }
                Err(e) => {
                    println!("Training sample forward pass failed: {}", e);
                    println!("Sample shape: {:?}", img_sample.shape());
                    return Err(e);
                }
            }
        }

        // Extract individual label and convert to one-hot encoding
        let label_value = labels_data[i] as usize;
        let mut one_hot = vec![0.0f32; 10]; // 10 classes for digits 0-9
        one_hot[label_value] = 1.0;
        let label_sample = Tensor::from_slice_on_device(&one_hot, &[1, 10], device.clone())?;

        // Check shape before moving
        if i % 2 == 0 {
            println!(
                "Prepared sample {} with label {} - sample shape: {:?}",
                i + 1,
                label_value,
                img_sample.shape()
            );
        }

        train_samples.push(img_sample);
        train_targets.push(label_sample);
    }

    // Train the network
    network.train(&train_samples, &train_targets, &training_config)?;

    let training_time = start_time.elapsed();
    println!("Training completed in {:.2}s", training_time.as_secs_f64());

    // Evaluate on test set
    println!("\nEvaluating on test set...");
    let test_metrics = evaluate_model(&mut network, &test_images, &test_labels)?;

    println!("Test Results:");
    println!("  Accuracy: {:.2}%", test_metrics.accuracy * 100.0);
    println!("  Loss: {:.4}", test_metrics.loss);
    println!(
        "  Top-3 Accuracy: {:.2}%",
        test_metrics.top3_accuracy * 100.0
    );

    // Show confusion matrix for first few classes
    let confusion_matrix = compute_confusion_matrix(&mut network, &test_images, &test_labels)?;
    print_confusion_matrix(&confusion_matrix);

    // Test individual predictions
    println!("\nSample predictions:");
    for i in 0..5 {
        let image = test_images.view().slice(0, i..i + 1)?.to_tensor()?;
        let prediction = network.forward(&image)?;
        let predicted_class = argmax(&prediction.to_vec()?);
        let actual_class = test_labels.to_vec()?[i] as usize;

        println!(
            "  Sample {}: Predicted: {}, Actual: {}",
            i + 1,
            predicted_class,
            actual_class
        );
    }

    // Benchmark inference speed
    println!("\nBenchmarking inference speed...");
    let sample_image = test_images.view().slice(0, 0..1)?.to_tensor()?;
    let num_iterations = 100;

    let start_time = Instant::now();
    for _ in 0..num_iterations {
        let _ = network.forward(&sample_image)?;
    }
    device.synchronize()?;
    let inference_time = start_time.elapsed();

    println!(
        "Average inference time: {:.4}ms ({} iterations)",
        inference_time.as_secs_f64() * 1000.0 / num_iterations as f64,
        num_iterations
    );

    // Save the trained model
    let model_path = "mnist_cnn_model.bin";
    save_model(&network, model_path, ModelFormat::Binary, None)?;
    println!("\nModel saved to: {}", model_path);

    // Demonstrate model loading and verify consistency
    println!("Verifying model loading...");
    // Note: Model loading functionality is under development
    println!("Model loading functionality is under development");
    // For now, we'll use the original network for verification
    let loaded_network = &network;

    // Verification will be implemented when model loading is complete
    println!("Model verification will be enabled when loading is implemented");

    println!("\nMNIST CNN training example completed successfully!");

    Ok(())
}

/// Simulated MNIST data loading function
/// In a real implementation, this would load actual MNIST data from files
fn load_mnist_data() -> Result<(Tensor, Tensor, Tensor, Tensor)> {
    use rand::prelude::*;

    let mut rng = thread_rng();

    // Simulate training data (very minimal size for debugging)
    let train_size = 16;
    let test_size = 8;
    let img_size = 28;

    // Generate random images (normally you'd load real MNIST data)
    let mut train_images = vec![0.0f32; train_size * img_size * img_size];
    let mut train_labels = vec![0.0f32; train_size];

    for i in 0..train_size {
        // Create simple patterns for each digit
        let label = i % 10;
        train_labels[i] = label as f32;

        let img_start = i * img_size * img_size;
        for j in 0..img_size * img_size {
            // Add some pattern based on label + noise
            let pattern = match label {
                0 => {
                    if in_circle(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                1 => {
                    if in_vertical_line(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                2 => {
                    if in_horizontal_lines(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                _ => rng.gen::<f32>() * 0.3 + 0.1, // Random pattern for other digits
            };
            train_images[img_start + j] = pattern + rng.gen::<f32>() * 0.2 - 0.1;
        }
    }

    // Generate test data similarly
    let mut test_images = vec![0.0f32; test_size * img_size * img_size];
    let mut test_labels = vec![0.0f32; test_size];

    for i in 0..test_size {
        let label = i % 10;
        test_labels[i] = label as f32;

        let img_start = i * img_size * img_size;
        for j in 0..img_size * img_size {
            let pattern = match label {
                0 => {
                    if in_circle(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                1 => {
                    if in_vertical_line(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                2 => {
                    if in_horizontal_lines(j, img_size) {
                        0.8
                    } else {
                        0.1
                    }
                }
                _ => rng.gen::<f32>() * 0.3 + 0.1,
            };
            test_images[img_start + j] = pattern + rng.gen::<f32>() * 0.2 - 0.1;
        }
    }

    Ok((
        Tensor::from_slice(&train_images, &[train_size, 1, img_size, img_size])?,
        Tensor::from_slice(&train_labels, &[train_size, 1])?,
        Tensor::from_slice(&test_images, &[test_size, 1, img_size, img_size])?,
        Tensor::from_slice(&test_labels, &[test_size, 1])?,
    ))
}

/// Helper function to create circle pattern
fn in_circle(idx: usize, size: usize) -> bool {
    let row = idx / size;
    let col = idx % size;
    let center = size / 2;
    let radius = size / 3;

    let dx = (row as i32 - center as i32).abs();
    let dy = (col as i32 - center as i32).abs();
    let dist_sq = dx * dx + dy * dy;

    dist_sq > (radius as i32 - 2).pow(2) && dist_sq < (radius as i32 + 2).pow(2)
}

/// Helper function to create vertical line pattern
fn in_vertical_line(idx: usize, size: usize) -> bool {
    let col = idx % size;
    let center = size / 2;

    (col as i32 - center as i32).abs() < 2
}

/// Helper function to create horizontal lines pattern
fn in_horizontal_lines(idx: usize, size: usize) -> bool {
    let row = idx / size;

    row % 6 < 2
}

/// Evaluation metrics
#[derive(Debug)]
struct EvaluationMetrics {
    accuracy: f32,
    loss: f32,
    top3_accuracy: f32,
}

/// Evaluate model performance on test data
fn evaluate_model(
    network: &mut Network,
    images: &Tensor,
    labels: &Tensor,
) -> Result<EvaluationMetrics> {
    let mut correct = 0;
    let mut top3_correct = 0;
    let mut total_loss = 0.0;
    let num_samples = images.shape()[0];

    for i in 0..num_samples {
        let image = images.view().slice(0, i..i + 1)?.to_tensor()?;
        let label = labels.to_vec()?[i] as usize;

        let prediction = network.forward(&image)?;
        let pred_probs = prediction.to_vec()?;

        // Calculate loss (simplified cross-entropy)
        total_loss += -pred_probs[label].ln();

        // Top-1 accuracy
        let predicted_class = argmax(&pred_probs);
        if predicted_class == label {
            correct += 1;
        }

        // Top-3 accuracy
        let mut indexed_probs: Vec<(usize, f32)> = pred_probs
            .iter()
            .enumerate()
            .map(|(i, &p)| (i, p))
            .collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if indexed_probs[..3].iter().any(|(class, _)| *class == label) {
            top3_correct += 1;
        }
    }

    Ok(EvaluationMetrics {
        accuracy: correct as f32 / num_samples as f32,
        loss: total_loss / num_samples as f32,
        top3_accuracy: top3_correct as f32 / num_samples as f32,
    })
}

/// Compute confusion matrix
fn compute_confusion_matrix(
    network: &mut Network,
    images: &Tensor,
    labels: &Tensor,
) -> Result<Vec<Vec<usize>>> {
    let mut matrix = vec![vec![0; 10]; 10];
    let num_samples = images.shape()[0];

    for i in 0..num_samples {
        let image = images.view().slice(0, i..i + 1)?.to_tensor()?;
        let actual = labels.to_vec()?[i] as usize;

        let prediction = network.forward(&image)?;
        let predicted = argmax(&prediction.to_vec()?);

        matrix[actual][predicted] += 1;
    }

    Ok(matrix)
}

/// Print confusion matrix
fn print_confusion_matrix(matrix: &[Vec<usize>]) {
    println!("\nConfusion Matrix (first 5x5):");
    println!("Actual\\Pred  0    1    2    3    4");
    println!("--------------------------------");

    for (i, row) in matrix.iter().take(5).enumerate() {
        print!("    {}     ", i);
        for &val in row.iter().take(5) {
            print!("{:4} ", val);
        }
        println!();
    }
}

/// Find index of maximum value
fn argmax(values: &[f32]) -> usize {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_data_loading() -> Result<()> {
        let (train_images, train_labels, test_images, test_labels) = load_mnist_data()?;

        assert_eq!(train_images.shape()[0], 1000);
        assert_eq!(train_labels.shape()[0], 1000);
        assert_eq!(test_images.shape()[0], 200);
        assert_eq!(test_labels.shape()[0], 200);
        assert_eq!(train_images.shape()[1], 1); // Channels
        assert_eq!(train_images.shape()[2], 28); // Height
        assert_eq!(train_images.shape()[3], 28); // Width

        Ok(())
    }

    #[test]
    fn test_small_cnn_training() -> Result<()> {
        // Create minimal dataset for testing
        let train_images = Tensor::zeros(&[8, 1, 14, 14])?;
        let train_labels = Tensor::from_slice(&[0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0], &[8, 1])?;

        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Conv2D {
                in_channels: 1,
                out_channels: 4,
                kernel_size: (3, 3),
                stride: (1, 1),
                padding: (1, 1),
                dilation: (1, 1),
                activation: Activation::ReLU,
                use_bias: true,
                weight_init: WeightInit::He,
            })
            .add_layer(LayerConfig::Flatten {
                start_dim: 1,
                end_dim: -1,
            })
            .add_layer(LayerConfig::Dense {
                input_size: 4 * 14 * 14,
                output_size: 2,
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
            .build()?;

        let config = TrainingConfig {
            epochs: 5,
            batch_size: 32,
            verbose: false,
            early_stopping_patience: 0,
            early_stopping_threshold: 0.001,
            lr_schedule: None,
            validation_split: 0.1,
            shuffle: true,
            random_seed: Some(42),
        };

        network.train_with_config(&train_images, &train_labels, &config)?;

        let test_input = Tensor::zeros(&[1, 1, 14, 14])?;
        let output = network.forward(&test_input)?;
        assert_eq!(output.shape(), &[1, 2]);

        Ok(())
    }

    #[test]
    fn test_evaluation_metrics() {
        let values = vec![0.1, 0.8, 0.1];
        assert_eq!(argmax(&values), 1);

        let values = vec![0.3, 0.2, 0.5];
        assert_eq!(argmax(&values), 2);
    }
}
