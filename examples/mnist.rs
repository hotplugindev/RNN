//! MNIST Training Example
//!
//! This example demonstrates training neural networks on the MNIST dataset
//! using both dense and convolutional approaches. It includes a custom MNIST
//! data loader for the IDX format files.

use rnn::prelude::*;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// MNIST dataset structure
#[derive(Debug)]
pub struct MnistDataset {
    pub train_images: Vec<Vec<f32>>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<Vec<f32>>,
    pub test_labels: Vec<u8>,
    pub image_size: (usize, usize),
}

impl MnistDataset {
    /// Load MNIST dataset from IDX files
    pub fn load(data_dir: &str) -> Result<Self> {
        println!("Loading MNIST dataset from: {}", data_dir);

        // Load training data
        let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        let (train_images, image_size) = Self::load_images(&train_images_path)?;
        let train_labels = Self::load_labels(&train_labels_path)?;
        let (test_images, _) = Self::load_images(&test_images_path)?;
        let test_labels = Self::load_labels(&test_labels_path)?;

        println!(
            "Loaded {} training samples and {} test samples",
            train_images.len(),
            test_images.len()
        );
        println!("Image size: {}x{}", image_size.0, image_size.1);

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
            image_size,
        })
    }

    /// Load images from IDX3 format
    fn load_images(path: &str) -> Result<(Vec<Vec<f32>>, (usize, usize))> {
        let file = File::open(path).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to open {}: {}", path, e),
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read magic number: {}", e),
            ))
        })?;
        let magic = u32::from_be_bytes(buffer);

        if magic != 0x00000803 {
            return Err(RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid magic number for images: expected 2051, got {}",
                    magic
                ),
            )));
        }

        // Read number of images
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read number of images: {}", e),
            ))
        })?;
        let num_images = u32::from_be_bytes(buffer) as usize;

        // Read number of rows
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read number of rows: {}", e),
            ))
        })?;
        let num_rows = u32::from_be_bytes(buffer) as usize;

        // Read number of columns
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read number of columns: {}", e),
            ))
        })?;
        let num_cols = u32::from_be_bytes(buffer) as usize;

        println!(
            "Reading {} images of size {}x{}",
            num_images, num_rows, num_cols
        );

        // Read image data
        let mut images = Vec::with_capacity(num_images);
        let image_size = num_rows * num_cols;

        for _ in 0..num_images {
            let mut image_data = vec![0u8; image_size];
            reader.read_exact(&mut image_data).map_err(|e| {
                RnnError::io(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    format!("Failed to read image data: {}", e),
                ))
            })?;

            // Convert to f32 and normalize to [0, 1]
            let image: Vec<f32> = image_data
                .into_iter()
                .map(|pixel| pixel as f32 / 255.0)
                .collect();
            images.push(image);
        }

        Ok((images, (num_rows, num_cols)))
    }

    /// Load labels from IDX1 format
    fn load_labels(path: &str) -> Result<Vec<u8>> {
        let file = File::open(path).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Failed to open {}: {}", path, e),
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut buffer = [0u8; 4];
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read magic number: {}", e),
            ))
        })?;
        let magic = u32::from_be_bytes(buffer);

        if magic != 0x00000801 {
            return Err(RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "Invalid magic number for labels: expected 2049, got {}",
                    magic
                ),
            )));
        }

        // Read number of labels
        reader.read_exact(&mut buffer).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read number of labels: {}", e),
            ))
        })?;
        let num_labels = u32::from_be_bytes(buffer) as usize;

        println!("Reading {} labels", num_labels);

        // Read label data
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("Failed to read label data: {}", e),
            ))
        })?;

        Ok(labels)
    }

    /// Convert to tensor format for training
    pub fn to_tensors(
        &self,
        device: &Device,
        batch_size: usize,
        use_cnn_format: bool,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
        println!(
            "Converting data to tensors (batch_size: {}, CNN format: {})",
            batch_size, use_cnn_format
        );

        let train_input_tensors =
            self.images_to_tensors(&self.train_images, device, batch_size, use_cnn_format)?;
        let train_target_tensors =
            self.labels_to_tensors(&self.train_labels, device, batch_size)?;
        let test_input_tensors =
            self.images_to_tensors(&self.test_images, device, batch_size, use_cnn_format)?;
        let test_target_tensors = self.labels_to_tensors(&self.test_labels, device, batch_size)?;

        println!(
            "Created {} train samples and {} test samples",
            train_input_tensors.len(),
            test_input_tensors.len()
        );

        Ok((
            train_input_tensors,
            train_target_tensors,
            test_input_tensors,
            test_target_tensors,
        ))
    }

    fn images_to_tensors(
        &self,
        images: &[Vec<f32>],
        device: &Device,
        _batch_size: usize,
        use_cnn_format: bool,
    ) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();

        // Create individual sample tensors instead of batched tensors
        for image in images {
            let tensor = if use_cnn_format {
                // Shape: [1, channels, height, width] - single sample
                let shape = vec![1, 1, self.image_size.0, self.image_size.1];
                Tensor::from_slice_on_device(image, &shape, device.clone())?
            } else {
                // Shape: [1, features] - single sample
                let shape = vec![1, 784];
                Tensor::from_slice_on_device(image, &shape, device.clone())?
            };

            tensors.push(tensor);
        }

        println!("Created {} individual sample tensors", tensors.len());

        Ok(tensors)
    }

    fn labels_to_tensors(
        &self,
        labels: &[u8],
        device: &Device,
        _batch_size: usize,
    ) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();

        // Create individual sample tensors with one-hot encoding
        for &label in labels {
            let mut one_hot = vec![0.0f32; 10];
            one_hot[label as usize] = 1.0;

            let shape = vec![1, 10]; // Single sample
            let tensor = Tensor::from_slice_on_device(&one_hot, &shape, device.clone())?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

/// Train a dense neural network on MNIST
fn train_dense_network(
    dataset: &MnistDataset,
    device: &Device,
    epochs: usize,
    batch_size: usize,
) -> Result<Network> {
    println!("\n=== Training Dense Neural Network ===");
    println!(
        "Dataset info: {} train images, {} test images",
        dataset.train_images.len(),
        dataset.test_images.len()
    );

    // Convert data to tensors
    let (train_inputs, train_targets, test_inputs, test_targets) =
        dataset.to_tensors(device, batch_size, false)?;

    println!("First batch shapes:");
    if !train_inputs.is_empty() {
        println!("  Input: {:?}", train_inputs[0].shape());
        println!("  Target: {:?}", train_targets[0].shape());

        // Debug: Check actual tensor data
        let input_data = train_inputs[0].to_vec()?;
        println!("  Input tensor size: {}", input_data.len());
        println!("  Expected input size for dense layer: 784");

        if train_inputs[0].shape().len() > 0 {
            let batch_dim = train_inputs[0].shape()[0];
            let feature_dim = if train_inputs[0].shape().len() > 1 {
                train_inputs[0].shape()[1]
            } else {
                1
            };
            println!(
                "  Interpreted as: batch_size={}, features={}",
                batch_dim, feature_dim
            );
        }
    }

    // Build network using preset
    let mut network = rnn::network::builder::presets::mnist_classifier()
        .device(device.clone())
        .build()?;

    println!("Network summary:");
    println!("  Input expected: [batch_size, 784]");
    println!("  Output: [batch_size, 10]");

    println!(
        "Dense network created with {} parameters",
        network.num_parameters()
    );

    // Training configuration - match the actual tensor batch size
    let training_config = TrainingConfig {
        epochs,
        batch_size,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_threshold: 0.001,
        lr_schedule: Some(LearningRateSchedule::StepLR {
            step_size: 10,
            gamma: 0.9,
        }),
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("Starting dense network training...");
    println!("Training with {} samples", train_inputs.len());
    println!("TrainingConfig batch_size: {}", training_config.batch_size);
    println!(
        "Individual tensor shape: {:?}",
        if !train_inputs.is_empty() {
            Some(train_inputs[0].shape())
        } else {
            None
        }
    );

    // Test forward pass before training
    if !train_inputs.is_empty() {
        println!("Testing forward pass with first batch...");
        let test_output = network.forward(&train_inputs[0])?;
        println!(
            "Forward pass successful! Output shape: {:?}",
            test_output.shape()
        );
    }

    let start_time = Instant::now();

    let history = network.train(&train_inputs, &train_targets, &training_config)?;

    let training_time = start_time.elapsed();
    println!(
        "Dense training completed in {:.2}s",
        training_time.as_secs_f64()
    );
    println!("Final training loss: {:.6}", history.final_loss());

    // Evaluate on test set
    evaluate_network(&mut network, &test_inputs, &test_targets, "Dense")?;

    Ok(network)
}

/// Train a CNN on MNIST
fn train_cnn_network(
    dataset: &MnistDataset,
    device: &Device,
    epochs: usize,
    batch_size: usize,
) -> Result<Network> {
    println!("\n=== Training Convolutional Neural Network ===");

    // Convert data to tensors
    let (train_inputs, train_targets, test_inputs, test_targets) =
        dataset.to_tensors(device, batch_size, true)?;

    // Build CNN network
    let mut network = NetworkBuilder::new()
        // First convolutional block
        .add_layer(LayerConfig::Conv2D {
            in_channels: 1,
            out_channels: 32,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 32,
            out_channels: 32,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::MaxPool2D {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.25 })
        // Second convolutional block
        .add_layer(LayerConfig::Conv2D {
            in_channels: 32,
            out_channels: 64,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Conv2D {
            in_channels: 64,
            out_channels: 64,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            dilation: (1, 1),
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::MaxPool2D {
            kernel_size: (2, 2),
            stride: Some((2, 2)),
            padding: (0, 0),
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.25 })
        // Flatten and dense layers
        .add_layer(LayerConfig::Flatten {
            start_dim: 1,
            end_dim: None,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 64 * 7 * 7, // 64 channels * 7x7 after two 2x2 pooling layers
            output_size: 512,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dropout { dropout_rate: 0.5 })
        .add_layer(LayerConfig::Dense {
            input_size: 512,
            output_size: 10,
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
            weight_decay: Some(1e-4),
            amsgrad: false,
        })
        .device(device.clone())
        .build()?;

    println!(
        "CNN network created with {} parameters",
        network.num_parameters()
    );

    // Training configuration - match the actual tensor batch size
    let training_config = TrainingConfig {
        epochs,
        batch_size,
        verbose: true,
        early_stopping_patience: 5,
        early_stopping_threshold: 0.001,
        lr_schedule: Some(LearningRateSchedule::StepLR {
            step_size: 5,
            gamma: 0.9,
        }),
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("Starting CNN training...");
    println!("Training with {} samples", train_inputs.len());
    println!("TrainingConfig batch_size: {}", training_config.batch_size);
    println!(
        "Individual tensor shape: {:?}",
        if !train_inputs.is_empty() {
            Some(train_inputs[0].shape())
        } else {
            None
        }
    );
    let start_time = Instant::now();

    let history = network.train(&train_inputs, &train_targets, &training_config)?;

    let training_time = start_time.elapsed();
    println!(
        "CNN training completed in {:.2}s",
        training_time.as_secs_f64()
    );
    println!("Final training loss: {:.6}", history.final_loss());

    // Evaluate on test set
    evaluate_network(&mut network, &test_inputs, &test_targets, "CNN")?;

    Ok(network)
}

/// Evaluate network performance
fn evaluate_network(
    network: &mut Network,
    test_inputs: &[Tensor],
    test_targets: &[Tensor],
    network_type: &str,
) -> Result<()> {
    println!("\nEvaluating {} network on test set...", network_type);

    let mut total_samples = 0;
    let mut correct_predictions = 0;
    let mut total_loss = 0.0;

    network.set_training(false);

    for (input_batch, target_batch) in test_inputs.iter().zip(test_targets.iter()) {
        let predictions = network.forward(input_batch)?;

        // Calculate accuracy
        let pred_data = predictions.to_vec()?;
        let target_data = target_batch.to_vec()?;

        let batch_size = target_batch.shape()[0];
        let num_classes = target_batch.shape()[1];

        for i in 0..batch_size {
            let pred_slice = &pred_data[i * num_classes..(i + 1) * num_classes];
            let target_slice = &target_data[i * num_classes..(i + 1) * num_classes];

            let predicted_class = pred_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let actual_class = target_slice
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            if predicted_class == actual_class {
                correct_predictions += 1;
            }
            total_samples += 1;
        }

        // Calculate loss (simplified cross-entropy)
        let mut batch_loss = 0.0;
        for i in 0..batch_size {
            for j in 0..num_classes {
                let pred_idx = i * num_classes + j;
                let target_val = target_data[pred_idx];
                let pred_val = pred_data[pred_idx].max(1e-15); // Prevent log(0)

                if target_val > 0.0 {
                    batch_loss -= target_val * pred_val.ln();
                }
            }
        }
        batch_loss /= batch_size as f32;
        total_loss += batch_loss;
    }

    let accuracy = correct_predictions as f64 / total_samples as f64;
    let avg_loss = total_loss / test_inputs.len() as f32;

    println!("{} Test Results:", network_type);
    println!(
        "  Accuracy: {:.2}% ({}/{})",
        accuracy * 100.0,
        correct_predictions,
        total_samples
    );
    println!("  Average Loss: {:.6}", avg_loss);
    println!("  Total samples evaluated: {}", total_samples);

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("MNIST Neural Network Training Example");
    println!("=====================================");

    // Setup device
    let device = Device::cpu()?;
    println!("Using device: {:?}", device.device_type());

    // Load MNIST dataset
    let data_dir = "examples/data";

    if !Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)).exists() {
        eprintln!("Error: MNIST data files not found in {}", data_dir);
        eprintln!("Please download MNIST data files:");
        eprintln!("  - train-images-idx3-ubyte");
        eprintln!("  - train-labels-idx1-ubyte");
        eprintln!("  - t10k-images-idx3-ubyte");
        eprintln!("  - t10k-labels-idx1-ubyte");
        eprintln!("From: http://yann.lecun.com/exdb/mnist/");
        return Err(RnnError::io(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "MNIST data files not found",
        )));
    }

    let dataset = MnistDataset::load(data_dir)?;

    // Training parameters
    let epochs = 2; // Reduced for testing
    let batch_size = 32; // Batch size for training (library will group individual samples)

    // Train dense network
    let dense_network = train_dense_network(&dataset, &device, epochs, batch_size)?;

    // Train CNN
    let cnn_network = train_cnn_network(&dataset, &device, epochs, batch_size)?;

    // Save trained models
    println!("\nSaving trained models...");

    rnn::io::save_model(
        &dense_network,
        "mnist_dense_model.bin",
        ModelFormat::Binary,
        Some(ModelMetadata {
            name: "MNIST Dense Classifier".to_string(),
            description: "Dense neural network trained on MNIST".to_string(),
            created_at: chrono::Utc::now().to_string(),
            modified_at: chrono::Utc::now().to_string(),
            training_info: TrainingInfo {
                epochs_trained: epochs,
                final_loss: 0.0, // Would need to get from history
                best_accuracy: 0.0,
                training_time_seconds: 0.0,
                dataset_info: Some(DatasetInfo {
                    name: "MNIST".to_string(),
                    train_samples: dataset.train_images.len(),
                    val_samples: Some(0),
                    test_samples: Some(dataset.test_images.len()),
                    num_classes: Some(10),
                }),
            },
            metrics: HashMap::new(),
            custom: HashMap::new(),
        }),
    )?;

    rnn::io::save_model(
        &cnn_network,
        "mnist_cnn_model.bin",
        ModelFormat::Binary,
        Some(ModelMetadata {
            name: "MNIST CNN Classifier".to_string(),
            description: "Convolutional neural network trained on MNIST".to_string(),
            created_at: chrono::Utc::now().to_string(),
            modified_at: chrono::Utc::now().to_string(),
            training_info: TrainingInfo {
                epochs_trained: epochs,
                final_loss: 0.0,
                best_accuracy: 0.0,
                training_time_seconds: 0.0,
                dataset_info: Some(DatasetInfo {
                    name: "MNIST".to_string(),
                    train_samples: dataset.train_images.len(),
                    val_samples: Some(0),
                    test_samples: Some(dataset.test_images.len()),
                    num_classes: Some(10),
                }),
            },
            metrics: HashMap::new(),
            custom: HashMap::new(),
        }),
    )?;

    println!("Models saved as mnist_dense_model.bin and mnist_cnn_model.bin");
    println!("\nMNIST training example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mnist_data_loading() -> Result<()> {
        // This test requires MNIST data files to be present
        if !Path::new("examples/data/train-images-idx3-ubyte").exists() {
            println!("Skipping MNIST test - data files not found");
            return Ok(());
        }

        let dataset = MnistDataset::load("examples/data")?;

        assert_eq!(dataset.train_images.len(), 60000);
        assert_eq!(dataset.test_images.len(), 10000);
        assert_eq!(dataset.train_labels.len(), 60000);
        assert_eq!(dataset.test_labels.len(), 10000);
        assert_eq!(dataset.image_size, (28, 28));

        // Check first image is valid
        assert_eq!(dataset.train_images[0].len(), 784);
        assert!(dataset.train_images[0]
            .iter()
            .all(|&x| x >= 0.0 && x <= 1.0));

        // Check first label is valid
        assert!(dataset.train_labels[0] < 10);

        Ok(())
    }

    #[test]
    fn test_tensor_conversion() -> Result<()> {
        if !Path::new("examples/data/train-images-idx3-ubyte").exists() {
            println!("Skipping tensor conversion test - data files not found");
            return Ok(());
        }

        let dataset = MnistDataset::load("examples/data")?;
        let device = Device::cpu()?;

        let (train_inputs, train_targets, _, _) = dataset.to_tensors(&device, 32, true)?;

        assert!(!train_inputs.is_empty());
        assert!(!train_targets.is_empty());
        assert_eq!(train_inputs.len(), train_targets.len());

        // Check shapes
        let first_input = &train_inputs[0];
        let first_target = &train_targets[0];

        assert_eq!(first_input.shape()[1], 1); // channels
        assert_eq!(first_input.shape()[2], 28); // height
        assert_eq!(first_input.shape()[3], 28); // width
        assert_eq!(first_target.shape()[1], 10); // num classes

        Ok(())
    }
}
