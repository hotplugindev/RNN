//! Small MNIST Training Example
//!
//! This example demonstrates training neural networks on a subset of the MNIST dataset
//! using only 600 training images and 10 test images for faster debugging.

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
    /// Load MNIST dataset from IDX files with size limits
    pub fn load(data_dir: &str, max_train: usize, max_test: usize) -> Result<Self> {
        println!("Loading MNIST dataset from: {}", data_dir);
        println!(
            "Limiting to {} train and {} test samples",
            max_train, max_test
        );

        // Load training data
        let train_images_path = format!("{}/train-images-idx3-ubyte", data_dir);
        let train_labels_path = format!("{}/train-labels-idx1-ubyte", data_dir);
        let test_images_path = format!("{}/t10k-images-idx3-ubyte", data_dir);
        let test_labels_path = format!("{}/t10k-labels-idx1-ubyte", data_dir);

        let (mut train_images, image_size) = Self::load_images(&train_images_path)?;
        let mut train_labels = Self::load_labels(&train_labels_path)?;
        let (mut test_images, _) = Self::load_images(&test_images_path)?;
        let mut test_labels = Self::load_labels(&test_labels_path)?;

        // Limit dataset size
        train_images.truncate(max_train);
        train_labels.truncate(max_train);
        test_images.truncate(max_test);
        test_labels.truncate(max_test);

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

    /// Load images from IDX3 format file
    fn load_images(path: &str) -> Result<(Vec<Vec<f32>>, (usize, usize))> {
        let file = File::open(path).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Could not open image file {}: {}", path, e),
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic_number = u32::from_be_bytes(magic);
        if magic_number != 2051 {
            return Err(RnnError::invalid_input(format!(
                "Invalid magic number for images: expected 2051, got {}",
                magic_number
            )));
        }

        // Read dimensions
        let mut num_images_bytes = [0u8; 4];
        reader.read_exact(&mut num_images_bytes)?;
        let num_images = u32::from_be_bytes(num_images_bytes) as usize;

        let mut num_rows_bytes = [0u8; 4];
        reader.read_exact(&mut num_rows_bytes)?;
        let num_rows = u32::from_be_bytes(num_rows_bytes) as usize;

        let mut num_cols_bytes = [0u8; 4];
        reader.read_exact(&mut num_cols_bytes)?;
        let num_cols = u32::from_be_bytes(num_cols_bytes) as usize;

        let image_size = num_rows * num_cols;
        let mut images = Vec::with_capacity(num_images);

        println!(
            "Reading {} images of size {}x{} from {}",
            num_images, num_rows, num_cols, path
        );

        // Read pixel data
        for i in 0..num_images {
            if i % 10000 == 0 {
                println!("Loading image {}/{}", i, num_images);
            }

            let mut pixels = vec![0u8; image_size];
            reader.read_exact(&mut pixels)?;

            // Convert to f32 and normalize to [0, 1]
            let normalized_pixels: Vec<f32> = pixels
                .into_iter()
                .map(|pixel| pixel as f32 / 255.0)
                .collect();

            images.push(normalized_pixels);
        }

        Ok((images, (num_rows, num_cols)))
    }

    /// Load labels from IDX1 format file
    fn load_labels(path: &str) -> Result<Vec<u8>> {
        let file = File::open(path).map_err(|e| {
            RnnError::io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Could not open label file {}: {}", path, e),
            ))
        })?;
        let mut reader = BufReader::new(file);

        // Read magic number
        let mut magic = [0u8; 4];
        reader.read_exact(&mut magic)?;
        let magic_number = u32::from_be_bytes(magic);
        if magic_number != 2049 {
            return Err(RnnError::invalid_input(format!(
                "Invalid magic number for labels: expected 2049, got {}",
                magic_number
            )));
        }

        // Read number of labels
        let mut num_labels_bytes = [0u8; 4];
        reader.read_exact(&mut num_labels_bytes)?;
        let num_labels = u32::from_be_bytes(num_labels_bytes) as usize;

        println!("Reading {} labels from {}", num_labels, path);

        // Read label data
        let mut labels = vec![0u8; num_labels];
        reader.read_exact(&mut labels)?;

        // Validate labels are in range [0, 9]
        for &label in &labels {
            if label > 9 {
                return Err(RnnError::invalid_input(format!(
                    "Invalid label: expected 0-9, got {}",
                    label
                )));
            }
        }

        Ok(labels)
    }

    /// Convert dataset to tensors for training
    pub fn to_tensors(
        &self,
        device: &Device,
        batch_size: usize,
        use_cnn_format: bool,
    ) -> Result<(Vec<Tensor>, Vec<Tensor>, Vec<Tensor>, Vec<Tensor>)> {
        println!("Converting data to tensors...");
        println!("Batch size: {}, CNN format: {}", batch_size, use_cnn_format);

        let train_inputs = self.images_to_tensors(
            &self.train_images,
            device,
            batch_size,
            use_cnn_format,
            "train inputs",
        )?;
        let train_targets = self.labels_to_tensors(&self.train_labels, device, batch_size)?;

        let test_inputs = self.images_to_tensors(
            &self.test_images,
            device,
            batch_size,
            use_cnn_format,
            "test inputs",
        )?;
        let test_targets = self.labels_to_tensors(&self.test_labels, device, batch_size)?;

        println!("Data conversion complete");
        println!(
            "Training samples: {}, Test samples: {}",
            train_inputs.len(),
            test_inputs.len()
        );

        Ok((train_inputs, train_targets, test_inputs, test_targets))
    }

    fn images_to_tensors(
        &self,
        images: &[Vec<f32>],
        device: &Device,
        _batch_size: usize,
        use_cnn_format: bool,
        name: &str,
    ) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();
        let total_samples = images.len();

        println!("Processing {} images for {}", total_samples, name);

        for image in images {
            let tensor = if use_cnn_format {
                // CNN format: [1, channels, height, width]
                let shape = vec![1, 1, self.image_size.0, self.image_size.1];
                Tensor::from_slice_on_device(image, &shape, device.clone())?
            } else {
                // Dense format: [1, features]
                let features = self.image_size.0 * self.image_size.1;
                let shape = vec![1, features];
                Tensor::from_slice_on_device(image, &shape, device.clone())?
            };

            tensors.push(tensor);
        }

        Ok(tensors)
    }

    fn labels_to_tensors(
        &self,
        labels: &[u8],
        device: &Device,
        _batch_size: usize,
    ) -> Result<Vec<Tensor>> {
        let mut tensors = Vec::new();

        for &label in labels {
            // One-hot encoding: [1, 10]
            let mut data = vec![0.0f32; 10];
            data[label as usize] = 1.0;

            let tensor = Tensor::from_slice_on_device(&data, &[1, 10], device.clone())?;
            tensors.push(tensor);
        }

        Ok(tensors)
    }
}

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

    println!("First sample shapes:");
    if !train_inputs.is_empty() {
        println!("  Input: {:?}", train_inputs[0].shape());
        println!("  Target: {:?}", train_targets[0].shape());
    }

    // Build network using preset
    let mut network = rnn::network::builder::presets::mnist_classifier()
        .device(device.clone())
        .build()?;

    println!(
        "Dense network created with {} parameters",
        network.num_parameters()
    );

    // Training configuration
    let training_config = TrainingConfig {
        epochs,
        batch_size,
        verbose: true,
        early_stopping_patience: 0, // Disable early stopping for small dataset
        early_stopping_threshold: 0.001,
        lr_schedule: None, // Disable learning rate scheduling
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("Starting dense network training...");
    println!("Training with {} samples", train_inputs.len());

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

fn evaluate_network(
    network: &mut Network,
    test_inputs: &[Tensor],
    test_targets: &[Tensor],
    network_type: &str,
) -> Result<()> {
    println!("\n=== Evaluating {} Network ===", network_type);

    let start_time = Instant::now();
    let metrics = network.evaluate(test_inputs, test_targets)?;
    let eval_time = start_time.elapsed();

    println!("Evaluation Results:");
    println!("  Test Loss: {:.6}", metrics.loss);
    println!(
        "  Test Accuracy: {:.4} ({:.2}%)",
        metrics.accuracy,
        metrics.accuracy * 100.0
    );
    println!("  Evaluation Time: {:.2}ms", eval_time.as_millis());

    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();

    println!("Small MNIST Neural Network Training Example");
    println!("==========================================");

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

    // Load only a small subset
    let dataset = MnistDataset::load(data_dir, 600, 10)?;

    // Training parameters
    let epochs = 2;
    let batch_size = 32;

    // Train dense network only
    let _dense_network = train_dense_network(&dataset, &device, epochs, batch_size)?;

    println!("\nSmall MNIST training example completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_small_mnist_data_loading() -> Result<()> {
        let data_dir = "examples/data";

        // Skip test if data files don't exist
        if !Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)).exists() {
            println!("Skipping test: MNIST data files not found");
            return Ok(());
        }

        let dataset = MnistDataset::load(data_dir, 10, 5)?;

        assert_eq!(dataset.train_images.len(), 10);
        assert_eq!(dataset.train_labels.len(), 10);
        assert_eq!(dataset.test_images.len(), 5);
        assert_eq!(dataset.test_labels.len(), 5);
        assert_eq!(dataset.image_size, (28, 28));

        // Check that images are normalized
        for image in &dataset.train_images {
            for &pixel in image {
                assert!(pixel >= 0.0 && pixel <= 1.0);
            }
        }

        Ok(())
    }

    #[test]
    fn test_small_tensor_conversion() -> Result<()> {
        let data_dir = "examples/data";

        // Skip test if data files don't exist
        if !Path::new(&format!("{}/train-images-idx3-ubyte", data_dir)).exists() {
            println!("Skipping test: MNIST data files not found");
            return Ok(());
        }

        let dataset = MnistDataset::load(data_dir, 8, 4)?;
        let device = Device::cpu()?;
        let batch_size = 4;

        let (train_inputs, train_targets, test_inputs, test_targets) =
            dataset.to_tensors(&device, batch_size, false)?;

        // Should have 2 training batches and 1 test batch
        assert_eq!(train_inputs.len(), 2);
        assert_eq!(train_targets.len(), 2);
        assert_eq!(test_inputs.len(), 1);
        assert_eq!(test_targets.len(), 1);

        // Check shapes
        assert_eq!(train_inputs[0].shape(), &[4, 784]);
        assert_eq!(train_targets[0].shape(), &[4, 10]);

        Ok(())
    }
}
