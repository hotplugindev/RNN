//! MNIST Training Example with GPU Acceleration
//!
//! This example demonstrates training a neural network on the MNIST dataset
//! using the RNN library's GPU acceleration capabilities. It includes:
//! - MNIST dataset loading and preprocessing
//! - Network architecture design for image classification
//! - GPU-accelerated training with performance monitoring
//! - Model evaluation and prediction visualization
//! - Comparison between different devices (GPU vs CPU)

use ndarray::{Array1, Array2, Array3, Axis};
use rnn::{
    ActivationFunction, GpuDeviceType, GpuManager, LayerBuilder, LossFunction, Network, Result,
    TrainingConfig,
};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

/// MNIST dataset structure
#[derive(Debug)]
pub struct MnistDataset {
    pub train_images: Array3<f64>,
    pub train_labels: Array1<u8>,
    pub test_images: Array3<f64>,
    pub test_labels: Array1<u8>,
}

impl MnistDataset {
    /// Load MNIST dataset from IDX files
    pub fn load() -> Result<Self> {
        println!("📁 Loading MNIST dataset...");

        let train_images = load_mnist_images("examples/data/train-images-idx3-ubyte")?;
        let train_labels = load_mnist_labels("examples/data/train-labels-idx1-ubyte")?;
        let test_images = load_mnist_images("examples/data/t10k-images-idx3-ubyte")?;
        let test_labels = load_mnist_labels("examples/data/t10k-labels-idx1-ubyte")?;

        println!(
            "✅ Dataset loaded: {} training samples, {} test samples",
            train_images.len_of(Axis(0)),
            test_images.len_of(Axis(0))
        );

        Ok(MnistDataset {
            train_images,
            train_labels,
            test_images,
            test_labels,
        })
    }

    /// Convert images to flattened format for neural network input
    pub fn flatten_images(&self) -> (Array2<f64>, Array2<f64>) {
        let train_flat = self
            .train_images
            .clone()
            .into_shape((self.train_images.len_of(Axis(0)), 784))
            .expect("Failed to flatten training images")
            .to_owned();

        let test_flat = self
            .test_images
            .clone()
            .into_shape((self.test_images.len_of(Axis(0)), 784))
            .expect("Failed to flatten test images")
            .to_owned();

        (train_flat, test_flat)
    }

    /// Convert labels to one-hot encoded format
    pub fn one_hot_labels(&self) -> (Array2<f64>, Array2<f64>) {
        let train_one_hot = labels_to_one_hot(&self.train_labels);
        let test_one_hot = labels_to_one_hot(&self.test_labels);
        (train_one_hot, test_one_hot)
    }

    /// Get a subset of the data for quick testing
    pub fn subset(&self, n_samples: usize) -> MnistDataset {
        let n_train = n_samples.min(self.train_images.len_of(Axis(0)));
        let n_test = (n_samples / 10).min(self.test_images.len_of(Axis(0)));

        MnistDataset {
            train_images: self
                .train_images
                .slice(ndarray::s![0..n_train, .., ..])
                .to_owned(),
            train_labels: self.train_labels.slice(ndarray::s![0..n_train]).to_owned(),
            test_images: self
                .test_images
                .slice(ndarray::s![0..n_test, .., ..])
                .to_owned(),
            test_labels: self.test_labels.slice(ndarray::s![0..n_test]).to_owned(),
        }
    }
}

fn main() -> Result<()> {
    println!("🎯 MNIST Training with GPU Acceleration");
    println!("========================================\n");

    // Initialize GPU manager
    let mut gpu_manager = GpuManager::new();
    display_gpu_info(&gpu_manager);

    // Load MNIST dataset
    let dataset = MnistDataset::load()?;

    // Show dataset statistics
    show_dataset_info(&dataset);

    // Quick demo with subset
    println!("\n🚀 Quick Demo (1000 samples):");
    let subset = dataset.subset(1000);
    demo_training(&mut gpu_manager, &subset)?;

    // Full training comparison
    println!("\n🏋️ Full Training Comparison:");
    if prompt_full_training() {
        compare_training_performance(&mut gpu_manager, &dataset)?;
    }

    // Interactive prediction demo
    println!("\n🔮 Prediction Demo:");
    prediction_demo(&dataset)?;

    Ok(())
}

fn display_gpu_info(gpu_manager: &GpuManager) {
    println!("🔧 GPU Information:");
    println!("===================");

    // Show backend availability
    println!("Backend Support:");
    println!(
        "  CUDA: {}",
        if GpuManager::is_cuda_available() {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  OpenCL: {}",
        if GpuManager::is_opencl_available() {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  ROCm: {}",
        if GpuManager::is_rocm_available() {
            "✅"
        } else {
            "❌"
        }
    );
    println!(
        "  Metal: {}",
        if GpuManager::is_metal_available() {
            "✅"
        } else {
            "❌"
        }
    );

    // Show available devices
    println!("\nAvailable Devices:");
    for (i, device) in gpu_manager.devices().iter().enumerate() {
        let icon = match device.device_type {
            GpuDeviceType::Cuda => "🟢",
            GpuDeviceType::OpenCL => "🔵",
            GpuDeviceType::ROCm => "🔴",
            GpuDeviceType::Metal => "🟡",
            GpuDeviceType::Intel => "🟣",
            GpuDeviceType::Generic => "⚪",
        };

        println!(
            "  [{}] {} {} - {:.1} GB ({:?})",
            i,
            icon,
            device.name,
            device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            device.device_type
        );
    }

    if let Some(default) = gpu_manager.default_device() {
        println!("\n🎯 Default device: {}", default.name);
    }
    println!();
}

fn show_dataset_info(dataset: &MnistDataset) {
    println!("📊 Dataset Information:");
    println!("======================");
    println!(
        "Training images: {} ({}×{})",
        dataset.train_images.len_of(Axis(0)),
        dataset.train_images.len_of(Axis(1)),
        dataset.train_images.len_of(Axis(2))
    );
    println!(
        "Test images: {} ({}×{})",
        dataset.test_images.len_of(Axis(0)),
        dataset.test_images.len_of(Axis(1)),
        dataset.test_images.len_of(Axis(2))
    );

    // Show class distribution
    let mut class_counts = vec![0u32; 10];
    for &label in dataset.train_labels.iter() {
        class_counts[label as usize] += 1;
    }

    println!("Class distribution:");
    for (digit, count) in class_counts.iter().enumerate() {
        println!("  Digit {}: {} samples", digit, count);
    }
    println!();
}

fn demo_training(gpu_manager: &mut GpuManager, dataset: &MnistDataset) -> Result<()> {
    let (train_images, test_images) = dataset.flatten_images();
    let (train_labels, test_labels) = dataset.one_hot_labels();

    println!("Building neural network architecture...");

    // Create a simple but effective MNIST network
    let mut network = Network::with_input_size(784)?
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
        .loss(LossFunction::CategoricalCrossEntropy)
        .name("MNIST Classifier")
        .build()?;

    println!("📐 Network Architecture:");
    network.print_summary();

    // Configure training
    let mut config = TrainingConfig::default();
    config.max_epochs = 10;
    config.batch_size = 32;
    config.validation_split = 0.2;
    config.verbose = true;

    // Use best available device
    if let Some(device) = gpu_manager.default_device() {
        println!("🔥 Training on: {}", device.name);

        if device.device_type != GpuDeviceType::Generic {
            let _context = gpu_manager.create_context(device.id)?;
            println!("✅ GPU context created");
        }
    }

    // Train the network
    println!("🏃 Starting training...");
    let start_time = Instant::now();
    let history = network.train(&train_images, &train_labels, &config)?;
    let training_duration = start_time.elapsed();

    // Show training results
    let final_loss = history.train_loss.last().unwrap_or(&f64::INFINITY);
    let final_val_loss = history.val_loss.last().unwrap_or(&f64::INFINITY);

    println!("\n📈 Training Results:");
    println!("===================");
    println!("Duration: {:.2} seconds", training_duration.as_secs_f64());
    println!("Final training loss: {:.6}", final_loss);
    println!("Final validation loss: {:.6}", final_val_loss);

    // Test the network
    println!("\n🧪 Testing on holdout set...");
    let test_start = Instant::now();
    let predictions = network.predict(&test_images)?;
    let test_duration = test_start.elapsed();

    let accuracy = calculate_accuracy(&predictions, &test_labels);
    println!("Test accuracy: {:.2}%", accuracy * 100.0);
    println!("Inference time: {:.2} ms", test_duration.as_millis());
    println!(
        "Throughput: {:.0} samples/sec",
        test_images.nrows() as f64 / test_duration.as_secs_f64()
    );

    Ok(())
}

fn compare_training_performance(
    gpu_manager: &mut GpuManager,
    dataset: &MnistDataset,
) -> Result<()> {
    let (train_images, _test_images) = dataset.flatten_images();
    let (train_labels, _test_labels) = dataset.one_hot_labels();

    println!("Comparing training performance across devices...");

    let mut results = Vec::new();

    // Collect device info first to avoid borrowing issues
    let devices: Vec<_> = gpu_manager
        .devices()
        .iter()
        .map(|d| (d.id, d.name.clone(), d.device_type))
        .collect();

    // Test each device
    for (device_id, device_name, device_type) in devices {
        println!("\n⏱️ Benchmarking: {}", device_name);

        // Create fresh network
        let mut network = Network::with_input_size(784)?
            .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
            .loss(LossFunction::CategoricalCrossEntropy)
            .build()?;

        // Create context for device
        let _context = gpu_manager.create_context(device_id)?;

        // Configure shorter training for benchmark
        let mut config = TrainingConfig::default();
        config.max_epochs = 3;
        config.batch_size = 64;
        config.verbose = false;

        // Time the training
        let start_time = Instant::now();
        let _history = network.train(&train_images, &train_labels, &config)?;
        let duration = start_time.elapsed();

        println!("  ✅ Completed in {:.2} seconds", duration.as_secs_f64());

        results.push((device_name, device_type, duration));
    }

    // Show performance comparison
    println!("\n🏆 Performance Comparison:");
    println!("==========================");

    results.sort_by(|a, b| a.2.cmp(&b.2));

    for (i, (name, device_type, time)) in results.iter().enumerate() {
        let icon = match i {
            0 => "🥇",
            1 => "🥈",
            2 => "🥉",
            _ => "  ",
        };

        println!(
            "{} {} ({:?}): {:.2}s",
            icon,
            name,
            device_type,
            time.as_secs_f64()
        );
    }

    if results.len() >= 2 {
        let fastest = results[0].2.as_secs_f64();
        let slowest = results[results.len() - 1].2.as_secs_f64();
        let speedup = slowest / fastest;
        println!("\n🚀 Speedup: {:.1}x faster on best device", speedup);
    }

    Ok(())
}

fn prediction_demo(dataset: &MnistDataset) -> Result<()> {
    println!("Creating trained model for prediction demo...");

    let (train_images, test_images) = dataset.flatten_images();
    let (train_labels, test_labels) = dataset.one_hot_labels();

    // Create and train a model
    let mut network = Network::with_input_size(784)?
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
        .loss(LossFunction::CategoricalCrossEntropy)
        .build()?;

    let mut config = TrainingConfig::default();
    config.max_epochs = 5;
    config.batch_size = 64;
    config.verbose = false;

    network.train(&train_images, &train_labels, &config)?;

    // Make predictions on test set
    let predictions = network.predict(&test_images)?;

    // Show some example predictions
    println!("\n🔮 Sample Predictions:");
    println!("=====================");

    for i in 0..10 {
        if i >= test_images.nrows() {
            break;
        }

        let predicted_class = predictions
            .row(i)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_class = test_labels
            .row(i)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let confidence = predictions[[i, predicted_class]];
        let correct = if predicted_class == true_class {
            "✅"
        } else {
            "❌"
        };

        println!(
            "Sample {}: Predicted {} (confidence: {:.1}%), True: {} {}",
            i + 1,
            predicted_class,
            confidence * 100.0,
            true_class,
            correct
        );

        // Show a simple ASCII representation of the digit
        if i < 3 {
            println!("  Image:");
            print_ascii_digit(test_images.row(i));
        }
    }

    let accuracy = calculate_accuracy(&predictions, &test_labels);
    println!("\nOverall test accuracy: {:.2}%", accuracy * 100.0);

    Ok(())
}

fn prompt_full_training() -> bool {
    println!("This will train on the full MNIST dataset (60k samples).");
    print!("Continue? [y/N]: ");
    std::io::Write::flush(&mut std::io::stdout()).unwrap();

    let mut input = String::new();
    std::io::stdin().read_line(&mut input).unwrap();

    matches!(input.trim().to_lowercase().as_str(), "y" | "yes")
}

fn calculate_accuracy(predictions: &Array2<f64>, labels: &Array2<f64>) -> f64 {
    let mut correct = 0;
    let total = predictions.nrows();

    for i in 0..total {
        let predicted_class = predictions
            .row(i)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        let true_class = labels
            .row(i)
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        if predicted_class == true_class {
            correct += 1;
        }
    }

    correct as f64 / total as f64
}

fn print_ascii_digit(image: ndarray::ArrayView1<f64>) {
    for row in 0..28 {
        print!("    ");
        for col in 0..28 {
            let pixel = image[row * 28 + col];
            let char = if pixel > 0.5 {
                "██"
            } else if pixel > 0.25 {
                "▓▓"
            } else if pixel > 0.1 {
                "░░"
            } else {
                "  "
            };
            print!("{}", char);
        }
        println!();
    }
    println!();
}

// MNIST file loading functions
fn load_mnist_images(path: &str) -> Result<Array3<f64>> {
    if !Path::new(path).exists() {
        return Err(rnn::NetworkError::data(format!(
            "MNIST file not found: {}",
            path
        )));
    }

    let mut file = BufReader::new(
        File::open(path)
            .map_err(|e| rnn::NetworkError::data(format!("Failed to open {}: {}", path, e)))?,
    );

    // Read header
    let mut header = [0u8; 16];
    file.read_exact(&mut header)
        .map_err(|e| rnn::NetworkError::data(format!("Failed to read header: {}", e)))?;

    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_images = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);
    let rows = u32::from_be_bytes([header[8], header[9], header[10], header[11]]);
    let cols = u32::from_be_bytes([header[12], header[13], header[14], header[15]]);

    if magic != 2051 {
        return Err(rnn::NetworkError::data(
            "Invalid magic number for images".to_string(),
        ));
    }

    println!("Loading {} images ({}×{})...", num_images, rows, cols);

    // Read image data
    let mut buffer = vec![0u8; (num_images * rows * cols) as usize];
    file.read_exact(&mut buffer)
        .map_err(|e| rnn::NetworkError::data(format!("Failed to read image data: {}", e)))?;

    // Convert to f64 and normalize
    let data: Vec<f64> = buffer.iter().map(|&x| x as f64 / 255.0).collect();

    Array3::from_shape_vec((num_images as usize, rows as usize, cols as usize), data)
        .map_err(|e| rnn::NetworkError::data(format!("Failed to reshape image data: {}", e)))
}

fn load_mnist_labels(path: &str) -> Result<Array1<u8>> {
    if !Path::new(path).exists() {
        return Err(rnn::NetworkError::data(format!(
            "MNIST file not found: {}",
            path
        )));
    }

    let mut file = BufReader::new(
        File::open(path)
            .map_err(|e| rnn::NetworkError::data(format!("Failed to open {}: {}", path, e)))?,
    );

    // Read header
    let mut header = [0u8; 8];
    file.read_exact(&mut header)
        .map_err(|e| rnn::NetworkError::data(format!("Failed to read header: {}", e)))?;

    let magic = u32::from_be_bytes([header[0], header[1], header[2], header[3]]);
    let num_labels = u32::from_be_bytes([header[4], header[5], header[6], header[7]]);

    if magic != 2049 {
        return Err(rnn::NetworkError::data(
            "Invalid magic number for labels".to_string(),
        ));
    }

    println!("Loading {} labels...", num_labels);

    // Read label data
    let mut buffer = vec![0u8; num_labels as usize];
    file.read_exact(&mut buffer)
        .map_err(|e| rnn::NetworkError::data(format!("Failed to read label data: {}", e)))?;

    Array1::from_vec(buffer)
        .into_dimensionality()
        .map_err(|e| rnn::NetworkError::data(format!("Failed to create label array: {}", e)))
}

fn labels_to_one_hot(labels: &Array1<u8>) -> Array2<f64> {
    let num_classes = 10;
    let num_samples = labels.len();
    let mut one_hot = Array2::zeros((num_samples, num_classes));

    for (i, &label) in labels.iter().enumerate() {
        one_hot[[i, label as usize]] = 1.0;
    }

    one_hot
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_one_hot_encoding() {
        let labels = Array1::from_vec(vec![0, 1, 2, 9]);
        let one_hot = labels_to_one_hot(&labels);

        assert_eq!(one_hot.shape(), &[4, 10]);
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
        assert_eq!(one_hot[[3, 9]], 1.0);
    }

    #[test]
    fn test_accuracy_calculation() {
        let predictions = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.1, 0.9, 0.0, // Predicted class 1
                0.8, 0.1, 0.1, // Predicted class 0
            ],
        )
        .unwrap();

        let labels = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.0, 1.0, 0.0, // True class 1
                1.0, 0.0, 0.0, // True class 0
            ],
        )
        .unwrap();

        let accuracy = calculate_accuracy(&predictions, &labels);
        assert_eq!(accuracy, 1.0); // 100% accuracy
    }

    #[test]
    fn test_dataset_subset() {
        // This test would require actual MNIST data files
        // In a real scenario, you would test with mock data
    }
}
