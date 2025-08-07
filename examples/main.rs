//! Example demonstrating the RNN neural network library capabilities.
//!
//! This example shows how to:
//! - Create different types of neural networks
//! - Train networks with various optimizers and training methods
//! - Use different activation functions and loss functions
//! - Save and load networks
//! - Evaluate network performance
//! - Use data preprocessing utilities

use rnn::{
    activation::ActivationFunction,
    layer::LayerBuilder,
    loss::LossFunction,
    network::Network,
    optimizer::{Optimizer, OptimizerType},
    training::{TrainingConfig, TrainingMethod},
    utils::{DataPreprocessing, MathUtils, RandomUtils},
    Result,
};

use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🚀 RNN Neural Network Library Demo");
    println!("=====================================\n");

    // Run different examples
    basic_regression_example()?;
    classification_example()?;
    optimizer_comparison()?;
    save_load_example()?;
    data_preprocessing_example()?;

    println!("✅ All examples completed successfully!");
    Ok(())
}

/// Basic regression example using a simple feedforward network
fn basic_regression_example() -> Result<()> {
    println!("📊 Example 1: Basic Regression");
    println!("-------------------------------");

    // Generate synthetic regression data: y = x1^2 + x2^2 + noise
    let (data, targets) = generate_regression_data(1000, 2)?;

    // Create a neural network
    let mut network = Network::with_input_size(2)?
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Linear))
        .loss(LossFunction::MeanSquaredError)
        .optimizer(Optimizer::adam(0.001)?)
        .name("regression_network")
        .description("Simple regression network")
        .build()?;

    network.compile()?;
    network.print_summary();

    // Configure training
    let mut config = TrainingConfig::default();
    config.max_epochs = 50;
    config.batch_size = 32;
    config.validation_split = 0.2;
    config.verbose = true;

    println!("\n🏋️ Training regression network...");
    let start_time = Instant::now();
    let history = network.train(&data, &targets, &config)?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2?}", training_time);
    println!(
        "Final training loss: {:.6}",
        history.train_loss.last().unwrap()
    );
    if let Some(val_loss) = history.val_loss.last() {
        println!("Final validation loss: {:.6}", val_loss);
    }

    // Test the network
    let test_data = Array2::from_shape_vec(
        (5, 2),
        vec![
            1.0, 1.0, // Expected: ~2.0
            2.0, 2.0, // Expected: ~8.0
            0.5, 0.5, // Expected: ~0.5
            -1.0, 1.0, // Expected: ~2.0
            0.0, 0.0, // Expected: ~0.0
        ],
    )?;

    let predictions = network.predict(&test_data)?;
    println!("\n🔮 Test predictions:");
    for (i, pred) in predictions.iter().enumerate() {
        let input = test_data.row(i);
        let expected = input[0].powi(2) + input[1].powi(2);
        println!(
            "Input: [{:.1}, {:.1}] -> Predicted: {:.3}, Expected: {:.3}",
            input[0], input[1], pred, expected
        );
    }

    println!();
    Ok(())
}

/// Classification example with multiple classes
fn classification_example() -> Result<()> {
    println!("🎯 Example 2: Multi-class Classification");
    println!("----------------------------------------");

    // Generate synthetic classification data (3 classes)
    let (data, targets) = generate_classification_data(800, 2, 3)?;

    // Create a classification network
    let mut network = Network::with_input_size(2)?
        .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dropout(0.3)?)
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(3).activation(ActivationFunction::Softmax))
        .loss(LossFunction::CategoricalCrossEntropy)
        .optimizer(Optimizer::adam(0.001)?)
        .name("classification_network")
        .build()?;

    network.compile()?;

    // Configure training
    let mut config = TrainingConfig::default();
    config.max_epochs = 100;
    config.batch_size = 64;
    config.validation_split = 0.25;
    config.early_stopping_patience = Some(15);
    config.early_stopping_min_delta = 1e-4;
    config.verbose = false; // Less verbose for this example

    println!("🏋️ Training classification network...");
    let start_time = Instant::now();
    let history = network.train(&data, &targets, &config)?;
    let training_time = start_time.elapsed();

    println!("Training completed in {:.2?}", training_time);
    println!("Epochs trained: {}", history.train_loss.len());
    println!(
        "Final training loss: {:.6}",
        history.train_loss.last().unwrap()
    );

    // Evaluate on test data
    let (test_data, test_targets) = generate_classification_data(200, 2, 3)?;
    let metrics = network.evaluate(&test_data, &test_targets)?;

    println!("\n📈 Test Results:");
    for (metric, value) in &metrics {
        println!("{}: {:.4}", metric, value);
    }

    println!();
    Ok(())
}

/// Compare different optimizers
fn optimizer_comparison() -> Result<()> {
    println!("⚡ Example 3: Optimizer Comparison");
    println!("---------------------------------");

    let (data, targets) = generate_regression_data(500, 3)?;

    let optimizers = vec![
        ("SGD", Optimizer::sgd(0.01)?),
        ("Adam", Optimizer::adam(0.001)?),
        ("RMSprop", Optimizer::rmsprop(0.001, 0.9)?),
        ("Momentum", Optimizer::momentum(0.01, 0.9)?),
    ];

    for (name, optimizer) in optimizers {
        println!("🔧 Training with {} optimizer...", name);

        let mut network = Network::with_input_size(3)?
            .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(16).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Linear))
            .loss(LossFunction::MeanSquaredError)
            .optimizer(optimizer)
            .build()?;

        network.compile()?;

        let mut config = TrainingConfig::default();
        config.max_epochs = 30;
        config.batch_size = 32;
        config.verbose = false;

        let start_time = Instant::now();
        let history = network.train(&data, &targets, &config)?;
        let training_time = start_time.elapsed();

        println!(
            "  - Final loss: {:.6}, Time: {:.2?}",
            history.train_loss.last().unwrap(),
            training_time
        );
    }

    println!();
    Ok(())
}

/// Demonstrate save and load functionality
fn save_load_example() -> Result<()> {
    println!("💾 Example 4: Save and Load Networks");
    println!("-----------------------------------");

    // Create and train a small network
    let (data, targets) = generate_regression_data(200, 2)?;

    let mut original_network = Network::with_input_size(2)?
        .add_layer(LayerBuilder::dense(16).activation(ActivationFunction::Tanh))
        .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Linear))
        .loss(LossFunction::MeanSquaredError)
        .name("save_load_test")
        .author("RNN Library Demo")
        .description("Test network for save/load functionality")
        .build()?;

    original_network.compile()?;

    let mut config = TrainingConfig::default();
    config.max_epochs = 20;
    config.verbose = false;
    original_network.train(&data, &targets, &config)?;

    // Test original network
    let test_input = Array2::from_shape_vec((1, 2), vec![1.5, 2.0])?;
    let original_prediction = original_network.predict(&test_input)?;

    // Save in different formats
    println!("💾 Saving network in different formats...");
    original_network.save("network.json")?;
    original_network.save_binary("network.bin")?;
    original_network.export_weights("network_weights")?;

    // Load and test
    println!("📂 Loading networks...");

    // Load from JSON
    let mut loaded_json = Network::load("network.json")?;
    loaded_json.compile()?;
    let json_prediction = loaded_json.predict(&test_input)?;

    // Load from binary
    let mut loaded_binary = Network::load_binary("network.bin")?;
    loaded_binary.compile()?;
    let binary_prediction = loaded_binary.predict(&test_input)?;

    println!("✅ Prediction comparison:");
    println!("Original:  {:.6}", original_prediction[[0, 0]]);
    println!("JSON:      {:.6}", json_prediction[[0, 0]]);
    println!("Binary:    {:.6}", binary_prediction[[0, 0]]);

    let json_diff = (original_prediction[[0, 0]] - json_prediction[[0, 0]]).abs();
    let binary_diff = (original_prediction[[0, 0]] - binary_prediction[[0, 0]]).abs();

    println!("JSON difference:   {:.10}", json_diff);
    println!("Binary difference: {:.10}", binary_diff);

    // Cleanup
    std::fs::remove_file("network.json").ok();
    std::fs::remove_file("network.bin").ok();
    std::fs::remove_file("network_weights_info.json").ok();
    std::fs::remove_file("network_weights_layer_0_weights.csv").ok();
    std::fs::remove_file("network_weights_layer_0_bias.csv").ok();
    std::fs::remove_file("network_weights_layer_1_weights.csv").ok();
    std::fs::remove_file("network_weights_layer_1_bias.csv").ok();

    println!();
    Ok(())
}

/// Demonstrate data preprocessing utilities
fn data_preprocessing_example() -> Result<()> {
    println!("🔧 Example 5: Data Preprocessing");
    println!("-------------------------------");

    // Generate some raw data
    let raw_data = Array2::from_shape_vec(
        (100, 3),
        (0..300)
            .map(|x| (x as f64 / 10.0) + (x % 7) as f64 * 100.0)
            .collect(),
    )?;

    println!("📊 Original data statistics:");
    for i in 0..3 {
        let col = raw_data.column(i);
        let min = col.fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max = col.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        let mean = col.mean().unwrap();
        println!(
            "  Column {}: min={:.2}, max={:.2}, mean={:.2}",
            i, min, max, mean
        );
    }

    // Min-max normalization
    let (normalized, min_vals, max_vals) = DataPreprocessing::min_max_normalize(&raw_data)?;
    println!("\n🔄 After min-max normalization:");
    for i in 0..3 {
        let col = normalized.column(i);
        let min = col.fold(f64::INFINITY, |acc, &x| acc.min(x));
        let max = col.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
        println!("  Column {}: min={:.2}, max={:.2}", i, min, max);
    }

    // Standardization
    let (standardized, means, stds) = DataPreprocessing::standardize(&raw_data)?;
    println!("\n📐 After standardization:");
    for i in 0..3 {
        let col = standardized.column(i);
        let mean = col.mean().unwrap();
        let variance = col.mapv(|x| (x - mean).powi(2)).mean().unwrap();
        let std = variance.sqrt();
        println!("  Column {}: mean={:.6}, std={:.6}", i, mean, std);
    }

    // Categorical encoding
    let labels = Array1::from(vec![0, 1, 2, 1, 0, 2, 1, 0]);
    let one_hot = DataPreprocessing::to_categorical(&labels, Some(3))?;
    println!("\n🏷️  Categorical encoding:");
    println!("Labels: {:?}", labels);
    println!("One-hot shape: {:?}", one_hot.shape());

    // Train-test split
    let dummy_targets = Array2::ones((100, 1));
    let (train_data, test_data, train_targets, test_targets) =
        DataPreprocessing::train_test_split(&raw_data, &dummy_targets, 0.2, true, Some(42))?;

    println!("\n✂️  Train-test split:");
    println!("Training data shape: {:?}", train_data.shape());
    println!("Test data shape: {:?}", test_data.shape());

    // Add noise for data augmentation
    let noisy_data = DataPreprocessing::add_noise(&normalized, 0.05, Some(42))?;
    let noise_level = MathUtils::l2_norm(&(noisy_data - &normalized));
    println!("\n🔊 Data augmentation:");
    println!("Noise level (L2 norm): {:.6}", noise_level);

    println!();
    Ok(())
}

/// Generate synthetic regression data
fn generate_regression_data(
    n_samples: usize,
    n_features: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let data = RandomUtils::normal_random((n_samples, n_features), 0.0, 1.0, Some(42))?;

    // Target: sum of squares + noise
    let mut targets = Array2::zeros((n_samples, 1));
    for (i, row) in data.rows().into_iter().enumerate() {
        let sum_of_squares: f64 = row.iter().map(|&x| x * x).sum();
        let noise = RandomUtils::normal_random((1, 1), 0.0, 0.1, None)?[[0, 0]];
        targets[[i, 0]] = sum_of_squares + noise;
    }

    Ok((data, targets))
}

/// Generate synthetic classification data
fn generate_classification_data(
    n_samples: usize,
    n_features: usize,
    n_classes: usize,
) -> Result<(Array2<f64>, Array2<f64>)> {
    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);

    let samples_per_class = n_samples / n_classes;

    for class in 0..n_classes {
        let start_idx = class * samples_per_class;
        let end_idx = if class == n_classes - 1 {
            n_samples
        } else {
            (class + 1) * samples_per_class
        };

        // Generate data points around different centers for each class
        let center_x = (class as f64 - n_classes as f64 / 2.0) * 2.0;
        let center_y = (class as f64 - n_classes as f64 / 2.0) * 1.5;

        for i in start_idx..end_idx {
            let noise = RandomUtils::normal_random((1, n_features), 0.0, 0.5, None)?;
            data[[i, 0]] = center_x + noise[[0, 0]];
            if n_features > 1 {
                data[[i, 1]] = center_y + noise[[0, 1 % n_features]];
            }
            for j in 2..n_features {
                data[[i, j]] = noise[[0, j % noise.ncols()]];
            }
            labels[i] = class;
        }
    }

    // Convert labels to one-hot encoding
    let targets = DataPreprocessing::to_categorical(&labels, Some(n_classes))?;

    Ok((data, targets))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_generation() {
        let (data, targets) = generate_regression_data(100, 3).unwrap();
        assert_eq!(data.shape(), [100, 3]);
        assert_eq!(targets.shape(), [100, 1]);

        let (data, targets) = generate_classification_data(150, 2, 3).unwrap();
        assert_eq!(data.shape(), [150, 2]);
        assert_eq!(targets.shape(), [150, 3]);
    }

    #[test]
    fn test_basic_network_creation() {
        let network = Network::with_input_size(10)
            .unwrap()
            .add_layer(LayerBuilder::dense(5))
            .build();
        assert!(network.is_ok());
    }
}
