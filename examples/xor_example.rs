//! XOR Neural Network Example
//!
//! This example demonstrates how to train a simple neural network to learn the XOR function.
//! The XOR function is a classic example that cannot be solved by a single perceptron,
//! requiring at least one hidden layer.
//!
//! XOR Truth Table:
//! | Input A | Input B | Output |
//! |---------|---------|--------|
//! |    0    |    0    |   0    |
//! |    1    |    0    |   1    |
//! |    0    |    1    |   1    |
//! |    1    |    1    |   0    |

use rnn::{
    activation::ActivationFunction, layer::LayerBuilder, loss::LossFunction, network::Network,
    optimizer::Optimizer, training::TrainingConfig, utils::DataPreprocessing, Result,
};

use ndarray::{Array1, Array2};
use std::time::Instant;

fn main() -> Result<()> {
    println!("🔄 XOR Neural Network Example");
    println!("============================\n");

    // Create XOR training data
    let (train_data, train_targets) = create_xor_dataset()?;

    println!("📊 Training Data:");
    println!("Inputs: {:?}", train_data);
    println!("Targets: {:?}", train_targets);
    println!();

    // Build the neural network
    let mut network = Network::with_input_size(2)?
        .add_layer(LayerBuilder::dense(4).activation(ActivationFunction::Sigmoid))
        .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
        .loss(LossFunction::MeanSquaredError)
        .optimizer(Optimizer::adam(0.1)?)
        .name("xor_network")
        .description("Simple network to learn XOR function")
        .build()?;

    // Compile the network
    network.compile()?;

    // Print network summary
    network.print_summary();

    // Configure training
    let mut config = TrainingConfig::default();
    config.max_epochs = 1000;
    config.batch_size = 4; // Use all samples in each batch
    config.validation_split = 0.0; // No validation split for this simple example
    config.verbose = true;

    // Train the network
    println!("🏋️ Training the network...");
    let start_time = Instant::now();
    let history = network.train(&train_data, &train_targets, &config)?;
    let training_time = start_time.elapsed();

    println!("\n✅ Training completed in {:.2?}", training_time);
    println!("Final loss: {:.6}", history.train_loss.last().unwrap());

    // Test the network on the same data (should predict XOR function)
    println!("\n🧪 Testing the trained network:");
    let predictions = network.predict(&train_data)?;

    println!("\nResults:");
    println!(
        "{:<8} {:<8} {:<12} {:<8}",
        "Input A", "Input B", "Predicted", "Expected"
    );
    println!("{}", "-".repeat(40));

    for i in 0..train_data.nrows() {
        let input_a = train_data[[i, 0]];
        let input_b = train_data[[i, 1]];
        let predicted = predictions[[i, 0]];
        let expected = train_targets[[i, 0]];

        // Round prediction to nearest integer for display
        let predicted_binary = if predicted > 0.5 { 1 } else { 0 };
        let expected_binary = expected as i32;

        println!(
            "{:<8.0} {:<8.0} {:<.6} ({}) {:<8}",
            input_a, input_b, predicted, predicted_binary, expected_binary
        );
    }

    // Calculate accuracy
    let mut correct = 0;
    for i in 0..predictions.nrows() {
        let predicted_binary = if predictions[[i, 0]] > 0.5 { 1.0 } else { 0.0 };
        if (predicted_binary - train_targets[[i, 0]]).abs() < 0.1 {
            correct += 1;
        }
    }

    let accuracy = correct as f64 / predictions.nrows() as f64;
    println!(
        "\n📈 Accuracy: {:.1}% ({}/{} correct)",
        accuracy * 100.0,
        correct,
        predictions.nrows()
    );

    // Test with additional edge cases
    println!("\n🔍 Testing edge cases:");
    let edge_cases = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.1, 0.1, // Close to (0, 0)
            0.9, 0.1, // Close to (1, 0)
            0.1, 0.9, // Close to (0, 1)
            0.9, 0.9, // Close to (1, 1)
        ],
    )?;

    let edge_predictions = network.predict(&edge_cases)?;

    println!("\nEdge case results:");
    for i in 0..edge_cases.nrows() {
        let input_a = edge_cases[[i, 0]];
        let input_b = edge_cases[[i, 1]];
        let predicted = edge_predictions[[i, 0]];
        let predicted_binary = if predicted > 0.5 { 1 } else { 0 };

        println!(
            "({:.1}, {:.1}) -> {:.6} ({})",
            input_a, input_b, predicted, predicted_binary
        );
    }

    // Save the trained model
    network.save("xor_model.json")?;
    println!("\n💾 Model saved as 'xor_model.json'");

    // Demonstrate loading the model
    println!("\n📂 Loading model from file...");
    let mut loaded_network = Network::load("xor_model.json")?;
    loaded_network.compile()?;

    let loaded_predictions = loaded_network.predict(&train_data)?;

    // Verify the loaded model works the same
    let mut predictions_match = true;
    for i in 0..predictions.nrows() {
        if (predictions[[i, 0]] - loaded_predictions[[i, 0]]).abs() > 1e-6 {
            predictions_match = false;
            break;
        }
    }

    if predictions_match {
        println!("✅ Loaded model predictions match original model!");
    } else {
        println!("❌ Loaded model predictions differ from original!");
    }

    // Show training progress
    if history.train_loss.len() > 10 {
        println!("\n📊 Training progress (every 100 epochs):");
        for (i, &loss) in history.train_loss.iter().enumerate() {
            if i % 100 == 0 || i == history.train_loss.len() - 1 {
                println!("Epoch {}: Loss = {:.6}", i + 1, loss);
            }
        }
    }

    // Clean up
    std::fs::remove_file("xor_model.json").ok();

    println!("\n🎉 XOR example completed successfully!");

    Ok(())
}

/// Create the XOR dataset
fn create_xor_dataset() -> Result<(Array2<f64>, Array2<f64>)> {
    // XOR inputs: all possible combinations of 0 and 1
    let inputs = Array2::from_shape_vec(
        (4, 2),
        vec![
            0.0, 0.0, // XOR(0, 0) = 0
            0.0, 1.0, // XOR(0, 1) = 1
            1.0, 0.0, // XOR(1, 0) = 1
            1.0, 1.0, // XOR(1, 1) = 0
        ],
    )?;

    // XOR outputs
    let outputs = Array2::from_shape_vec(
        (4, 1),
        vec![
            0.0, // 0 XOR 0 = 0
            1.0, // 0 XOR 1 = 1
            1.0, // 1 XOR 0 = 1
            0.0, // 1 XOR 1 = 0
        ],
    )?;

    Ok((inputs, outputs))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_xor_dataset_creation() {
        let (inputs, outputs) = create_xor_dataset().unwrap();

        assert_eq!(inputs.shape(), [4, 2]);
        assert_eq!(outputs.shape(), [4, 1]);

        // Verify XOR logic
        assert_abs_diff_eq!(outputs[[0, 0]], 0.0, epsilon = 1e-10); // 0 XOR 0
        assert_abs_diff_eq!(outputs[[1, 0]], 1.0, epsilon = 1e-10); // 0 XOR 1
        assert_abs_diff_eq!(outputs[[2, 0]], 1.0, epsilon = 1e-10); // 1 XOR 0
        assert_abs_diff_eq!(outputs[[3, 0]], 0.0, epsilon = 1e-10); // 1 XOR 1
    }

    #[test]
    fn test_network_creation() {
        let network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(4).activation(ActivationFunction::Sigmoid))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .build();

        assert!(network.is_ok());
        let net = network.unwrap();
        assert_eq!(net.input_dim, 2);
        assert_eq!(net.output_dim, 1);
        assert_eq!(net.layers.len(), 2);
    }

    #[test]
    fn test_training_basic() {
        let (train_data, train_targets) = create_xor_dataset().unwrap();

        let mut network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(4).activation(ActivationFunction::Sigmoid))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .optimizer(Optimizer::adam(0.1).unwrap())
            .build()
            .unwrap();

        network.compile().unwrap();

        let mut config = TrainingConfig::default();
        config.max_epochs = 10; // Short training for test
        config.verbose = false;

        let history = network.train(&train_data, &train_targets, &config);
        assert!(history.is_ok());

        let hist = history.unwrap();
        assert_eq!(hist.train_loss.len(), 10);

        // Loss should generally decrease (though not guaranteed in just 10 epochs)
        assert!(hist.train_loss[0] > 0.0);
    }
}
