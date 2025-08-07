//! Simple demo to verify basic RNN library functionality
//!
//! This is a minimal example to test that the library works correctly
//! without external dependencies like BLAS/LAPACK.

use rnn::{
    activation::ActivationFunction, layer::LayerBuilder, loss::LossFunction, network::Network,
    optimizer::Optimizer, training::TrainingConfig, Result,
};

use ndarray::Array2;

fn main() -> Result<()> {
    println!("🚀 Simple RNN Library Demo");
    println!("==========================\n");

    // Create simple training data (XOR-like problem)
    let train_data = Array2::from_shape_vec((4, 2), vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0])?;

    let train_targets = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 1.0, 0.0])?;

    println!("📊 Training data:");
    println!("Inputs:\n{}", train_data);
    println!("Targets:\n{}", train_targets);

    // Create a simple neural network
    let mut network = Network::with_input_size(2)?
        .add_layer(LayerBuilder::dense(4).activation(ActivationFunction::Sigmoid))
        .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
        .loss(LossFunction::MeanSquaredError)
        .optimizer(Optimizer::adam(0.1)?)
        .name("simple_demo_network")
        .build()?;

    // Compile the network
    network.compile()?;

    println!("\n🏗️  Network architecture:");
    network.print_summary();

    // Configure training with minimal epochs for demo
    let mut config = TrainingConfig::default();
    config.max_epochs = 100;
    config.batch_size = 4;
    config.validation_split = 0.0;
    config.verbose = false; // Keep output clean

    println!("\n🏋️  Training network for {} epochs...", config.max_epochs);

    // Train the network
    let history = network.train(&train_data, &train_targets, &config)?;

    println!("✅ Training completed!");
    println!("Final loss: {:.6}", history.train_loss.last().unwrap());

    // Test the trained network
    println!("\n🧪 Testing predictions:");
    let predictions = network.predict(&train_data)?;

    println!("\nResults:");
    println!(
        "{:<8} {:<8} {:<12} {:<8} {:<8}",
        "Input A", "Input B", "Predicted", "Target", "Correct?"
    );
    println!("{}", "-".repeat(50));

    for i in 0..train_data.nrows() {
        let input_a = train_data[[i, 0]];
        let input_b = train_data[[i, 1]];
        let predicted = predictions[[i, 0]];
        let target = train_targets[[i, 0]];

        // Simple threshold at 0.5
        let predicted_binary = if predicted > 0.5 { 1.0 } else { 0.0 };
        let correct = (predicted_binary - target).abs() < 0.1;

        println!(
            "{:<8.0} {:<8.0} {:<12.6} {:<8.0} {:<8}",
            input_a,
            input_b,
            predicted,
            target,
            if correct { "✅" } else { "❌" }
        );
    }

    // Calculate simple accuracy
    let mut correct_count = 0;
    for i in 0..predictions.nrows() {
        let predicted_binary = if predictions[[i, 0]] > 0.5 { 1.0 } else { 0.0 };
        if (predicted_binary - train_targets[[i, 0]]).abs() < 0.1 {
            correct_count += 1;
        }
    }

    let accuracy = correct_count as f64 / predictions.nrows() as f64;
    println!(
        "\n📈 Accuracy: {:.1}% ({}/{} correct)",
        accuracy * 100.0,
        correct_count,
        predictions.nrows()
    );

    // Demonstrate save/load functionality
    println!("\n💾 Testing save/load functionality...");
    network.save("simple_demo_model.json")?;

    let mut loaded_network = Network::load("simple_demo_model.json")?;
    loaded_network.compile()?;

    let loaded_predictions = loaded_network.predict(&train_data)?;

    // Verify predictions match
    let mut predictions_match = true;
    for i in 0..predictions.nrows() {
        if (predictions[[i, 0]] - loaded_predictions[[i, 0]]).abs() > 1e-6 {
            predictions_match = false;
            break;
        }
    }

    if predictions_match {
        println!("✅ Save/load test passed - predictions match!");
    } else {
        println!("❌ Save/load test failed - predictions differ!");
    }

    // Clean up
    std::fs::remove_file("simple_demo_model.json").ok();

    // Show parameter count
    println!("\n📊 Network statistics:");
    println!("Total parameters: {}", network.parameter_count());
    println!("Training epochs: {}", history.train_loss.len());
    println!("Input dimension: {}", network.input_dim);
    println!("Output dimension: {}", network.output_dim);

    println!("\n🎉 Demo completed successfully!");

    Ok(())
}
