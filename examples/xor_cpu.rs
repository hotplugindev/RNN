//! XOR CPU Example
//!
//! This example demonstrates training a simple neural network to solve the XOR problem
//! using CPU computation. The XOR problem is a classic test for neural networks as it
//! requires learning a non-linearly separable function.

use num_traits::Float;
use rnn::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    // Initialize environment logger for debugging
    env_logger::init();

    println!("🧠 RNN XOR CPU Example");
    println!("======================");
    println!("Training a neural network to solve the XOR problem on CPU...\n");

    // Ensure we're using CPU device
    let device = Device::cpu()?;
    println!("🖥️  Using device: {:?}", device.device_type());

    // Create XOR dataset
    let (inputs, targets) = create_xor_dataset(&device)?;

    println!("📊 Dataset:");
    for (input, target) in inputs.iter().zip(targets.iter()) {
        let input_data = input.to_vec()?;
        let target_data = target.to_vec()?;
        println!(
            "   [{:.0}, {:.0}] -> {:.0}",
            input_data[0], input_data[1], target_data[0]
        );
    }
    println!();

    // Build the neural network
    let mut network = NetworkBuilder::new()
        .name("XOR Solver")
        .description("Simple MLP for XOR problem")
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 8,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 8,
            output_size: 4,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::HeNormal,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 4,
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .device(device)
        .build()?;

    // Print network summary
    println!("🏗️  Network Architecture:");
    println!("{}", network.summary());
    println!();

    // Training configuration
    let config = TrainingConfig {
        epochs: 10,
        batch_size: 4,
        verbose: true,
        early_stopping_patience: 50,
        early_stopping_threshold: 1e-6,
        lr_schedule: Some(LearningRateSchedule::StepLR {
            step_size: 200,
            gamma: 0.8,
        }),
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    println!("🏋️  Training Configuration:");
    println!("   Epochs: {}", config.epochs);
    println!("   Batch Size: {}", config.batch_size);
    println!("   Learning Rate Schedule: {:?}", config.lr_schedule);
    println!(
        "   Early Stopping Patience: {}",
        config.early_stopping_patience
    );
    println!();

    // Train the network
    println!("🚀 Starting training...");
    let start_time = Instant::now();

    let history = network.train(&inputs, &targets, &config)?;

    let training_time = start_time.elapsed();
    println!(
        "✅ Training completed in {:.2}s",
        training_time.as_secs_f32()
    );
    println!();

    // Print training summary
    let summary = history.summary();
    println!("📈 Training Summary:");
    println!("{}", summary);
    println!();

    // Test the trained network
    println!("🧪 Testing the trained network:");
    network.set_training(false);

    let test_cases = vec![
        ([0.0, 0.0], 0.0),
        ([0.0, 1.0], 1.0),
        ([1.0, 0.0], 1.0),
        ([1.0, 1.0], 0.0),
    ];

    let mut correct = 0;
    let threshold = 0.5;
    let total_cases = test_cases.len();

    for (inputs_arr, expected) in test_cases {
        let test_input = Tensor::from_slice(&inputs_arr, &[1, 2])?;
        let prediction = network.forward(&test_input)?;
        let pred_value = prediction.to_vec()?[0];

        let predicted_class = if pred_value > threshold { 1.0 } else { 0.0 };
        let is_correct = (predicted_class - expected).abs() < 0.1_f32;

        if is_correct {
            correct += 1;
        }

        println!(
            "   Input: [{:.0}, {:.0}] -> Prediction: {:.4} -> Class: {:.0} (Expected: {:.0}) {}",
            inputs_arr[0],
            inputs_arr[1],
            pred_value,
            predicted_class,
            expected,
            if is_correct { "✅" } else { "❌" }
        );
    }

    let accuracy = correct as f32 / total_cases as f32;
    println!();
    println!(
        "🎯 Final Accuracy: {:.1}% ({}/{})",
        accuracy * 100.0,
        correct,
        total_cases
    );

    // Demonstrate model saving
    println!();
    println!("💾 Saving model to disk...");

    let model_path = "xor_model_cpu.bin";
    let metadata = rnn::io::ModelMetadata {
        name: "XOR CPU Model".to_string(),
        description: "Trained XOR solver using CPU".to_string(),
        created_at: chrono::Utc::now().to_rfc3339(),
        modified_at: chrono::Utc::now().to_rfc3339(),
        training_info: rnn::io::TrainingInfo {
            epochs_trained: history.epochs(),
            final_loss: history.final_loss(),
            best_accuracy: history.best_accuracy(),
            training_time_seconds: training_time.as_secs_f32(),
            dataset_info: Some(rnn::io::DatasetInfo {
                name: "XOR".to_string(),
                train_samples: inputs.len(),
                val_samples: None,
                test_samples: Some(4),
                num_classes: Some(2),
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
    println!("✅ Model saved to: {}", model_path);

    // Performance analysis
    println!();
    println!("⚡ Performance Analysis:");
    let metrics = network.metrics();
    println!("   Total Parameters: {}", metrics.total_parameters);
    println!("   Training Time: {:.2}s", training_time.as_secs_f32());
    println!("   Final Loss: {:.6}", history.final_loss());
    println!(
        "   Best Loss: {:.6} (epoch {})",
        history.best_loss(),
        history.best_loss_epoch()
    );

    if accuracy >= 1.0 {
        println!("🎉 Perfect! The network learned the XOR function completely!");
    } else if accuracy >= 0.75 {
        println!("👍 Good! The network mostly learned the XOR function.");
    } else {
        println!(
            "🤔 The network struggled to learn the XOR function. Try adjusting hyperparameters."
        );
    }

    Ok(())
}

/// Create the XOR dataset
fn create_xor_dataset(device: &Device) -> Result<(Vec<Tensor>, Vec<Tensor>)> {
    let input_data = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];

    let target_data = vec![
        vec![0.0], // 0 XOR 0 = 0
        vec![1.0], // 0 XOR 1 = 1
        vec![1.0], // 1 XOR 0 = 1
        vec![0.0], // 1 XOR 1 = 0
    ];

    let inputs = input_data
        .into_iter()
        .map(|data| Tensor::from_slice_on_device(&data, &[1, 2], device.clone()))
        .collect::<Result<Vec<_>>>()?;

    let targets = target_data
        .into_iter()
        .map(|data| Tensor::from_slice_on_device(&data, &[1, 1], device.clone()))
        .collect::<Result<Vec<_>>>()?;

    Ok((inputs, targets))
}
