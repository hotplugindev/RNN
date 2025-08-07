//! Device Selection Example for RNN Library
//!
//! This example demonstrates how to manually select between different GPU devices
//! and CPU, showcasing the flexibility of the runtime detection system.

use ndarray::Array2;
use rnn::{
    ActivationFunction, GpuDeviceType, GpuManager, LayerBuilder, LossFunction, Network, Result,
    TrainingConfig,
};
use std::io::{self, Write};

fn main() -> Result<()> {
    println!("🎯 RNN Device Selection Example");
    println!("================================\n");

    // Initialize GPU manager with runtime detection
    let mut gpu_manager = GpuManager::new();

    // Show available devices
    display_available_devices(&gpu_manager);

    // Let user choose device
    let selected_device_id = if gpu_manager.devices().len() > 1 {
        prompt_device_selection(&gpu_manager)?
    } else {
        gpu_manager.devices()[0].id
    };

    // Demonstrate with selected device
    demonstrate_with_device(&mut gpu_manager, selected_device_id)?;

    // Show performance comparison if multiple devices available
    if gpu_manager.devices().len() > 1 {
        println!("\n🏁 Performance Comparison:");
        println!("==========================");
        compare_device_performance(&mut gpu_manager)?;
    }

    Ok(())
}

fn display_available_devices(gpu_manager: &GpuManager) {
    println!("📱 Available Devices:");

    if gpu_manager.devices().is_empty() {
        println!("  ❌ No devices found");
        return;
    }

    for (i, device) in gpu_manager.devices().iter().enumerate() {
        let device_type_icon = match device.device_type {
            GpuDeviceType::Cuda => "🟢",
            GpuDeviceType::OpenCL => "🔵",
            GpuDeviceType::ROCm => "🔴",
            GpuDeviceType::Metal => "🟡",
            GpuDeviceType::Intel => "🟣",
            GpuDeviceType::Generic => "⚪",
        };

        println!(
            "  [{}] {} {} - {:.1} GB memory ({:?})",
            i,
            device_type_icon,
            device.name,
            device.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
            device.device_type
        );

        if device.device_type != GpuDeviceType::Generic {
            println!(
                "      💻 Compute Units: {}, Max Work Group: {}",
                device.multiprocessor_count, device.max_work_group_size
            );
        }
    }

    // Show default device
    if let Some(default_device) = gpu_manager.default_device() {
        println!(
            "\n🎯 Default device: {} (ID: {})",
            default_device.name, default_device.id
        );
    }

    println!();
}

fn prompt_device_selection(gpu_manager: &GpuManager) -> Result<usize> {
    loop {
        print!(
            "🔍 Select device [0-{}] or 'auto' for automatic selection: ",
            gpu_manager.devices().len() - 1
        );
        io::stdout().flush().unwrap();

        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        let input = input.trim();

        if input == "auto" {
            if let Some(default_device) = gpu_manager.default_device() {
                println!("✅ Using automatic selection: {}", default_device.name);
                return Ok(default_device.id);
            } else {
                println!("❌ No default device available");
                continue;
            }
        }

        match input.parse::<usize>() {
            Ok(index) if index < gpu_manager.devices().len() => {
                let device = &gpu_manager.devices()[index];
                println!("✅ Selected: {}", device.name);
                return Ok(device.id);
            }
            _ => {
                println!(
                    "❌ Invalid selection. Please enter a number between 0 and {} or 'auto'",
                    gpu_manager.devices().len() - 1
                );
            }
        }
    }
}

fn demonstrate_with_device(gpu_manager: &mut GpuManager, device_id: usize) -> Result<()> {
    // Find the device info
    let device_name = gpu_manager
        .devices()
        .iter()
        .find(|d| d.id == device_id)
        .map(|d| d.name.clone())
        .expect("Device not found");

    let device_type = gpu_manager
        .devices()
        .iter()
        .find(|d| d.id == device_id)
        .map(|d| d.device_type)
        .expect("Device not found");

    let device_memory = gpu_manager
        .devices()
        .iter()
        .find(|d| d.id == device_id)
        .map(|d| d.total_memory)
        .expect("Device not found");

    println!("🔥 Demonstrating with device: {}", device_name);
    println!("   Type: {:?}", device_type);
    println!(
        "   Memory: {:.1} GB",
        device_memory as f64 / (1024.0 * 1024.0 * 1024.0)
    );

    // Create context for the device
    let context = gpu_manager.create_context(device_id)?;
    println!("✅ Created context for device");

    // Create sample data
    let (input_data, target_data) = create_sample_data();
    println!("📊 Created sample dataset: {} samples", input_data.nrows());

    // Create and test GPU tensor operations
    if device_type != GpuDeviceType::Generic {
        println!("🧮 Testing GPU tensor operations...");
        test_gpu_operations(&input_data, device_id, context)?;
    } else {
        println!("💻 Using CPU computations");
    }

    // Train a small network
    println!("🎓 Training neural network...");
    train_sample_network(&input_data, &target_data, &device_name)?;

    Ok(())
}

fn test_gpu_operations(
    data: &Array2<f64>,
    device_id: usize,
    context: &mut dyn rnn::gpu::GpuContext,
) -> Result<()> {
    // Create GPU tensor
    let gpu_tensor = rnn::gpu::GpuTensor::from_cpu(data, device_id, context)?;
    println!("  ✅ Created GPU tensor: {:?}", gpu_tensor.shape());

    // Test memory operations
    if let Ok(stats) = context.memory_stats() {
        println!(
            "  💾 Memory: {} bytes allocated ({} allocations)",
            stats.allocated, stats.allocation_count
        );
    }

    // Test tensor reshape
    let original_shape = gpu_tensor.shape().to_vec();
    let reshaped = gpu_tensor.reshape(vec![data.len(), 1])?;
    println!(
        "  🔄 Reshaped from {:?} to {:?}",
        original_shape,
        reshaped.shape()
    );

    // Transfer back to CPU
    let cpu_result = gpu_tensor.to_cpu(context)?;
    println!(
        "  ⬅️ Successfully transferred back to CPU: {:?}",
        cpu_result.shape()
    );

    Ok(())
}

fn train_sample_network(
    input_data: &Array2<f64>,
    target_data: &Array2<f64>,
    device_name: &str,
) -> Result<()> {
    let start_time = std::time::Instant::now();

    // Create a simple network
    let mut network = Network::with_input_size(input_data.ncols())?
        .add_layer(LayerBuilder::dense(32).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(16).activation(ActivationFunction::ReLU))
        .add_layer(LayerBuilder::dense(target_data.ncols()).activation(ActivationFunction::Sigmoid))
        .loss(LossFunction::BinaryCrossEntropy)
        .name(&format!("Network on {}", device_name))
        .build()?;

    // Configure training
    let mut config = TrainingConfig::default();
    config.max_epochs = 5;
    config.batch_size = 32;
    config.verbose = false; // Quiet training for this demo

    // Train the network
    let history = network.train(input_data, target_data, &config)?;
    let training_time = start_time.elapsed();

    // Show results
    let final_loss = history.train_loss.last().unwrap_or(&f64::INFINITY);
    println!(
        "  📈 Training completed in {:.2}s - Final loss: {:.6}",
        training_time.as_secs_f64(),
        final_loss
    );

    // Make predictions
    let predictions = network.predict(input_data)?;
    println!("  🔮 Generated {} predictions", predictions.nrows());

    Ok(())
}

fn compare_device_performance(gpu_manager: &mut GpuManager) -> Result<()> {
    let (input_data, target_data) = create_sample_data();
    let mut results = Vec::new();

    // Collect device info first to avoid borrowing issues
    let devices: Vec<_> = gpu_manager
        .devices()
        .iter()
        .map(|d| (d.id, d.name.clone(), d.device_type))
        .collect();

    for (device_id, device_name, device_type) in devices {
        println!("⏱️ Benchmarking {}...", device_name);

        let start_time = std::time::Instant::now();

        // Create context
        let _context = gpu_manager.create_context(device_id)?;

        // Quick training benchmark
        let mut network = Network::with_input_size(input_data.ncols())?
            .add_layer(LayerBuilder::dense(16).activation(ActivationFunction::ReLU))
            .add_layer(
                LayerBuilder::dense(target_data.ncols()).activation(ActivationFunction::Sigmoid),
            )
            .loss(LossFunction::BinaryCrossEntropy)
            .build()?;

        let mut config = TrainingConfig::default();
        config.max_epochs = 3;
        config.batch_size = 32;
        config.verbose = false;

        let _history = network.train(&input_data, &target_data, &config)?;
        let elapsed = start_time.elapsed();

        results.push((device_name, device_type, elapsed));
        println!("  ✅ Completed in {:.2}s", elapsed.as_secs_f64());
    }

    // Show comparison
    println!("\n📊 Performance Results:");
    println!("=======================");

    // Sort by performance (fastest first)
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

    // Show speedup
    if results.len() >= 2 {
        let fastest = results[0].2.as_secs_f64();
        let slowest = results[results.len() - 1].2.as_secs_f64();
        let speedup = slowest / fastest;

        println!("\n🚀 Best device is {:.1}x faster than slowest", speedup);
    }

    Ok(())
}

fn create_sample_data() -> (Array2<f64>, Array2<f64>) {
    // Create a simple XOR-like classification problem
    let n_samples = 1000;

    let mut input_data = Vec::new();
    let mut target_data = Vec::new();

    for i in 0..n_samples {
        let x1 = (i as f64 / n_samples as f64) * 2.0 - 1.0; // Range [-1, 1]
        let x2 = ((i * 3) as f64 / n_samples as f64) * 2.0 - 1.0; // Range [-1, 1]

        input_data.extend(&[x1, x2]);

        // XOR-like function: output 1 if signs are different
        let output = if (x1 > 0.0) != (x2 > 0.0) { 1.0 } else { 0.0 };
        target_data.push(output);
    }

    let input_array =
        Array2::from_shape_vec((n_samples, 2), input_data).expect("Failed to create input array");
    let target_array =
        Array2::from_shape_vec((n_samples, 1), target_data).expect("Failed to create target array");

    (input_array, target_array)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_data_creation() {
        let (inputs, targets) = create_sample_data();
        assert_eq!(inputs.nrows(), 1000);
        assert_eq!(inputs.ncols(), 2);
        assert_eq!(targets.nrows(), 1000);
        assert_eq!(targets.ncols(), 1);

        // Check that we have both classes
        let sum: f64 = targets.iter().sum();
        assert!(sum > 0.0 && sum < targets.len() as f64);
    }

    #[test]
    fn test_gpu_manager_has_devices() {
        let manager = GpuManager::new();
        assert!(
            !manager.devices().is_empty(),
            "Should have at least CPU device"
        );
    }
}
