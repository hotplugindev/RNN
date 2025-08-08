# NNL - Neural Network Library

![nnl Logo](https://raw.githubusercontent.com/hotplugindev/NNL/main/img/nnl.png)

A high-performance neural network library for Rust with CPU and Vulkan GPU support.

[![Crates.io](https://img.shields.io/crates/v/nnl.svg)](https://crates.io/crates/nnl)
[![Documentation](https://docs.rs/nnl/badge.svg)](https://docs.rs/nnl)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](#license)

## Features

- 🚀 **Dual Backend Support**: Optimized CPU execution and Vulkan compute shaders
- 🎯 **Automatic Hardware Detection**: Seamlessly selects between CPU and Vulkan GPU
- 🧠 **Advanced Optimizers**: Adam, SGD, and other optimization algorithms
- 🏗️ **Flexible Architecture**: Dense layers, CNN, batch normalization, dropout, and custom layers
- 💾 **Model Persistence**: Save/load models with metadata in multiple formats (Binary, JSON, MessagePack)
- ⚡ **Production Ready**: SIMD optimizations, parallel processing, and zero-copy operations
- 🔧 **Comprehensive Training**: Learning rate scheduling, early stopping, metrics tracking
- 🎛️ **Fine-grained Control**: Custom loss functions, weight initialization, and gradient computation

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
nnl = "0.1.0"
```

### Basic XOR Example

```rust
use nnl::prelude::*;

fn main() -> Result<()> {
    // Select the CPU device
    let device = Device::cpu()?;

    // Create XOR training data
    // Each input and target is a separate Tensor, typically a batch of size 1 for this example.
    let train_inputs = vec![
        Tensor::from_slice_on_device(&[0.0, 0.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?,
        Tensor::from_slice_on_device(&[1.0, 1.0], &[1, 2], device.clone())?,
    ];

    let train_targets = vec![
        Tensor::from_slice_on_device(&[0.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[1.0], &[1, 1], device.clone())?,
        Tensor::from_slice_on_device(&[0.0], &[1, 1], device.clone())?,
    ];

    // Create a simple neural network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 8, // Hidden layer size
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 8, // Input to output layer
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::MeanSquaredError) // Common for regression/binary classification
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .device(device.clone()) // Specify the device for the network
        .build()?;

    // Configure training
    let training_config = TrainingConfig {
        epochs: 1000,
        batch_size: 4, // Train on all 4 samples at once
        verbose: false, // Set to true for detailed epoch logs
        ..Default::default() // Use default values for other config fields
    };

    // Train the network
    network.train(&train_inputs, &train_targets, &training_config)?;

    // Make predictions and evaluate
    let test_input_00 = Tensor::from_slice_on_device(&[0.0, 0.0], &[1, 2], device.clone())?;
    let test_input_01 = Tensor::from_slice_on_device(&[0.0, 1.0], &[1, 2], device.clone())?;
    let test_input_10 = Tensor::from_slice_on_device(&[1.0, 0.0], &[1, 2], device.clone())?;
    let test_input_11 = Tensor::from_slice_on_device(&[1.0, 1.0], &[1, 2], device)?;

    let pred_00 = network.forward(&test_input_00)?.to_vec()?[0];
    let pred_01 = network.forward(&test_input_01)?.to_vec()?[0];
    let pred_10 = network.forward(&test_input_10)?.to_vec()?[0];
    let pred_11 = network.forward(&test_input_11)?.to_vec()?[0];

    // Print predictions, converting to binary (0 or 1)
    println!("\n--- XOR Predictions ---");
    println!("XOR(0,0) = {:.4} (class: {:.0})", pred_00, if pred_00 > 0.5 { 1.0 } else { 0.0 });
    println!("XOR(0,1) = {:.4} (class: {:.0})", pred_01, if pred_01 > 0.5 { 1.0 } else { 0.0 });
    println!("XOR(1,0) = {:.4} (class: {:.0})", pred_10, if pred_10 > 0.5 { 1.0 } else { 0.0 });
    println!("XOR(1,1) = {:.4} (class: {:.0})", pred_11, if pred_11 > 0.5 { 1.0 } else { 0.0 });
    println!("-------------------------");

    Ok(())
}
```

## Installation

### CPU-only

```toml
[dependencies]
nnl = "0.1.0"
```

### With OpenBLAS optimization

```toml
[dependencies]
nnl = { version = "0.1.0", features = ["cpu-optimized"] }
```

### With Intel MKL optimization

```toml
[dependencies]
nnl = { version = "0.1.0", features = ["intel-mkl"] }
```

### System Requirements

- **Rust**: 1.70 or later (edition 2024)
- **CPU**: Any modern x86_64 or ARM64 processor
- **GPU (optional)**: Any Vulkan 1.2+ compatible GPU (AMD, Intel, NVIDIA)
- **OS**: Linux, Windows, macOS

### GPU Support

NNL uses Vulkan compute shaders for GPU acceleration, which works on:
- **AMD GPUs**: Radeon RX 400 series and newer
- **NVIDIA GPUs**: GTX 900 series and newer  
- **Intel GPUs**: Arc series and modern integrated graphics

## Examples

Run the included examples to see the library in action:

```bash
# Basic XOR problem (CPU)
cargo run --example xor

# XOR with GPU acceleration (if Vulkan GPU available)
cargo run --example xor_gpu

# MNIST digit classification
cargo run --example mnist

# MNIST with GPU
cargo run --example mnist_gpu

# Convolutional Neural Network
cargo run --example simple_cnn

# CNN with GPU support
cargo run --example simple_cnn_gpu

# Small MNIST examples for testing
cargo run --example mnist_small
cargo run --example mnist_small_gpu
```

### Available Examples

- [`xor.rs`](examples/xor.rs) - Solve XOR problem with a simple neural network (CPU)
- [`xor_gpu.rs`](examples/xor_gpu.rs) - XOR with Vulkan GPU acceleration
- [`mnist.rs`](examples/mnist.rs) - MNIST handwritten digit classification (CPU)
- [`mnist_gpu.rs`](examples/mnist_gpu.rs) - MNIST with GPU acceleration
- [`mnist_small.rs`](examples/mnist_small.rs) - Smaller MNIST dataset for testing (CPU)
- [`mnist_small_gpu.rs`](examples/mnist_small_gpu.rs) - Small MNIST with GPU
- [`simple_cnn.rs`](examples/simple_cnn.rs) - Convolutional neural network (CPU)
- [`simple_cnn_gpu.rs`](examples/simple_cnn_gpu.rs) - CNN with GPU acceleration

## Core Concepts

### Device Management

```rust
// Automatic device selection (prefers GPU if available, falls back to CPU)
let device = Device::auto_select()?;

// Specific device types
let cpu_device = Device::cpu()?;
let vulkan_device = Device::vulkan()?;  // May fail if no Vulkan GPU available

// Check device capabilities
println!("Device: {}", device.device_type());
println!("Memory: {:?}", device.info().memory_size);
```

### Tensors

```rust
// Create tensors (uses auto-selected device)
let zeros = Tensor::zeros(&[3, 4])?;
let ones = Tensor::ones(&[2, 2])?;
let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;

// Create tensors on specific device
let device = Device::vulkan()?;
let gpu_tensor = Tensor::from_slice_on_device(&[1.0, 2.0, 3.0], &[3], device)?;

// Tensor operations
let a = Tensor::randn(&[2, 3])?;
let b = Tensor::randn(&[2, 3])?;
let result = a.add(&b)?;  // Element-wise addition
let matmul = a.matmul(&b.transpose(&[1, 0])?)?;  // Matrix multiplication
```

### Network Architecture

```rust
let network = NetworkBuilder::new()
    .add_layer(LayerConfig::Dense {
        input_size: 784,
        output_size: 128,
        activation: Activation::ReLU,
        use_bias: true,
        weight_init: WeightInit::Xavier,
    })
    .add_layer(LayerConfig::Dropout { dropout_rate: 0.2 })
    .add_layer(LayerConfig::Dense {
        input_size: 128,
        output_size: 10,
        activation: Activation::Softmax,
        use_bias: true,
        weight_init: WeightInit::Xavier,
    })
    .loss(LossFunction::CategoricalCrossEntropy)
    .optimizer(OptimizerConfig::Adam {
        learning_rate: 0.001,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
        weight_decay: Some(1e-4),
        amsgrad: false,
    })
    .device(Device::auto_select()?)  // Automatically choose best device
    .build()?;
```

### Training with Advanced Features

```rust
let config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    verbose: true,
    early_stopping_patience: Some(10),
    early_stopping_threshold: 1e-4,
    validation_split: 0.2,
    shuffle: true,
    random_seed: Some(42),
    ..Default::default()
};

let history = network.train(&train_data, &train_labels, &config)?;
println!("Final loss: {:.4}", history.final_loss());
```

### Model Persistence

```rust
use nnl::io::{save_model, load_model, ModelFormat};

// Save model
save_model(&network, "my_model.bin", ModelFormat::Binary, None)?;

// Load model
let loaded_network = load_model("my_model.bin")?;

// Save with metadata
let metadata = ModelMetadata {
    name: "MNIST Classifier".to_string(),
    description: "CNN for digit classification".to_string(),
    ..Default::default()
};
save_model(&network, "model_with_meta.json", ModelFormat::Json, Some(&metadata))?;
```

## Performance

### Benchmarks

Performance comparison on common tasks (Intel i7-10700K, RTX 3060 via Vulkan):

| Task | CPU (8 threads) | Vulkan GPU | Speedup |
|------|----------------|------------|---------|
| Dense 1000x1000 MatMul | 12.5ms | 3.2ms | 3.9x |
| Conv2D 224x224x64 | 145ms | 28ms | 5.2x |
| MNIST Training (60k samples) | 45s | 18s | 2.5x |

*Note: Performance varies significantly based on GPU model and driver quality. Vulkan performance on NVIDIA may be lower than native CUDA.*

### Optimization Tips

1. **Use appropriate batch sizes**: 32-128 for GPU, 8-32 for CPU
2. **Enable CPU optimizations**: Use `features = ["cpu-optimized"]` for OpenBLAS
3. **Intel CPUs**: Use `features = ["intel-mkl"]` for maximum CPU performance
4. **Memory management**: Call `network.zero_grad()` regularly to free unused memory
5. **Data loading**: Use parallel data loading for large datasets
6. **GPU memory**: Monitor GPU memory usage, reduce batch size if running out

## Feature Flags

| Feature | Description | Dependencies |
|---------|-------------|--------------|
| `default` | CPU-optimized + examples | `["cpu-optimized", "examples"]` |
| `cpu-optimized` | OpenBLAS acceleration | `openblas-src` |
| `intel-mkl` | Intel MKL acceleration | `intel-mkl-src` |
| `examples` | Example binaries and utilities | `clap`, `image` |

Note: Vulkan support is always enabled and does not require a feature flag.

## Troubleshooting

### Common Issues

**Vulkan not available**
```bash
# Install Vulkan drivers and loader
# Ubuntu/Debian:
sudo apt install vulkan-tools vulkan-utils mesa-vulkan-drivers

# Verify Vulkan works:
vulkaninfo

# For NVIDIA GPUs, ensure latest drivers are installed
# For AMD GPUs on Linux, ensure AMDGPU driver is loaded
```

**Slow CPU performance**
```toml
# Enable OpenBLAS optimizations
nnl = { version = "0.1.0", features = ["cpu-optimized"] }

# Or for Intel CPUs, use MKL:
nnl = { version = "0.1.0", features = ["intel-mkl"] }
```

**Out of memory on GPU**
- Reduce batch size in `TrainingConfig`
- Use smaller model architectures
- Monitor GPU memory usage with `nvidia-smi` or similar tools

**Compilation errors with MKL**
```bash
# Ensure Intel MKL is properly installed
# Or switch to OpenBLAS:
nnl = { version = "0.1.0", features = ["cpu-optimized"] }
```

**Poor GPU performance**
- Ensure you're using `Device::vulkan()` or `Device::auto_select()`
- Check that Vulkan drivers are up to date
- Some operations may not be optimized for GPU yet
- Consider using CPU with optimizations for small models

## API Documentation

For detailed API documentation, see [docs.rs/nnl](https://docs.rs/nnl).

Key modules:
- [`tensor`](https://docs.rs/nnl/latest/nnl/tensor/) - Tensor operations and data structures
- [`network`](https://docs.rs/nnl/latest/nnl/network/) - Neural network building and training
- [`layers`](https://docs.rs/nnl/latest/nnl/layers/) - Layer implementations and configurations
- [`optimizers`](https://docs.rs/nnl/latest/nnl/optimizers/) - Optimization algorithms
- [`device`](https://docs.rs/nnl/latest/nnl/device/) - Device management and backend selection
- [`io`](https://docs.rs/nnl/latest/nnl/io/) - Model saving and loading

## Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run `cargo test` and `cargo clippy`
5. Submit a pull request

For major changes, please open an issue first to discuss the proposed changes.

### Development Setup

```bash
git clone https://github.com/hotplugindev/NNL.git
cd NNL
cargo build
cargo test
cargo run --example xor

# Test GPU functionality (requires Vulkan)
cargo run --example xor_gpu
```

## Roadmap

- [ ] **CUDA Support**: Native NVIDIA CUDA backend for better performance
- [ ] **ROCm Support**: AMD ROCm backend for compute-focused workloads  
- [ ] **Distributed Training**: Multi-GPU support
- [ ] **Mobile Deployment**: ARM optimization and model quantization
- [ ] **Web Assembly**: Browser-based inference
- [ ] **Model Zoo**: Pre-trained models for common tasks
- [ ] **Auto-ML**: Neural architecture search
- [ ] **Graph Optimization**: Operator fusion and memory optimization

## Limitations

- **CUDA**: Not yet supported (Vulkan used for NVIDIA GPUs)
- **ROCm**: Not yet supported (Vulkan used for AMD GPUs)
- **Distributed Training**: Single device only
- **Model Formats**: Limited compared to PyTorch/TensorFlow
- **Layer Types**: Growing but not comprehensive
- **Performance**: Vulkan overhead may impact small models

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE-MIT) file for details.

## Acknowledgments

- Built on excellent Rust ecosystem crates: `ndarray`, `rayon`, `vulkano`
- Inspired by PyTorch and TensorFlow APIs
- Thanks to the Rust ML community and all contributors

---

**Questions?** Open an [issue](https://github.com/hotplugindev/nnl/issues) on GitHub.