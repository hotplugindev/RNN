# NNL - Neural Network Library

![nnl Logo](nnl.png)

A high-performance neural network library for Rust with comprehensive GPU and CPU support.

[![Crates.io](https://img.shields.io/crates/v/nnl.svg)](https://crates.io/crates/nnl)
[![Documentation](https://docs.rs/nnl/badge.svg)](https://docs.rs/nnl)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](#license)
[![Build Status](https://github.com/hotplugindev/nnl/workflows/CI/badge.svg)](https://github.com/hotplugindev/nnl/actions)

## Features

- 🚀 **Multi-backend Support**: NVIDIA CUDA, AMD ROCm/Vulkan, and optimized CPU execution
- 🎯 **Automatic Hardware Detection**: Seamlessly selects the best available compute backend
- 🧠 **Advanced Optimizers**: Adam, SGD, AdaGrad, RMSprop, AdamW, LBFGS, and more
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
    // Create a simple neural network
    let mut network = NetworkBuilder::new()
        .add_layer(LayerConfig::Dense {
            input_size: 2,
            output_size: 4,
            activation: Activation::ReLU,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .add_layer(LayerConfig::Dense {
            input_size: 4,
            output_size: 1,
            activation: Activation::Sigmoid,
            use_bias: true,
            weight_init: WeightInit::Xavier,
        })
        .loss(LossFunction::BinaryCrossEntropy)
        .optimizer(OptimizerConfig::Adam { learning_rate: 0.01 })
        .build()?;

    // Training data for XOR problem
    let inputs = Tensor::from_slice(&[
        0.0, 0.0,  // XOR(0,0) = 0
        0.0, 1.0,  // XOR(0,1) = 1
        1.0, 0.0,  // XOR(1,0) = 1
        1.0, 1.0,  // XOR(1,1) = 0
    ], &[4, 2])?;

    let targets = Tensor::from_slice(&[0.0, 1.0, 1.0, 0.0], &[4, 1])?;

    // Train the network
    network.train(&inputs, &targets, 1000)?;

    // Make predictions
    let test_input = Tensor::from_slice(&[1.0, 0.0], &[1, 2])?;
    let prediction = network.forward(&test_input)?;
    println!("XOR(1,0) = {:.4}", prediction.to_vec()?[0]);

    Ok(())
}
```

## Installation

### CPU-only (default)

```toml
[dependencies]
nnl = "0.1.0"
```

### With GPU Support

```toml
[dependencies]
nnl = { version = "0.1.0", features = ["cuda"] }  # NVIDIA CUDA
# or
nnl = { version = "0.1.0", features = ["vulkan"] } # Vulkan (AMD/Intel/NVIDIA)
# or
nnl = { version = "0.1.0", features = ["all-backends"] } # All GPU backends
```

### System Requirements

- **Rust**: 1.70 or later
- **CPU**: Any modern x86_64 or ARM64 processor
- **GPU (optional)**:
  - CUDA: NVIDIA GPU with compute capability 3.5+, CUDA 11.0+
  - Vulkan: Any Vulkan 1.2+ compatible GPU
  - ROCm: AMD GPU with ROCm 4.0+ (experimental)

## Examples

Run the included examples to see the library in action:

```bash
# Basic XOR problem (CPU)
cargo run --example xor

# XOR with GPU acceleration
cargo run --example xor_gpu --features cuda

# MNIST digit classification
cargo run --example mnist

# Convolutional Neural Network
cargo run --example simple_cnn

# CNN with GPU support
cargo run --example simple_cnn_gpu --features cuda
```

### Available Examples

- [`xor.rs`](examples/xor.rs) - Solve XOR problem with a simple neural network
- [`mnist.rs`](examples/mnist.rs) - MNIST handwritten digit classification
- [`simple_cnn.rs`](examples/simple_cnn.rs) - Convolutional neural network example
- GPU variants: `*_gpu.rs` - Same examples with GPU acceleration

## Core Concepts

### Device Management

```rust
// Automatic device selection (CPU/GPU)
let device = Device::auto_select()?;

// Specific device types
let cpu_device = Device::cpu()?;
let cuda_device = Device::cuda(0)?;  // GPU 0
let vulkan_device = Device::vulkan()?;
```

### Tensors

```rust
// Create tensors
let zeros = Tensor::zeros(&[3, 4])?;
let ones = Tensor::ones(&[2, 2])?;
let from_data = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;

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
    .build()?;
```

### Training with Advanced Features

```rust
let config = TrainingConfig {
    epochs: 100,
    batch_size: 32,
    verbose: true,
    early_stopping_patience: 10,
    early_stopping_threshold: 1e-4,
    lr_schedule: Some(LearningRateSchedule::StepLR {
        step_size: 30,
        gamma: 0.1
    }),
    validation_split: 0.2,
    shuffle: true,
    random_seed: Some(42),
};

let history = network.train(&train_data, &train_labels, &config)?;
println!("Best accuracy: {:.4}", history.best_accuracy());
```

### Model Persistence

```rust
// Save model
save_model(&network, "my_model.bin", ModelFormat::Binary)?;

// Load model
let loaded_network = load_model("my_model.bin")?;

// Save with metadata
let metadata = ModelMetadata {
    name: "MNIST Classifier".to_string(),
    description: "CNN for digit classification".to_string(),
    training_info: Some(training_info),
    ..Default::default()
};
save_model_with_metadata(&network, "model_with_meta.json", ModelFormat::Json, &metadata)?;
```

## Performance

### Benchmarks

Performance comparison on common tasks (Intel i7-10700K, RTX 3080):

| Task | CPU (8 threads) | CUDA GPU | Speedup |
|------|----------------|----------|---------|
| Dense 1000x1000 MatMul | 12.5ms | 0.8ms | 15.6x |
| Conv2D 224x224x64 | 145ms | 8.2ms | 17.7x |
| MNIST Training (60k samples) | 45s | 3.2s | 14.1x |

### Optimization Tips

1. **Use appropriate batch sizes**: 32-256 for GPU, 8-32 for CPU
2. **Enable CPU optimizations**: Use `features = ["cpu-optimized"]` for Intel MKL
3. **Memory management**: Call `network.zero_grad()` regularly to free unused memory
4. **Data loading**: Use parallel data loading for large datasets
5. **Mixed precision**: Enable f16 on supported GPUs for 2x speedup

## Feature Flags

| Feature | Description | Example |
|---------|-------------|---------|
| `default` | CPU-optimized backend | `nnl = "0.1.0"` |
| `cuda` | NVIDIA CUDA support | `features = ["cuda"]` |
| `vulkan` | Vulkan compute support | `features = ["vulkan"]` |
| `rocm` | AMD ROCm support (experimental) | `features = ["rocm"]` |
| `cpu-optimized` | Intel MKL/OpenBLAS acceleration | `features = ["cpu-optimized"]` |
| `all-backends` | All GPU backends | `features = ["all-backends"]` |
| `examples` | Example binaries | `features = ["examples"]` |

## Troubleshooting

### Common Issues

**CUDA not found**
```bash
# Install CUDA toolkit 11.0+
# Add to ~/.bashrc:
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Vulkan not available**
```bash
# Install Vulkan drivers
sudo apt install vulkan-tools vulkan-loader-dev  # Ubuntu/Debian
# Verify: vulkaninfo
```

**Slow CPU performance**
```toml
# Enable CPU optimizations
nnl = { version = "0.1.0", features = ["cpu-optimized"] }
```

**Out of memory on GPU**
- Reduce batch size
- Use gradient accumulation
- Enable mixed precision training

## API Documentation

For detailed API documentation, see [docs.rs/nnl](https://docs.rs/nnl).

Key modules:
- [`tensor`](https://docs.rs/nnl/latest/nnl/tensor/) - Tensor operations and data structures
- [`network`](https://docs.rs/nnl/latest/nnl/network/) - Neural network building and training
- [`layers`](https://docs.rs/nnl/latest/nnl/layers/) - Layer implementations and configurations
- [`optimizers`](https://docs.rs/nnl/latest/nnl/optimizers/) - Optimization algorithms
- [`device`](https://docs.rs/nnl/latest/nnl/device/) - Device management and backend selection

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
```

## Roadmap

- [ ] **Distributed Training**: Multi-GPU and multi-node support
- [ ] **Mobile Deployment**: ARM optimization and model quantization
- [ ] **Web Assembly**: Browser-based inference
- [ ] **Model Zoo**: Pre-trained models for common tasks
- [ ] **Auto-ML**: Neural architecture search
- [ ] **Graph Optimization**: Operator fusion and memory optimization

## License

This project is dual-licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT License ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Inspired by PyTorch and TensorFlow APIs
- Built on excellent Rust ecosystem crates: `ndarray`, `rayon`, `vulkano`, `cudarc`
- Thanks to the Rust ML community and all contributors

---

**Questions?** Check out our [FAQ](docs/FAQ.md) or open an [issue](https://github.com/hotplugindev/nnl/issues).
