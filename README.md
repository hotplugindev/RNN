# RNN - Rust Neural Network Library

A high-performance neural network library for Rust with comprehensive GPU and CPU support.

[![Crates.io](https://img.shields.io/crates/v/rnn.svg)](https://crates.io/crates/rnn)
[![Documentation](https://docs.rs/rnn/badge.svg)](https://docs.rs/rnn)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](https://github.com/yourusername/rnn#license)

## Features

- 🚀 **Multi-backend Support**: NVIDIA CUDA, AMD ROCm/Vulkan, and optimized CPU execution
- 🎯 **Automatic Hardware Detection**: Seamlessly selects the best available compute backend
- 🧠 **Multiple Training Methods**: Backpropagation, Newton's method, and advanced optimizers
- 🏗️ **Flexible Architecture**: Support for both linear and convolutional networks
- 💾 **Model Persistence**: Import/export trained models to disk
- ⚡ **Production Ready**: Zero-copy operations, SIMD optimizations, and batched processing
- 🔧 **Comprehensive APIs**: Full control over every aspect of neural network training

## Quick Start

Add this to your `Cargo.toml`:

```toml
[dependencies]
rnn = "0.1.0"
```

### Basic Example

```rust
use rnn::prelude::*;

fn main() -> Result<()> {
    // Create a simple XOR network
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
        .optimizer(OptimizerConfig::Adam {
            learning_rate: 0.01,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        })
        .build()?;

    // Create training data
    let inputs = vec![
        Tensor::from_slice(&[0.0, 0.0], &[1, 2])?,
        Tensor::from_slice(&[0.0, 1.0], &[1, 2])?,
        Tensor::from_slice(&[1.0, 0.0], &[1, 2])?,
        Tensor::from_slice(&[1.0, 1.0], &[1, 2])?,
    ];
    
    let targets = vec![
        Tensor::from_slice(&[0.0], &[1, 1])?,
        Tensor::from_slice(&[1.0], &[1, 1])?,
        Tensor::from_slice(&[1.0], &[1, 1])?,
        Tensor::from_slice(&[0.0], &[1, 1])?,
    ];

    // Training configuration
    let config = TrainingConfig {
        epochs: 1000,
        batch_size: 4,
        verbose: true,
        early_stopping_patience: 50,
        early_stopping_threshold: 1e-6,
        lr_schedule: None,
        validation_split: 0.0,
        shuffle: true,
        random_seed: Some(42),
    };

    // Train the network
    let history = network.train(&inputs, &targets, &config)?;
    println!("Final loss: {:.6}", history.final_loss());

    // Make predictions
    let test_input = Tensor::from_slice(&[1.0, 0.0], &[1, 2])?;
    let prediction = network.forward(&test_input)?;
    println!("Prediction: {:.4}", prediction.to_vec()?[0]);

    Ok(())
}
```

## Complete API Reference

### Core Types

#### Result and Error Handling

```rust
pub type Result<T> = std::result::Result<T, RnnError>;

// Error types
pub enum RnnError {
    InvalidShape(String),
    DeviceError(String),
    ComputationError(String),
    IoError(String),
    TrainingError(String),
    // ... more variants
}
```

### Device Management

#### Device Types

```rust
pub enum DeviceType {
    CPU,
    CUDA,
    Vulkan,
    WebGPU,
}

pub struct DeviceInfo {
    pub name: String,
    pub device_type: DeviceType,
    pub memory_size: Option<u64>,
    pub compute_units: Option<u32>,
    pub supports_f16: bool,
    pub supports_f64: bool,
}
```

#### Device Creation and Management

```rust
impl Device {
    // Device creation
    pub fn auto_select() -> Result<Self>;
    pub fn cpu() -> Result<Self>;
    pub fn cuda() -> Result<Self>;
    pub fn vulkan() -> Result<Self>;
    pub fn webgpu() -> Result<Self>;

    // Device information
    pub fn info(&self) -> &DeviceInfo;
    pub fn device_type(&self) -> DeviceType;
    pub fn supports_f16(&self) -> bool;
    pub fn supports_f64(&self) -> bool;
    pub fn memory_size(&self) -> Option<u64>;
    pub fn synchronize(&self) -> Result<()>;
}

// Device utilities
pub mod device::utils {
    pub fn list_devices() -> Vec<DeviceInfo>;
    pub fn benchmark_devices() -> Result<Vec<(DeviceInfo, f64)>>;
}
```

### Tensors

#### Tensor Creation

```rust
impl Tensor {
    // Basic creation
    pub fn zeros(shape: &[usize]) -> Result<Self>;
    pub fn ones(shape: &[usize]) -> Result<Self>;
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Result<Self>;
    pub fn from_slice_on_device(data: &[f32], shape: &[usize], device: Device) -> Result<Self>;
    
    // Random initialization
    pub fn randn(shape: &[usize]) -> Result<Self>;
    pub fn uniform(shape: &[usize], low: f32, high: f32) -> Result<Self>;
    pub fn normal(shape: &[usize], mean: f32, std: f32) -> Result<Self>;
    
    // Special tensors
    pub fn eye(size: usize) -> Result<Self>;
    pub fn arange(start: f32, end: f32, step: f32) -> Result<Self>;
    pub fn linspace(start: f32, end: f32, steps: usize) -> Result<Self>;
}
```

#### Tensor Operations

```rust
impl Tensor {
    // Shape and metadata
    pub fn shape(&self) -> &[usize];
    pub fn ndim(&self) -> usize;
    pub fn size(&self) -> usize;
    pub fn is_contiguous(&self) -> bool;
    
    // Data access
    pub fn to_vec(&self) -> Result<Vec<f32>>;
    pub fn item(&self) -> Result<f32>; // For scalar tensors
    
    // Shape manipulation
    pub fn reshape(&self, shape: &[usize]) -> Result<Self>;
    pub fn transpose(&self) -> Result<Self>;
    pub fn permute(&self, dims: &[usize]) -> Result<Self>;
    pub fn squeeze(&self) -> Result<Self>;
    pub fn unsqueeze(&self, dim: usize) -> Result<Self>;
    
    // Slicing and indexing
    pub fn slice(&self, ranges: &[std::ops::Range<usize>]) -> Result<Self>;
    pub fn select(&self, dim: usize, index: usize) -> Result<Self>;
    
    // Arithmetic operations
    pub fn add(&self, other: &Tensor) -> Result<Self>;
    pub fn sub(&self, other: &Tensor) -> Result<Self>;
    pub fn mul(&self, other: &Tensor) -> Result<Self>;
    pub fn div(&self, other: &Tensor) -> Result<Self>;
    pub fn matmul(&self, other: &Tensor) -> Result<Self>;
    
    // Scalar operations
    pub fn add_scalar(&self, scalar: f32) -> Result<Self>;
    pub fn mul_scalar(&self, scalar: f32) -> Result<Self>;
    pub fn pow(&self, exponent: f32) -> Result<Self>;
    
    // Reductions
    pub fn sum(&self) -> Result<f32>;
    pub fn mean(&self) -> Result<f32>;
    pub fn max(&self) -> Result<f32>;
    pub fn min(&self) -> Result<f32>;
    pub fn sum_axis(&self, axis: usize) -> Result<Self>;
    pub fn mean_axis(&self, axis: usize) -> Result<Self>;
    
    // Comparisons
    pub fn eq(&self, other: &Tensor) -> Result<Self>;
    pub fn gt(&self, other: &Tensor) -> Result<Self>;
    pub fn lt(&self, other: &Tensor) -> Result<Self>;
    
    // Activation functions
    pub fn activation(&self, activation: Activation) -> Result<Self>;
    pub fn relu(&self) -> Result<Self>;
    pub fn sigmoid(&self) -> Result<Self>;
    pub fn tanh(&self) -> Result<Self>;
    pub fn softmax(&self, dim: usize) -> Result<Self>;
}
```

#### Operator Overloading

```rust
// Arithmetic operators are overloaded for convenience
let c = &a + &b;  // Addition
let c = &a - &b;  // Subtraction  
let c = &a * &b;  // Element-wise multiplication
let c = &a / &b;  // Element-wise division
```

### Activation Functions

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    Linear,
    ReLU,
    LeakyReLU(f32),
    Sigmoid,
    Tanh,
    Softmax,
    ELU(f32),
    Swish,
    GELU,
    Mish,
    PReLU(f32),
}

impl Activation {
    // Apply activation
    pub fn forward(&self, x: f32) -> f32;
    pub fn backward(&self, x: f32) -> f32;
    pub fn forward_slice(&self, input: &[f32], output: &mut [f32]) -> Result<()>;
    
    // Convenience constructors
    pub fn relu() -> Self;
    pub fn leaky_relu() -> Self;
    pub fn leaky_relu_with_slope(alpha: f32) -> Self;
    pub fn sigmoid() -> Self;
    pub fn tanh() -> Self;
    pub fn softmax() -> Self;
    pub fn elu() -> Self;
    pub fn elu_with_alpha(alpha: f32) -> Self;
    pub fn swish() -> Self;
    pub fn gelu() -> Self;
    pub fn mish() -> Self;
    pub fn prelu() -> Self;
    pub fn prelu_with_alpha(alpha: f32) -> Self;
    
    // Metadata
    pub fn name(&self) -> &'static str;
    pub fn has_parameters(&self) -> bool;
    pub fn parameters(&self) -> Vec<f32>;
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<()>;
}
```

### Loss Functions

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LossFunction {
    MeanSquaredError,
    MeanAbsoluteError,
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    SparseCategoricalCrossEntropy,
    Hinge,
    SquaredHinge,
    Huber { delta: f32 },
    FocalLoss { alpha: f32, gamma: f32 },
    KLDivergence,
}

impl LossFunction {
    // Compute loss
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32>;
    pub fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor>;
    
    // Convenience constructors
    pub fn mse() -> Self;
    pub fn mae() -> Self;
    pub fn binary_cross_entropy() -> Self;
    pub fn categorical_cross_entropy() -> Self;
    pub fn sparse_categorical_cross_entropy() -> Self;
    pub fn hinge() -> Self;
    pub fn squared_hinge() -> Self;
    pub fn huber(delta: f32) -> Self;
    pub fn focal_loss(alpha: f32, gamma: f32) -> Self;
    pub fn kl_divergence() -> Self;
    
    // Metadata
    pub fn name(&self) -> &'static str;
    pub fn requires_probabilities(&self) -> bool;
}
```

### Optimizers

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerConfig {
    SGD {
        learning_rate: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: bool,
    },
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
        amsgrad: bool,
    },
    AdaGrad {
        learning_rate: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
    },
    RMSprop {
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: bool,
    },
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
}

pub trait Optimizer {
    fn step(&mut self, gradients: &[Tensor]) -> Result<Vec<Tensor>>;
    fn zero_grad(&mut self);
    fn learning_rate(&self) -> f32;
    fn set_learning_rate(&mut self, lr: f32);
}
```

### Layers

#### Layer Configuration

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum LayerConfig {
    Dense {
        input_size: usize,
        output_size: usize,
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
    },
    Conv2D {
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
        activation: Activation,
        use_bias: bool,
        weight_init: WeightInit,
    },
    Dropout { dropout_rate: f32 },
    BatchNorm {
        num_features: usize,
        eps: f32,
        momentum: f32,
        affine: bool,
    },
    LayerNorm {
        normalized_shape: Vec<usize>,
        eps: f32,
        elementwise_affine: bool,
    },
    MaxPool2D {
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    },
    AvgPool2D {
        kernel_size: (usize, usize),
        stride: Option<(usize, usize)>,
        padding: (usize, usize),
    },
    Flatten {
        start_dim: usize,
        end_dim: Option<usize>,
    },
    Reshape { target_shape: Vec<usize> },
}
```

#### Weight Initialization

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum WeightInit {
    Zeros,
    Ones,
    Uniform { low: f32, high: f32 },
    Normal { mean: f32, std: f32 },
    Xavier,
    XavierUniform,
    Kaiming,
    KaimingUniform,
    HeNormal,
    HeUniform,
    LecunNormal,
    LecunUniform,
}
```

#### Layer Trait

```rust
pub trait Layer: Send + Sync {
    fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    fn backward(&mut self, grad_output: &Tensor) -> Result<Tensor>;
    fn parameters(&self) -> Vec<&Tensor>;
    fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    fn num_parameters(&self) -> usize;
    fn set_training(&mut self, training: bool);
    fn training(&self) -> bool;
}
```

### Neural Networks

#### Network Builder

```rust
impl NetworkBuilder {
    pub fn new() -> Self;
    
    // Layer management
    pub fn add_layer(self, config: LayerConfig) -> Self;
    pub fn add_layers(self, configs: Vec<LayerConfig>) -> Self;
    
    // Configuration
    pub fn loss(self, loss: LossFunction) -> Self;
    pub fn optimizer(self, optimizer: OptimizerConfig) -> Self;
    pub fn device(self, device: Device) -> Self;
    pub fn name(self, name: impl Into<String>) -> Self;
    pub fn description(self, description: impl Into<String>) -> Self;
    
    // Build the network
    pub fn build(self) -> Result<Network>;
    
    // Validation
    pub fn validate_architecture(&self) -> Result<()>;
    pub fn estimate_memory_usage(&self) -> Result<usize>;
}
```

#### Network Operations

```rust
impl Network {
    // Inference
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor>;
    pub fn predict(&mut self, input: &Tensor) -> Result<Tensor>;
    pub fn predict_batch(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>>;
    
    // Training
    pub fn train(&mut self, inputs: &[Tensor], targets: &[Tensor], config: &TrainingConfig) -> Result<TrainingHistory>;
    pub fn train_epoch(&mut self, inputs: &[Tensor], targets: &[Tensor], config: &TrainingConfig) -> Result<TrainingMetrics>;
    pub fn evaluate(&mut self, inputs: &[Tensor], targets: &[Tensor]) -> Result<TrainingMetrics>;
    
    // State management
    pub fn set_training(&mut self, training: bool);
    pub fn training(&self) -> bool;
    pub fn zero_grad(&mut self);
    
    // Network information
    pub fn summary(&self) -> String;
    pub fn num_parameters(&self) -> usize;
    pub fn memory_usage(&self) -> usize;
    pub fn metrics(&self) -> &NetworkMetrics;
    
    // Parameter access
    pub fn parameters(&self) -> Vec<&Tensor>;
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor>;
    pub fn named_parameters(&self) -> Vec<(String, &Tensor)>;
    
    // Device management
    pub fn device(&self) -> &Device;
    pub fn to_device(&mut self, device: Device) -> Result<()>;
}
```

### Training

#### Training Configuration

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub verbose: bool,
    pub early_stopping_patience: usize,
    pub early_stopping_threshold: f32,
    pub lr_schedule: Option<LearningRateSchedule>,
    pub validation_split: f32,
    pub shuffle: bool,
    pub random_seed: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    StepLR { step_size: usize, gamma: f32 },
    ExponentialLR { gamma: f32 },
    ReduceOnPlateau {
        factor: f32,
        patience: usize,
        threshold: f32,
        min_lr: f32,
    },
    CosineAnnealingLR { t_max: usize, eta_min: f32 },
    PolynomialLR { total_epochs: usize, power: f32 },
}
```

#### Training Metrics and History

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    pub loss: f32,
    pub accuracy: f32,
    pub learning_rate: f32,
    pub epoch_time_ms: f32,
}

impl TrainingHistory {
    // Epoch management
    pub fn add_epoch(&mut self, metrics: TrainingMetrics);
    pub fn get_epoch(&self, epoch: usize) -> Option<&TrainingMetrics>;
    pub fn all_epochs(&self) -> &[TrainingMetrics];
    pub fn latest(&self) -> Option<&TrainingMetrics>;
    
    // Statistics
    pub fn epochs(&self) -> usize;
    pub fn final_loss(&self) -> f32;
    pub fn best_loss(&self) -> f32;
    pub fn best_loss_epoch(&self) -> usize;
    pub fn best_accuracy(&self) -> f32;
    pub fn best_accuracy_epoch(&self) -> usize;
    pub fn average_epoch_time(&self) -> f32;
    pub fn total_training_time(&self) -> f32;
    
    // Analysis
    pub fn summary(&self) -> TrainingSummary;
    pub fn plot_data(&self) -> PlotData;
    pub fn convergence_analysis(&self) -> ConvergenceAnalysis;
}
```

### Model I/O

#### Model Serialization

```rust
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ModelFormat {
    Binary,      // Fast binary format
    Json,        // Human-readable JSON
    MessagePack, // Compact binary format
}

// Model metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub description: String,
    pub created_at: String,
    pub modified_at: String,
    pub training_info: TrainingInfo,
    pub metrics: HashMap<String, f32>,
    pub custom: HashMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingInfo {
    pub epochs_trained: usize,
    pub final_loss: f32,
    pub best_accuracy: f32,
    pub training_time_seconds: f32,
    pub dataset_info: Option<DatasetInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub train_samples: usize,
    pub val_samples: Option<usize>,
    pub test_samples: Option<usize>,
    pub num_classes: Option<usize>,
}
```

#### Save and Load Functions

```rust
// Save model
pub fn save_model(
    network: &Network,
    path: &str,
    format: ModelFormat,
    metadata: Option<ModelMetadata>,
) -> Result<()>;

// Load model
pub fn load_model(
    path: &str,
    format: ModelFormat,
) -> Result<Network>;

// Metadata operations
pub fn load_metadata(path: &str, format: ModelFormat) -> Result<ModelMetadata>;
pub fn update_metadata(
    path: &str,
    format: ModelFormat,
    metadata: ModelMetadata,
) -> Result<()>;
```

### Utilities

```rust
pub mod utils {
    // Data preprocessing
    pub fn normalize(data: &mut [f32], mean: f32, std: f32);
    pub fn standardize(data: &mut [f32]);
    pub fn one_hot_encode(labels: &[usize], num_classes: usize) -> Vec<Vec<f32>>;
    
    // Dataset utilities
    pub fn train_test_split<T>(
        data: Vec<T>,
        test_size: f32,
        shuffle: bool,
        random_seed: Option<u64>,
    ) -> (Vec<T>, Vec<T>);
    
    pub fn create_batches<T>(data: Vec<T>, batch_size: usize) -> Vec<Vec<T>>;
    
    // Performance monitoring
    pub struct PerformanceMonitor;
    impl PerformanceMonitor {
        pub fn new() -> Self;
        pub fn start_timer(&mut self, name: &str);
        pub fn end_timer(&mut self, name: &str) -> f32;
        pub fn get_stats(&self) -> HashMap<String, f32>;
    }
    
    // Memory management
    pub fn get_memory_usage() -> Result<usize>;
    pub fn clear_cache();
    
    // Debugging
    pub fn set_debug_mode(enabled: bool);
    pub fn print_tensor_stats(tensor: &Tensor);
    pub fn validate_gradients(network: &Network) -> Result<bool>;
}
```

## Examples

The library includes comprehensive examples in the `examples/` directory:

- **`basic_usage.rs`** - Basic library usage and tensor operations
- **`xor_cpu.rs`** - Simple XOR problem solved on CPU
- **`xor_gpu.rs`** - XOR problem with GPU acceleration
- **`mnist_cnn.rs`** - CNN for MNIST digit classification

### Running Examples

```bash
# Basic usage example
cargo run --example basic_usage

# XOR problem on CPU
cargo run --example xor_cpu

# XOR problem on GPU (requires compatible GPU)
cargo run --example xor_gpu

# MNIST CNN (requires dataset download)
cargo run --example mnist_cnn
```

## Feature Flags

```toml
[dependencies]
rnn = { version = "0.1.0", features = ["cuda", "intel-mkl"] }
```

Available features:

- **`default`** - CPU optimizations and examples
- **`cuda`** - NVIDIA CUDA support
- **`rocm`** - AMD ROCm/HIP support  
- **`cpu-optimized`** - OpenBLAS CPU optimizations
- **`intel-mkl`** - Intel MKL optimizations
- **`examples`** - Include example dependencies
- **`all-backends`** - Enable all compute backends

## GPU Support

### CUDA

Requires NVIDIA GPU with CUDA Toolkit 11.0+:

```toml
rnn = { version = "0.1.0", features = ["cuda"] }
```

### Vulkan

Cross-platform GPU support:

```toml
rnn = { version = "0.1.0", default-features = false, features = ["vulkan"] }
```

### WebGPU

For web deployment and cross-platform compatibility:

```toml
rnn = { version = "0.1.0", default-features = false, features = ["webgpu"] }
```

## Performance Tips

1. **Use appropriate device**: GPU for large models, CPU for small ones
2. **Batch processing**: Larger batches improve GPU utilization
3. **Memory management**: Reuse tensors when possible
4. **Mixed precision**: Use f16 on supported hardware
5. **Parallel data loading**: Use multiple threads for data preprocessing

## Architecture Support

- **x86_64**: Full support with optimizations
- **ARM64**: Full support (Apple Silicon, ARM servers)
- **WebAssembly**: Basic support via WebGPU

## Minimum Supported Rust Version (MSRV)

Rust 1.70 or later.

## Contributing

Contributions are welcome! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under either of

- Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Acknowledgments

- Inspired by PyTorch and TensorFlow APIs
- Built on top of excellent Rust ecosystem crates
- Thanks to all contributors and the Rust ML community