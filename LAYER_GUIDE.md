# Neural Network Layer Guide

This guide provides comprehensive documentation for using layers in the NNL library, including shape calculations, configuration guidelines, and common pitfalls.

## Table of Contents
- [Shape Flow Basics](#shape-flow-basics)
- [Convolutional Layers](#convolutional-layers)
- [Pooling Layers](#pooling-layers)
- [Dense Layers](#dense-layers)
- [Normalization Layers](#normalization-layers)
- [Utility Layers](#utility-layers)
- [Common Patterns](#common-patterns)
- [Troubleshooting](#troubleshooting)

## Shape Flow Basics

Understanding how tensor shapes flow through your network is crucial for building working models. Each layer transforms the input shape according to its specific rules.

### Tensor Format
- **4D tensors**: `[batch_size, channels, height, width]` (for images)
- **2D tensors**: `[batch_size, features]` (for dense layers)
- **3D tensors**: `[batch_size, sequence_length, features]` (for sequences)

### Shape Calculation Functions
```rust
// Calculate convolution output size
fn conv_output_size(input_size: usize, kernel_size: usize, stride: usize, padding: usize, dilation: usize) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

// Calculate pooling output size
fn pool_output_size(input_size: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    (input_size + 2 * padding - kernel_size) / stride + 1
}
```

## Convolutional Layers

### Conv2D Configuration

```rust
LayerConfig::Conv2D {
    in_channels: 3,        // Input channels (e.g., 3 for RGB images)
    out_channels: 64,      // Output channels (number of filters)
    kernel_size: (3, 3),   // Filter size (height, width)
    stride: (1, 1),        // Step size (height, width)
    padding: (1, 1),       // Padding (height, width)
    dilation: (1, 1),      // Dilation factor (height, width)
    activation: Activation::ReLU,
    use_bias: true,
    weight_init: WeightInit::HeNormal,
}
```

### Shape Calculation Example

**Input**: `[1, 3, 32, 32]` (CIFAR-10 image)
**Conv2D**: `kernel_size=(3,3), stride=(1,1), padding=(1,1)`

```
Output Height = (32 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 32
Output Width  = (32 + 2*1 - 1*(3-1) - 1) / 1 + 1 = 32
Output Shape  = [1, 64, 32, 32]
```

### Common Patterns

#### Same Padding (preserve spatial dimensions)
```rust
LayerConfig::Conv2D {
    kernel_size: (3, 3),
    stride: (1, 1),
    padding: (1, 1),    // padding = (kernel_size - 1) / 2
    // ...
}
```

#### Downsampling (reduce spatial dimensions by 2)
```rust
LayerConfig::Conv2D {
    kernel_size: (3, 3),
    stride: (2, 2),     // stride = 2 for 2x downsampling
    padding: (1, 1),
    // ...
}
```

## Pooling Layers

### MaxPool2D and AvgPool2D

```rust
LayerConfig::MaxPool2D {
    kernel_size: (2, 2),     // Pooling window size
    stride: Some((2, 2)),    // Step size (None = kernel_size)
    padding: (0, 0),         // Padding
}
```

### Global Average Pooling

For global average pooling, use `stride: None` to make stride equal to kernel_size:

```rust
// Convert [batch, channels, H, W] -> [batch, channels, 1, 1]
LayerConfig::AvgPool2D {
    kernel_size: (H, W),     // Same as input spatial size
    stride: None,            // stride = kernel_size for global pooling
    padding: (0, 0),
}
```

### Shape Examples

#### Regular Pooling
**Input**: `[1, 64, 32, 32]`
**MaxPool2D**: `kernel_size=(2,2), stride=Some((2,2))`
**Output**: `[1, 64, 16, 16]`

#### Global Average Pooling
**Input**: `[1, 512, 4, 4]`
**AvgPool2D**: `kernel_size=(4,4), stride=None`
**Output**: `[1, 512, 1, 1]`

⚠️ **Common Mistake**: Using `stride: Some((1, 1))` for global pooling will not reduce spatial dimensions properly.

## Dense Layers

### Configuration

```rust
LayerConfig::Dense {
    input_size: 512,          // Must match flattened input size
    output_size: 256,         // Number of output neurons
    activation: Activation::ReLU,
    use_bias: true,
    weight_init: WeightInit::HeNormal,
}
```

### Shape Calculation

Dense layers expect 2D input: `[batch_size, input_size]`

**Example**:
- Input: `[batch_size, 512]`
- Dense layer: `input_size=512, output_size=256`
- Output: `[batch_size, 256]`

### Calculating Input Size After Convolutions

After convolution and pooling layers, you need to flatten to feed into dense layers:

```rust
// CIFAR-10 example shape flow:
// [1, 3, 32, 32]    <- Input image
// [1, 64, 32, 32]   <- After Conv2D (same padding)
// [1, 128, 16, 16]  <- After Conv2D (stride=2)
// [1, 256, 8, 8]    <- After Conv2D (stride=2)
// [1, 512, 4, 4]    <- After Conv2D (stride=2)
// [1, 512, 1, 1]    <- After AvgPool2D global pooling
// [1, 512]          <- After Flatten
```

The Dense layer input_size should be: `512 * 1 * 1 = 512`

## Normalization Layers

### BatchNorm

```rust
LayerConfig::BatchNorm {
    num_features: 64,         // Same as input channels
    eps: 1e-5,
    momentum: 0.1,
    affine: true,             // Enable learnable parameters
}
```

**Shape**: Input and output shapes are identical.

### LayerNorm

```rust
LayerConfig::LayerNorm {
    normalized_shape: vec![512],  // Shape of dimensions to normalize
    eps: 1e-5,
    elementwise_affine: true,
}
```

## Utility Layers

### Flatten

```rust
LayerConfig::Flatten {
    start_dim: 1,            // Start flattening from dimension 1
    end_dim: None,           // Flatten to the end (None = last dim)
}
```

**Shape Examples**:
- Input: `[1, 512, 4, 4]`
- Flatten: `start_dim=1, end_dim=None`
- Output: `[1, 8192]` (1 * 512 * 4 * 4 = 8192)

### Dropout

```rust
LayerConfig::Dropout {
    dropout_rate: 0.5,       // Probability of setting elements to zero
}
```

**Shape**: Input and output shapes are identical.

## Common Patterns

### CIFAR-10 CNN Architecture

```rust
NetworkBuilder::new()
    // Input: [batch, 3, 32, 32]
    .add_layer(LayerConfig::Conv2D {
        in_channels: 3, out_channels: 64,
        kernel_size: (3, 3), stride: (1, 1), padding: (1, 1),
        activation: Activation::ReLU, use_bias: true,
        weight_init: WeightInit::HeNormal,
    })
    // Shape: [batch, 64, 32, 32]
    
    .add_layer(LayerConfig::Conv2D {
        in_channels: 64, out_channels: 128,
        kernel_size: (3, 3), stride: (2, 2), padding: (1, 1),
        activation: Activation::ReLU, use_bias: true,
        weight_init: WeightInit::HeNormal,
    })
    // Shape: [batch, 128, 16, 16]
    
    .add_layer(LayerConfig::Conv2D {
        in_channels: 128, out_channels: 256,
        kernel_size: (3, 3), stride: (2, 2), padding: (1, 1),
        activation: Activation::ReLU, use_bias: true,
        weight_init: WeightInit::HeNormal,
    })
    // Shape: [batch, 256, 8, 8]
    
    .add_layer(LayerConfig::Conv2D {
        in_channels: 256, out_channels: 512,
        kernel_size: (3, 3), stride: (2, 2), padding: (1, 1),
        activation: Activation::ReLU, use_bias: true,
        weight_init: WeightInit::HeNormal,
    })
    // Shape: [batch, 512, 4, 4]
    
    .add_layer(LayerConfig::AvgPool2D {
        kernel_size: (4, 4),
        stride: None,  // Global average pooling
        padding: (0, 0),
    })
    // Shape: [batch, 512, 1, 1]
    
    .add_layer(LayerConfig::Flatten {
        start_dim: 1,
        end_dim: None,
    })
    // Shape: [batch, 512]
    
    .add_layer(LayerConfig::Dense {
        input_size: 512,  // Must match flattened size
        output_size: 10,  // Number of classes
        activation: Activation::Softmax,
        use_bias: true,
        weight_init: WeightInit::Xavier,
    })
    // Shape: [batch, 10]
```

### ResNet-style Block

```rust
// Residual block pattern
.add_layer(LayerConfig::Conv2D { /* ... */ })
.add_layer(LayerConfig::BatchNorm { /* ... */ })
.add_layer(LayerConfig::Conv2D { 
    activation: Activation::Linear, // No activation here
    /* ... */ 
})
.add_layer(LayerConfig::BatchNorm { /* ... */ })
// Add skip connection here (not shown - would need custom layer)
.add_layer(LayerConfig::Conv2D {
    kernel_size: (1, 1), // Activation via 1x1 conv
    activation: Activation::ReLU,
    /* ... */
})
```

## Troubleshooting

### Common Errors and Solutions

#### 1. Shape Mismatch in Dense Layer

**Error**: `Shape mismatch: expected [512], got [8192]`

**Cause**: The flattened tensor size doesn't match the Dense layer's `input_size`.

**Solution**: Calculate the correct flattened size:
```rust
// After convolutions ending with shape [batch, 512, 4, 4]:
// Flattened size = 512 * 4 * 4 = 8192
LayerConfig::Dense {
    input_size: 8192,  // Not 512!
    // ...
}
```

Or use proper global average pooling to get [batch, 512, 1, 1] → [batch, 512].

#### 2. Global Average Pooling Not Working

**Problem**: Using `stride: Some((1, 1))` doesn't reduce spatial dimensions.

**Solution**: Use `stride: None` for global average pooling:
```rust
LayerConfig::AvgPool2D {
    kernel_size: (4, 4),  // Same as input spatial size
    stride: None,         // This makes stride = kernel_size
    padding: (0, 0),
}
```

#### 3. Negative Output Dimensions

**Error**: Calculation results in negative or zero output dimensions.

**Causes**:
- Kernel size larger than input size
- Insufficient padding
- Stride too large

**Solution**: Adjust parameters:
```rust
// For input size 32x32 with kernel 5x5:
LayerConfig::Conv2D {
    kernel_size: (5, 5),
    stride: (1, 1),
    padding: (2, 2),  // padding = (kernel_size - 1) / 2
    // ...
}
```

#### 4. Channel Mismatch

**Error**: `expected [64], actual [128]`

**Cause**: Output channels of one layer don't match input channels of the next.

**Solution**: Ensure channel continuity:
```rust
.add_layer(LayerConfig::Conv2D {
    in_channels: 64,
    out_channels: 128,  // Output 128 channels
    // ...
})
.add_layer(LayerConfig::Conv2D {
    in_channels: 128,   // Input must be 128 channels
    out_channels: 256,
    // ...
})
```

### Debugging Shape Flow

Add debug prints to trace shapes:
```rust
// Test network with dummy input
let test_input = Tensor::zeros(&[1, 3, 32, 32])?;
let output = network.forward(&test_input)?;
println!("Final output shape: {:?}", output.shape());
```

Or create a simple debug network that stops at each layer to check intermediate shapes.

### Best Practices

1. **Plan your architecture**: Calculate shapes on paper before implementing
2. **Use same padding**: For preserving spatial dimensions in early layers
3. **Power-of-2 channels**: Use 32, 64, 128, 256, 512 for better hardware utilization
4. **Gradual downsampling**: Reduce spatial dimensions while increasing channels
5. **Global pooling**: Prefer global average pooling over large dense layers
6. **Batch normalization**: Add after Conv2D layers (before activation)
7. **Proper initialization**: Use HeNormal for ReLU, Xavier for Sigmoid/Tanh

### Weight Initialization Guidelines

```rust
// For ReLU activations
weight_init: WeightInit::HeNormal,

// For Sigmoid/Tanh activations  
weight_init: WeightInit::Xavier,

// For linear layers (no activation)
weight_init: WeightInit::Xavier,

// For final classification layer
weight_init: WeightInit::Xavier,
```

This guide should help you build robust neural networks with proper shape flow and avoid common pitfalls. Remember to always verify your shape calculations and test with dummy inputs before training.