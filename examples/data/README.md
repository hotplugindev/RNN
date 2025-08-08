# Examples Data Directory

This directory contains data files and utilities for the NNL library examples.

## Contents

- `README.md` - This file
- Sample datasets and configuration files used by examples
- Data loading utilities and preprocessing scripts

## Usage

The examples in the parent directory will automatically look for data files in this directory. Most examples generate synthetic data for demonstration purposes, but this directory can be used to store:

- Real datasets (MNIST, CIFAR-10, etc.)
- Pre-trained model weights
- Configuration files
- Benchmark results

## Data Sources

For real-world datasets, please download them from their official sources:

- **MNIST**: http://yann.lecun.com/exdb/mnist/
- **CIFAR-10**: https://www.cs.toronto.edu/~kriz/cifar.html
- **Fashion-MNIST**: https://github.com/zalandoresearch/fashion-mnist

## File Formats

The NNL library supports various data formats:
- Raw binary data (.bin)
- JSON (.json)
- MessagePack (.msgpack)
- NumPy arrays (via conversion)

## Example Usage

```rust
// Load data from this directory
let data_path = "examples/data/mnist_train.bin";
let dataset = load_dataset(data_path)?;
```

## Note

Example scripts generate synthetic data by default to ensure they run without requiring large dataset downloads. For production use, replace with real datasets.
