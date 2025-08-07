//! Input/Output operations for neural networks.
//!
//! This module provides functionality for importing and exporting neural networks
//! in various formats, including JSON, binary, ONNX-compatible, and other common formats.

use crate::error::{NetworkError, Result};
use crate::network::{Network, NetworkMetadata};
use ndarray::{Array1, Array2};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

/// Supported export/import formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkFormat {
    /// JSON format (human-readable)
    Json,
    /// Binary format (compact)
    Binary,
    /// HDF5 format (hierarchical data)
    Hdf5,
    /// NumPy format (.npz for weights)
    Numpy,
    /// ONNX-compatible format
    Onnx,
    /// Custom RNN format
    Rnn,
}

/// Export configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportConfig {
    /// Format to export to
    pub format: NetworkFormat,
    /// Include training history
    pub include_history: bool,
    /// Include metadata
    pub include_metadata: bool,
    /// Compress the output
    pub compress: bool,
    /// Export only weights (exclude architecture)
    pub weights_only: bool,
    /// Custom export options
    pub custom_options: HashMap<String, serde_json::Value>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: NetworkFormat::Json,
            include_history: true,
            include_metadata: true,
            compress: false,
            weights_only: false,
            custom_options: HashMap::new(),
        }
    }
}

/// Import configuration options.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportConfig {
    /// Expected format
    pub format: NetworkFormat,
    /// Strict mode (fail on missing fields)
    pub strict: bool,
    /// Map layer names if different
    pub layer_name_mapping: HashMap<String, String>,
    /// Custom import options
    pub custom_options: HashMap<String, serde_json::Value>,
}

impl Default for ImportConfig {
    fn default() -> Self {
        Self {
            format: NetworkFormat::Json,
            strict: true,
            layer_name_mapping: HashMap::new(),
            custom_options: HashMap::new(),
        }
    }
}

/// Network export functionality.
pub struct NetworkExporter;

impl NetworkExporter {
    /// Export network to file with specified configuration.
    pub fn export<P: AsRef<Path>>(network: &Network, path: P, config: &ExportConfig) -> Result<()> {
        match config.format {
            NetworkFormat::Json => Self::export_json(network, path, config),
            NetworkFormat::Binary => Self::export_binary(network, path, config),
            NetworkFormat::Hdf5 => Self::export_hdf5(network, path, config),
            NetworkFormat::Numpy => Self::export_numpy(network, path, config),
            NetworkFormat::Onnx => Self::export_onnx(network, path, config),
            NetworkFormat::Rnn => Self::export_rnn(network, path, config),
        }
    }

    /// Export network in JSON format.
    fn export_json<P: AsRef<Path>>(
        network: &Network,
        path: P,
        config: &ExportConfig,
    ) -> Result<()> {
        let export_data = Self::prepare_export_data(network, config)?;
        let file = File::create(path).map_err(NetworkError::from)?;
        let writer = BufWriter::new(file);

        if config.compress {
            let compressed_data = Self::compress_data(&export_data)?;
            serde_json::to_writer_pretty(writer, &compressed_data).map_err(NetworkError::from)?;
        } else {
            serde_json::to_writer_pretty(writer, &export_data).map_err(NetworkError::from)?;
        }

        Ok(())
    }

    /// Export network in binary format.
    fn export_binary<P: AsRef<Path>>(
        network: &Network,
        path: P,
        config: &ExportConfig,
    ) -> Result<()> {
        let export_data = Self::prepare_export_data(network, config)?;
        let file = File::create(path).map_err(NetworkError::from)?;
        let writer = BufWriter::new(file);

        bincode::serialize_into(writer, &export_data).map_err(NetworkError::from)?;
        Ok(())
    }

    /// Export network in HDF5 format (placeholder implementation).
    fn export_hdf5<P: AsRef<Path>>(
        _network: &Network,
        _path: P,
        _config: &ExportConfig,
    ) -> Result<()> {
        Err(NetworkError::configuration(
            "HDF5 export not yet implemented",
        ))
    }

    /// Export network weights in NumPy format.
    fn export_numpy<P: AsRef<Path>>(
        network: &Network,
        path: P,
        _config: &ExportConfig,
    ) -> Result<()> {
        let path_ref = path.as_ref();
        let base_name = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("network");

        // Export each layer's weights as separate .npy files
        for (i, layer) in network.layers.iter().enumerate() {
            let weight_path = path_ref
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("{}_layer_{}_weights.csv", base_name, i));

            Self::save_array_as_csv(&layer.weights, &weight_path)?;

            if layer.use_bias {
                let bias_path = path_ref
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(format!("{}_layer_{}_bias.csv", base_name, i));

                // Convert bias to 2D for consistent saving
                let bias_2d = layer.bias.clone().into_shape((layer.bias.len(), 1))?;
                Self::save_array_as_csv(&bias_2d, &bias_path)?;
            }
        }

        // Save network architecture info
        let info_path = path_ref
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}_info.json", base_name));

        let info = NetworkArchitectureInfo {
            input_dim: network.input_dim,
            output_dim: network.output_dim,
            layers: network
                .layers
                .iter()
                .map(|layer| LayerInfo {
                    layer_type: format!("{:?}", layer.layer_type),
                    input_dim: layer.input_dim,
                    output_dim: layer.output_dim,
                    activation: layer.activation.name().to_string(),
                    use_bias: layer.use_bias,
                })
                .collect(),
            loss_function: network.loss_function.name().to_string(),
        };

        let file = File::create(info_path).map_err(NetworkError::from)?;
        serde_json::to_writer_pretty(file, &info).map_err(NetworkError::from)?;

        Ok(())
    }

    /// Export network in ONNX-compatible format (simplified).
    fn export_onnx<P: AsRef<Path>>(
        _network: &Network,
        _path: P,
        _config: &ExportConfig,
    ) -> Result<()> {
        Err(NetworkError::configuration(
            "ONNX export not yet implemented",
        ))
    }

    /// Export network in custom RNN format.
    fn export_rnn<P: AsRef<Path>>(network: &Network, path: P, config: &ExportConfig) -> Result<()> {
        let rnn_data = RnnFormat {
            version: "1.0".to_string(),
            metadata: if config.include_metadata {
                Some(network.metadata.clone())
            } else {
                None
            },
            architecture: NetworkArchitecture {
                input_dim: network.input_dim,
                output_dim: network.output_dim,
                layers: network
                    .layers
                    .iter()
                    .map(|layer| RnnLayer {
                        layer_type: format!("{:?}", layer.layer_type),
                        input_dim: layer.input_dim,
                        output_dim: layer.output_dim,
                        activation: layer.activation.name().to_string(),
                        use_bias: layer.use_bias,
                        trainable: layer.trainable,
                        weights: if config.weights_only {
                            layer.weights.clone()
                        } else {
                            layer.weights.clone()
                        },
                        bias: if layer.use_bias {
                            Some(layer.bias.clone())
                        } else {
                            None
                        },
                    })
                    .collect(),
                loss_function: network.loss_function.name().to_string(),
                optimizer_type: format!("{:?}", network.optimizer.optimizer_type),
            },
            training_history: if config.include_history {
                Some(network.history.clone())
            } else {
                None
            },
        };

        let file = File::create(path).map_err(NetworkError::from)?;
        let writer = BufWriter::new(file);

        if config.compress {
            // Simple compression placeholder
            bincode::serialize_into(writer, &rnn_data).map_err(NetworkError::from)?;
        } else {
            serde_json::to_writer_pretty(writer, &rnn_data).map_err(NetworkError::from)?;
        }

        Ok(())
    }

    /// Prepare export data based on configuration.
    fn prepare_export_data(network: &Network, config: &ExportConfig) -> Result<ExportData> {
        Ok(ExportData {
            metadata: if config.include_metadata {
                Some(network.metadata.clone())
            } else {
                None
            },
            architecture: if !config.weights_only {
                Some(NetworkArchitecture {
                    input_dim: network.input_dim,
                    output_dim: network.output_dim,
                    layers: network
                        .layers
                        .iter()
                        .map(|layer| RnnLayer {
                            layer_type: format!("{:?}", layer.layer_type),
                            input_dim: layer.input_dim,
                            output_dim: layer.output_dim,
                            activation: layer.activation.name().to_string(),
                            use_bias: layer.use_bias,
                            trainable: layer.trainable,
                            weights: layer.weights.clone(),
                            bias: if layer.use_bias {
                                Some(layer.bias.clone())
                            } else {
                                None
                            },
                        })
                        .collect(),
                    loss_function: network.loss_function.name().to_string(),
                    optimizer_type: format!("{:?}", network.optimizer.optimizer_type),
                })
            } else {
                None
            },
            weights: network.get_weights(),
            history: if config.include_history {
                Some(network.history.clone())
            } else {
                None
            },
        })
    }

    /// Save 2D array as CSV file.
    fn save_array_as_csv<P: AsRef<Path>>(array: &Array2<f64>, path: P) -> Result<()> {
        let mut file = File::create(path).map_err(NetworkError::from)?;

        for row in array.rows() {
            let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            writeln!(file, "{}", row_str.join(",")).map_err(NetworkError::from)?;
        }

        Ok(())
    }

    /// Compress data (placeholder implementation).
    fn compress_data(data: &ExportData) -> Result<CompressedExportData> {
        // Simple compression placeholder - in practice, use proper compression
        let serialized = bincode::serialize(data).map_err(NetworkError::from)?;
        Ok(CompressedExportData {
            data: serialized,
            compression_method: "bincode".to_string(),
        })
    }
}

/// Network import functionality.
pub struct NetworkImporter;

impl NetworkImporter {
    /// Import network from file with specified configuration.
    pub fn import<P: AsRef<Path>>(path: P, config: &ImportConfig) -> Result<Network> {
        match config.format {
            NetworkFormat::Json => Self::import_json(path, config),
            NetworkFormat::Binary => Self::import_binary(path, config),
            NetworkFormat::Hdf5 => Self::import_hdf5(path, config),
            NetworkFormat::Numpy => Self::import_numpy(path, config),
            NetworkFormat::Onnx => Self::import_onnx(path, config),
            NetworkFormat::Rnn => Self::import_rnn(path, config),
        }
    }

    /// Import network from JSON format.
    fn import_json<P: AsRef<Path>>(path: P, _config: &ImportConfig) -> Result<Network> {
        let file = File::open(path).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);
        let network: Network = serde_json::from_reader(reader).map_err(NetworkError::from)?;
        Ok(network)
    }

    /// Import network from binary format.
    fn import_binary<P: AsRef<Path>>(path: P, _config: &ImportConfig) -> Result<Network> {
        let file = File::open(path).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);
        let network: Network = bincode::deserialize_from(reader).map_err(NetworkError::from)?;
        Ok(network)
    }

    /// Import network from HDF5 format (placeholder).
    fn import_hdf5<P: AsRef<Path>>(_path: P, _config: &ImportConfig) -> Result<Network> {
        Err(NetworkError::configuration(
            "HDF5 import not yet implemented",
        ))
    }

    /// Import network from NumPy format.
    fn import_numpy<P: AsRef<Path>>(path: P, _config: &ImportConfig) -> Result<Network> {
        let path_ref = path.as_ref();
        let base_name = path_ref
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("network");

        // Load architecture info
        let info_path = path_ref
            .parent()
            .unwrap_or(Path::new("."))
            .join(format!("{}_info.json", base_name));

        let file = File::open(info_path).map_err(NetworkError::from)?;
        let info: NetworkArchitectureInfo =
            serde_json::from_reader(file).map_err(NetworkError::from)?;

        // Create network builder
        let mut builder = crate::network::NetworkBuilder::new(info.input_dim)?;

        // Add layers based on info
        for (i, layer_info) in info.layers.iter().enumerate() {
            let activation = Self::parse_activation(&layer_info.activation)?;

            let layer_builder = crate::layer::LayerBuilder::dense(layer_info.output_dim)
                .activation(activation)
                .use_bias(layer_info.use_bias);

            builder = builder.add_layer(layer_builder);
        }

        // Set loss function
        let loss_function = Self::parse_loss_function(&info.loss_function)?;
        builder = builder.loss(loss_function);

        let mut network = builder.build()?;

        // Load weights
        for (i, layer) in network.layers.iter_mut().enumerate() {
            let weight_path = path_ref
                .parent()
                .unwrap_or(Path::new("."))
                .join(format!("{}_layer_{}_weights.csv", base_name, i));

            layer.weights = Self::load_array_from_csv(&weight_path)?;

            if layer.use_bias {
                let bias_path = path_ref
                    .parent()
                    .unwrap_or(Path::new("."))
                    .join(format!("{}_layer_{}_bias.csv", base_name, i));

                let bias_2d = Self::load_array_from_csv(&bias_path)?;
                let bias_len = bias_2d.len();
                layer.bias = bias_2d.into_shape(bias_len)?;
            }
        }

        Ok(network)
    }

    /// Import network from ONNX format (placeholder).
    fn import_onnx<P: AsRef<Path>>(_path: P, _config: &ImportConfig) -> Result<Network> {
        Err(NetworkError::configuration(
            "ONNX import not yet implemented",
        ))
    }

    /// Import network from custom RNN format.
    fn import_rnn<P: AsRef<Path>>(path: P, _config: &ImportConfig) -> Result<Network> {
        let path_ref = path.as_ref();
        let file = File::open(path_ref).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);

        // Try JSON first, then binary
        let mut content = String::new();
        let mut file_reader = BufReader::new(File::open(path_ref).map_err(NetworkError::from)?);
        if file_reader.read_to_string(&mut content).is_ok() {
            // Try JSON
            if let Ok(rnn_data) = serde_json::from_str::<RnnFormat>(&content) {
                return Self::build_network_from_rnn_format(&rnn_data);
            }
        }

        // Try binary
        let rnn_data: RnnFormat = bincode::deserialize_from(reader).map_err(NetworkError::from)?;
        Self::build_network_from_rnn_format(&rnn_data)
    }

    /// Build network from RNN format data.
    fn build_network_from_rnn_format(data: &RnnFormat) -> Result<Network> {
        let mut builder = crate::network::NetworkBuilder::new(data.architecture.input_dim)?;

        for rnn_layer in &data.architecture.layers {
            let activation = Self::parse_activation(&rnn_layer.activation)?;

            let layer_builder = crate::layer::LayerBuilder::dense(rnn_layer.output_dim)
                .activation(activation)
                .use_bias(rnn_layer.use_bias)
                .trainable(rnn_layer.trainable);

            builder = builder.add_layer(layer_builder);
        }

        let loss_function = Self::parse_loss_function(&data.architecture.loss_function)?;
        builder = builder.loss(loss_function);

        if let Some(metadata) = &data.metadata {
            builder = builder.metadata(metadata.clone());
        }

        let mut network = builder.build()?;

        // Set weights
        for (i, (layer, rnn_layer)) in network
            .layers
            .iter_mut()
            .zip(data.architecture.layers.iter())
            .enumerate()
        {
            layer.weights = rnn_layer.weights.clone();
            if let Some(bias) = &rnn_layer.bias {
                layer.bias = bias.clone();
            }
        }

        // Set training history if available
        if let Some(history) = &data.training_history {
            network.history = history.clone();
        }

        Ok(network)
    }

    /// Load 2D array from CSV file.
    fn load_array_from_csv<P: AsRef<Path>>(path: P) -> Result<Array2<f64>> {
        let file = File::open(path).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);

        let mut rows = Vec::new();
        let mut num_cols = 0;

        for line in std::io::BufRead::lines(reader) {
            let line = line.map_err(NetworkError::from)?;
            let values: Result<Vec<f64>> = line
                .split(',')
                .map(|s| {
                    s.trim()
                        .parse::<f64>()
                        .map_err(|e| NetworkError::data(format!("Parse error: {}", e)))
                })
                .collect();

            let row_values = values?;
            if num_cols == 0 {
                num_cols = row_values.len();
            } else if row_values.len() != num_cols {
                return Err(NetworkError::data("Inconsistent number of columns in CSV"));
            }

            rows.extend(row_values);
        }

        if rows.is_empty() {
            return Err(NetworkError::data("Empty CSV file"));
        }

        let num_rows = rows.len() / num_cols;
        Array2::from_shape_vec((num_rows, num_cols), rows).map_err(NetworkError::from)
    }

    /// Parse activation function from string.
    fn parse_activation(name: &str) -> Result<crate::activation::ActivationFunction> {
        match name.to_lowercase().as_str() {
            "linear" => Ok(crate::activation::ActivationFunction::Linear),
            "sigmoid" => Ok(crate::activation::ActivationFunction::Sigmoid),
            "tanh" => Ok(crate::activation::ActivationFunction::Tanh),
            "relu" => Ok(crate::activation::ActivationFunction::ReLU),
            "leakyrelu" => Ok(crate::activation::ActivationFunction::LeakyReLU),
            "elu" => Ok(crate::activation::ActivationFunction::ELU),
            "swish" => Ok(crate::activation::ActivationFunction::Swish),
            "gelu" => Ok(crate::activation::ActivationFunction::GELU),
            "softmax" => Ok(crate::activation::ActivationFunction::Softmax),
            "softplus" => Ok(crate::activation::ActivationFunction::Softplus),
            "mish" => Ok(crate::activation::ActivationFunction::Mish),
            "hardsigmoid" => Ok(crate::activation::ActivationFunction::HardSigmoid),
            "hardtanh" => Ok(crate::activation::ActivationFunction::HardTanh),
            "prelu" => Ok(crate::activation::ActivationFunction::PReLU),
            _ => Err(NetworkError::configuration(format!(
                "Unknown activation function: {}",
                name
            ))),
        }
    }

    /// Parse loss function from string.
    fn parse_loss_function(name: &str) -> Result<crate::loss::LossFunction> {
        match name.to_lowercase().as_str() {
            "meansquarederror" => Ok(crate::loss::LossFunction::MeanSquaredError),
            "meanabsoluteerror" => Ok(crate::loss::LossFunction::MeanAbsoluteError),
            "binarycrossentropy" => Ok(crate::loss::LossFunction::BinaryCrossEntropy),
            "categoricalcrossentropy" => Ok(crate::loss::LossFunction::CategoricalCrossEntropy),
            "sparsecategoricalcrossentropy" => {
                Ok(crate::loss::LossFunction::SparseCategoricalCrossEntropy)
            }
            "huberloss" => Ok(crate::loss::LossFunction::HuberLoss),
            "hingeloss" => Ok(crate::loss::LossFunction::HingeLoss),
            "squaredhingeloss" => Ok(crate::loss::LossFunction::SquaredHingeLoss),
            "kldivergence" => Ok(crate::loss::LossFunction::KLDivergence),
            "poissonloss" => Ok(crate::loss::LossFunction::PoissonLoss),
            "cosinesimilarityloss" => Ok(crate::loss::LossFunction::CosineSimilarityLoss),
            "logcoshloss" => Ok(crate::loss::LossFunction::LogCoshLoss),
            "quantileloss" => Ok(crate::loss::LossFunction::QuantileLoss),
            _ => Err(NetworkError::configuration(format!(
                "Unknown loss function: {}",
                name
            ))),
        }
    }
}

/// Data structures for serialization.

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExportData {
    metadata: Option<NetworkMetadata>,
    architecture: Option<NetworkArchitecture>,
    weights: Vec<Array2<f64>>,
    history: Option<crate::training::TrainingHistory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CompressedExportData {
    data: Vec<u8>,
    compression_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkArchitecture {
    input_dim: usize,
    output_dim: usize,
    layers: Vec<RnnLayer>,
    loss_function: String,
    optimizer_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RnnLayer {
    layer_type: String,
    input_dim: usize,
    output_dim: usize,
    activation: String,
    use_bias: bool,
    trainable: bool,
    weights: Array2<f64>,
    bias: Option<Array1<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct NetworkArchitectureInfo {
    input_dim: usize,
    output_dim: usize,
    layers: Vec<LayerInfo>,
    loss_function: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct LayerInfo {
    layer_type: String,
    input_dim: usize,
    output_dim: usize,
    activation: String,
    use_bias: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RnnFormat {
    version: String,
    metadata: Option<NetworkMetadata>,
    architecture: NetworkArchitecture,
    training_history: Option<crate::training::TrainingHistory>,
}

/// Convenience functions for common import/export operations.
impl Network {
    /// Export network to JSON format.
    pub fn export_json<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let config = ExportConfig {
            format: NetworkFormat::Json,
            ..Default::default()
        };
        NetworkExporter::export(self, path, &config)
    }

    /// Export network to binary format.
    pub fn export_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let config = ExportConfig {
            format: NetworkFormat::Binary,
            ..Default::default()
        };
        NetworkExporter::export(self, path, &config)
    }

    /// Export weights only to NumPy format.
    pub fn export_weights<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let config = ExportConfig {
            format: NetworkFormat::Numpy,
            weights_only: true,
            include_history: false,
            include_metadata: false,
            ..Default::default()
        };
        NetworkExporter::export(self, path, &config)
    }

    /// Import network from JSON format.
    pub fn import_json<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = ImportConfig {
            format: NetworkFormat::Json,
            ..Default::default()
        };
        NetworkImporter::import(path, &config)
    }

    /// Import network from binary format.
    pub fn import_binary<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = ImportConfig {
            format: NetworkFormat::Binary,
            ..Default::default()
        };
        NetworkImporter::import(path, &config)
    }

    /// Import network from NumPy format.
    pub fn import_weights<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = ImportConfig {
            format: NetworkFormat::Numpy,
            ..Default::default()
        };
        NetworkImporter::import(path, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ActivationFunction;
    use crate::layer::LayerBuilder;
    use crate::loss::LossFunction;
    use std::fs;

    #[test]
    fn test_export_import_json() {
        let network = crate::network::Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(3).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .loss(LossFunction::BinaryCrossEntropy)
            .name("test_network")
            .build()
            .unwrap();

        let path = "test_network_export.json";

        // Export
        network.export_json(path).unwrap();
        assert!(std::path::Path::new(path).exists());

        // Import
        let imported_network = Network::import_json(path).unwrap();

        // Verify
        assert_eq!(network.input_dim, imported_network.input_dim);
        assert_eq!(network.output_dim, imported_network.output_dim);
        assert_eq!(network.layers.len(), imported_network.layers.len());
        assert_eq!(network.name, imported_network.name);
        assert_eq!(network.loss_function, imported_network.loss_function);

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_import_binary() {
        let network = crate::network::Network::with_input_size(3)
            .unwrap()
            .add_layer(LayerBuilder::dense(2).activation(ActivationFunction::Tanh))
            .loss(LossFunction::MeanSquaredError)
            .build()
            .unwrap();

        let path = "test_network_binary.bin";

        // Export
        network.export_binary(path).unwrap();
        assert!(std::path::Path::new(path).exists());

        // Import
        let imported_network = Network::import_binary(path).unwrap();

        // Verify
        assert_eq!(network.input_dim, imported_network.input_dim);
        assert_eq!(network.output_dim, imported_network.output_dim);
        assert_eq!(network.layers.len(), imported_network.layers.len());

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_export_weights_numpy() {
        let network = crate::network::Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(2))
            .build()
            .unwrap();

        let path = "test_weights";

        // Export weights
        network.export_weights(path).unwrap();

        // Check that files were created
        assert!(std::path::Path::new("test_weights_info.json").exists());
        assert!(std::path::Path::new("test_weights_layer_0_weights.csv").exists());
        assert!(std::path::Path::new("test_weights_layer_0_bias.csv").exists());

        // Import back
        let imported_network = Network::import_weights(path).unwrap();
        assert_eq!(network.input_dim, imported_network.input_dim);
        assert_eq!(network.output_dim, imported_network.output_dim);

        // Cleanup
        fs::remove_file("test_weights_info.json").ok();
        fs::remove_file("test_weights_layer_0_weights.csv").ok();
        fs::remove_file("test_weights_layer_0_bias.csv").ok();
    }

    #[test]
    fn test_export_config_options() {
        let network = crate::network::Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(1))
            .build()
            .unwrap();

        let config = ExportConfig {
            format: NetworkFormat::Json,
            include_history: false,
            include_metadata: false,
            weights_only: true,
            ..Default::default()
        };

        let path = "test_config_export.json";
        NetworkExporter::export(&network, path, &config).unwrap();

        assert!(std::path::Path::new(path).exists());

        // Cleanup
        fs::remove_file(path).ok();
    }

    #[test]
    fn test_parse_activation_functions() {
        assert!(NetworkImporter::parse_activation("relu").is_ok());
        assert!(NetworkImporter::parse_activation("sigmoid").is_ok());
        assert!(NetworkImporter::parse_activation("unknown").is_err());
    }

    #[test]
    fn test_parse_loss_functions() {
        assert!(NetworkImporter::parse_loss_function("meansquarederror").is_ok());
        assert!(NetworkImporter::parse_loss_function("binarycrossentropy").is_ok());
        assert!(NetworkImporter::parse_loss_function("unknown").is_err());
    }

    #[test]
    fn test_network_format_enum() {
        let format = NetworkFormat::Json;
        assert_eq!(format, NetworkFormat::Json);

        let serialized = serde_json::to_string(&format).unwrap();
        let deserialized: NetworkFormat = serde_json::from_str(&serialized).unwrap();
        assert_eq!(format, deserialized);
    }

    #[test]
    fn test_export_import_configs() {
        let export_config = ExportConfig::default();
        assert_eq!(export_config.format, NetworkFormat::Json);
        assert!(export_config.include_history);

        let import_config = ImportConfig::default();
        assert_eq!(import_config.format, NetworkFormat::Json);
        assert!(import_config.strict);
    }
}
