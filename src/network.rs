//! Neural network implementation with support for various architectures and training methods.
//!
//! This module provides the main `Network` struct that combines layers, loss functions,
//! optimizers, and training algorithms into a complete neural network system.

use crate::activation::ActivationFunction;
use crate::error::{NetworkError, Result};
use crate::layer::{BackwardResult, Layer, LayerBuilder, LayerSummary};
use crate::loss::LossFunction;
use crate::optimizer::{Optimizer, OptimizerType};
use crate::training::{
    DataLoader, EarlyStopping, LearningRateScheduler, Metrics, TrainingCallback, TrainingConfig,
    TrainingHistory, TrainingMethod, TrainingState,
};
use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;
use std::time::Instant;

/// The main neural network struct.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Network {
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
    /// Network layers
    pub layers: Vec<Layer>,
    /// Loss function
    pub loss_function: LossFunction,
    /// Network optimizer
    pub optimizer: Optimizer,
    /// Network name
    pub name: Option<String>,
    /// Whether the network is compiled
    pub compiled: bool,
    /// Training history
    #[serde(skip)]
    pub history: TrainingHistory,
    /// Network metadata
    pub metadata: NetworkMetadata,
}

/// Metadata about the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkMetadata {
    /// Creation timestamp
    pub created_at: String,
    /// Last modified timestamp
    pub modified_at: String,
    /// Network version
    pub version: String,
    /// Author/creator
    pub author: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Tags for categorization
    pub tags: Vec<String>,
    /// Training configuration used
    pub training_config: Option<TrainingConfig>,
    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

impl Default for NetworkMetadata {
    fn default() -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            created_at: now.clone(),
            modified_at: now,
            version: "1.0.0".to_string(),
            author: None,
            description: None,
            tags: Vec::new(),
            training_config: None,
            performance_metrics: HashMap::new(),
        }
    }
}

/// Builder for creating neural networks with a fluent interface.
#[derive(Debug, Clone)]
pub struct NetworkBuilder {
    input_dim: usize,
    layers: Vec<LayerBuilder>,
    loss_function: LossFunction,
    optimizer: Option<Optimizer>,
    name: Option<String>,
    metadata: NetworkMetadata,
}

impl NetworkBuilder {
    /// Create a new network builder with the specified input dimension.
    pub fn new(input_dim: usize) -> Result<Self> {
        if input_dim == 0 {
            return Err(NetworkError::architecture(
                "Input dimension must be greater than 0",
            ));
        }

        Ok(Self {
            input_dim,
            layers: Vec::new(),
            loss_function: LossFunction::MeanSquaredError,
            optimizer: None,
            name: None,
            metadata: NetworkMetadata::default(),
        })
    }

    /// Add a layer to the network.
    pub fn add_layer(mut self, layer_builder: LayerBuilder) -> Self {
        self.layers.push(layer_builder);
        self
    }

    /// Set the loss function.
    pub fn loss(mut self, loss_function: LossFunction) -> Self {
        self.loss_function = loss_function;
        self
    }

    /// Set the optimizer.
    pub fn optimizer(mut self, optimizer: Optimizer) -> Self {
        self.optimizer = Some(optimizer);
        self
    }

    /// Set the network name.
    pub fn name<S: Into<String>>(mut self, name: S) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set network metadata.
    pub fn metadata(mut self, metadata: NetworkMetadata) -> Self {
        self.metadata = metadata;
        self
    }

    /// Set author.
    pub fn author<S: Into<String>>(mut self, author: S) -> Self {
        self.metadata.author = Some(author.into());
        self
    }

    /// Set description.
    pub fn description<S: Into<String>>(mut self, description: S) -> Self {
        self.metadata.description = Some(description.into());
        self
    }

    /// Add tags.
    pub fn tags(mut self, tags: Vec<String>) -> Self {
        self.metadata.tags = tags;
        self
    }

    /// Build the network.
    pub fn build(self) -> Result<Network> {
        if self.layers.is_empty() {
            return Err(NetworkError::architecture(
                "Network must have at least one layer",
            ));
        }

        let mut built_layers = Vec::new();
        let mut current_input_dim = self.input_dim;

        // Build all layers with correct input dimensions
        for layer_builder in self.layers {
            let layer = layer_builder.build(current_input_dim)?;
            current_input_dim = layer.output_dim;
            built_layers.push(layer);
        }

        let output_dim = current_input_dim;

        // Use provided optimizer or create default Adam optimizer
        let optimizer = self
            .optimizer
            .unwrap_or_else(|| Optimizer::adam(0.001).unwrap());

        Ok(Network {
            input_dim: self.input_dim,
            output_dim,
            layers: built_layers,
            loss_function: self.loss_function,
            optimizer,
            name: self.name,
            compiled: false,
            history: TrainingHistory::default(),
            metadata: self.metadata,
        })
    }
}

impl Network {
    /// Create a new network builder.
    pub fn builder() -> NetworkBuilder {
        NetworkBuilder::new(1).unwrap() // Will be overridden by input_size call
    }

    /// Create a new network builder with input size.
    pub fn with_input_size(input_size: usize) -> Result<NetworkBuilder> {
        NetworkBuilder::new(input_size)
    }

    /// Compile the network for training.
    pub fn compile(&mut self) -> Result<()> {
        // Validate network architecture
        self.validate_architecture()?;

        // Initialize optimizer state if needed
        self.compiled = true;
        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        Ok(())
    }

    /// Validate the network architecture.
    fn validate_architecture(&self) -> Result<()> {
        if self.layers.is_empty() {
            return Err(NetworkError::architecture("Network has no layers"));
        }

        // Check that layer dimensions are consistent
        let mut current_dim = self.input_dim;
        for (i, layer) in self.layers.iter().enumerate() {
            if layer.input_dim != current_dim {
                return Err(NetworkError::architecture(format!(
                    "Layer {} input dimension ({}) doesn't match expected ({})",
                    i, layer.input_dim, current_dim
                )));
            }
            current_dim = layer.output_dim;
        }

        if current_dim != self.output_dim {
            return Err(NetworkError::architecture(format!(
                "Final layer output dimension ({}) doesn't match network output dimension ({})",
                current_dim, self.output_dim
            )));
        }

        Ok(())
    }

    /// Forward pass through the network.
    pub fn forward(&mut self, input: &Array2<f64>, training: bool) -> Result<Array2<f64>> {
        if !self.compiled {
            return Err(NetworkError::configuration(
                "Network must be compiled before use",
            ));
        }

        if input.ncols() != self.input_dim {
            return Err(NetworkError::dimension_mismatch(
                format!("input columns: {}", self.input_dim),
                format!("actual input columns: {}", input.ncols()),
            ));
        }

        let mut current_output = input.clone();

        for layer in &mut self.layers {
            current_output = layer.forward(&current_output, training)?;
        }

        Ok(current_output)
    }

    /// Backward pass through the network.
    pub fn backward(&mut self, grad_output: &Array2<f64>) -> Result<Vec<BackwardResult>> {
        if !self.compiled {
            return Err(NetworkError::configuration(
                "Network must be compiled before use",
            ));
        }

        let mut gradients = Vec::new();
        let mut current_grad = grad_output.clone();

        // Backward pass through layers in reverse order
        for layer in self.layers.iter_mut().rev() {
            let backward_result = layer.backward(&current_grad)?;
            current_grad = backward_result.grad_input.clone();
            gradients.push(backward_result);
        }

        // Reverse gradients to match layer order
        gradients.reverse();
        Ok(gradients)
    }

    /// Predict outputs for given inputs.
    pub fn predict(&mut self, input: &Array2<f64>) -> Result<Array2<f64>> {
        self.forward(input, false)
    }

    /// Predict single sample.
    pub fn predict_one(&mut self, input: &Array1<f64>) -> Result<Array1<f64>> {
        if input.len() != self.input_dim {
            return Err(NetworkError::dimension_mismatch(
                format!("input size: {}", self.input_dim),
                format!("actual input size: {}", input.len()),
            ));
        }

        let input_matrix = input.clone().into_shape((1, input.len()))?;
        let output_matrix = self.predict(&input_matrix)?;
        Ok(output_matrix.row(0).to_owned())
    }

    /// Train the network using the specified configuration.
    pub fn train(
        &mut self,
        train_data: &Array2<f64>,
        train_targets: &Array2<f64>,
        config: &TrainingConfig,
    ) -> Result<TrainingHistory> {
        self.train_with_validation(train_data, train_targets, None, None, config)
    }

    /// Train the network with validation data.
    pub fn train_with_validation(
        &mut self,
        train_data: &Array2<f64>,
        train_targets: &Array2<f64>,
        val_data: Option<&Array2<f64>>,
        val_targets: Option<&Array2<f64>>,
        config: &TrainingConfig,
    ) -> Result<TrainingHistory> {
        if !self.compiled {
            self.compile()?;
        }

        // Validate inputs
        if train_data.nrows() != train_targets.nrows() {
            return Err(NetworkError::dimension_mismatch(
                format!("train_data rows: {}", train_data.nrows()),
                format!("train_targets rows: {}", train_targets.nrows()),
            ));
        }

        if let (Some(vd), Some(vt)) = (val_data, val_targets) {
            if vd.nrows() != vt.nrows() {
                return Err(NetworkError::dimension_mismatch(
                    format!("val_data rows: {}", vd.nrows()),
                    format!("val_targets rows: {}", vt.nrows()),
                ));
            }
        }

        // Initialize training state
        let mut state = TrainingState::default();
        state.history.parameter_count = self.parameter_count();
        state.current_lr = self.optimizer.get_learning_rate();

        // Split data if validation data not provided
        let (train_data_split, train_targets_split, val_data_split, val_targets_split) =
            if val_data.is_none() && config.validation_split > 0.0 {
                self.split_data(train_data, train_targets, config.validation_split)?
            } else {
                (
                    train_data.clone(),
                    train_targets.clone(),
                    val_data.map(|d| d.clone()),
                    val_targets.map(|t| t.clone()),
                )
            };

        // Initialize learning rate scheduler
        let mut lr_scheduler = config.lr_schedule.as_ref().map(|schedule| {
            LearningRateScheduler::new(schedule.clone(), self.optimizer.get_learning_rate())
        });

        // Initialize early stopping
        let mut early_stopping = config
            .early_stopping_patience
            .map(|patience| EarlyStopping::new(patience, config.early_stopping_min_delta));

        // Training loop
        for epoch in 0..config.max_epochs {
            let epoch_start = Instant::now();
            state.epoch = epoch;

            // Create data loader
            let mut data_loader = DataLoader::new(
                train_data_split.clone(),
                train_targets_split.clone(),
                config.batch_size,
                config.shuffle,
            )?;

            // Training phase
            let mut epoch_train_loss = 0.0;
            let mut batch_count = 0;

            while let Some((batch_data, batch_targets)) = data_loader.next_batch() {
                state.batch = batch_count;

                // Forward pass
                let predictions = self.forward(&batch_data, true)?;

                // Compute loss
                let loss = self.loss_function.compute(&predictions, &batch_targets)?;
                epoch_train_loss += loss;

                // Backward pass
                let loss_grad = self.loss_function.gradient(&predictions, &batch_targets)?;
                let gradients = self.backward(&loss_grad)?;

                // Update parameters
                self.update_parameters(&gradients)?;

                batch_count += 1;

                // Check training time limit
                if let Some(max_time) = config.max_training_time {
                    if state.start_time.elapsed() > max_time {
                        if config.verbose {
                            println!("Training stopped due to time limit");
                        }
                        break;
                    }
                }
            }

            epoch_train_loss /= batch_count as f64;
            state.history.train_loss.push(epoch_train_loss);

            // Validation phase
            let epoch_val_loss = if let (Some(val_data), Some(val_targets)) =
                (&val_data_split, &val_targets_split)
            {
                let val_predictions = self.predict(val_data)?;
                let val_loss = self.loss_function.compute(&val_predictions, val_targets)?;
                state.history.val_loss.push(val_loss);
                Some(val_loss)
            } else {
                None
            };

            // Update learning rate
            if let Some(ref mut scheduler) = lr_scheduler {
                let metric = epoch_val_loss.or(Some(epoch_train_loss));
                let new_lr = scheduler.step(metric);
                self.optimizer.update_learning_rate(new_lr)?;
                state.current_lr = new_lr;
            }

            state.history.learning_rate.push(state.current_lr);
            state.history.epoch_times.push(epoch_start.elapsed());

            // Early stopping check
            if let Some(ref mut early_stop) = early_stopping {
                let stop_metric = epoch_val_loss.unwrap_or(epoch_train_loss);
                let current_weights = self.get_weights();

                if early_stop.should_stop(stop_metric, Some(&current_weights)) {
                    if config.verbose {
                        println!(
                            "Early stopping at epoch {} with best score: {:.6}",
                            epoch,
                            early_stop.best_score()
                        );
                    }

                    // Restore best weights
                    if let Some(best_weights) = early_stop.best_weights() {
                        self.set_weights(best_weights)?;
                    }

                    state.history.best_epoch = Some(epoch);
                    state.history.best_val_loss = Some(early_stop.best_score());
                    break;
                }
            }

            // Verbose output
            if config.verbose && (epoch % 10 == 0 || epoch == config.max_epochs - 1) {
                let val_str = epoch_val_loss
                    .map(|v| format!(" - val_loss: {:.6}", v))
                    .unwrap_or_default();
                println!(
                    "Epoch {}/{} - loss: {:.6}{} - lr: {:.6}",
                    epoch + 1,
                    config.max_epochs,
                    epoch_train_loss,
                    val_str,
                    state.current_lr
                );
            }
        }

        state.history.total_time = state.start_time.elapsed();
        self.history = state.history.clone();
        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        Ok(state.history)
    }

    /// Update network parameters using computed gradients.
    fn update_parameters(&mut self, gradients: &[BackwardResult]) -> Result<()> {
        for (i, (layer, grad_result)) in self.layers.iter_mut().zip(gradients.iter()).enumerate() {
            if let (Some(grad_weights), Some(grad_bias)) =
                (&grad_result.grad_weights, &grad_result.grad_bias)
            {
                let param_name = format!("layer_{}", i);

                // Update weights
                self.optimizer.update(
                    &format!("{}_weights", param_name),
                    &mut layer.weights,
                    grad_weights,
                    None,
                )?;

                // Update bias if used
                if layer.use_bias {
                    // Convert bias gradient to 2D for optimizer compatibility
                    let grad_bias_2d = grad_bias.clone().into_shape((grad_bias.len(), 1))?;
                    let mut bias_2d = layer.bias.clone().into_shape((layer.bias.len(), 1))?;

                    self.optimizer.update(
                        &format!("{}_bias", param_name),
                        &mut bias_2d,
                        &grad_bias_2d,
                        None,
                    )?;

                    // Convert back to 1D
                    layer.bias = bias_2d.into_shape(layer.bias.len())?;
                }
            }
        }
        Ok(())
    }

    /// Split data into training and validation sets.
    fn split_data(
        &self,
        data: &Array2<f64>,
        targets: &Array2<f64>,
        validation_split: f64,
    ) -> Result<(
        Array2<f64>,
        Array2<f64>,
        Option<Array2<f64>>,
        Option<Array2<f64>>,
    )> {
        if validation_split <= 0.0 || validation_split >= 1.0 {
            return Err(NetworkError::invalid_parameter(
                "validation_split",
                &validation_split.to_string(),
                "must be between 0 and 1",
            ));
        }

        let total_samples = data.nrows();
        let val_samples = (total_samples as f64 * validation_split).round() as usize;
        let train_samples = total_samples - val_samples;

        let train_data = data.slice(ndarray::s![..train_samples, ..]).to_owned();
        let train_targets = targets.slice(ndarray::s![..train_samples, ..]).to_owned();
        let val_data = data.slice(ndarray::s![train_samples.., ..]).to_owned();
        let val_targets = targets.slice(ndarray::s![train_samples.., ..]).to_owned();

        Ok((train_data, train_targets, Some(val_data), Some(val_targets)))
    }

    /// Evaluate the network on test data.
    pub fn evaluate(
        &mut self,
        test_data: &Array2<f64>,
        test_targets: &Array2<f64>,
    ) -> Result<HashMap<String, f64>> {
        let predictions = self.predict(test_data)?;
        let mut metrics = HashMap::new();

        // Compute loss
        let loss = self.loss_function.compute(&predictions, test_targets)?;
        metrics.insert("loss".to_string(), loss);

        // Compute additional metrics based on problem type
        if self.loss_function.is_classification_loss() {
            let accuracy = Metrics::accuracy(&predictions, test_targets)?;
            metrics.insert("accuracy".to_string(), accuracy);

            // For binary classification
            if test_targets.ncols() == 1 || (test_targets.ncols() == 2) {
                let precision = Metrics::precision(&predictions, test_targets, 0.5)?;
                let recall = Metrics::recall(&predictions, test_targets, 0.5)?;
                let f1 = Metrics::f1_score(&predictions, test_targets, 0.5)?;

                metrics.insert("precision".to_string(), precision);
                metrics.insert("recall".to_string(), recall);
                metrics.insert("f1_score".to_string(), f1);
            }
        } else if self.loss_function.is_regression_loss() {
            let mae = Metrics::mae(&predictions, test_targets)?;
            let rmse = Metrics::rmse(&predictions, test_targets)?;
            let r2 = Metrics::r2_score(&predictions, test_targets)?;

            metrics.insert("mae".to_string(), mae);
            metrics.insert("rmse".to_string(), rmse);
            metrics.insert("r2_score".to_string(), r2);
        }

        Ok(metrics)
    }

    /// Get network weights.
    pub fn get_weights(&self) -> Vec<Array2<f64>> {
        self.layers
            .iter()
            .map(|layer| layer.weights.clone())
            .collect()
    }

    /// Set network weights.
    pub fn set_weights(&mut self, weights: &[Array2<f64>]) -> Result<()> {
        if weights.len() != self.layers.len() {
            return Err(NetworkError::dimension_mismatch(
                format!("expected {} weight matrices", self.layers.len()),
                format!("got {} weight matrices", weights.len()),
            ));
        }

        for (layer, weight_matrix) in self.layers.iter_mut().zip(weights.iter()) {
            if weight_matrix.shape() != layer.weights.shape() {
                return Err(NetworkError::dimension_mismatch(
                    format!("{:?}", layer.weights.shape()),
                    format!("{:?}", weight_matrix.shape()),
                ));
            }
            layer.weights = weight_matrix.clone();
        }

        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Get the total number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.layers
            .iter()
            .map(|layer| layer.parameter_count())
            .sum()
    }

    /// Get network summary.
    pub fn summary(&self) -> NetworkSummary {
        let layer_summaries: Vec<LayerSummary> =
            self.layers.iter().map(|layer| layer.summary()).collect();

        NetworkSummary {
            name: self.name.clone(),
            input_dim: self.input_dim,
            output_dim: self.output_dim,
            total_parameters: self.parameter_count(),
            trainable_parameters: self
                .layers
                .iter()
                .filter(|layer| layer.trainable)
                .map(|layer| layer.parameter_count())
                .sum(),
            layers: layer_summaries,
            loss_function: self.loss_function.name().to_string(),
            optimizer: self.optimizer.summary(),
            compiled: self.compiled,
            metadata: self.metadata.clone(),
        }
    }

    /// Print a detailed summary of the network.
    pub fn print_summary(&self) {
        let summary = self.summary();

        println!("\n{}", "=".repeat(80));
        println!("Neural Network Summary");
        println!("{}", "=".repeat(80));

        if let Some(name) = &summary.name {
            println!("Name: {}", name);
        }

        println!("Input dimension: {}", summary.input_dim);
        println!("Output dimension: {}", summary.output_dim);
        println!("Loss function: {}", summary.loss_function);
        println!("Optimizer: {:?}", summary.optimizer.optimizer_type);
        println!("Compiled: {}", summary.compiled);
        println!();

        println!(
            "{:<20} {:<15} {:<15} {:<15} {:<15}",
            "Layer (type)", "Output Shape", "Param #", "Activation", "Trainable"
        );
        println!("{}", "-".repeat(80));

        for (i, layer) in summary.layers.iter().enumerate() {
            println!(
                "{:<20} {:<15} {:<15} {:<15} {:<15}",
                format!(
                    "{}_{} ({:?})",
                    layer.name.split('_').next().unwrap_or("layer"),
                    i,
                    layer.layer_type
                ),
                format!("{:?}", layer.output_shape),
                layer.parameter_count,
                layer.activation,
                layer.trainable
            );
        }

        println!("{}", "-".repeat(80));
        println!("Total params: {}", summary.total_parameters);
        println!("Trainable params: {}", summary.trainable_parameters);
        println!(
            "Non-trainable params: {}",
            summary.total_parameters - summary.trainable_parameters
        );
        println!("{}", "=".repeat(80));
    }

    /// Save the network to a file.
    pub fn save<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(NetworkError::from)?;
        let writer = BufWriter::new(file);

        // Update metadata before saving
        let mut network_to_save = self.clone();
        network_to_save.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        serde_json::to_writer_pretty(writer, &network_to_save).map_err(NetworkError::from)?;

        Ok(())
    }

    /// Load a network from a file.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);
        let network: Network = serde_json::from_reader(reader).map_err(NetworkError::from)?;

        Ok(network)
    }

    /// Save network in binary format.
    pub fn save_binary<P: AsRef<Path>>(&self, path: P) -> Result<()> {
        let file = File::create(path).map_err(NetworkError::from)?;
        let writer = BufWriter::new(file);

        let mut network_to_save = self.clone();
        network_to_save.metadata.modified_at = chrono::Utc::now().to_rfc3339();

        bincode::serialize_into(writer, &network_to_save).map_err(NetworkError::from)?;

        Ok(())
    }

    /// Load network from binary format.
    pub fn load_binary<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path).map_err(NetworkError::from)?;
        let reader = BufReader::new(file);
        let network: Network = bincode::deserialize_from(reader).map_err(NetworkError::from)?;

        Ok(network)
    }

    /// Clone the network architecture without weights.
    pub fn clone_architecture(&self) -> Result<Network> {
        let mut builder = NetworkBuilder::new(self.input_dim)?;

        // Recreate layer builders from existing layers
        for layer in &self.layers {
            let layer_builder = match layer.layer_type {
                crate::layer::LayerType::Dense => LayerBuilder::dense(layer.output_dim)
                    .activation(layer.activation)
                    .weight_init(layer.weight_init)
                    .use_bias(layer.use_bias),
                crate::layer::LayerType::Dropout => {
                    if let Some(dropout_config) = layer.dropout {
                        LayerBuilder::dropout(dropout_config.rate)?
                    } else {
                        return Err(NetworkError::architecture(
                            "Dropout layer missing configuration",
                        ));
                    }
                }
                _ => {
                    return Err(NetworkError::architecture(format!(
                        "Cloning not supported for layer type: {:?}",
                        layer.layer_type
                    )));
                }
            };

            builder = builder.add_layer(layer_builder);
        }

        builder
            .loss(self.loss_function)
            .optimizer(self.optimizer.clone())
            .name(self.name.clone().unwrap_or_default())
            .metadata(self.metadata.clone())
            .build()
    }

    /// Reset network weights to random initialization.
    pub fn reset_weights(&mut self) -> Result<()> {
        for layer in &mut self.layers {
            layer.reset_parameters()?;
        }
        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Freeze layers (make them non-trainable).
    pub fn freeze_layers(&mut self, layer_indices: &[usize]) -> Result<()> {
        for &index in layer_indices {
            if index >= self.layers.len() {
                return Err(NetworkError::architecture(format!(
                    "Layer index {} out of bounds (network has {} layers)",
                    index,
                    self.layers.len()
                )));
            }
            self.layers[index].trainable = false;
        }
        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Unfreeze layers (make them trainable).
    pub fn unfreeze_layers(&mut self, layer_indices: &[usize]) -> Result<()> {
        for &index in layer_indices {
            if index >= self.layers.len() {
                return Err(NetworkError::architecture(format!(
                    "Layer index {} out of bounds (network has {} layers)",
                    index,
                    self.layers.len()
                )));
            }
            self.layers[index].trainable = true;
        }
        self.metadata.modified_at = chrono::Utc::now().to_rfc3339();
        Ok(())
    }

    /// Get layer by index.
    pub fn get_layer(&self, index: usize) -> Result<&Layer> {
        self.layers.get(index).ok_or_else(|| {
            NetworkError::architecture(format!(
                "Layer index {} out of bounds (network has {} layers)",
                index,
                self.layers.len()
            ))
        })
    }

    /// Get mutable layer by index.
    pub fn get_layer_mut(&mut self, index: usize) -> Result<&mut Layer> {
        let layer_count = self.layers.len();
        self.layers.get_mut(index).ok_or_else(|| {
            NetworkError::architecture(format!(
                "Layer index {} out of bounds (network has {} layers)",
                index, layer_count
            ))
        })
    }
}

/// Summary information about the network.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSummary {
    pub name: Option<String>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub total_parameters: usize,
    pub trainable_parameters: usize,
    pub layers: Vec<LayerSummary>,
    pub loss_function: String,
    pub optimizer: crate::optimizer::OptimizerSummary,
    pub compiled: bool,
    pub metadata: NetworkMetadata,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::ActivationFunction;
    use crate::layer::LayerBuilder;
    use crate::optimizer::OptimizerType;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_network_builder() {
        let network = Network::with_input_size(784)
            .unwrap()
            .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(64).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
            .loss(LossFunction::CategoricalCrossEntropy)
            .name("test_network")
            .build()
            .unwrap();

        assert_eq!(network.input_dim, 784);
        assert_eq!(network.output_dim, 10);
        assert_eq!(network.layers.len(), 3);
        assert_eq!(network.name, Some("test_network".to_string()));
        assert_eq!(network.loss_function, LossFunction::CategoricalCrossEntropy);
    }

    #[test]
    fn test_network_forward_pass() {
        let mut network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(3).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .build()
            .unwrap();

        network.compile().unwrap();

        let input = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let output = network.forward(&input, false).unwrap();

        assert_eq!(output.shape(), &[1, 1]);
        assert!(output[[0, 0]] >= 0.0 && output[[0, 0]] <= 1.0); // Sigmoid output
    }

    #[test]
    fn test_network_prediction() {
        let mut network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Linear))
            .build()
            .unwrap();

        network.compile().unwrap();

        let input = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let predictions = network.predict(&input).unwrap();

        assert_eq!(predictions.shape(), &[2, 1]);
    }

    #[test]
    fn test_network_parameter_count() {
        let network = Network::with_input_size(10)
            .unwrap()
            .add_layer(LayerBuilder::dense(5).use_bias(true))
            .add_layer(LayerBuilder::dense(1).use_bias(true))
            .build()
            .unwrap();

        // First layer: 10*5 + 5 = 55 parameters
        // Second layer: 5*1 + 1 = 6 parameters
        // Total: 61 parameters
        assert_eq!(network.parameter_count(), 61);
    }

    #[test]
    fn test_network_save_load() {
        let network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(3))
            .add_layer(LayerBuilder::dense(1))
            .name("test_save_load")
            .build()
            .unwrap();

        let temp_path = "test_network.json";

        // Save network
        network.save(temp_path).unwrap();

        // Load network
        let loaded_network = Network::load(temp_path).unwrap();

        assert_eq!(network.input_dim, loaded_network.input_dim);
        assert_eq!(network.output_dim, loaded_network.output_dim);
        assert_eq!(network.layers.len(), loaded_network.layers.len());
        assert_eq!(network.name, loaded_network.name);

        // Clean up
        std::fs::remove_file(temp_path).ok();
    }

    #[test]
    fn test_network_clone_architecture() {
        let original = Network::with_input_size(10)
            .unwrap()
            .add_layer(LayerBuilder::dense(5).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(1).activation(ActivationFunction::Sigmoid))
            .build()
            .unwrap();

        let cloned = original.clone_architecture().unwrap();

        assert_eq!(original.input_dim, cloned.input_dim);
        assert_eq!(original.output_dim, cloned.output_dim);
        assert_eq!(original.layers.len(), cloned.layers.len());

        // Weights should be different (newly initialized)
        assert_ne!(original.layers[0].weights, cloned.layers[0].weights);
    }

    #[test]
    fn test_network_freeze_unfreeze() {
        let mut network = Network::with_input_size(10)
            .unwrap()
            .add_layer(LayerBuilder::dense(5))
            .add_layer(LayerBuilder::dense(1))
            .build()
            .unwrap();

        // Initially all layers should be trainable
        assert!(network.layers.iter().all(|layer| layer.trainable));

        // Freeze first layer
        network.freeze_layers(&[0]).unwrap();
        assert!(!network.layers[0].trainable);
        assert!(network.layers[1].trainable);

        // Unfreeze first layer
        network.unfreeze_layers(&[0]).unwrap();
        assert!(network.layers[0].trainable);
        assert!(network.layers[1].trainable);
    }

    #[test]
    fn test_network_validation() {
        // Test invalid input dimension
        let result = Network::with_input_size(0);
        assert!(result.is_err());

        // Test empty network
        let result = Network::with_input_size(10).unwrap().build();
        assert!(result.is_err());
    }

    #[test]
    fn test_network_summary() {
        let network = Network::with_input_size(784)
            .unwrap()
            .add_layer(LayerBuilder::dense(128).activation(ActivationFunction::ReLU))
            .add_layer(LayerBuilder::dense(10).activation(ActivationFunction::Softmax))
            .name("mnist_classifier")
            .description("MNIST digit classifier")
            .build()
            .unwrap();

        let summary = network.summary();
        assert_eq!(summary.name, Some("mnist_classifier".to_string()));
        assert_eq!(summary.input_dim, 784);
        assert_eq!(summary.output_dim, 10);
        assert_eq!(summary.layers.len(), 2);
        assert!(summary.total_parameters > 0);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let mut network = Network::with_input_size(2)
            .unwrap()
            .add_layer(LayerBuilder::dense(1))
            .build()
            .unwrap();

        network.compile().unwrap();

        // Wrong input dimension
        let wrong_input = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();
        assert!(network.forward(&wrong_input, false).is_err());
    }
}
