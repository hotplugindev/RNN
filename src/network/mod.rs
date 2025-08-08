//! Neural Network module with builder pattern
//!
//! This module provides a comprehensive neural network implementation with
//! a flexible builder pattern for constructing networks, training capabilities,
//! and inference functionality across different device backends.

use crate::device::Device;
use crate::error::{Result, RnnError};
#[cfg(test)]
use crate::layers::create_layer;
use crate::layers::{Layer, TrainingMode};
use crate::losses::LossFunction;
#[cfg(test)]
use crate::optimizers::create_optimizer;
use crate::optimizers::Optimizer;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

use std::fmt;
use std::time::Instant;

pub mod builder;
pub mod training;

pub use builder::NetworkBuilder;
pub use training::{LearningRateSchedule, TrainingConfig, TrainingHistory, TrainingMetrics};

/// A neural network consisting of sequential layers
#[derive(Debug)]
pub struct Network {
    /// Network layers
    layers: Vec<Box<dyn Layer>>,
    /// Loss function
    loss_function: LossFunction,
    /// Optimizer
    optimizer: Box<dyn Optimizer>,
    /// Device for computation
    device: Device,
    /// Training mode flag
    training: bool,
    /// Network metrics
    metrics: NetworkMetrics,
}

/// Network performance metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NetworkMetrics {
    /// Total number of trainable parameters
    pub total_parameters: usize,
    /// Total number of layers in the network
    pub total_layers: usize,
    /// Memory usage in megabytes
    pub memory_usage_mb: f32,
    /// Average inference time in milliseconds
    pub inference_time_ms: f32,
    /// Total training time in milliseconds
    pub training_time_ms: f32,
    /// Number of epochs the network has been trained
    pub epochs_trained: usize,
    /// Best loss achieved during training
    pub best_loss: f32,
    /// Best accuracy achieved during training
    pub best_accuracy: f32,
}

/// Training progress callback
pub trait TrainingCallback: Send + Sync {
    /// Called at the beginning of each epoch
    fn on_epoch_start(&mut self, epoch: usize, total_epochs: usize);
    /// Called at the end of each epoch with training metrics
    fn on_epoch_end(&mut self, epoch: usize, metrics: &TrainingMetrics);
    /// Called at the beginning of each batch
    fn on_batch_start(&mut self, batch: usize, total_batches: usize);
    /// Called at the end of each batch with computed loss
    fn on_batch_end(&mut self, batch: usize, loss: f32);
    fn should_stop(&self) -> bool {
        false
    }
}

/// Simple progress callback implementation
#[derive(Debug)]
pub struct ProgressCallback {
    pub verbose: bool,
    pub print_every: usize,
}

impl ProgressCallback {
    pub fn new(verbose: bool) -> Self {
        Self {
            verbose,
            print_every: 10,
        }
    }
}

impl TrainingCallback for ProgressCallback {
    fn on_epoch_start(&mut self, epoch: usize, total_epochs: usize) {
        if self.verbose {
            println!("Epoch {}/{}", epoch + 1, total_epochs);
        }
    }

    fn on_epoch_end(&mut self, epoch: usize, metrics: &TrainingMetrics) {
        if self.verbose && (epoch + 1) % self.print_every == 0 {
            println!(
                "Epoch {}: Loss = {:.6}, Accuracy = {:.4}",
                epoch + 1,
                metrics.loss,
                metrics.accuracy
            );
        }
    }

    fn on_batch_start(&mut self, _batch: usize, _total_batches: usize) {}

    fn on_batch_end(&mut self, _batch: usize, _loss: f32) {}
}

impl Network {
    /// Create a new network
    pub fn new(
        layers: Vec<Box<dyn Layer>>,
        loss_function: LossFunction,
        optimizer: Box<dyn Optimizer>,
        device: Device,
    ) -> Result<Self> {
        if layers.is_empty() {
            return Err(RnnError::network("Network must have at least one layer"));
        }

        let total_parameters = layers.iter().map(|layer| layer.num_parameters()).sum();
        let total_layers = layers.len();

        let metrics = NetworkMetrics {
            total_parameters,
            total_layers,
            memory_usage_mb: 0.0,
            inference_time_ms: 0.0,
            training_time_ms: 0.0,
            epochs_trained: 0,
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
        };

        Ok(Self {
            layers,
            loss_function,
            optimizer,
            device,
            training: true,
            metrics,
        })
    }

    /// Forward pass through the network
    pub fn forward(&mut self, input: &Tensor) -> Result<Tensor> {
        let start_time = Instant::now();

        let training_mode = if self.training {
            TrainingMode::Training
        } else {
            TrainingMode::Inference
        };

        let mut current = input.clone_data()?;

        for layer in &mut self.layers {
            current = layer.forward(&current, training_mode)?;
        }

        let elapsed = start_time.elapsed();
        if !self.training {
            self.metrics.inference_time_ms = elapsed.as_millis() as f32;
        }

        Ok(current)
    }

    /// Backward pass through the network
    pub fn backward(&mut self, target: &Tensor, prediction: &Tensor) -> Result<f32> {
        // Compute loss
        let loss = self.loss_function.forward(prediction, target)?;

        // Compute loss gradient
        let mut grad = self.loss_function.backward(prediction, target)?;

        // Backward pass through layers
        for layer in self.layers.iter_mut().rev() {
            grad = layer.backward(&grad)?;
        }

        Ok(loss)
    }

    /// Train the network for one epoch
    pub fn train_epoch(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        batch_size: usize,
        mut callback: Option<&mut dyn TrainingCallback>,
    ) -> Result<TrainingMetrics> {
        if inputs.len() != targets.len() {
            return Err(RnnError::training("Inputs and targets length mismatch"));
        }

        self.set_training(true);
        let start_time = Instant::now();

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let mut total_samples = 0;

        let num_batches = (inputs.len() + batch_size - 1) / batch_size;

        for (batch_idx, batch_start) in (0..inputs.len()).step_by(batch_size).enumerate() {
            if let Some(callback) = callback.as_mut() {
                callback.on_batch_start(batch_idx, num_batches);
                if callback.should_stop() {
                    break;
                }
            }

            let batch_end = (batch_start + batch_size).min(inputs.len());
            let batch_inputs = &inputs[batch_start..batch_end];
            let batch_targets = &targets[batch_start..batch_end];

            // Zero gradients
            for layer in &mut self.layers {
                layer.zero_grad();
            }

            let mut batch_loss = 0.0;
            let mut batch_correct = 0;

            // Process batch
            for (input, target) in batch_inputs.iter().zip(batch_targets.iter()) {
                let prediction = self.forward(input)?;
                let loss = self.backward(target, &prediction)?;

                batch_loss += loss;
                total_samples += 1;

                // Calculate accuracy for classification tasks
                if self.is_classification_task() {
                    if self.is_correct_prediction(&prediction, target)? {
                        batch_correct += 1;
                    }
                }
            }

            // Update parameters
            let mut params = self.collect_parameters();
            let grads = self.collect_gradients();
            self.optimizer.step(&mut params, &grads)?;
            self.update_parameters(params)?;

            let avg_batch_loss = batch_loss / batch_inputs.len() as f32;
            total_loss += batch_loss;
            correct_predictions += batch_correct;

            if let Some(callback) = callback.as_mut() {
                callback.on_batch_end(batch_idx, avg_batch_loss);
            }
        }

        let avg_loss = total_loss / total_samples as f32;
        let accuracy = if total_samples > 0 {
            correct_predictions as f32 / total_samples as f32
        } else {
            0.0
        };

        let elapsed = start_time.elapsed();
        self.metrics.training_time_ms = elapsed.as_millis() as f32;
        self.metrics.epochs_trained += 1;

        if avg_loss < self.metrics.best_loss {
            self.metrics.best_loss = avg_loss;
        }
        if accuracy > self.metrics.best_accuracy {
            self.metrics.best_accuracy = accuracy;
        }

        Ok(TrainingMetrics {
            loss: avg_loss,
            accuracy,
            learning_rate: self.optimizer.learning_rate(),
            epoch_time_ms: elapsed.as_millis() as f32,
            val_loss: None,
            val_accuracy: None,
            custom_metrics: std::collections::HashMap::new(),
        })
    }

    /// Train the network for multiple epochs
    pub fn train(
        &mut self,
        inputs: &[Tensor],
        targets: &[Tensor],
        config: &TrainingConfig,
    ) -> Result<TrainingHistory> {
        let mut history = TrainingHistory::new();
        let mut callback = if config.verbose {
            Some(ProgressCallback::new(true))
        } else {
            None
        };

        for epoch in 0..config.epochs {
            if let Some(callback) = callback.as_mut() {
                callback.on_epoch_start(epoch, config.epochs);
            }

            let metrics = self.train_epoch(
                inputs,
                targets,
                config.batch_size,
                callback.as_mut().map(|c| c as &mut dyn TrainingCallback),
            )?;

            history.add_epoch(metrics.clone());

            if let Some(callback) = callback.as_mut() {
                callback.on_epoch_end(epoch, &metrics);
            }

            // Early stopping
            if config.early_stopping_patience > 0 {
                if history.should_early_stop(
                    config.early_stopping_patience,
                    config.early_stopping_threshold,
                ) {
                    println!("Early stopping triggered at epoch {}", epoch + 1);
                    break;
                }
            }

            // Learning rate scheduling
            if let Some(schedule) = &config.lr_schedule {
                self.apply_lr_schedule(&schedule, epoch, &metrics)?;
            }
        }

        Ok(history)
    }

    /// Evaluate the network on test data
    pub fn evaluate(&mut self, inputs: &[Tensor], targets: &[Tensor]) -> Result<TrainingMetrics> {
        if inputs.len() != targets.len() {
            return Err(RnnError::training("Inputs and targets length mismatch"));
        }

        self.set_training(false);
        let start_time = Instant::now();

        let mut total_loss = 0.0;
        let mut correct_predictions = 0;
        let total_samples = inputs.len();

        for (input, target) in inputs.iter().zip(targets.iter()) {
            let prediction = self.forward(input)?;
            let loss = self.loss_function.forward(&prediction, target)?;

            total_loss += loss;

            if self.is_classification_task() {
                if self.is_correct_prediction(&prediction, target)? {
                    correct_predictions += 1;
                }
            }
        }

        let avg_loss = total_loss / total_samples as f32;
        let accuracy = if total_samples > 0 {
            correct_predictions as f32 / total_samples as f32
        } else {
            0.0
        };

        let elapsed = start_time.elapsed();

        Ok(TrainingMetrics {
            loss: avg_loss,
            accuracy,
            learning_rate: self.optimizer.learning_rate(),
            epoch_time_ms: elapsed.as_millis() as f32,
            val_loss: None,
            val_accuracy: None,
            custom_metrics: std::collections::HashMap::new(),
        })
    }

    /// Make predictions on new data
    pub fn predict(&mut self, inputs: &[Tensor]) -> Result<Vec<Tensor>> {
        self.set_training(false);
        let mut predictions = Vec::with_capacity(inputs.len());

        for input in inputs {
            let prediction = self.forward(input)?;
            predictions.push(prediction);
        }

        Ok(predictions)
    }

    /// Set training mode
    pub fn set_training(&mut self, training: bool) {
        self.training = training;
        for layer in &mut self.layers {
            layer.set_training(training);
        }
    }

    /// Check if network is in training mode
    pub fn training(&self) -> bool {
        self.training
    }

    /// Get network metrics
    pub fn metrics(&self) -> &NetworkMetrics {
        &self.metrics
    }

    /// Get number of layers
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Get total number of parameters
    pub fn num_parameters(&self) -> usize {
        self.metrics.total_parameters
    }

    /// Get loss function
    pub fn loss_function(&self) -> &LossFunction {
        &self.loss_function
    }

    /// Get device
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Move network to device
    pub fn to_device(&mut self, device: Device) -> Result<()> {
        for layer in &mut self.layers {
            layer.to_device(device.clone())?;
        }
        self.device = device;
        Ok(())
    }

    /// Get network summary
    pub fn summary(&self) -> NetworkSummary {
        let mut layer_info = Vec::new();
        let mut total_params = 0;

        for (i, layer) in self.layers.iter().enumerate() {
            let params = layer.num_parameters();
            total_params += params;

            layer_info.push(LayerInfo {
                index: i,
                name: layer.name().to_string(),
                parameters: params,
                output_shape: Vec::new(), // Would need input shape to calculate
            });
        }

        NetworkSummary {
            layers: layer_info,
            total_parameters: total_params,
            loss_function: format!("{}", self.loss_function),
            optimizer: self.optimizer.name().to_string(),
            device: format!("{:?}", self.device.device_type()),
        }
    }

    // Helper methods
    fn collect_parameters(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| {
                layer
                    .parameters()
                    .into_iter()
                    .map(|p| p.clone_data().unwrap())
            })
            .collect()
    }

    fn collect_gradients(&self) -> Vec<Tensor> {
        self.layers
            .iter()
            .flat_map(|layer| {
                layer
                    .gradients()
                    .into_iter()
                    .map(|g| g.clone_data().unwrap())
            })
            .collect()
    }

    fn update_parameters(&mut self, params: Vec<Tensor>) -> Result<()> {
        let mut param_idx = 0;
        for layer in &mut self.layers {
            let mut layer_params = layer.parameters_mut();
            for param in layer_params.iter_mut() {
                if param_idx < params.len() {
                    let new_data = params[param_idx].to_vec()?;
                    param.copy_from_slice(&new_data)?;
                    param_idx += 1;
                }
            }
        }
        Ok(())
    }

    /// Debug helper: check if weights are properly initialized
    pub fn check_weight_initialization(&self) -> Result<()> {
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let params = layer.parameters();
            for (param_idx, param) in params.iter().enumerate() {
                let data = param.to_vec()?;
                let all_zeros = data.iter().all(|&x| x == 0.0);
                if all_zeros {
                    println!(
                        "WARNING: Layer {} parameter {} is all zeros!",
                        layer_idx, param_idx
                    );
                } else {
                    println!(
                        "Layer {} parameter {}: first few values [{:.6}, {:.6}, {:.6}, ...]",
                        layer_idx,
                        param_idx,
                        data.get(0).unwrap_or(&0.0),
                        data.get(1).unwrap_or(&0.0),
                        data.get(2).unwrap_or(&0.0)
                    );
                }
            }
        }
        Ok(())
    }

    /// Get read-only access to layers for debugging
    pub fn layers(&self) -> &[Box<dyn Layer>] {
        &self.layers
    }

    fn is_classification_task(&self) -> bool {
        matches!(
            self.loss_function,
            LossFunction::CrossEntropy | LossFunction::BinaryCrossEntropy
        )
    }

    fn is_correct_prediction(&self, prediction: &Tensor, target: &Tensor) -> Result<bool> {
        let pred_data = prediction.to_vec()?;
        let target_data = target.to_vec()?;

        match self.loss_function {
            LossFunction::CrossEntropy => {
                // Multi-class classification: argmax
                let pred_class = pred_data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                let target_class = target_data
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);

                Ok(pred_class == target_class)
            }
            LossFunction::BinaryCrossEntropy => {
                // Binary classification: threshold at 0.5
                let predicted = if pred_data[0] > 0.5 { 1.0 } else { 0.0 };
                Ok((predicted - target_data[0]).abs() < 0.5)
            }
            _ => Ok(false), // Not a classification task
        }
    }

    fn apply_lr_schedule(
        &mut self,
        schedule: &LearningRateSchedule,
        epoch: usize,
        metrics: &TrainingMetrics,
    ) -> Result<()> {
        let new_lr = match schedule {
            LearningRateSchedule::StepLR { step_size, gamma } => {
                if (epoch + 1) % step_size == 0 {
                    self.optimizer.learning_rate() * gamma
                } else {
                    self.optimizer.learning_rate()
                }
            }
            LearningRateSchedule::ExponentialLR { gamma } => self.optimizer.learning_rate() * gamma,
            LearningRateSchedule::ReduceOnPlateau {
                factor,
                patience: _,
                threshold,
                ..
            } => {
                // Simplified implementation - would need loss history
                if metrics.loss > *threshold {
                    self.optimizer.learning_rate() * factor
                } else {
                    self.optimizer.learning_rate()
                }
            }
            LearningRateSchedule::CosineAnnealingLR { t_max, eta_min } => {
                let base_lr = self.optimizer.learning_rate();
                eta_min
                    + (base_lr - eta_min)
                        * (1.0 + (std::f32::consts::PI * epoch as f32 / *t_max as f32).cos())
                        / 2.0
            }
            LearningRateSchedule::PolynomialLR {
                total_epochs,
                power,
            } => {
                let base_lr = self.optimizer.learning_rate();
                base_lr * (1.0 - epoch as f32 / *total_epochs as f32).powf(*power)
            }
        };

        if new_lr != self.optimizer.learning_rate() {
            self.optimizer.set_learning_rate(new_lr);
        }

        Ok(())
    }
}

/// Network architecture summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkSummary {
    pub layers: Vec<LayerInfo>,
    pub total_parameters: usize,
    pub loss_function: String,
    pub optimizer: String,
    pub device: String,
}

/// Layer information for summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    pub index: usize,
    pub name: String,
    pub parameters: usize,
    pub output_shape: Vec<usize>,
}

impl fmt::Display for Network {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Neural Network Summary:")?;
        writeln!(f, "======================")?;
        writeln!(f, "Total Layers: {}", self.num_layers())?;
        writeln!(f, "Total Parameters: {}", self.num_parameters())?;
        writeln!(f, "Loss Function: {}", self.loss_function)?;
        writeln!(f, "Optimizer: {}", self.optimizer.name())?;
        writeln!(f, "Device: {:?}", self.device.device_type())?;
        writeln!(f)?;
        writeln!(f, "Layers:")?;
        for (i, layer) in self.layers.iter().enumerate() {
            writeln!(
                f,
                "  {}: {} ({} params)",
                i,
                layer.name(),
                layer.num_parameters()
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for NetworkSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Network Summary")?;
        writeln!(f, "===============")?;
        writeln!(f, "Total Parameters: {}", self.total_parameters)?;
        writeln!(f, "Loss Function: {}", self.loss_function)?;
        writeln!(f, "Optimizer: {}", self.optimizer)?;
        writeln!(f, "Device: {}", self.device)?;
        writeln!(f)?;
        writeln!(f, "Layers:")?;
        for layer in &self.layers {
            writeln!(
                f,
                "  {}: {} ({} params)",
                layer.index, layer.name, layer.parameters
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activations::Activation;
    use crate::layers::{LayerConfig, WeightInit};
    use crate::losses::LossFunction;
    use crate::optimizers::OptimizerConfig;

    #[test]
    fn test_network_creation() {
        let layers = vec![
            create_layer(
                LayerConfig::Dense {
                    input_size: 2,
                    output_size: 4,
                    activation: crate::activations::Activation::ReLU,
                    use_bias: true,
                    weight_init: crate::layers::WeightInit::Xavier,
                },
                Device::cpu().unwrap(),
            )
            .unwrap(),
            create_layer(
                LayerConfig::Dense {
                    input_size: 4,
                    output_size: 1,
                    activation: crate::activations::Activation::Sigmoid,
                    use_bias: true,
                    weight_init: crate::layers::WeightInit::Xavier,
                },
                Device::cpu().unwrap(),
            )
            .unwrap(),
        ];

        let loss = LossFunction::MeanSquaredError;
        let optimizer = create_optimizer(crate::optimizers::OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        })
        .unwrap();
        let device = Device::cpu().unwrap();

        let network = Network::new(layers, loss, optimizer, device);
        assert!(network.is_ok());

        let network = network.unwrap();
        assert_eq!(network.num_layers(), 2);
        assert_eq!(network.num_parameters(), 2 * 4 + 4 + 4 * 1 + 1); // weights + biases
    }

    #[test]
    fn test_network_forward() {
        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 1,
                activation: Activation::Linear,
                use_bias: false,
                weight_init: WeightInit::Ones,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::sgd(0.1))
            .build()
            .unwrap();

        let input = Tensor::from_slice(&[1.0, 2.0], &[1, 2]).unwrap();
        let output = network.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 1]);
        // With weights = [1, 1] and input = [1, 2], output should be [3]
        let output_data = output.to_vec().unwrap();
        assert!((output_data[0] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_network_training() {
        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 1,
                activation: Activation::Sigmoid,
                use_bias: true,
                weight_init: WeightInit::Xavier,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::sgd(0.1))
            .build()
            .unwrap();

        // XOR problem
        let inputs = vec![
            Tensor::from_slice(&[0.0, 0.0], &[1, 2]).unwrap(),
            Tensor::from_slice(&[0.0, 1.0], &[1, 2]).unwrap(),
            Tensor::from_slice(&[1.0, 0.0], &[1, 2]).unwrap(),
            Tensor::from_slice(&[1.0, 1.0], &[1, 2]).unwrap(),
        ];
        let targets = vec![
            Tensor::from_slice(&[0.0], &[1, 1]).unwrap(),
            Tensor::from_slice(&[1.0], &[1, 1]).unwrap(),
            Tensor::from_slice(&[1.0], &[1, 1]).unwrap(),
            Tensor::from_slice(&[0.0], &[1, 1]).unwrap(),
        ];

        let config = TrainingConfig {
            epochs: 5,
            batch_size: 4,
            verbose: false,
            early_stopping_patience: 0,
            early_stopping_threshold: 0.0,
            lr_schedule: None,
            validation_split: 0.0,
            shuffle: false,
            random_seed: None,
        };

        let history = network.train(&inputs, &targets, &config);
        assert!(history.is_ok());

        let history = history.unwrap();
        assert_eq!(history.epochs(), 5);
        assert!(history.final_loss() > 0.0);
    }

    #[test]
    fn test_network_evaluation() {
        let mut network = NetworkBuilder::new()
            .add_layer(LayerConfig::Dense {
                input_size: 2,
                output_size: 1,
                activation: Activation::Linear,
                use_bias: false,
                weight_init: WeightInit::Ones,
            })
            .loss(LossFunction::MeanSquaredError)
            .optimizer(OptimizerConfig::sgd(0.1))
            .build()
            .unwrap();

        let inputs = vec![
            Tensor::from_slice(&[1.0, 1.0], &[1, 2]).unwrap(),
            Tensor::from_slice(&[2.0, 2.0], &[1, 2]).unwrap(),
        ];
        let targets = vec![
            Tensor::from_slice(&[2.0], &[1, 1]).unwrap(),
            Tensor::from_slice(&[4.0], &[1, 1]).unwrap(),
        ];

        let metrics = network.evaluate(&inputs, &targets).unwrap();
        assert!(metrics.loss >= 0.0);
    }

    #[test]
    fn test_network_summary() {
        let network = NetworkBuilder::new()
            .add_layer(LayerConfig::dense_relu(784, 128))
            .add_layer(LayerConfig::dense_relu(128, 64))
            .add_layer(LayerConfig::dense_linear(64, 10))
            .loss(LossFunction::CrossEntropy)
            .optimizer(OptimizerConfig::adam(0.001))
            .build()
            .unwrap();

        let summary = network.summary();
        assert_eq!(summary.layers.len(), 3);
        assert_eq!(
            summary.total_parameters,
            784 * 128 + 128 + 128 * 64 + 64 + 64 * 10 + 10
        );
        assert!(summary.loss_function.contains("Cross Entropy"));
    }
}
