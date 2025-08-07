//! Training algorithms and configurations for neural networks.
//!
//! This module provides various training methods including backpropagation, Newton's method,
//! and other advanced training algorithms with support for different training configurations.

use crate::error::{NetworkError, Result};
use crate::loss::LossFunction;
use crate::optimizer::{Optimizer, OptimizerType};
use ndarray::{Array1, Array2, Axis};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Enumeration of available training methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrainingMethod {
    /// Standard backpropagation with gradient descent
    Backpropagation,
    /// Newton's method (requires Hessian computation)
    Newton,
    /// Quasi-Newton methods (BFGS, L-BFGS)
    QuasiNewton,
    /// Conjugate Gradient
    ConjugateGradient,
    /// Levenberg-Marquardt algorithm
    LevenbergMarquardt,
    /// Evolutionary algorithms
    Evolutionary,
    /// Particle Swarm Optimization
    ParticleSwarm,
    /// Genetic Algorithm
    GeneticAlgorithm,
    /// Simulated Annealing
    SimulatedAnnealing,
}

/// Configuration for training process.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Maximum number of epochs
    pub max_epochs: usize,
    /// Batch size for mini-batch training
    pub batch_size: usize,
    /// Loss function to use
    pub loss_function: LossFunction,
    /// Optimizer configuration
    pub optimizer: Optimizer,
    /// Training method
    pub method: TrainingMethod,
    /// Validation split (0.0 to 1.0)
    pub validation_split: f64,
    /// Early stopping patience (epochs)
    pub early_stopping_patience: Option<usize>,
    /// Early stopping minimum delta
    pub early_stopping_min_delta: f64,
    /// Learning rate schedule
    pub lr_schedule: Option<LearningRateSchedule>,
    /// Verbose training output
    pub verbose: bool,
    /// Shuffle training data
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
    /// Metrics to track during training
    pub metrics: Vec<String>,
    /// Checkpoint frequency (epochs)
    pub checkpoint_frequency: Option<usize>,
    /// Maximum training time
    pub max_training_time: Option<Duration>,
    /// Gradient accumulation steps
    pub gradient_accumulation_steps: usize,
    /// Mixed precision training
    pub mixed_precision: bool,
    /// Data augmentation settings
    pub data_augmentation: Option<DataAugmentationConfig>,
    /// Regularization settings
    pub regularization: RegularizationConfig,
}

/// Learning rate scheduling strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Step decay: multiply by factor every step_size epochs
    StepDecay { step_size: usize, gamma: f64 },
    /// Exponential decay
    ExponentialDecay { gamma: f64 },
    /// Cosine annealing
    CosineAnnealing { t_max: usize, eta_min: f64 },
    /// Reduce on plateau
    ReduceOnPlateau {
        factor: f64,
        patience: usize,
        min_lr: f64,
    },
    /// Polynomial decay
    PolynomialDecay { power: f64, end_lr: f64 },
    /// Cyclic learning rate
    CyclicLR {
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        mode: String,
    },
    /// One cycle learning rate
    OneCycleLR {
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        anneal_strategy: String,
    },
}

/// Data augmentation configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataAugmentationConfig {
    /// Add Gaussian noise
    pub gaussian_noise: Option<f64>,
    /// Dropout rate for inputs
    pub input_dropout: Option<f64>,
    /// Label smoothing factor
    pub label_smoothing: Option<f64>,
    /// Mixup alpha parameter
    pub mixup_alpha: Option<f64>,
    /// Cutout probability
    pub cutout_prob: Option<f64>,
}

/// Regularization configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegularizationConfig {
    /// L1 regularization strength
    pub l1_lambda: Option<f64>,
    /// L2 regularization strength
    pub l2_lambda: Option<f64>,
    /// Dropout rate
    pub dropout_rate: Option<f64>,
    /// Batch normalization
    pub batch_norm: bool,
    /// Layer normalization
    pub layer_norm: bool,
    /// Spectral normalization
    pub spectral_norm: bool,
    /// Gradient penalty weight
    pub gradient_penalty: Option<f64>,
}

impl Default for RegularizationConfig {
    fn default() -> Self {
        Self {
            l1_lambda: None,
            l2_lambda: None,
            dropout_rate: None,
            batch_norm: false,
            layer_norm: false,
            spectral_norm: false,
            gradient_penalty: None,
        }
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            batch_size: 32,
            loss_function: LossFunction::MeanSquaredError,
            optimizer: Optimizer::adam(0.001).unwrap(),
            method: TrainingMethod::Backpropagation,
            validation_split: 0.2,
            early_stopping_patience: Some(10),
            early_stopping_min_delta: 1e-4,
            lr_schedule: None,
            verbose: true,
            shuffle: true,
            seed: None,
            metrics: vec!["loss".to_string()],
            checkpoint_frequency: None,
            max_training_time: None,
            gradient_accumulation_steps: 1,
            mixed_precision: false,
            data_augmentation: None,
            regularization: RegularizationConfig::default(),
        }
    }
}

/// Training history and metrics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Loss values for each epoch
    pub train_loss: Vec<f64>,
    /// Validation loss values
    pub val_loss: Vec<f64>,
    /// Training metrics
    pub train_metrics: HashMap<String, Vec<f64>>,
    /// Validation metrics
    pub val_metrics: HashMap<String, Vec<f64>>,
    /// Learning rate history
    pub learning_rate: Vec<f64>,
    /// Epoch times
    pub epoch_times: Vec<Duration>,
    /// Total training time
    pub total_time: Duration,
    /// Number of parameters
    pub parameter_count: usize,
    /// Best epoch (lowest validation loss)
    pub best_epoch: Option<usize>,
    /// Best validation loss
    pub best_val_loss: Option<f64>,
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self {
            train_loss: Vec::new(),
            val_loss: Vec::new(),
            train_metrics: HashMap::new(),
            val_metrics: HashMap::new(),
            learning_rate: Vec::new(),
            epoch_times: Vec::new(),
            total_time: Duration::new(0, 0),
            parameter_count: 0,
            best_epoch: None,
            best_val_loss: None,
        }
    }
}

/// Training state for tracking progress.
#[derive(Debug, Clone)]
pub struct TrainingState {
    /// Current epoch
    pub epoch: usize,
    /// Current batch
    pub batch: usize,
    /// Training history
    pub history: TrainingHistory,
    /// Early stopping counter
    pub early_stopping_counter: usize,
    /// Current learning rate
    pub current_lr: f64,
    /// Training start time
    pub start_time: Instant,
    /// Last checkpoint epoch
    pub last_checkpoint: Option<usize>,
    /// Best model weights (for early stopping)
    pub best_weights: Option<Vec<Array2<f64>>>,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            batch: 0,
            history: TrainingHistory::default(),
            early_stopping_counter: 0,
            current_lr: 0.001,
            start_time: Instant::now(),
            last_checkpoint: None,
            best_weights: None,
        }
    }
}

/// Training progress callback trait.
pub trait TrainingCallback {
    /// Called at the beginning of training
    fn on_train_begin(&mut self, _config: &TrainingConfig) {}

    /// Called at the end of training
    fn on_train_end(&mut self, _state: &TrainingState) {}

    /// Called at the beginning of each epoch
    fn on_epoch_begin(&mut self, _epoch: usize, _state: &TrainingState) {}

    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, _epoch: usize, _state: &mut TrainingState) {}

    /// Called at the beginning of each batch
    fn on_batch_begin(&mut self, _batch: usize, _state: &TrainingState) {}

    /// Called at the end of each batch
    fn on_batch_end(&mut self, _batch: usize, _state: &mut TrainingState) {}
}

/// Metrics computation utilities.
pub struct Metrics;

impl Metrics {
    /// Compute accuracy for classification
    pub fn accuracy(predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        let mut correct = 0;
        let mut total = 0;

        for (pred_row, target_row) in predictions.rows().into_iter().zip(targets.rows()) {
            let pred_class = pred_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            let target_class = target_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            if pred_class == target_class {
                correct += 1;
            }
            total += 1;
        }

        Ok(correct as f64 / total as f64)
    }

    /// Compute precision for binary classification
    pub fn precision(
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        threshold: f64,
    ) -> Result<f64> {
        let pred_binary = predictions.mapv(|x| if x > threshold { 1.0 } else { 0.0 });

        let mut true_positives = 0.0;
        let mut false_positives = 0.0;

        for (pred, target) in pred_binary.iter().zip(targets.iter()) {
            if *pred == 1.0 && *target == 1.0 {
                true_positives += 1.0;
            } else if *pred == 1.0 && *target == 0.0 {
                false_positives += 1.0;
            }
        }

        if true_positives + false_positives == 0.0 {
            Ok(0.0)
        } else {
            Ok(true_positives / (true_positives + false_positives))
        }
    }

    /// Compute recall for binary classification
    pub fn recall(predictions: &Array2<f64>, targets: &Array2<f64>, threshold: f64) -> Result<f64> {
        let pred_binary = predictions.mapv(|x| if x > threshold { 1.0 } else { 0.0 });

        let mut true_positives = 0.0;
        let mut false_negatives = 0.0;

        for (pred, target) in pred_binary.iter().zip(targets.iter()) {
            if *pred == 1.0 && *target == 1.0 {
                true_positives += 1.0;
            } else if *pred == 0.0 && *target == 1.0 {
                false_negatives += 1.0;
            }
        }

        if true_positives + false_negatives == 0.0 {
            Ok(0.0)
        } else {
            Ok(true_positives / (true_positives + false_negatives))
        }
    }

    /// Compute F1 score
    pub fn f1_score(
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        threshold: f64,
    ) -> Result<f64> {
        let precision = Self::precision(predictions, targets, threshold)?;
        let recall = Self::recall(predictions, targets, threshold)?;

        if precision + recall == 0.0 {
            Ok(0.0)
        } else {
            Ok(2.0 * precision * recall / (precision + recall))
        }
    }

    /// Compute Mean Absolute Error
    pub fn mae(predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());
        Ok(abs_diff.mean().unwrap())
    }

    /// Compute Root Mean Squared Error
    pub fn rmse(predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        let diff = predictions - targets;
        let squared_diff = diff.mapv(|x| x * x);
        Ok(squared_diff.mean().unwrap().sqrt())
    }

    /// Compute R² score (coefficient of determination)
    pub fn r2_score(predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        let target_mean = targets.mean().unwrap();
        let ss_tot = targets.mapv(|x| (x - target_mean).powi(2)).sum();
        let ss_res = (targets - predictions).mapv(|x| x.powi(2)).sum();

        if ss_tot == 0.0 {
            Ok(1.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}

/// Learning rate scheduler implementation.
pub struct LearningRateScheduler {
    schedule: LearningRateSchedule,
    initial_lr: f64,
    current_lr: f64,
    step_count: usize,
    plateau_history: Vec<f64>,
    best_metric: f64,
    patience_counter: usize,
}

impl LearningRateScheduler {
    /// Create a new learning rate scheduler.
    pub fn new(schedule: LearningRateSchedule, initial_lr: f64) -> Self {
        Self {
            schedule,
            initial_lr,
            current_lr: initial_lr,
            step_count: 0,
            plateau_history: Vec::new(),
            best_metric: f64::INFINITY,
            patience_counter: 0,
        }
    }

    /// Update the learning rate based on the schedule.
    pub fn step(&mut self, metric: Option<f64>) -> f64 {
        self.step_count += 1;

        match &self.schedule {
            LearningRateSchedule::StepDecay { step_size, gamma } => {
                if self.step_count % step_size == 0 {
                    self.current_lr *= gamma;
                }
            }
            LearningRateSchedule::ExponentialDecay { gamma } => {
                self.current_lr = self.initial_lr * gamma.powi(self.step_count as i32);
            }
            LearningRateSchedule::CosineAnnealing { t_max, eta_min } => {
                let t = self.step_count % t_max;
                self.current_lr = eta_min
                    + (self.initial_lr - eta_min)
                        * (1.0 + (std::f64::consts::PI * t as f64 / *t_max as f64).cos())
                        / 2.0;
            }
            LearningRateSchedule::ReduceOnPlateau {
                factor,
                patience,
                min_lr,
            } => {
                if let Some(current_metric) = metric {
                    if current_metric < self.best_metric {
                        self.best_metric = current_metric;
                        self.patience_counter = 0;
                    } else {
                        self.patience_counter += 1;
                        if self.patience_counter >= *patience {
                            self.current_lr = (self.current_lr * factor).max(*min_lr);
                            self.patience_counter = 0;
                        }
                    }
                }
            }
            LearningRateSchedule::PolynomialDecay { power, end_lr } => {
                let decay_factor = (1.0 - self.step_count as f64 / 1000.0).max(0.0); // Assuming 1000 total steps
                self.current_lr = (self.initial_lr - end_lr) * decay_factor.powf(*power) + end_lr;
            }
            LearningRateSchedule::CyclicLR {
                base_lr,
                max_lr,
                step_size_up,
                mode: _,
            } => {
                let cycle = (1.0 + self.step_count as f64 / (2.0 * *step_size_up as f64)).floor();
                let x = (self.step_count as f64 / *step_size_up as f64 - 2.0 * cycle + 1.0).abs();
                self.current_lr = base_lr + (max_lr - base_lr) * (1.0 - x).max(0.0);
            }
            LearningRateSchedule::OneCycleLR {
                max_lr,
                total_steps,
                pct_start,
                anneal_strategy: _,
            } => {
                let step_num = self.step_count.min(*total_steps);
                let pct = step_num as f64 / *total_steps as f64;

                if pct <= *pct_start {
                    self.current_lr =
                        self.initial_lr + (max_lr - self.initial_lr) * (pct / pct_start);
                } else {
                    let remaining_pct = (pct - pct_start) / (1.0 - pct_start);
                    self.current_lr = max_lr - (max_lr - self.initial_lr) * remaining_pct;
                }
            }
        }

        self.current_lr
    }

    /// Get the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Reset the scheduler.
    pub fn reset(&mut self) {
        self.current_lr = self.initial_lr;
        self.step_count = 0;
        self.plateau_history.clear();
        self.best_metric = f64::INFINITY;
        self.patience_counter = 0;
    }
}

/// Data loader for batch processing.
pub struct DataLoader {
    data: Array2<f64>,
    targets: Array2<f64>,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    current_index: usize,
}

impl DataLoader {
    /// Create a new data loader.
    pub fn new(
        data: Array2<f64>,
        targets: Array2<f64>,
        batch_size: usize,
        shuffle: bool,
    ) -> Result<Self> {
        if data.nrows() != targets.nrows() {
            return Err(NetworkError::dimension_mismatch(
                format!("data rows: {}", data.nrows()),
                format!("target rows: {}", targets.nrows()),
            ));
        }

        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        if shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }

        Ok(Self {
            data,
            targets,
            batch_size,
            shuffle,
            indices,
            current_index: 0,
        })
    }

    /// Get the next batch.
    pub fn next_batch(&mut self) -> Option<(Array2<f64>, Array2<f64>)> {
        if self.current_index >= self.indices.len() {
            return None;
        }

        let end_index = (self.current_index + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.current_index..end_index];

        let batch_data = self.data.select(Axis(0), batch_indices);
        let batch_targets = self.targets.select(Axis(0), batch_indices);

        self.current_index = end_index;

        Some((batch_data, batch_targets))
    }

    /// Reset the data loader for a new epoch.
    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            self.indices.shuffle(&mut rng);
        }
    }

    /// Get the number of batches.
    pub fn num_batches(&self) -> usize {
        (self.data.nrows() + self.batch_size - 1) / self.batch_size
    }

    /// Check if there are more batches.
    pub fn has_next(&self) -> bool {
        self.current_index < self.indices.len()
    }
}

/// Early stopping implementation.
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best_score: f64,
    counter: usize,
    best_weights: Option<Vec<Array2<f64>>>,
}

impl EarlyStopping {
    /// Create a new early stopping instance.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_score: f64::INFINITY,
            counter: 0,
            best_weights: None,
        }
    }

    /// Check if training should stop.
    pub fn should_stop(&mut self, score: f64, weights: Option<&[Array2<f64>]>) -> bool {
        if score < self.best_score - self.min_delta {
            self.best_score = score;
            self.counter = 0;
            if let Some(w) = weights {
                self.best_weights = Some(w.to_vec());
            }
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }

    /// Get the best weights.
    pub fn best_weights(&self) -> Option<&[Array2<f64>]> {
        self.best_weights.as_deref()
    }

    /// Get the best score.
    pub fn best_score(&self) -> f64 {
        self.best_score
    }

    /// Reset early stopping.
    pub fn reset(&mut self) {
        self.best_score = f64::INFINITY;
        self.counter = 0;
        self.best_weights = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_metrics_accuracy() {
        let predictions = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.1, 0.8, 0.1, // Predicted class 1
                0.7, 0.2, 0.1, // Predicted class 0
            ],
        )
        .unwrap();

        let targets = Array2::from_shape_vec(
            (2, 3),
            vec![
                0.0, 1.0, 0.0, // True class 1
                1.0, 0.0, 0.0, // True class 0
            ],
        )
        .unwrap();

        let accuracy = Metrics::accuracy(&predictions, &targets).unwrap();
        assert_abs_diff_eq!(accuracy, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_mae() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 2.5, 3.5]).unwrap();

        let mae = Metrics::mae(&predictions, &targets).unwrap();
        assert_abs_diff_eq!(mae, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_metrics_rmse() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 2.5, 3.5]).unwrap();

        let rmse = Metrics::rmse(&predictions, &targets).unwrap();
        assert_abs_diff_eq!(rmse, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_learning_rate_scheduler_step_decay() {
        let schedule = LearningRateSchedule::StepDecay {
            step_size: 2,
            gamma: 0.5,
        };
        let mut scheduler = LearningRateScheduler::new(schedule, 1.0);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.5);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.5);

        scheduler.step(None);
        assert_eq!(scheduler.get_lr(), 0.25);
    }

    #[test]
    fn test_learning_rate_scheduler_exponential_decay() {
        let schedule = LearningRateSchedule::ExponentialDecay { gamma: 0.9 };
        let mut scheduler = LearningRateScheduler::new(schedule, 1.0);

        assert_eq!(scheduler.get_lr(), 1.0);

        scheduler.step(None);
        assert_abs_diff_eq!(scheduler.get_lr(), 0.9, epsilon = 1e-10);

        scheduler.step(None);
        assert_abs_diff_eq!(scheduler.get_lr(), 0.81, epsilon = 1e-10);
    }

    #[test]
    fn test_data_loader() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let mut loader = DataLoader::new(data, targets, 2, false).unwrap();

        assert_eq!(loader.num_batches(), 2);
        assert!(loader.has_next());

        let (batch_data, batch_targets) = loader.next_batch().unwrap();
        assert_eq!(batch_data.nrows(), 2);
        assert_eq!(batch_targets.nrows(), 2);

        let (batch_data, batch_targets) = loader.next_batch().unwrap();
        assert_eq!(batch_data.nrows(), 2);
        assert_eq!(batch_targets.nrows(), 2);

        assert!(!loader.has_next());
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_early_stopping() {
        let mut early_stopping = EarlyStopping::new(2, 0.01);

        // First improvement
        assert!(!early_stopping.should_stop(1.0, None));
        assert_eq!(early_stopping.best_score(), 1.0);

        // Second improvement
        assert!(!early_stopping.should_stop(0.5, None));
        assert_eq!(early_stopping.best_score(), 0.5);

        // No improvement, but within patience
        assert!(!early_stopping.should_stop(0.6, None));

        // No improvement, still within patience
        assert!(!early_stopping.should_stop(0.7, None));

        // No improvement, exceeds patience
        assert!(early_stopping.should_stop(0.8, None));
    }

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.max_epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.validation_split, 0.2);
        assert!(config.verbose);
        assert!(config.shuffle);
    }

    #[test]
    fn test_training_history_default() {
        let history = TrainingHistory::default();
        assert!(history.train_loss.is_empty());
        assert!(history.val_loss.is_empty());
        assert!(history.train_metrics.is_empty());
        assert!(history.val_metrics.is_empty());
        assert_eq!(history.parameter_count, 0);
    }

    #[test]
    fn test_dimension_mismatch_errors() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        assert!(Metrics::accuracy(&predictions, &targets).is_err());
        assert!(Metrics::mae(&predictions, &targets).is_err());
        assert!(Metrics::rmse(&predictions, &targets).is_err());
    }
}
