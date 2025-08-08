//! Training module for neural networks
//!
//! This module provides training configuration, metrics tracking, and
//! utilities for monitoring and controlling the training process.

use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fmt;

/// Training configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    /// Number of training epochs
    pub epochs: usize,
    /// Batch size for training
    pub batch_size: usize,
    /// Whether to print training progress
    pub verbose: bool,
    /// Early stopping patience (0 to disable)
    pub early_stopping_patience: usize,
    /// Early stopping threshold
    pub early_stopping_threshold: f32,
    /// Learning rate schedule
    pub lr_schedule: Option<LearningRateSchedule>,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: f32,
    /// Whether to shuffle data each epoch
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            verbose: true,
            early_stopping_patience: 0,
            early_stopping_threshold: 1e-4,
            lr_schedule: None,
            validation_split: 0.0,
            shuffle: true,
            random_seed: None,
        }
    }
}

/// Learning rate scheduling strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningRateSchedule {
    /// Step decay: multiply by gamma every step_size epochs
    StepLR {
        /// Number of epochs between each step
        step_size: usize,
        /// Multiplicative factor for learning rate decay
        gamma: f32,
    },
    /// Exponential decay: multiply by gamma each epoch
    ExponentialLR {
        /// Multiplicative factor for learning rate decay
        gamma: f32,
    },
    /// Reduce on plateau: reduce when metric stops improving
    ReduceOnPlateau {
        /// Factor by which the learning rate is reduced
        factor: f32,
        /// Number of epochs with no improvement after which learning rate is reduced
        patience: usize,
        /// Threshold for measuring the new optimum, to only focus on significant changes
        threshold: f32,
        /// Lower bound on the learning rate
        min_lr: f32,
    },
    /// Cosine annealing
    CosineAnnealingLR {
        /// Maximum number of iterations
        t_max: usize,
        /// Minimum learning rate
        eta_min: f32,
    },
    /// Polynomial decay
    PolynomialLR {
        /// Total number of training epochs
        total_epochs: usize,
        /// Power of the polynomial
        power: f32,
    },
}

/// Training metrics for a single epoch
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingMetrics {
    /// Training loss
    pub loss: f32,
    /// Training accuracy (if applicable)
    pub accuracy: f32,
    /// Current learning rate
    pub learning_rate: f32,
    /// Time taken for this epoch in milliseconds
    pub epoch_time_ms: f32,
    /// Validation loss (if validation data provided)
    pub val_loss: Option<f32>,
    /// Validation accuracy (if validation data provided)
    pub val_accuracy: Option<f32>,
    /// Additional custom metrics
    pub custom_metrics: std::collections::HashMap<String, f32>,
}

impl TrainingMetrics {
    /// Create new training metrics
    pub fn new(loss: f32, accuracy: f32, learning_rate: f32, epoch_time_ms: f32) -> Self {
        Self {
            loss,
            accuracy,
            learning_rate,
            epoch_time_ms,
            val_loss: None,
            val_accuracy: None,
            custom_metrics: std::collections::HashMap::new(),
        }
    }

    /// Add a custom metric
    pub fn add_metric<S: Into<String>>(&mut self, name: S, value: f32) {
        self.custom_metrics.insert(name.into(), value);
    }

    /// Get a custom metric
    pub fn get_metric(&self, name: &str) -> Option<f32> {
        self.custom_metrics.get(name).copied()
    }

    /// Set validation metrics
    pub fn set_validation(&mut self, val_loss: f32, val_accuracy: f32) {
        self.val_loss = Some(val_loss);
        self.val_accuracy = Some(val_accuracy);
    }
}

impl fmt::Display for TrainingMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Loss: {:.6}, Accuracy: {:.4}, LR: {:.6}, Time: {:.2}ms",
            self.loss, self.accuracy, self.learning_rate, self.epoch_time_ms
        )?;

        if let (Some(val_loss), Some(val_acc)) = (self.val_loss, self.val_accuracy) {
            write!(f, ", Val Loss: {:.6}, Val Acc: {:.4}", val_loss, val_acc)?;
        }

        Ok(())
    }
}

/// Training history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingHistory {
    /// Metrics for each epoch
    epochs: Vec<TrainingMetrics>,
    /// Best loss achieved
    best_loss: f32,
    /// Best accuracy achieved
    best_accuracy: f32,
    /// Epoch when best loss was achieved
    best_loss_epoch: usize,
    /// Epoch when best accuracy was achieved
    best_accuracy_epoch: usize,
    /// Early stopping counter
    early_stopping_counter: usize,
}

impl TrainingHistory {
    /// Create new training history
    pub fn new() -> Self {
        Self {
            epochs: Vec::new(),
            best_loss: f32::INFINITY,
            best_accuracy: 0.0,
            best_loss_epoch: 0,
            best_accuracy_epoch: 0,
            early_stopping_counter: 0,
        }
    }

    /// Add metrics for an epoch
    pub fn add_epoch(&mut self, metrics: TrainingMetrics) {
        let epoch = self.epochs.len();

        // Update best metrics
        if metrics.loss < self.best_loss {
            self.best_loss = metrics.loss;
            self.best_loss_epoch = epoch;
            self.early_stopping_counter = 0; // Reset early stopping counter
        } else {
            self.early_stopping_counter += 1;
        }

        if metrics.accuracy > self.best_accuracy {
            self.best_accuracy = metrics.accuracy;
            self.best_accuracy_epoch = epoch;
        }

        self.epochs.push(metrics);
    }

    /// Get number of epochs recorded
    pub fn epochs(&self) -> usize {
        self.epochs.len()
    }

    /// Get metrics for a specific epoch
    pub fn get_epoch(&self, epoch: usize) -> Option<&TrainingMetrics> {
        self.epochs.get(epoch)
    }

    /// Get all epoch metrics
    pub fn all_epochs(&self) -> &[TrainingMetrics] {
        &self.epochs
    }

    /// Get the latest epoch metrics
    pub fn latest(&self) -> Option<&TrainingMetrics> {
        self.epochs.last()
    }

    /// Get final loss
    pub fn final_loss(&self) -> f32 {
        self.epochs.last().map(|m| m.loss).unwrap_or(f32::INFINITY)
    }

    /// Get final accuracy
    pub fn final_accuracy(&self) -> f32 {
        self.epochs.last().map(|m| m.accuracy).unwrap_or(0.0)
    }

    /// Get best loss achieved
    pub fn best_loss(&self) -> f32 {
        self.best_loss
    }

    /// Get best accuracy achieved
    pub fn best_accuracy(&self) -> f32 {
        self.best_accuracy
    }

    /// Get epoch when best loss was achieved
    pub fn best_loss_epoch(&self) -> usize {
        self.best_loss_epoch
    }

    /// Get epoch when best accuracy was achieved
    pub fn best_accuracy_epoch(&self) -> usize {
        self.best_accuracy_epoch
    }

    /// Check if early stopping should be triggered
    pub fn should_early_stop(&self, patience: usize, threshold: f32) -> bool {
        if patience == 0 {
            return false;
        }

        self.early_stopping_counter >= patience && self.best_loss > threshold
    }

    /// Get loss history
    pub fn loss_history(&self) -> Vec<f32> {
        self.epochs.iter().map(|m| m.loss).collect()
    }

    /// Get accuracy history
    pub fn accuracy_history(&self) -> Vec<f32> {
        self.epochs.iter().map(|m| m.accuracy).collect()
    }

    /// Get learning rate history
    pub fn lr_history(&self) -> Vec<f32> {
        self.epochs.iter().map(|m| m.learning_rate).collect()
    }

    /// Get validation loss history (if available)
    pub fn val_loss_history(&self) -> Vec<f32> {
        self.epochs.iter().filter_map(|m| m.val_loss).collect()
    }

    /// Get validation accuracy history (if available)
    pub fn val_accuracy_history(&self) -> Vec<f32> {
        self.epochs.iter().filter_map(|m| m.val_accuracy).collect()
    }

    /// Calculate average loss over last N epochs
    pub fn average_loss(&self, n: usize) -> f32 {
        if self.epochs.is_empty() {
            return f32::INFINITY;
        }

        let start = self.epochs.len().saturating_sub(n);
        let losses: Vec<f32> = self.epochs[start..].iter().map(|m| m.loss).collect();
        losses.iter().sum::<f32>() / losses.len() as f32
    }

    /// Calculate average accuracy over last N epochs
    pub fn average_accuracy(&self, n: usize) -> f32 {
        if self.epochs.is_empty() {
            return 0.0;
        }

        let start = self.epochs.len().saturating_sub(n);
        let accuracies: Vec<f32> = self.epochs[start..].iter().map(|m| m.accuracy).collect();
        accuracies.iter().sum::<f32>() / accuracies.len() as f32
    }

    /// Check if training is improving (loss decreasing trend)
    pub fn is_improving(&self, window: usize) -> bool {
        if self.epochs.len() < window * 2 {
            return true; // Not enough data to determine trend
        }

        let recent_avg = self.average_loss(window);
        let older_avg = self.average_loss(window * 2) - recent_avg;

        recent_avg < older_avg
    }

    /// Get training statistics summary
    pub fn summary(&self) -> TrainingSummary {
        TrainingSummary {
            total_epochs: self.epochs(),
            best_loss: self.best_loss,
            best_accuracy: self.best_accuracy,
            final_loss: self.final_loss(),
            final_accuracy: self.final_accuracy(),
            best_loss_epoch: self.best_loss_epoch,
            best_accuracy_epoch: self.best_accuracy_epoch,
            total_time_ms: self.epochs.iter().map(|m| m.epoch_time_ms).sum(),
            average_epoch_time_ms: if self.epochs.is_empty() {
                0.0
            } else {
                self.epochs.iter().map(|m| m.epoch_time_ms).sum::<f32>() / self.epochs.len() as f32
            },
        }
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Training summary statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingSummary {
    /// Total number of training epochs completed
    pub total_epochs: usize,
    /// Best (lowest) loss achieved during training
    pub best_loss: f32,
    /// Best (highest) accuracy achieved during training
    pub best_accuracy: f32,
    /// Final loss at the end of training
    pub final_loss: f32,
    /// Final accuracy at the end of training
    pub final_accuracy: f32,
    /// Epoch number where the best loss was achieved
    pub best_loss_epoch: usize,
    /// Epoch number where the best accuracy was achieved
    pub best_accuracy_epoch: usize,
    /// Total training time in milliseconds
    pub total_time_ms: f32,
    /// Average time per epoch in milliseconds
    pub average_epoch_time_ms: f32,
}

impl fmt::Display for TrainingSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Training Summary")?;
        writeln!(f, "===============")?;
        writeln!(f, "Total Epochs: {}", self.total_epochs)?;
        writeln!(
            f,
            "Best Loss: {:.6} (epoch {})",
            self.best_loss, self.best_loss_epoch
        )?;
        writeln!(
            f,
            "Best Accuracy: {:.4} (epoch {})",
            self.best_accuracy, self.best_accuracy_epoch
        )?;
        writeln!(f, "Final Loss: {:.6}", self.final_loss)?;
        writeln!(f, "Final Accuracy: {:.4}", self.final_accuracy)?;
        writeln!(f, "Total Time: {:.2}s", self.total_time_ms / 1000.0)?;
        writeln!(f, "Average Epoch Time: {:.2}ms", self.average_epoch_time_ms)?;
        Ok(())
    }
}

/// Moving average tracker for smoothing metrics
#[derive(Debug, Clone)]
pub struct MovingAverage {
    window_size: usize,
    values: VecDeque<f32>,
    sum: f32,
}

impl MovingAverage {
    /// Create new moving average tracker
    pub fn new(window_size: usize) -> Self {
        Self {
            window_size,
            values: VecDeque::new(),
            sum: 0.0,
        }
    }

    /// Add a new value and get the current average
    pub fn update(&mut self, value: f32) -> f32 {
        self.values.push_back(value);
        self.sum += value;

        if self.values.len() > self.window_size {
            if let Some(old_value) = self.values.pop_front() {
                self.sum -= old_value;
            }
        }

        self.average()
    }

    /// Get current average
    pub fn average(&self) -> f32 {
        if self.values.is_empty() {
            0.0
        } else {
            self.sum / self.values.len() as f32
        }
    }

    /// Reset the moving average
    pub fn reset(&mut self) {
        self.values.clear();
        self.sum = 0.0;
    }
}

/// Training utilities
pub mod utils {

    /// Calculate exponential moving average
    pub fn ema(current: f32, previous: f32, alpha: f32) -> f32 {
        alpha * current + (1.0 - alpha) * previous
    }

    /// Calculate learning rate for cosine annealing
    pub fn cosine_annealing_lr(
        initial_lr: f32,
        current_epoch: usize,
        total_epochs: usize,
        eta_min: f32,
    ) -> f32 {
        if total_epochs == 0 {
            return initial_lr;
        }

        let progress = current_epoch as f32 / total_epochs as f32;
        eta_min + (initial_lr - eta_min) * (1.0 + (std::f32::consts::PI * progress).cos()) / 2.0
    }

    /// Calculate learning rate for polynomial decay
    pub fn polynomial_lr(
        initial_lr: f32,
        current_epoch: usize,
        total_epochs: usize,
        power: f32,
    ) -> f32 {
        if total_epochs == 0 {
            return initial_lr;
        }

        let progress = (current_epoch as f32 / total_epochs as f32).min(1.0);
        initial_lr * (1.0 - progress).powf(power)
    }

    /// Calculate warmup learning rate
    pub fn warmup_lr(target_lr: f32, current_step: usize, warmup_steps: usize) -> f32 {
        if warmup_steps == 0 {
            return target_lr;
        }

        let progress = (current_step as f32 / warmup_steps as f32).min(1.0);
        target_lr * progress
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert!(config.verbose);
        assert_eq!(config.early_stopping_patience, 0);
    }

    #[test]
    fn test_training_metrics() {
        let mut metrics = TrainingMetrics::new(0.5, 0.8, 0.001, 100.0);
        assert_eq!(metrics.loss, 0.5);
        assert_eq!(metrics.accuracy, 0.8);
        assert_eq!(metrics.learning_rate, 0.001);

        metrics.add_metric("precision", 0.85);
        assert_eq!(metrics.get_metric("precision"), Some(0.85));

        metrics.set_validation(0.6, 0.75);
        assert_eq!(metrics.val_loss, Some(0.6));
        assert_eq!(metrics.val_accuracy, Some(0.75));
    }

    #[test]
    fn test_training_history() {
        let mut history = TrainingHistory::new();

        // Add first epoch
        let metrics1 = TrainingMetrics::new(1.0, 0.5, 0.01, 100.0);
        history.add_epoch(metrics1);

        assert_eq!(history.epochs(), 1);
        assert_eq!(history.best_loss(), 1.0);
        assert_eq!(history.best_accuracy(), 0.5);

        // Add second epoch with better metrics
        let metrics2 = TrainingMetrics::new(0.8, 0.7, 0.01, 100.0);
        history.add_epoch(metrics2);

        assert_eq!(history.epochs(), 2);
        assert_eq!(history.best_loss(), 0.8);
        assert_eq!(history.best_accuracy(), 0.7);
        assert_eq!(history.best_loss_epoch(), 1);
        assert_eq!(history.best_accuracy_epoch(), 1);
    }

    #[test]
    fn test_early_stopping() {
        let mut history = TrainingHistory::new();

        // Add epochs with worsening loss
        for i in 0..5 {
            let loss = 1.0 + i as f32 * 0.1; // Increasing loss
            let metrics = TrainingMetrics::new(loss, 0.5, 0.01, 100.0);
            history.add_epoch(metrics);
        }

        // Should trigger early stopping with patience 3
        assert!(history.should_early_stop(3, 0.5));
        assert!(!history.should_early_stop(10, 0.5)); // Higher patience
    }

    #[test]
    fn test_moving_average() {
        let mut ma = MovingAverage::new(3);

        assert_eq!(ma.update(1.0), 1.0);
        assert_eq!(ma.update(2.0), 1.5);
        assert_eq!(ma.update(3.0), 2.0);
        assert_eq!(ma.update(4.0), 3.0); // (2+3+4)/3

        ma.reset();
        assert_eq!(ma.average(), 0.0);
    }

    #[test]
    fn test_cosine_annealing_lr() {
        let initial_lr = 0.1;
        let total_epochs = 100;
        let eta_min = 0.001;

        // At epoch 0, should be close to initial_lr
        let lr_0 = utils::cosine_annealing_lr(initial_lr, 0, total_epochs, eta_min);
        assert!((lr_0 - initial_lr).abs() < 1e-6);

        // At half way, should be somewhere in between
        let lr_50 = utils::cosine_annealing_lr(initial_lr, 50, total_epochs, eta_min);
        assert!(lr_50 > eta_min && lr_50 < initial_lr);

        // At the end, should be close to eta_min
        let lr_100 = utils::cosine_annealing_lr(initial_lr, 100, total_epochs, eta_min);
        assert!((lr_100 - eta_min).abs() < 1e-6);
    }

    #[test]
    fn test_polynomial_lr() {
        let initial_lr = 0.1;
        let total_epochs = 100;
        let power = 2.0;

        // At epoch 0, should be initial_lr
        let lr_0 = utils::polynomial_lr(initial_lr, 0, total_epochs, power);
        assert!((lr_0 - initial_lr).abs() < 1e-6);

        // At the end, should be close to 0
        let lr_100 = utils::polynomial_lr(initial_lr, 100, total_epochs, power);
        assert!(lr_100 < 1e-6);
    }

    #[test]
    fn test_warmup_lr() {
        let target_lr = 0.01;
        let warmup_steps = 1000;

        // At step 0, should be 0
        let lr_0 = utils::warmup_lr(target_lr, 0, warmup_steps);
        assert!(lr_0 < 1e-6);

        // At half warmup, should be half target
        let lr_500 = utils::warmup_lr(target_lr, 500, warmup_steps);
        assert!((lr_500 - target_lr / 2.0).abs() < 1e-6);

        // At end of warmup, should be target
        let lr_1000 = utils::warmup_lr(target_lr, 1000, warmup_steps);
        assert!((lr_1000 - target_lr).abs() < 1e-6);
    }

    #[test]
    fn test_training_summary() {
        let mut history = TrainingHistory::new();

        for i in 0..5 {
            let loss = 1.0 - i as f32 * 0.1; // Decreasing loss
            let accuracy = 0.5 + i as f32 * 0.1; // Increasing accuracy
            let metrics = TrainingMetrics::new(loss, accuracy, 0.01, 100.0);
            history.add_epoch(metrics);
        }

        let summary = history.summary();
        assert_eq!(summary.total_epochs, 5);
        assert_eq!(summary.best_loss, 0.6); // Last epoch had best loss
        assert_eq!(summary.best_accuracy, 0.9); // Last epoch had best accuracy
        assert_eq!(summary.total_time_ms, 500.0); // 5 epochs * 100ms each
        assert_eq!(summary.average_epoch_time_ms, 100.0);
    }
}
