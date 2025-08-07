//! Loss functions for neural network training.
//!
//! This module provides various loss functions commonly used in neural networks,
//! along with their derivatives for backpropagation.

use crate::error::{NetworkError, Result};
use ndarray::{Array1, Array2, Zip};
use serde::{Deserialize, Serialize};
use std::f64::consts::E;

/// Enumeration of available loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error (MSE) - for regression
    MeanSquaredError,
    /// Mean Absolute Error (MAE) - for regression
    MeanAbsoluteError,
    /// Binary Cross-Entropy - for binary classification
    BinaryCrossEntropy,
    /// Categorical Cross-Entropy - for multi-class classification
    CategoricalCrossEntropy,
    /// Sparse Categorical Cross-Entropy - for multi-class with integer labels
    SparseCategoricalCrossEntropy,
    /// Huber Loss - robust regression loss
    HuberLoss,
    /// Hinge Loss - for SVM-style classification
    HingeLoss,
    /// Squared Hinge Loss
    SquaredHingeLoss,
    /// Kullback-Leibler Divergence
    KLDivergence,
    /// Poisson Loss - for count data
    PoissonLoss,
    /// Cosine Similarity Loss
    CosineSimilarityLoss,
    /// Log-Cosh Loss - smooth approximation to MAE
    LogCoshLoss,
    /// Quantile Loss - for quantile regression
    QuantileLoss,
}

impl LossFunction {
    /// Compute the loss between predictions and targets.
    pub fn compute(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        if predictions.is_empty() {
            return Err(NetworkError::computation(
                "Cannot compute loss on empty arrays",
            ));
        }

        let loss = match self {
            LossFunction::MeanSquaredError => self.mse(predictions, targets)?,
            LossFunction::MeanAbsoluteError => self.mae(predictions, targets)?,
            LossFunction::BinaryCrossEntropy => self.binary_cross_entropy(predictions, targets)?,
            LossFunction::CategoricalCrossEntropy => {
                self.categorical_cross_entropy(predictions, targets)?
            }
            LossFunction::SparseCategoricalCrossEntropy => {
                self.sparse_categorical_cross_entropy(predictions, targets)?
            }
            LossFunction::HuberLoss => self.huber_loss(predictions, targets, 1.0)?,
            LossFunction::HingeLoss => self.hinge_loss(predictions, targets)?,
            LossFunction::SquaredHingeLoss => self.squared_hinge_loss(predictions, targets)?,
            LossFunction::KLDivergence => self.kl_divergence(predictions, targets)?,
            LossFunction::PoissonLoss => self.poisson_loss(predictions, targets)?,
            LossFunction::CosineSimilarityLoss => {
                self.cosine_similarity_loss(predictions, targets)?
            }
            LossFunction::LogCoshLoss => self.log_cosh_loss(predictions, targets)?,
            LossFunction::QuantileLoss => self.quantile_loss(predictions, targets, 0.5)?,
        };

        if !loss.is_finite() {
            return Err(NetworkError::numerical(format!(
                "Loss computation resulted in non-finite value: {}",
                loss
            )));
        }

        Ok(loss)
    }

    /// Compute the gradient of the loss with respect to predictions.
    pub fn gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        if predictions.shape() != targets.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", predictions.shape()),
                format!("{:?}", targets.shape()),
            ));
        }

        let gradient = match self {
            LossFunction::MeanSquaredError => self.mse_gradient(predictions, targets),
            LossFunction::MeanAbsoluteError => self.mae_gradient(predictions, targets),
            LossFunction::BinaryCrossEntropy => {
                self.binary_cross_entropy_gradient(predictions, targets)?
            }
            LossFunction::CategoricalCrossEntropy => {
                self.categorical_cross_entropy_gradient(predictions, targets)?
            }
            LossFunction::SparseCategoricalCrossEntropy => {
                self.sparse_categorical_cross_entropy_gradient(predictions, targets)?
            }
            LossFunction::HuberLoss => self.huber_loss_gradient(predictions, targets, 1.0),
            LossFunction::HingeLoss => self.hinge_loss_gradient(predictions, targets),
            LossFunction::SquaredHingeLoss => {
                self.squared_hinge_loss_gradient(predictions, targets)
            }
            LossFunction::KLDivergence => self.kl_divergence_gradient(predictions, targets)?,
            LossFunction::PoissonLoss => self.poisson_loss_gradient(predictions, targets),
            LossFunction::CosineSimilarityLoss => {
                self.cosine_similarity_loss_gradient(predictions, targets)?
            }
            LossFunction::LogCoshLoss => self.log_cosh_loss_gradient(predictions, targets),
            LossFunction::QuantileLoss => self.quantile_loss_gradient(predictions, targets, 0.5),
        };

        // Check for non-finite gradients
        if gradient.iter().any(|&g| !g.is_finite()) {
            return Err(NetworkError::numerical(
                "Gradient computation resulted in non-finite values",
            ));
        }

        Ok(gradient)
    }

    /// Get the name of the loss function.
    pub fn name(&self) -> &'static str {
        match self {
            LossFunction::MeanSquaredError => "MeanSquaredError",
            LossFunction::MeanAbsoluteError => "MeanAbsoluteError",
            LossFunction::BinaryCrossEntropy => "BinaryCrossEntropy",
            LossFunction::CategoricalCrossEntropy => "CategoricalCrossEntropy",
            LossFunction::SparseCategoricalCrossEntropy => "SparseCategoricalCrossEntropy",
            LossFunction::HuberLoss => "HuberLoss",
            LossFunction::HingeLoss => "HingeLoss",
            LossFunction::SquaredHingeLoss => "SquaredHingeLoss",
            LossFunction::KLDivergence => "KLDivergence",
            LossFunction::PoissonLoss => "PoissonLoss",
            LossFunction::CosineSimilarityLoss => "CosineSimilarityLoss",
            LossFunction::LogCoshLoss => "LogCoshLoss",
            LossFunction::QuantileLoss => "QuantileLoss",
        }
    }

    /// Check if this loss function is suitable for classification tasks.
    pub fn is_classification_loss(&self) -> bool {
        matches!(
            self,
            LossFunction::BinaryCrossEntropy
                | LossFunction::CategoricalCrossEntropy
                | LossFunction::SparseCategoricalCrossEntropy
                | LossFunction::HingeLoss
                | LossFunction::SquaredHingeLoss
        )
    }

    /// Check if this loss function is suitable for regression tasks.
    pub fn is_regression_loss(&self) -> bool {
        matches!(
            self,
            LossFunction::MeanSquaredError
                | LossFunction::MeanAbsoluteError
                | LossFunction::HuberLoss
                | LossFunction::LogCoshLoss
                | LossFunction::QuantileLoss
        )
    }

    // Individual loss function implementations

    /// Mean Squared Error loss
    fn mse(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let diff = predictions - targets;
        let squared_diff = &diff * &diff;
        Ok(squared_diff.mean().unwrap())
    }

    /// MSE gradient
    fn mse_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let n = predictions.len() as f64;
        2.0 * (predictions - targets) / n
    }

    /// Mean Absolute Error loss
    fn mae(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());
        Ok(abs_diff.mean().unwrap())
    }

    /// MAE gradient
    fn mae_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let n = predictions.len() as f64;
        let diff = predictions - targets;
        diff.mapv(|x| x.signum()) / n
    }

    /// Binary Cross-Entropy loss
    fn binary_cross_entropy(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        let epsilon = 1e-15; // Small value to prevent log(0)
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));

        let loss = Zip::from(targets)
            .and(&clipped_predictions)
            .fold(0.0, |acc, &t, &p| {
                acc - (t * p.ln() + (1.0 - t) * (1.0 - p).ln())
            });

        Ok(loss / predictions.len() as f64)
    }

    /// Binary Cross-Entropy gradient
    fn binary_cross_entropy_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        let n = predictions.len() as f64;

        Ok(Zip::from(targets)
            .and(&clipped_predictions)
            .map_collect(|&t, &p| (p - t) / (p * (1.0 - p)) / n))
    }

    /// Categorical Cross-Entropy loss
    fn categorical_cross_entropy(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));

        let loss = Zip::from(targets)
            .and(&clipped_predictions)
            .fold(0.0, |acc, &t, &p| acc - t * p.ln());

        Ok(loss / predictions.nrows() as f64)
    }

    /// Categorical Cross-Entropy gradient
    fn categorical_cross_entropy_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon).min(1.0 - epsilon));
        let n = predictions.nrows() as f64;

        Ok(Zip::from(targets)
            .and(&clipped_predictions)
            .map_collect(|&t, &p| -t / p / n))
    }

    /// Sparse Categorical Cross-Entropy loss
    fn sparse_categorical_cross_entropy(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        // For simplicity, treat as categorical cross-entropy
        // In practice, targets would be integer indices
        self.categorical_cross_entropy(predictions, targets)
    }

    /// Sparse Categorical Cross-Entropy gradient
    fn sparse_categorical_cross_entropy_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        self.categorical_cross_entropy_gradient(predictions, targets)
    }

    /// Huber loss (smooth combination of MSE and MAE)
    fn huber_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        delta: f64,
    ) -> Result<f64> {
        let diff = predictions - targets;
        let abs_diff = diff.mapv(|x| x.abs());

        let loss = Zip::from(&abs_diff)
            .and(&diff)
            .fold(0.0, |acc, &abs_d, &d| {
                if abs_d <= delta {
                    acc + 0.5 * d * d
                } else {
                    acc + delta * (abs_d - 0.5 * delta)
                }
            });

        Ok(loss / predictions.len() as f64)
    }

    /// Huber loss gradient
    fn huber_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        delta: f64,
    ) -> Array2<f64> {
        let diff = predictions - targets;
        let n = predictions.len() as f64;

        diff.mapv(|d| {
            if d.abs() <= delta {
                d / n
            } else {
                delta * d.signum() / n
            }
        })
    }

    /// Hinge loss for SVM-style classification
    fn hinge_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let loss = Zip::from(targets)
            .and(predictions)
            .fold(0.0, |acc, &t, &p| acc + (1.0 - t * p).max(0.0));

        Ok(loss / predictions.len() as f64)
    }

    /// Hinge loss gradient
    fn hinge_loss_gradient(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Array2<f64> {
        let n = predictions.len() as f64;

        Zip::from(targets)
            .and(predictions)
            .map_collect(|&t, &p| if t * p < 1.0 { -t / n } else { 0.0 })
    }

    /// Squared Hinge loss
    fn squared_hinge_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let loss = Zip::from(targets)
            .and(predictions)
            .fold(0.0, |acc, &t, &p| {
                let margin = (1.0 - t * p).max(0.0);
                acc + margin * margin
            });

        Ok(loss / predictions.len() as f64)
    }

    /// Squared Hinge loss gradient
    fn squared_hinge_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let n = predictions.len() as f64;

        Zip::from(targets).and(predictions).map_collect(|&t, &p| {
            if t * p < 1.0 {
                -2.0 * t * (1.0 - t * p) / n
            } else {
                0.0
            }
        })
    }

    /// Kullback-Leibler Divergence
    fn kl_divergence(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon));
        let clipped_targets = targets.mapv(|t| t.max(epsilon));

        let loss = Zip::from(&clipped_targets)
            .and(&clipped_predictions)
            .fold(0.0, |acc, &t, &p| acc + t * (t / p).ln());

        Ok(loss / predictions.len() as f64)
    }

    /// KL Divergence gradient
    fn kl_divergence_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon));
        let clipped_targets = targets.mapv(|t| t.max(epsilon));
        let n = predictions.len() as f64;

        Ok(Zip::from(&clipped_targets)
            .and(&clipped_predictions)
            .map_collect(|&t, &p| -t / p / n))
    }

    /// Poisson loss for count data
    fn poisson_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon));

        let loss = Zip::from(targets)
            .and(&clipped_predictions)
            .fold(0.0, |acc, &t, &p| acc + p - t * p.ln());

        Ok(loss / predictions.len() as f64)
    }

    /// Poisson loss gradient
    fn poisson_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let epsilon = 1e-15;
        let clipped_predictions = predictions.mapv(|p| p.max(epsilon));
        let n = predictions.len() as f64;

        Zip::from(targets)
            .and(&clipped_predictions)
            .map_collect(|&t, &p| (1.0 - t / p) / n)
    }

    /// Cosine Similarity loss
    fn cosine_similarity_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<f64> {
        let mut total_loss = 0.0;

        for (pred_row, target_row) in predictions.rows().into_iter().zip(targets.rows()) {
            let pred_norm = (pred_row.mapv(|x| x * x).sum()).sqrt();
            let target_norm = (target_row.mapv(|x| x * x).sum()).sqrt();
            let dot_product = pred_row.dot(&target_row);

            if pred_norm == 0.0 || target_norm == 0.0 {
                total_loss += 1.0; // Maximum dissimilarity
            } else {
                let cosine_sim = dot_product / (pred_norm * target_norm);
                total_loss += 1.0 - cosine_sim; // Convert similarity to loss
            }
        }

        Ok(total_loss / predictions.nrows() as f64)
    }

    /// Cosine Similarity loss gradient
    fn cosine_similarity_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Result<Array2<f64>> {
        let mut gradient = Array2::zeros(predictions.raw_dim());

        for (i, (pred_row, target_row)) in predictions
            .rows()
            .into_iter()
            .zip(targets.rows())
            .enumerate()
        {
            let pred_norm = (pred_row.mapv(|x| x * x).sum()).sqrt();
            let target_norm = (target_row.mapv(|x| x * x).sum()).sqrt();

            if pred_norm != 0.0 && target_norm != 0.0 {
                let dot_product = pred_row.dot(&target_row);
                let cosine_sim = dot_product / (pred_norm * target_norm);

                for j in 0..pred_row.len() {
                    let grad = -(target_row[j] / (pred_norm * target_norm)
                        - cosine_sim * pred_row[j] / (pred_norm * pred_norm))
                        / predictions.nrows() as f64;
                    gradient[[i, j]] = grad;
                }
            }
        }

        Ok(gradient)
    }

    /// Log-Cosh loss (smooth approximation to MAE)
    fn log_cosh_loss(&self, predictions: &Array2<f64>, targets: &Array2<f64>) -> Result<f64> {
        let diff = predictions - targets;
        let loss = diff.mapv(|x| x.cosh().ln()).sum();
        Ok(loss / predictions.len() as f64)
    }

    /// Log-Cosh loss gradient
    fn log_cosh_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> Array2<f64> {
        let diff = predictions - targets;
        let n = predictions.len() as f64;
        diff.mapv(|x| x.tanh() / n)
    }

    /// Quantile loss for quantile regression
    fn quantile_loss(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        quantile: f64,
    ) -> Result<f64> {
        let diff = targets - predictions;
        let loss = Zip::from(&diff).fold(0.0, |acc, &d| {
            if d >= 0.0 {
                acc + quantile * d
            } else {
                acc + (quantile - 1.0) * d
            }
        });

        Ok(loss / predictions.len() as f64)
    }

    /// Quantile loss gradient
    fn quantile_loss_gradient(
        &self,
        predictions: &Array2<f64>,
        targets: &Array2<f64>,
        quantile: f64,
    ) -> Array2<f64> {
        let diff = targets - predictions;
        let n = predictions.len() as f64;

        diff.mapv(|d| {
            if d >= 0.0 {
                -quantile / n
            } else {
                -(quantile - 1.0) / n
            }
        })
    }
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::MeanSquaredError
    }
}

/// Configuration for parameterized loss functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossConfig {
    pub function: LossFunction,
    pub parameters: LossParameters,
}

/// Parameters for loss functions that require additional configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LossParameters {
    /// Delta parameter for Huber loss
    pub huber_delta: Option<f64>,
    /// Quantile parameter for Quantile loss
    pub quantile: Option<f64>,
    /// Label smoothing factor for cross-entropy losses
    pub label_smoothing: Option<f64>,
    /// Class weights for imbalanced classification
    pub class_weights: Option<Vec<f64>>,
}

impl Default for LossParameters {
    fn default() -> Self {
        Self {
            huber_delta: Some(1.0),
            quantile: Some(0.5),
            label_smoothing: None,
            class_weights: None,
        }
    }
}

impl LossConfig {
    /// Create a new loss configuration.
    pub fn new(function: LossFunction) -> Self {
        Self {
            function,
            parameters: LossParameters::default(),
        }
    }

    /// Set Huber loss delta parameter.
    pub fn with_huber_delta(mut self, delta: f64) -> Result<Self> {
        if delta <= 0.0 {
            return Err(NetworkError::invalid_parameter(
                "huber_delta",
                &delta.to_string(),
                "must be positive",
            ));
        }
        self.parameters.huber_delta = Some(delta);
        Ok(self)
    }

    /// Set quantile parameter.
    pub fn with_quantile(mut self, quantile: f64) -> Result<Self> {
        if quantile <= 0.0 || quantile >= 1.0 {
            return Err(NetworkError::invalid_parameter(
                "quantile",
                &quantile.to_string(),
                "must be in range (0, 1)",
            ));
        }
        self.parameters.quantile = Some(quantile);
        Ok(self)
    }

    /// Set label smoothing factor.
    pub fn with_label_smoothing(mut self, smoothing: f64) -> Result<Self> {
        if smoothing < 0.0 || smoothing >= 1.0 {
            return Err(NetworkError::invalid_parameter(
                "label_smoothing",
                &smoothing.to_string(),
                "must be in range [0, 1)",
            ));
        }
        self.parameters.label_smoothing = Some(smoothing);
        Ok(self)
    }

    /// Set class weights.
    pub fn with_class_weights(mut self, weights: Vec<f64>) -> Result<Self> {
        if weights.iter().any(|&w| w <= 0.0) {
            return Err(NetworkError::invalid_parameter(
                "class_weights",
                &format!("{:?}", weights),
                "all weights must be positive",
            ));
        }
        self.parameters.class_weights = Some(weights);
        Ok(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_mse_loss() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 2.5, 3.5]).unwrap();

        let loss = LossFunction::MeanSquaredError
            .compute(&predictions, &targets)
            .unwrap();

        // Expected: ((1-1.5)² + (2-1.5)² + (3-2.5)² + (4-3.5)²) / 4 = (0.25 + 0.25 + 0.25 + 0.25) / 4 = 0.25
        assert_abs_diff_eq!(loss, 0.25, epsilon = 1e-10);
    }

    #[test]
    fn test_mse_gradient() {
        let predictions = Array2::from_shape_vec((1, 2), vec![2.0, 3.0]).unwrap();
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        let gradient = LossFunction::MeanSquaredError
            .gradient(&predictions, &targets)
            .unwrap();

        // Expected gradient: 2 * (pred - target) / n = 2 * ([2-1, 3-2]) / 2 = [1.0, 1.0]
        assert_abs_diff_eq!(gradient[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(gradient[[0, 1]], 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mae_loss() {
        let predictions = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let targets = Array2::from_shape_vec((2, 2), vec![1.5, 1.5, 2.5, 3.5]).unwrap();

        let loss = LossFunction::MeanAbsoluteError
            .compute(&predictions, &targets)
            .unwrap();

        // Expected: (|1-1.5| + |2-1.5| + |3-2.5| + |4-3.5|) / 4 = (0.5 + 0.5 + 0.5 + 0.5) / 4 = 0.5
        assert_abs_diff_eq!(loss, 0.5, epsilon = 1e-10);
    }

    #[test]
    fn test_binary_cross_entropy() {
        let predictions = Array2::from_shape_vec((1, 2), vec![0.8, 0.2]).unwrap();
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, 0.0]).unwrap();

        let loss = LossFunction::BinaryCrossEntropy
            .compute(&predictions, &targets)
            .unwrap();

        // Should be finite and positive
        assert!(loss.is_finite());
        assert!(loss > 0.0);
    }

    #[test]
    fn test_categorical_cross_entropy() {
        let predictions = Array2::from_shape_vec((1, 3), vec![0.7, 0.2, 0.1]).unwrap();
        let targets = Array2::from_shape_vec((1, 3), vec![1.0, 0.0, 0.0]).unwrap();

        let loss = LossFunction::CategoricalCrossEntropy
            .compute(&predictions, &targets)
            .unwrap();

        let expected = -1.0_f64 * 0.7_f64.ln();
        assert_abs_diff_eq!(loss, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = Array2::from_shape_vec((1, 2), vec![1.0, 3.0]).unwrap();
        let targets = Array2::from_shape_vec((1, 2), vec![1.5, 1.0]).unwrap();

        let loss = LossFunction::HuberLoss
            .compute(&predictions, &targets)
            .unwrap();

        // Should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_hinge_loss() {
        let predictions = Array2::from_shape_vec((1, 2), vec![0.5, -0.3]).unwrap();
        let targets = Array2::from_shape_vec((1, 2), vec![1.0, -1.0]).unwrap();

        let loss = LossFunction::HingeLoss
            .compute(&predictions, &targets)
            .unwrap();

        // Should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_loss_properties() {
        assert!(LossFunction::BinaryCrossEntropy.is_classification_loss());
        assert!(!LossFunction::MeanSquaredError.is_classification_loss());

        assert!(LossFunction::MeanSquaredError.is_regression_loss());
        assert!(!LossFunction::BinaryCrossEntropy.is_regression_loss());

        assert_eq!(LossFunction::MeanSquaredError.name(), "MeanSquaredError");
        assert_eq!(
            LossFunction::BinaryCrossEntropy.name(),
            "BinaryCrossEntropy"
        );
    }

    #[test]
    fn test_loss_config() {
        let config = LossConfig::new(LossFunction::HuberLoss)
            .with_huber_delta(2.0)
            .unwrap();

        assert_eq!(config.function, LossFunction::HuberLoss);
        assert_eq!(config.parameters.huber_delta, Some(2.0));

        // Test invalid delta
        assert!(LossConfig::new(LossFunction::HuberLoss)
            .with_huber_delta(-1.0)
            .is_err());
    }

    #[test]
    fn test_quantile_loss() {
        let predictions = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let targets = Array2::from_shape_vec((1, 2), vec![1.5, 1.5]).unwrap();

        let loss = LossFunction::QuantileLoss
            .compute(&predictions, &targets)
            .unwrap();

        // Should be finite and non-negative
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let predictions = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let targets = Array2::from_shape_vec((1, 3), vec![1.0, 2.0, 3.0]).unwrap();

        let result = LossFunction::MeanSquaredError.compute(&predictions, &targets);
        assert!(result.is_err());
    }

    #[test]
    fn test_numerical_stability() {
        // Test with very small predictions for binary cross-entropy
        let predictions = Array2::from_shape_vec((1, 1), vec![1e-10]).unwrap();
        let targets = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();

        let loss = LossFunction::BinaryCrossEntropy
            .compute(&predictions, &targets)
            .unwrap();

        assert!(loss.is_finite());
        assert!(loss > 0.0);

        // Test gradient
        let gradient = LossFunction::BinaryCrossEntropy
            .gradient(&predictions, &targets)
            .unwrap();

        assert!(gradient.iter().all(|&g| g.is_finite()));
    }
}
