//! Loss functions for neural networks
//!
//! This module provides a comprehensive set of loss functions commonly used
//! in neural network training, including their forward and backward passes
//! for gradient computation.

use crate::error::{Result, RnnError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Enumeration of available loss functions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LossFunction {
    /// Mean Squared Error - for regression tasks
    MeanSquaredError,
    /// Mean Absolute Error - for regression tasks
    MeanAbsoluteError,
    /// Cross Entropy Loss - for classification tasks
    CrossEntropy,
    /// Binary Cross Entropy - for binary classification
    BinaryCrossEntropy,
    /// Huber Loss - robust regression loss
    HuberLoss(f32),
    /// Hinge Loss - for SVM-style classification
    HingeLoss,
    /// KL Divergence - for probability distributions
    KLDivergence,
    /// Focal Loss - for imbalanced classification
    FocalLoss {
        /// Weighting factor for rare class
        alpha: f32,
        /// Focusing parameter to down-weight easy examples
        gamma: f32,
    },
    /// Smooth L1 Loss - for object detection
    SmoothL1Loss(f32),
    /// Cosine Embedding Loss - for similarity learning
    CosineEmbeddingLoss,
    /// Triplet Loss - for metric learning
    TripletLoss(f32), // margin
}

impl LossFunction {
    /// Compute the loss value
    pub fn forward(&self, predictions: &Tensor, targets: &Tensor) -> Result<f32> {
        if predictions.shape() != targets.shape() {
            return Err(RnnError::shape_mismatch(
                predictions.shape(),
                targets.shape(),
            ));
        }

        let pred_data = predictions.to_vec()?;
        let target_data = targets.to_vec()?;

        match self {
            LossFunction::MeanSquaredError => self.mse_forward(&pred_data, &target_data),
            LossFunction::MeanAbsoluteError => self.mae_forward(&pred_data, &target_data),
            LossFunction::CrossEntropy => self.cross_entropy_forward(&pred_data, &target_data),
            LossFunction::BinaryCrossEntropy => self.bce_forward(&pred_data, &target_data),
            LossFunction::HuberLoss(delta) => self.huber_forward(&pred_data, &target_data, *delta),
            LossFunction::HingeLoss => self.hinge_forward(&pred_data, &target_data),
            LossFunction::KLDivergence => self.kl_div_forward(&pred_data, &target_data),
            LossFunction::FocalLoss { alpha, gamma } => {
                self.focal_forward(&pred_data, &target_data, *alpha, *gamma)
            }
            LossFunction::SmoothL1Loss(beta) => {
                self.smooth_l1_forward(&pred_data, &target_data, *beta)
            }
            LossFunction::CosineEmbeddingLoss => {
                self.cosine_embedding_forward(&pred_data, &target_data)
            }
            LossFunction::TripletLoss(margin) => {
                self.triplet_forward(&pred_data, &target_data, *margin)
            }
        }
    }

    /// Compute the gradient of the loss with respect to predictions
    pub fn backward(&self, predictions: &Tensor, targets: &Tensor) -> Result<Tensor> {
        if predictions.shape() != targets.shape() {
            return Err(RnnError::shape_mismatch(
                predictions.shape(),
                targets.shape(),
            ));
        }

        let pred_data = predictions.to_vec()?;
        let target_data = targets.to_vec()?;

        let grad_data = match self {
            LossFunction::MeanSquaredError => self.mse_backward(&pred_data, &target_data)?,
            LossFunction::MeanAbsoluteError => self.mae_backward(&pred_data, &target_data)?,
            LossFunction::CrossEntropy => self.cross_entropy_backward(&pred_data, &target_data)?,
            LossFunction::BinaryCrossEntropy => self.bce_backward(&pred_data, &target_data)?,
            LossFunction::HuberLoss(delta) => {
                self.huber_backward(&pred_data, &target_data, *delta)?
            }
            LossFunction::HingeLoss => self.hinge_backward(&pred_data, &target_data)?,
            LossFunction::KLDivergence => self.kl_div_backward(&pred_data, &target_data)?,
            LossFunction::FocalLoss { alpha, gamma } => {
                self.focal_backward(&pred_data, &target_data, *alpha, *gamma)?
            }
            LossFunction::SmoothL1Loss(beta) => {
                self.smooth_l1_backward(&pred_data, &target_data, *beta)?
            }
            LossFunction::CosineEmbeddingLoss => {
                self.cosine_embedding_backward(&pred_data, &target_data)?
            }
            LossFunction::TripletLoss(margin) => {
                self.triplet_backward(&pred_data, &target_data, *margin)?
            }
        };

        Tensor::from_slice_on_device(
            &grad_data,
            predictions.shape(),
            predictions.device().clone(),
        )
    }

    /// Mean Squared Error forward pass
    fn mse_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let sum_squared_error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum();

        Ok(sum_squared_error / predictions.len() as f32)
    }

    /// Mean Squared Error backward pass
    fn mse_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let n = predictions.len() as f32;
        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| 2.0 * (pred - target) / n)
            .collect();

        Ok(gradients)
    }

    /// Mean Absolute Error forward pass
    fn mae_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let sum_absolute_error: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .sum();

        Ok(sum_absolute_error / predictions.len() as f32)
    }

    /// Mean Absolute Error backward pass
    fn mae_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let n = predictions.len() as f32;
        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if pred > target {
                    1.0 / n
                } else if pred < target {
                    -1.0 / n
                } else {
                    0.0
                }
            })
            .collect();

        Ok(gradients)
    }

    /// Cross Entropy Loss forward pass (assumes softmax predictions and one-hot targets)
    fn cross_entropy_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if target > 0.0 {
                if pred <= 0.0 {
                    return Err(RnnError::math(
                        "Prediction must be positive for cross entropy",
                    ));
                }
                loss -= target * pred.ln();
            }
        }

        Ok(loss)
    }

    /// Cross Entropy Loss backward pass
    fn cross_entropy_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if pred <= 0.0 {
                    0.0 // Avoid division by zero
                } else {
                    -target / pred
                }
            })
            .collect();

        Ok(gradients)
    }

    /// Binary Cross Entropy forward pass
    fn bce_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let eps = 1e-7; // Small epsilon to prevent log(0)
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clamped = pred.max(eps).min(1.0 - eps);
            loss -= target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln();
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Binary Cross Entropy backward pass
    fn bce_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let eps = 1e-7;
        let n = predictions.len() as f32;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let pred_clamped = pred.max(eps).min(1.0 - eps);
                (-target / pred_clamped + (1.0 - target) / (1.0 - pred_clamped)) / n
            })
            .collect();

        Ok(gradients)
    }

    /// Huber Loss forward pass
    fn huber_forward(&self, predictions: &[f32], targets: &[f32], delta: f32) -> Result<f32> {
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let diff = (pred - target).abs();
            if diff <= delta {
                loss += 0.5 * diff.powi(2);
            } else {
                loss += delta * (diff - 0.5 * delta);
            }
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Huber Loss backward pass
    fn huber_backward(&self, predictions: &[f32], targets: &[f32], delta: f32) -> Result<Vec<f32>> {
        let n = predictions.len() as f32;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                let abs_diff = diff.abs();

                if abs_diff <= delta {
                    diff / n
                } else {
                    delta * diff.signum() / n
                }
            })
            .collect();

        Ok(gradients)
    }

    /// Hinge Loss forward pass
    fn hinge_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            loss += (1.0 - target * pred).max(0.0);
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Hinge Loss backward pass
    fn hinge_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let n = predictions.len() as f32;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if 1.0 - target * pred > 0.0 {
                    -target / n
                } else {
                    0.0
                }
            })
            .collect();

        Ok(gradients)
    }

    /// KL Divergence forward pass
    fn kl_div_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        let eps = 1e-7;
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if target > eps {
                let pred_clamped = pred.max(eps);
                loss += target * (target / pred_clamped).ln();
            }
        }

        Ok(loss)
    }

    /// KL Divergence backward pass
    fn kl_div_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let eps = 1e-7;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                if target > eps && pred > eps {
                    -target / pred
                } else {
                    0.0
                }
            })
            .collect();

        Ok(gradients)
    }

    /// Focal Loss forward pass
    fn focal_forward(
        &self,
        predictions: &[f32],
        targets: &[f32],
        alpha: f32,
        gamma: f32,
    ) -> Result<f32> {
        let eps = 1e-7;
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clamped = pred.max(eps).min(1.0 - eps);
            let ce_loss = if target == 1.0 {
                -pred_clamped.ln()
            } else {
                -(1.0 - pred_clamped).ln()
            };

            let pt = if target == 1.0 {
                pred_clamped
            } else {
                1.0 - pred_clamped
            };
            let focal_weight = alpha * (1.0 - pt).powf(gamma);
            loss += focal_weight * ce_loss;
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Focal Loss backward pass
    fn focal_backward(
        &self,
        predictions: &[f32],
        targets: &[f32],
        alpha: f32,
        gamma: f32,
    ) -> Result<Vec<f32>> {
        let eps = 1e-7;
        let n = predictions.len() as f32;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let pred_clamped = pred.max(eps).min(1.0 - eps);
                let pt = if target == 1.0 {
                    pred_clamped
                } else {
                    1.0 - pred_clamped
                };

                let ce_grad = if target == 1.0 {
                    -1.0 / pred_clamped
                } else {
                    1.0 / (1.0 - pred_clamped)
                };

                let focal_weight = alpha * (1.0 - pt).powf(gamma);
                let focal_grad = alpha
                    * gamma
                    * (1.0 - pt).powf(gamma - 1.0)
                    * if target == 1.0 { -1.0 } else { 1.0 };

                (focal_weight * ce_grad
                    + focal_grad
                        * if target == 1.0 {
                            -pred_clamped.ln()
                        } else {
                            -(1.0 - pred_clamped).ln()
                        })
                    / n
            })
            .collect();

        Ok(gradients)
    }

    /// Smooth L1 Loss forward pass
    fn smooth_l1_forward(&self, predictions: &[f32], targets: &[f32], beta: f32) -> Result<f32> {
        let mut loss = 0.0;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let diff = (pred - target).abs();
            if diff < beta {
                loss += 0.5 * diff.powi(2) / beta;
            } else {
                loss += diff - 0.5 * beta;
            }
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Smooth L1 Loss backward pass
    fn smooth_l1_backward(
        &self,
        predictions: &[f32],
        targets: &[f32],
        beta: f32,
    ) -> Result<Vec<f32>> {
        let n = predictions.len() as f32;

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let diff = pred - target;
                let abs_diff = diff.abs();

                if abs_diff < beta {
                    diff / beta / n
                } else {
                    diff.signum() / n
                }
            })
            .collect();

        Ok(gradients)
    }

    /// Cosine Embedding Loss forward pass
    fn cosine_embedding_forward(&self, predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        let dot_product: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| p * t)
            .sum();

        let pred_norm: f32 = predictions.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let target_norm: f32 = targets.iter().map(|&x| x * x).sum::<f32>().sqrt();

        if pred_norm == 0.0 || target_norm == 0.0 {
            return Ok(1.0); // Maximum dissimilarity
        }

        let cosine_similarity = dot_product / (pred_norm * target_norm);
        Ok(1.0 - cosine_similarity)
    }

    /// Cosine Embedding Loss backward pass
    fn cosine_embedding_backward(&self, predictions: &[f32], targets: &[f32]) -> Result<Vec<f32>> {
        let dot_product: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&p, &t)| p * t)
            .sum();

        let pred_norm_sq: f32 = predictions.iter().map(|&x| x * x).sum();
        let target_norm_sq: f32 = targets.iter().map(|&x| x * x).sum();

        let pred_norm = pred_norm_sq.sqrt();
        let target_norm = target_norm_sq.sqrt();

        if pred_norm == 0.0 || target_norm == 0.0 {
            return Ok(vec![0.0; predictions.len()]);
        }

        let gradients = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| {
                let term1 = -target / (pred_norm * target_norm);
                let term2 = dot_product * pred / (pred_norm_sq * pred_norm * target_norm);
                term1 + term2
            })
            .collect();

        Ok(gradients)
    }

    /// Triplet Loss forward pass (simplified - assumes anchor, positive, negative in sequence)
    fn triplet_forward(&self, predictions: &[f32], _targets: &[f32], margin: f32) -> Result<f32> {
        if predictions.len() % 3 != 0 {
            return Err(RnnError::tensor(
                "Triplet loss requires inputs in groups of 3",
            ));
        }

        let mut loss = 0.0;
        let num_triplets = predictions.len() / 3;

        for i in 0..num_triplets {
            let anchor = predictions[i * 3];
            let positive = predictions[i * 3 + 1];
            let negative = predictions[i * 3 + 2];

            let pos_dist = (anchor - positive).powi(2);
            let neg_dist = (anchor - negative).powi(2);

            loss += (pos_dist - neg_dist + margin).max(0.0);
        }

        Ok(loss / num_triplets as f32)
    }

    /// Triplet Loss backward pass
    fn triplet_backward(
        &self,
        predictions: &[f32],
        _targets: &[f32],
        margin: f32,
    ) -> Result<Vec<f32>> {
        if predictions.len() % 3 != 0 {
            return Err(RnnError::tensor(
                "Triplet loss requires inputs in groups of 3",
            ));
        }

        let mut gradients = vec![0.0; predictions.len()];
        let num_triplets = predictions.len() / 3;

        for i in 0..num_triplets {
            let anchor = predictions[i * 3];
            let positive = predictions[i * 3 + 1];
            let negative = predictions[i * 3 + 2];

            let pos_dist = (anchor - positive).powi(2);
            let neg_dist = (anchor - negative).powi(2);

            if pos_dist - neg_dist + margin > 0.0 {
                // Gradients for anchor
                gradients[i * 3] += 2.0 * (positive - negative) / num_triplets as f32;
                // Gradients for positive
                gradients[i * 3 + 1] += 2.0 * (anchor - positive) / num_triplets as f32;
                // Gradients for negative
                gradients[i * 3 + 2] += 2.0 * (negative - anchor) / num_triplets as f32;
            }
        }

        Ok(gradients)
    }

    /// Get the name of the loss function
    pub fn name(&self) -> &'static str {
        match self {
            LossFunction::MeanSquaredError => "mse",
            LossFunction::MeanAbsoluteError => "mae",
            LossFunction::CrossEntropy => "cross_entropy",
            LossFunction::BinaryCrossEntropy => "binary_cross_entropy",
            LossFunction::HuberLoss(_) => "huber",
            LossFunction::HingeLoss => "hinge",
            LossFunction::KLDivergence => "kl_divergence",
            LossFunction::FocalLoss { .. } => "focal",
            LossFunction::SmoothL1Loss(_) => "smooth_l1",
            LossFunction::CosineEmbeddingLoss => "cosine_embedding",
            LossFunction::TripletLoss(_) => "triplet",
        }
    }
}

impl fmt::Display for LossFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LossFunction::MeanSquaredError => write!(f, "Mean Squared Error"),
            LossFunction::MeanAbsoluteError => write!(f, "Mean Absolute Error"),
            LossFunction::CrossEntropy => write!(f, "Cross Entropy"),
            LossFunction::BinaryCrossEntropy => write!(f, "Binary Cross Entropy"),
            LossFunction::HuberLoss(delta) => write!(f, "Huber Loss (δ={})", delta),
            LossFunction::HingeLoss => write!(f, "Hinge Loss"),
            LossFunction::KLDivergence => write!(f, "KL Divergence"),
            LossFunction::FocalLoss { alpha, gamma } => {
                write!(f, "Focal Loss (α={}, γ={})", alpha, gamma)
            }
            LossFunction::SmoothL1Loss(beta) => write!(f, "Smooth L1 Loss (β={})", beta),
            LossFunction::CosineEmbeddingLoss => write!(f, "Cosine Embedding Loss"),
            LossFunction::TripletLoss(margin) => write!(f, "Triplet Loss (margin={})", margin),
        }
    }
}

impl Default for LossFunction {
    fn default() -> Self {
        LossFunction::MeanSquaredError
    }
}

/// Common loss function presets
impl LossFunction {
    /// Mean Squared Error
    pub fn mse() -> Self {
        LossFunction::MeanSquaredError
    }

    /// Mean Absolute Error
    pub fn mae() -> Self {
        LossFunction::MeanAbsoluteError
    }

    /// Cross Entropy Loss
    pub fn cross_entropy() -> Self {
        LossFunction::CrossEntropy
    }

    /// Binary Cross Entropy Loss
    pub fn binary_cross_entropy() -> Self {
        LossFunction::BinaryCrossEntropy
    }

    /// Huber Loss with default delta of 1.0
    pub fn huber() -> Self {
        LossFunction::HuberLoss(1.0)
    }

    /// Huber Loss with custom delta
    pub fn huber_with_delta(delta: f32) -> Self {
        LossFunction::HuberLoss(delta)
    }

    /// Hinge Loss
    pub fn hinge() -> Self {
        LossFunction::HingeLoss
    }

    /// KL Divergence
    pub fn kl_divergence() -> Self {
        LossFunction::KLDivergence
    }

    /// Focal Loss with default parameters
    pub fn focal() -> Self {
        LossFunction::FocalLoss {
            alpha: 1.0,
            gamma: 2.0,
        }
    }

    /// Focal Loss with custom parameters
    pub fn focal_with_params(alpha: f32, gamma: f32) -> Self {
        LossFunction::FocalLoss { alpha, gamma }
    }

    /// Smooth L1 Loss with default beta of 1.0
    pub fn smooth_l1() -> Self {
        LossFunction::SmoothL1Loss(1.0)
    }

    /// Smooth L1 Loss with custom beta
    pub fn smooth_l1_with_beta(beta: f32) -> Self {
        LossFunction::SmoothL1Loss(beta)
    }

    /// Cosine Embedding Loss
    pub fn cosine_embedding() -> Self {
        LossFunction::CosineEmbeddingLoss
    }

    /// Triplet Loss with default margin of 1.0
    pub fn triplet() -> Self {
        LossFunction::TripletLoss(1.0)
    }

    /// Triplet Loss with custom margin
    pub fn triplet_with_margin(margin: f32) -> Self {
        LossFunction::TripletLoss(margin)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use approx::assert_relative_eq;

    #[test]
    fn test_mse_forward() {
        let predictions = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.5, 2.5, 2.5], &[3]).unwrap();

        let loss = LossFunction::MeanSquaredError;
        let result = loss.forward(&predictions, &targets).unwrap();

        // Expected: ((1-1.5)^2 + (2-2.5)^2 + (3-2.5)^2) / 3 = (0.25 + 0.25 + 0.25) / 3 = 0.25
        assert_relative_eq!(result, 0.25, epsilon = 1e-6);
    }

    #[test]
    fn test_mse_backward() {
        let predictions = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.5, 2.5, 2.5], &[3]).unwrap();

        let loss = LossFunction::MeanSquaredError;
        let grad = loss.backward(&predictions, &targets).unwrap();
        let grad_data = grad.to_vec().unwrap();

        // Expected gradients: 2 * (pred - target) / n
        let expected = vec![
            2.0 * (1.0 - 1.5) / 3.0,
            2.0 * (2.0 - 2.5) / 3.0,
            2.0 * (3.0 - 2.5) / 3.0,
        ];
        for (actual, expected) in grad_data.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_mae_forward() {
        let predictions = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.5, 2.5, 2.5], &[3]).unwrap();

        let loss = LossFunction::MeanAbsoluteError;
        let result = loss.forward(&predictions, &targets).unwrap();

        // Expected: (|1-1.5| + |2-2.5| + |3-2.5|) / 3 = (0.5 + 0.5 + 0.5) / 3 = 0.5
        assert_relative_eq!(result, 0.5, epsilon = 1e-6);
    }

    #[test]
    fn test_bce_forward() {
        let predictions = Tensor::from_slice(&[0.8, 0.2, 0.9], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.0, 0.0, 1.0], &[3]).unwrap();

        let loss = LossFunction::BinaryCrossEntropy;
        let result = loss.forward(&predictions, &targets).unwrap();

        // Should be a positive value
        assert!(result > 0.0);
    }

    #[test]
    fn test_cross_entropy_forward() {
        let predictions = Tensor::from_slice(&[0.7, 0.2, 0.1], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.0, 0.0, 0.0], &[3]).unwrap();

        let loss = LossFunction::CrossEntropy;
        let result = loss.forward(&predictions, &targets).unwrap();

        // Should be -ln(0.7)
        let expected = -0.7_f32.ln();
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_huber_loss() {
        let predictions = Tensor::from_slice(&[1.0, 2.0, 5.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.5, 2.5, 2.0], &[3]).unwrap();

        let loss = LossFunction::HuberLoss(1.0);
        let result = loss.forward(&predictions, &targets).unwrap();

        // For delta=1.0: first two use quadratic, third uses linear
        assert!(result > 0.0);
    }

    #[test]
    fn test_focal_loss() {
        let predictions = Tensor::from_slice(&[0.9, 0.1], &[2]).unwrap();
        let targets = Tensor::from_slice(&[1.0, 0.0], &[2]).unwrap();

        let loss = LossFunction::FocalLoss {
            alpha: 1.0,
            gamma: 2.0,
        };
        let result = loss.forward(&predictions, &targets).unwrap();

        assert!(result > 0.0);
    }

    #[test]
    fn test_cosine_embedding_loss() {
        let predictions = Tensor::from_slice(&[1.0, 0.0, 0.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[0.0, 1.0, 0.0], &[3]).unwrap();

        let loss = LossFunction::CosineEmbeddingLoss;
        let result = loss.forward(&predictions, &targets).unwrap();

        // Orthogonal vectors should give loss of 1.0
        assert_relative_eq!(result, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_triplet_loss() {
        // Anchor, positive, negative triplet
        let predictions = Tensor::from_slice(&[1.0, 1.1, 2.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[0.0, 0.0, 0.0], &[3]).unwrap(); // Dummy targets

        let loss = LossFunction::TripletLoss(0.5);
        let result = loss.forward(&predictions, &targets).unwrap();

        assert!(result >= 0.0);
    }

    #[test]
    fn test_loss_function_names() {
        assert_eq!(LossFunction::MeanSquaredError.name(), "mse");
        assert_eq!(LossFunction::CrossEntropy.name(), "cross_entropy");
        assert_eq!(
            LossFunction::BinaryCrossEntropy.name(),
            "binary_cross_entropy"
        );
    }

    #[test]
    fn test_loss_function_display() {
        assert_eq!(
            format!("{}", LossFunction::MeanSquaredError),
            "Mean Squared Error"
        );
        assert_eq!(
            format!("{}", LossFunction::HuberLoss(1.5)),
            "Huber Loss (δ=1.5)"
        );
        assert_eq!(
            format!(
                "{}",
                LossFunction::FocalLoss {
                    alpha: 0.25,
                    gamma: 2.0
                }
            ),
            "Focal Loss (α=0.25, γ=2)"
        );
    }

    #[test]
    fn test_shape_mismatch_error() {
        let predictions = Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap();
        let targets = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();

        let loss = LossFunction::MeanSquaredError;
        let result = loss.forward(&predictions, &targets);

        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            RnnError::ShapeMismatch { .. }
        ));
    }

    #[test]
    fn test_smooth_l1_loss() {
        let predictions = Tensor::from_slice(&[1.0, 2.0, 4.0], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.2, 2.3, 2.0], &[3]).unwrap();

        let loss = LossFunction::SmoothL1Loss(1.0);
        let result = loss.forward(&predictions, &targets).unwrap();

        assert!(result > 0.0);

        // Test gradient
        let grad = loss.backward(&predictions, &targets).unwrap();
        assert_eq!(grad.shape(), predictions.shape());
    }

    #[test]
    fn test_hinge_loss() {
        let predictions = Tensor::from_slice(&[0.5, -0.3, 0.8], &[3]).unwrap();
        let targets = Tensor::from_slice(&[1.0, -1.0, 1.0], &[3]).unwrap();

        let loss = LossFunction::HingeLoss;
        let result = loss.forward(&predictions, &targets).unwrap();

        assert!(result >= 0.0);
    }

    #[test]
    fn test_kl_divergence() {
        let predictions = Tensor::from_slice(&[0.4, 0.3, 0.3], &[3]).unwrap();
        let targets = Tensor::from_slice(&[0.5, 0.3, 0.2], &[3]).unwrap();

        let loss = LossFunction::KLDivergence;
        let result = loss.forward(&predictions, &targets).unwrap();

        assert!(result >= 0.0);
    }
}
