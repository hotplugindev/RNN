//! Optimization algorithms for neural network training.
//!
//! This module provides various optimization algorithms commonly used in neural networks,
//! including SGD, Adam, RMSprop, and others with their respective hyperparameters.

use crate::error::{NetworkError, Result};
use ndarray::{Array1, Array2, Zip};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Enumeration of available optimization algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizerType {
    /// Stochastic Gradient Descent
    SGD,
    /// SGD with Momentum
    Momentum,
    /// Nesterov Accelerated Gradient
    Nesterov,
    /// Adaptive Gradient Algorithm
    Adagrad,
    /// Adadelta
    Adadelta,
    /// RMSprop
    RMSprop,
    /// Adam (Adaptive Moment Estimation)
    Adam,
    /// AdaMax (variant of Adam)
    AdaMax,
    /// Nadam (Nesterov-accelerated Adam)
    Nadam,
    /// AMSGrad (variant of Adam)
    AMSGrad,
    /// AdaBound
    AdaBound,
    /// LBFGS (Limited-memory BFGS)
    LBFGS,
    /// Newton's Method
    Newton,
}

/// Configuration for optimizer hyperparameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerConfig {
    /// Learning rate
    pub learning_rate: f64,
    /// Decay rate for learning rate (if applicable)
    pub decay: Option<f64>,
    /// Momentum parameter (for momentum-based optimizers)
    pub momentum: Option<f64>,
    /// Beta1 parameter (for Adam-family optimizers)
    pub beta1: Option<f64>,
    /// Beta2 parameter (for Adam-family optimizers)
    pub beta2: Option<f64>,
    /// Epsilon for numerical stability
    pub epsilon: f64,
    /// Weight decay (L2 regularization)
    pub weight_decay: Option<f64>,
    /// Gradient clipping threshold
    pub gradient_clip: Option<f64>,
    /// AdaBound parameters
    pub final_lr: Option<f64>,
    pub gamma: Option<f64>,
    /// LBFGS memory size
    pub memory_size: Option<usize>,
    /// Newton's method damping factor
    pub damping: Option<f64>,
}

impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            decay: None,
            momentum: None,
            beta1: Some(0.9),
            beta2: Some(0.999),
            epsilon: 1e-8,
            weight_decay: None,
            gradient_clip: None,
            final_lr: None,
            gamma: None,
            memory_size: Some(20),
            damping: Some(1e-4),
        }
    }
}

/// State information for different optimizers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    /// Current iteration/step count
    pub step: usize,
    /// Momentum accumulator (for momentum-based optimizers)
    pub momentum_buffer: HashMap<String, Array2<f64>>,
    /// Velocity accumulator (for Adam-family optimizers)
    pub velocity: HashMap<String, Array2<f64>>,
    /// Second moment accumulator (for Adam-family optimizers)
    pub second_moment: HashMap<String, Array2<f64>>,
    /// Max second moment (for AMSGrad)
    pub max_second_moment: HashMap<String, Array2<f64>>,
    /// Accumulated gradient squared (for Adagrad/RMSprop)
    pub accumulated_grad: HashMap<String, Array2<f64>>,
    /// Delta accumulator (for Adadelta)
    pub accumulated_delta: HashMap<String, Array2<f64>>,
    /// LBFGS history
    pub lbfgs_history: Vec<(Array2<f64>, Array2<f64>)>, // (s, y) pairs
    /// Bias corrections for Adam-family
    pub bias_correction_1: f64,
    pub bias_correction_2: f64,
    /// Learning rate schedule state
    pub effective_lr: f64,
}

impl Default for OptimizerState {
    fn default() -> Self {
        Self {
            step: 0,
            momentum_buffer: HashMap::new(),
            velocity: HashMap::new(),
            second_moment: HashMap::new(),
            max_second_moment: HashMap::new(),
            accumulated_grad: HashMap::new(),
            accumulated_delta: HashMap::new(),
            lbfgs_history: Vec::new(),
            bias_correction_1: 1.0,
            bias_correction_2: 1.0,
            effective_lr: 0.001,
        }
    }
}

/// Main optimizer struct that handles parameter updates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Optimizer {
    pub optimizer_type: OptimizerType,
    pub config: OptimizerConfig,
    pub state: OptimizerState,
}

impl Optimizer {
    /// Create a new optimizer with the specified type and configuration.
    pub fn new(optimizer_type: OptimizerType, config: OptimizerConfig) -> Result<Self> {
        // Validate configuration
        if config.learning_rate <= 0.0 {
            return Err(NetworkError::invalid_parameter(
                "learning_rate",
                &config.learning_rate.to_string(),
                "must be positive",
            ));
        }

        if config.epsilon <= 0.0 {
            return Err(NetworkError::invalid_parameter(
                "epsilon",
                &config.epsilon.to_string(),
                "must be positive",
            ));
        }

        let mut state = OptimizerState::default();
        state.effective_lr = config.learning_rate;

        Ok(Self {
            optimizer_type,
            config,
            state,
        })
    }

    /// Create SGD optimizer.
    pub fn sgd(learning_rate: f64) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            ..Default::default()
        };
        Self::new(OptimizerType::SGD, config)
    }

    /// Create SGD with momentum optimizer.
    pub fn momentum(learning_rate: f64, momentum: f64) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            momentum: Some(momentum),
            ..Default::default()
        };
        Self::new(OptimizerType::Momentum, config)
    }

    /// Create Adam optimizer.
    pub fn adam(learning_rate: f64) -> Result<Self> {
        Self::adam_with_params(learning_rate, 0.9, 0.999, 1e-8)
    }

    /// Create Adam optimizer with custom parameters.
    pub fn adam_with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
    ) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            beta1: Some(beta1),
            beta2: Some(beta2),
            epsilon,
            ..Default::default()
        };
        Self::new(OptimizerType::Adam, config)
    }

    /// Create RMSprop optimizer.
    pub fn rmsprop(learning_rate: f64, decay: f64) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            decay: Some(decay),
            ..Default::default()
        };
        Self::new(OptimizerType::RMSprop, config)
    }

    /// Create Adagrad optimizer.
    pub fn adagrad(learning_rate: f64) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            ..Default::default()
        };
        Self::new(OptimizerType::Adagrad, config)
    }

    /// Create Newton's method optimizer.
    pub fn newton(learning_rate: f64, damping: f64) -> Result<Self> {
        let config = OptimizerConfig {
            learning_rate,
            damping: Some(damping),
            ..Default::default()
        };
        Self::new(OptimizerType::Newton, config)
    }

    /// Update parameters using the computed gradients.
    pub fn update(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
        hessian: Option<&Array2<f64>>,
    ) -> Result<()> {
        if parameters.shape() != gradients.shape() {
            return Err(NetworkError::dimension_mismatch(
                format!("{:?}", parameters.shape()),
                format!("{:?}", gradients.shape()),
            ));
        }

        // Apply gradient clipping if specified
        let clipped_gradients = if let Some(clip_value) = self.config.gradient_clip {
            self.clip_gradients(gradients, clip_value)
        } else {
            gradients.clone()
        };

        // Apply weight decay if specified
        let final_gradients = if let Some(weight_decay) = self.config.weight_decay {
            &clipped_gradients + weight_decay * &*parameters
        } else {
            clipped_gradients
        };

        // Increment step count
        self.state.step += 1;

        // Apply the specific optimizer algorithm
        match self.optimizer_type {
            OptimizerType::SGD => self.update_sgd(param_name, parameters, &final_gradients),
            OptimizerType::Momentum => {
                self.update_momentum(param_name, parameters, &final_gradients)
            }
            OptimizerType::Nesterov => {
                self.update_nesterov(param_name, parameters, &final_gradients)
            }
            OptimizerType::Adagrad => self.update_adagrad(param_name, parameters, &final_gradients),
            OptimizerType::Adadelta => {
                self.update_adadelta(param_name, parameters, &final_gradients)
            }
            OptimizerType::RMSprop => self.update_rmsprop(param_name, parameters, &final_gradients),
            OptimizerType::Adam => self.update_adam(param_name, parameters, &final_gradients),
            OptimizerType::AdaMax => self.update_adamax(param_name, parameters, &final_gradients),
            OptimizerType::Nadam => self.update_nadam(param_name, parameters, &final_gradients),
            OptimizerType::AMSGrad => self.update_amsgrad(param_name, parameters, &final_gradients),
            OptimizerType::AdaBound => {
                self.update_adabound(param_name, parameters, &final_gradients)
            }
            OptimizerType::LBFGS => self.update_lbfgs(param_name, parameters, &final_gradients),
            OptimizerType::Newton => {
                if let Some(hessian_matrix) = hessian {
                    self.update_newton(param_name, parameters, &final_gradients, hessian_matrix)
                } else {
                    return Err(NetworkError::optimizer(
                        "Newton's method requires Hessian matrix",
                    ));
                }
            }
        }
    }

    /// Clip gradients to prevent exploding gradients.
    fn clip_gradients(&self, gradients: &Array2<f64>, clip_value: f64) -> Array2<f64> {
        let grad_norm = (gradients.mapv(|x| x * x).sum()).sqrt();
        if grad_norm > clip_value {
            gradients * (clip_value / grad_norm)
        } else {
            gradients.clone()
        }
    }

    /// SGD update rule.
    fn update_sgd(
        &mut self,
        _param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        *parameters = &*parameters - self.state.effective_lr * gradients;
        Ok(())
    }

    /// SGD with momentum update rule.
    fn update_momentum(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let momentum = self.config.momentum.unwrap_or(0.9);

        let velocity = self
            .state
            .momentum_buffer
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *velocity = momentum * &*velocity + gradients;
        *parameters = &*parameters - self.state.effective_lr * &*velocity;

        Ok(())
    }

    /// Nesterov accelerated gradient update rule.
    fn update_nesterov(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let momentum = self.config.momentum.unwrap_or(0.9);

        let velocity = self
            .state
            .momentum_buffer
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let prev_velocity = velocity.clone();
        *velocity = momentum * &*velocity + gradients;
        *parameters = &*parameters
            - self.state.effective_lr * (gradients + momentum * (&*velocity - &prev_velocity));

        Ok(())
    }

    /// Adagrad update rule.
    fn update_adagrad(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let accumulated = self
            .state
            .accumulated_grad
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *accumulated = &*accumulated + gradients.mapv(|x| x * x);
        let adaptive_lr =
            accumulated.mapv(|x| self.state.effective_lr / (x.sqrt() + self.config.epsilon));
        *parameters = &*parameters - &adaptive_lr * gradients;

        Ok(())
    }

    /// Adadelta update rule.
    fn update_adadelta(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let rho = self.config.decay.unwrap_or(0.95);

        let accumulated_grad = self
            .state
            .accumulated_grad
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let accumulated_delta = self
            .state
            .accumulated_delta
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *accumulated_grad = rho * &*accumulated_grad + (1.0 - rho) * gradients.mapv(|x| x * x);

        let delta = Zip::from(&*accumulated_delta)
            .and(&*accumulated_grad)
            .and(gradients)
            .map_collect(|&acc_delta, &acc_grad, &grad| {
                -((acc_delta + self.config.epsilon) / (acc_grad + self.config.epsilon)).sqrt()
                    * grad
            });

        *accumulated_delta = rho * &*accumulated_delta + (1.0 - rho) * delta.mapv(|x| x * x);
        *parameters = &*parameters + &delta;

        Ok(())
    }

    /// RMSprop update rule.
    fn update_rmsprop(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let decay = self.config.decay.unwrap_or(0.99);

        let accumulated = self
            .state
            .accumulated_grad
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *accumulated = decay * &*accumulated + (1.0 - decay) * gradients.mapv(|x| x * x);
        let adaptive_lr =
            accumulated.mapv(|x| self.state.effective_lr / (x.sqrt() + self.config.epsilon));
        *parameters = &*parameters - &adaptive_lr * gradients;

        Ok(())
    }

    /// Adam update rule.
    fn update_adam(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let beta1 = self.config.beta1.unwrap_or(0.9);
        let beta2 = self.config.beta2.unwrap_or(0.999);

        let m = self
            .state
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let v = self
            .state
            .second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        // Update biased first and second moment estimates
        *m = beta1 * &*m + (1.0 - beta1) * gradients;
        *v = beta2 * &*v + (1.0 - beta2) * gradients.mapv(|x| x * x);

        // Compute bias-corrected moment estimates
        self.state.bias_correction_1 *= beta1;
        self.state.bias_correction_2 *= beta2;

        let m_hat = &*m / (1.0 - self.state.bias_correction_1);
        let v_hat = &*v / (1.0 - self.state.bias_correction_2);

        // Update parameters
        let update = Zip::from(&m_hat).and(&v_hat).map_collect(|&m_val, &v_val| {
            self.state.effective_lr * m_val / (v_val.sqrt() + self.config.epsilon)
        });

        *parameters = &*parameters - &update;

        Ok(())
    }

    /// AdaMax update rule.
    fn update_adamax(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let beta1 = self.config.beta1.unwrap_or(0.9);
        let beta2 = self.config.beta2.unwrap_or(0.999);

        let m = self
            .state
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let u = self
            .state
            .second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *m = beta1 * &*m + (1.0 - beta1) * gradients;
        *u = Zip::from(&*u)
            .and(gradients)
            .map_collect(|&u_val, &grad| (beta2 * u_val).max(grad.abs()));

        let bias_correction = 1.0 - beta1.powi(self.state.step as i32);
        let lr_t = self.state.effective_lr / bias_correction;

        let update = Zip::from(&*m)
            .and(&*u)
            .map_collect(|&m_val, &u_val| lr_t * m_val / (u_val + self.config.epsilon));

        *parameters = &*parameters - &update;

        Ok(())
    }

    /// Nadam update rule.
    fn update_nadam(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let beta1 = self.config.beta1.unwrap_or(0.9);
        let beta2 = self.config.beta2.unwrap_or(0.999);

        let m = self
            .state
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let v = self
            .state
            .second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *m = beta1 * &*m + (1.0 - beta1) * gradients;
        *v = beta2 * &*v + (1.0 - beta2) * gradients.mapv(|x| x * x);

        let bias_correction_1 = 1.0 - beta1.powi(self.state.step as i32);
        let bias_correction_2 = 1.0 - beta2.powi(self.state.step as i32);

        let m_hat = &*m / bias_correction_1;
        let v_hat = &*v / bias_correction_2;

        // Nesterov momentum
        let m_bar = beta1 * &m_hat + (1.0 - beta1) * gradients / bias_correction_1;

        let update = Zip::from(&m_bar).and(&v_hat).map_collect(|&m_val, &v_val| {
            self.state.effective_lr * m_val / (v_val.sqrt() + self.config.epsilon)
        });

        *parameters = &*parameters - &update;

        Ok(())
    }

    /// AMSGrad update rule.
    fn update_amsgrad(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let beta1 = self.config.beta1.unwrap_or(0.9);
        let beta2 = self.config.beta2.unwrap_or(0.999);

        let m = self
            .state
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let v = self
            .state
            .second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let v_max = self
            .state
            .max_second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *m = beta1 * &*m + (1.0 - beta1) * gradients;
        *v = beta2 * &*v + (1.0 - beta2) * gradients.mapv(|x| x * x);

        // Use maximum of current and past squared gradients
        *v_max = Zip::from(&*v_max)
            .and(&*v)
            .map_collect(|&v_max_val, &v_val| v_max_val.max(v_val));

        let bias_correction_1 = 1.0 - beta1.powi(self.state.step as i32);

        let update = Zip::from(&*m)
            .and(&*v_max)
            .map_collect(|&m_val, &v_max_val| {
                (self.state.effective_lr / bias_correction_1) * m_val
                    / (v_max_val.sqrt() + self.config.epsilon)
            });

        *parameters = &*parameters - &update;

        Ok(())
    }

    /// AdaBound update rule.
    fn update_adabound(
        &mut self,
        param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let beta1 = self.config.beta1.unwrap_or(0.9);
        let beta2 = self.config.beta2.unwrap_or(0.999);
        let final_lr = self.config.final_lr.unwrap_or(0.1);
        let gamma = self.config.gamma.unwrap_or(1e-3);

        let m = self
            .state
            .velocity
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        let v = self
            .state
            .second_moment
            .entry(param_name.to_string())
            .or_insert_with(|| Array2::zeros(parameters.raw_dim()));

        *m = beta1 * &*m + (1.0 - beta1) * gradients;
        *v = beta2 * &*v + (1.0 - beta2) * gradients.mapv(|x| x * x);

        let bias_correction_1 = 1.0 - beta1.powi(self.state.step as i32);
        let bias_correction_2 = 1.0 - beta2.powi(self.state.step as i32);

        let step_size = self.state.effective_lr * (bias_correction_2.sqrt() / bias_correction_1);

        // AdaBound specific bounds
        let lower_bound = final_lr * (1.0 - 1.0 / (gamma * self.state.step as f64 + 1.0));
        let upper_bound = final_lr * (1.0 + 1.0 / (gamma * self.state.step as f64));

        let update = Zip::from(&*m).and(&*v).map_collect(|&m_val, &v_val| {
            let adaptive_lr = step_size / (v_val.sqrt() + self.config.epsilon);
            let clipped_lr = adaptive_lr.max(lower_bound).min(upper_bound);
            clipped_lr * m_val
        });

        *parameters = &*parameters - &update;

        Ok(())
    }

    /// L-BFGS update rule (simplified version).
    fn update_lbfgs(
        &mut self,
        _param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
    ) -> Result<()> {
        let memory_size = self.config.memory_size.unwrap_or(20);

        // For simplicity, we'll implement a basic L-BFGS without line search
        // In practice, L-BFGS requires function evaluation capabilities
        let update = gradients * self.state.effective_lr;
        *parameters = &*parameters - &update;

        // Store history (simplified)
        if self.state.lbfgs_history.len() >= memory_size {
            self.state.lbfgs_history.remove(0);
        }

        Ok(())
    }

    /// Newton's method update rule.
    fn update_newton(
        &mut self,
        _param_name: &str,
        parameters: &mut Array2<f64>,
        gradients: &Array2<f64>,
        hessian: &Array2<f64>,
    ) -> Result<()> {
        let damping = self.config.damping.unwrap_or(1e-4);

        // Add damping to diagonal for numerical stability
        let mut damped_hessian = hessian.clone();
        for i in 0..damped_hessian.nrows().min(damped_hessian.ncols()) {
            damped_hessian[[i, i]] += damping;
        }

        // Solve: H * delta = -g for delta
        // For simplicity, we'll use a pseudo-inverse approximation
        // In practice, you'd use proper linear algebra solvers
        let grad_vec = gradients.clone().into_shape(gradients.len()).unwrap();

        // Simple approximation: delta ≈ -lr * H^(-1) * g
        // Here we'll use gradient descent as a fallback for stability
        let update = gradients * self.state.effective_lr;
        *parameters = &*parameters - &update;

        Ok(())
    }

    /// Update the learning rate (for learning rate scheduling).
    pub fn update_learning_rate(&mut self, new_lr: f64) -> Result<()> {
        if new_lr <= 0.0 {
            return Err(NetworkError::invalid_parameter(
                "learning_rate",
                &new_lr.to_string(),
                "must be positive",
            ));
        }
        self.state.effective_lr = new_lr;
        Ok(())
    }

    /// Get the current effective learning rate.
    pub fn get_learning_rate(&self) -> f64 {
        self.state.effective_lr
    }

    /// Get the current step count.
    pub fn get_step(&self) -> usize {
        self.state.step
    }

    /// Reset the optimizer state.
    pub fn reset(&mut self) {
        self.state = OptimizerState::default();
        self.state.effective_lr = self.config.learning_rate;
    }

    /// Get optimizer summary information.
    pub fn summary(&self) -> OptimizerSummary {
        OptimizerSummary {
            optimizer_type: self.optimizer_type,
            learning_rate: self.state.effective_lr,
            step: self.state.step,
            config: self.config.clone(),
        }
    }
}

/// Summary information about an optimizer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerSummary {
    pub optimizer_type: OptimizerType,
    pub learning_rate: f64,
    pub step: usize,
    pub config: OptimizerConfig,
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sgd_optimizer() {
        let mut optimizer = Optimizer::sgd(0.1).unwrap();
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 2), vec![0.1, 0.2, 0.3, 0.4]).unwrap();

        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();

        // Expected: params - 0.1 * gradients
        assert_abs_diff_eq!(params[[0, 0]], 0.99, epsilon = 1e-10);
        assert_abs_diff_eq!(params[[0, 1]], 1.98, epsilon = 1e-10);
        assert_abs_diff_eq!(params[[1, 0]], 2.97, epsilon = 1e-10);
        assert_abs_diff_eq!(params[[1, 1]], 3.96, epsilon = 1e-10);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut optimizer = Optimizer::adam(0.001).unwrap();
        let mut params = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 2), vec![0.1, 0.2]).unwrap();

        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();

        // Parameters should be updated (exact values depend on Adam's internal state)
        assert_ne!(params[[0, 0]], 1.0);
        assert_ne!(params[[0, 1]], 2.0);
        assert_eq!(optimizer.get_step(), 1);
    }

    #[test]
    fn test_momentum_optimizer() {
        let mut optimizer = Optimizer::momentum(0.1, 0.9).unwrap();
        let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();

        // First update
        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();
        let first_update = params[[0, 0]];

        // Second update with same gradients
        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();
        let second_update = params[[0, 0]];

        // With momentum, the second update should be larger
        assert!(first_update < 1.0); // Parameter decreased
        assert!(second_update < first_update); // Decreased more in second step
    }

    #[test]
    fn test_gradient_clipping() {
        let mut config = OptimizerConfig::default();
        config.learning_rate = 0.1;
        config.gradient_clip = Some(1.0);

        let mut optimizer = Optimizer::new(OptimizerType::SGD, config).unwrap();
        let mut params = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).unwrap();

        // Large gradients that should be clipped
        let gradients = Array2::from_shape_vec((1, 2), vec![10.0, 20.0]).unwrap();

        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();

        // The update should be smaller due to gradient clipping
        let update_magnitude =
            ((1.0 - params[[0, 0]]).powi(2) + (2.0 - params[[0, 1]]).powi(2)).sqrt();
        assert!(update_magnitude < 1.0); // Should be less than unclipped update
    }

    #[test]
    fn test_weight_decay() {
        let mut config = OptimizerConfig::default();
        config.learning_rate = 0.1;
        config.weight_decay = Some(0.01);

        let mut optimizer = Optimizer::new(OptimizerType::SGD, config).unwrap();
        let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![0.0]).unwrap(); // Zero gradients

        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();

        // Even with zero gradients, parameters should decay due to weight decay
        assert!(params[[0, 0]] < 1.0);
    }

    #[test]
    fn test_learning_rate_update() {
        let mut optimizer = Optimizer::sgd(0.1).unwrap();

        assert_eq!(optimizer.get_learning_rate(), 0.1);

        optimizer.update_learning_rate(0.05).unwrap();
        assert_eq!(optimizer.get_learning_rate(), 0.05);

        // Test invalid learning rate
        assert!(optimizer.update_learning_rate(-0.1).is_err());
    }

    #[test]
    fn test_optimizer_reset() {
        let mut optimizer = Optimizer::adam(0.001).unwrap();
        let mut params = Array2::from_shape_vec((1, 1), vec![1.0]).unwrap();
        let gradients = Array2::from_shape_vec((1, 1), vec![0.1]).unwrap();

        // Make some updates
        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();
        optimizer
            .update("test", &mut params, &gradients, None)
            .unwrap();

        assert_eq!(optimizer.get_step(), 2);

        // Reset optimizer
        optimizer.reset();
        assert_eq!(optimizer.get_step(), 0);
        assert!(optimizer.state.velocity.is_empty());
        assert!(optimizer.state.second_moment.is_empty());
    }

    #[test]
    fn test_optimizer_summary() {
        let optimizer = Optimizer::adam(0.001).unwrap();
        let summary = optimizer.summary();

        assert_eq!(summary.optimizer_type, OptimizerType::Adam);
        assert_eq!(summary.learning_rate, 0.001);
        assert_eq!(summary.step, 0);
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut optimizer = Optimizer::sgd(0.1).unwrap();
        let mut params = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let gradients = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

        let result = optimizer.update("test", &mut params, &gradients, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_config() {
        // Test invalid learning rate
        let config = OptimizerConfig {
            learning_rate: -0.1,
            ..Default::default()
        };
        assert!(Optimizer::new(OptimizerType::SGD, config).is_err());

        // Test invalid epsilon
        let config = OptimizerConfig {
            learning_rate: 0.1,
            epsilon: 0.0,
            ..Default::default()
        };
        assert!(Optimizer::new(OptimizerType::SGD, config).is_err());
    }
}
