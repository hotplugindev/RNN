//! Optimizers for neural network training
//!
//! This module provides a comprehensive set of optimization algorithms for
//! training neural networks, including gradient descent variants and adaptive
//! methods with momentum, learning rate scheduling, and regularization.

use crate::error::{Result, RnnError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Configuration for different optimizers
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OptimizerConfig {
    /// Stochastic Gradient Descent
    SGD {
        learning_rate: f32,
        momentum: Option<f32>,
        weight_decay: Option<f32>,
        nesterov: bool,
    },
    /// Adam optimizer
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
        amsgrad: bool,
    },
    /// AdaGrad optimizer
    AdaGrad {
        learning_rate: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
    },
    /// RMSprop optimizer
    RMSprop {
        learning_rate: f32,
        alpha: f32,
        epsilon: f32,
        weight_decay: Option<f32>,
        momentum: Option<f32>,
        centered: bool,
    },
    /// AdamW optimizer (Adam with decoupled weight decay)
    AdamW {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        weight_decay: f32,
    },
    /// LBFGS optimizer (Newton's method approximation)
    LBFGS {
        learning_rate: f32,
        max_iter: usize,
        max_eval: Option<usize>,
        tolerance_grad: f32,
        tolerance_change: f32,
        history_size: usize,
        line_search_fn: Option<String>,
    },
    /// Adabound optimizer
    AdaBound {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
        final_lr: f32,
        gamma: f32,
        weight_decay: Option<f32>,
    },
    /// Lookahead optimizer wrapper
    Lookahead {
        base_optimizer: Box<OptimizerConfig>,
        k: usize,
        alpha: f32,
    },
}

/// Optimizer trait for parameter updates
pub trait Optimizer: Send + Sync + std::fmt::Debug {
    /// Update parameters given gradients
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()>;

    /// Get current learning rate
    fn learning_rate(&self) -> f32;

    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);

    /// Zero gradients (reset state if needed)
    fn zero_grad(&mut self);

    /// Get optimizer state for serialization
    fn state_dict(&self) -> HashMap<String, Tensor>;

    /// Load optimizer state from serialization
    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()>;

    /// Get optimizer name
    fn name(&self) -> &str;
}

/// SGD optimizer implementation
#[derive(Debug)]
pub struct SGD {
    learning_rate: f32,
    momentum: Option<f32>,
    weight_decay: Option<f32>,
    nesterov: bool,
    velocity: Vec<Option<Tensor>>,
}

impl SGD {
    pub fn new(config: &OptimizerConfig) -> Result<Self> {
        match config {
            OptimizerConfig::SGD {
                learning_rate,
                momentum,
                weight_decay,
                nesterov,
            } => Ok(Self {
                learning_rate: *learning_rate,
                momentum: *momentum,
                weight_decay: *weight_decay,
                nesterov: *nesterov,
                velocity: Vec::new(),
            }),
            _ => Err(RnnError::config("Invalid config for SGD optimizer")),
        }
    }

    fn ensure_velocity_initialized(&mut self, parameters: &[Tensor]) -> Result<()> {
        if self.velocity.len() != parameters.len() {
            self.velocity = parameters
                .iter()
                .map(|param| {
                    if self.momentum.is_some() {
                        Some(
                            Tensor::zeros_on_device(param.shape(), param.device().clone()).unwrap(),
                        )
                    } else {
                        None
                    }
                })
                .collect();
        }
        Ok(())
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(RnnError::config("Parameters and gradients length mismatch"));
        }

        self.ensure_velocity_initialized(parameters)?;

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let mut update = grad.clone_data()?;

            // Apply weight decay
            if let Some(decay) = self.weight_decay {
                let weight_penalty = param.mul_scalar(decay)?;
                update = update.add(&weight_penalty)?;
            }

            // Apply momentum
            if let Some(momentum_factor) = self.momentum {
                if let Some(ref mut velocity) = self.velocity[i] {
                    *velocity = velocity.mul_scalar(momentum_factor)?.add(&update)?;

                    if self.nesterov {
                        // Nesterov momentum: param = param - lr * (momentum * velocity + grad)
                        let nesterov_update = velocity
                            .mul_scalar(momentum_factor)?
                            .add(&update)?
                            .mul_scalar(self.learning_rate)?;
                        *param = param.sub(&nesterov_update)?;
                    } else {
                        // Standard momentum: param = param - lr * velocity
                        let momentum_update = velocity.mul_scalar(self.learning_rate)?;
                        *param = param.sub(&momentum_update)?;
                    }
                } else {
                    return Err(RnnError::config("Velocity not initialized for momentum"));
                }
            } else {
                // Standard SGD: param = param - lr * grad
                let sgd_update = update.mul_scalar(self.learning_rate)?;
                *param = param.sub(&sgd_update)?;
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn zero_grad(&mut self) {
        // SGD doesn't accumulate gradients, so nothing to zero
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        for (i, velocity) in self.velocity.iter().enumerate() {
            if let Some(v) = velocity {
                state.insert(format!("velocity_{}", i), v.clone_data().unwrap());
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        for (key, tensor) in state {
            if let Some(index_str) = key.strip_prefix("velocity_") {
                if let Ok(index) = index_str.parse::<usize>() {
                    if index < self.velocity.len() {
                        self.velocity[index] = Some(tensor);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "SGD"
    }
}

/// Adam optimizer implementation
#[derive(Debug)]
pub struct Adam {
    learning_rate: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    weight_decay: Option<f32>,
    amsgrad: bool,
    step_count: usize,
    m: Vec<Option<Tensor>>,     // First moment
    v: Vec<Option<Tensor>>,     // Second moment
    v_max: Vec<Option<Tensor>>, // For AMSGrad
}

impl Adam {
    pub fn new(config: &OptimizerConfig) -> Result<Self> {
        match config {
            OptimizerConfig::Adam {
                learning_rate,
                beta1,
                beta2,
                epsilon,
                weight_decay,
                amsgrad,
            } => Ok(Self {
                learning_rate: *learning_rate,
                beta1: *beta1,
                beta2: *beta2,
                epsilon: *epsilon,
                weight_decay: *weight_decay,
                amsgrad: *amsgrad,
                step_count: 0,
                m: Vec::new(),
                v: Vec::new(),
                v_max: Vec::new(),
            }),
            _ => Err(RnnError::config("Invalid config for Adam optimizer")),
        }
    }

    fn ensure_moments_initialized(&mut self, parameters: &[Tensor]) -> Result<()> {
        if self.m.len() != parameters.len() {
            self.m = parameters
                .iter()
                .map(|param| {
                    Some(Tensor::zeros_on_device(param.shape(), param.device().clone()).unwrap())
                })
                .collect();
        }
        if self.v.len() != parameters.len() {
            self.v = parameters
                .iter()
                .map(|param| {
                    Some(Tensor::zeros_on_device(param.shape(), param.device().clone()).unwrap())
                })
                .collect();
        }
        if self.amsgrad && self.v_max.len() != parameters.len() {
            self.v_max = parameters
                .iter()
                .map(|param| {
                    Some(Tensor::zeros_on_device(param.shape(), param.device().clone()).unwrap())
                })
                .collect();
        }
        Ok(())
    }
}

impl Optimizer for Adam {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(RnnError::config("Parameters and gradients length mismatch"));
        }

        self.ensure_moments_initialized(parameters)?;
        self.step_count += 1;

        let bias_correction1 = 1.0 - self.beta1.powi(self.step_count as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.step_count as i32);

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let mut grad_with_decay = grad.clone_data()?;

            // Apply weight decay
            if let Some(decay) = self.weight_decay {
                let weight_penalty = param.mul_scalar(decay)?;
                grad_with_decay = grad_with_decay.add(&weight_penalty)?;
            }

            // Update first moment (momentum)
            if let Some(ref mut m) = self.m[i] {
                *m = m
                    .mul_scalar(self.beta1)?
                    .add(&grad_with_decay.mul_scalar(1.0 - self.beta1)?)?;
            }

            // Update second moment (RMSprop)
            if let Some(ref mut v) = self.v[i] {
                let grad_squared = grad_with_decay.mul(&grad_with_decay)?;
                *v = v
                    .mul_scalar(self.beta2)?
                    .add(&grad_squared.mul_scalar(1.0 - self.beta2)?)?;
            }

            // Get moments for this parameter
            let m = self.m[i].as_ref().unwrap();
            let v = self.v[i].as_ref().unwrap();

            // Bias correction
            let m_hat = m.mul_scalar(1.0 / bias_correction1)?;
            let v_hat = if self.amsgrad {
                // AMSGrad: use max of current and previous v_hat
                let current_v_hat = v.mul_scalar(1.0 / bias_correction2)?;
                if let Some(ref mut v_max) = self.v_max[i] {
                    // Element-wise max
                    let v_max_data = v_max.to_vec()?;
                    let current_data = current_v_hat.to_vec()?;
                    let max_data: Vec<f32> = v_max_data
                        .iter()
                        .zip(current_data.iter())
                        .map(|(&a, &b)| a.max(b))
                        .collect();
                    *v_max = Tensor::from_slice_on_device(
                        &max_data,
                        v_max.shape(),
                        v_max.device().clone(),
                    )?;
                    v_max.clone_data()?
                } else {
                    current_v_hat
                }
            } else {
                v.mul_scalar(1.0 / bias_correction2)?
            };

            // Compute update: lr * m_hat / (sqrt(v_hat) + epsilon)
            let v_hat_sqrt = self.element_wise_sqrt(&v_hat)?;
            let denominator = v_hat_sqrt.add_scalar(self.epsilon)?;
            let update = m_hat.div(&denominator)?.mul_scalar(self.learning_rate)?;

            // Update parameter
            *param = param.sub(&update)?;
        }

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn zero_grad(&mut self) {
        // Adam doesn't accumulate gradients, so nothing to zero
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        for (i, m) in self.m.iter().enumerate() {
            if let Some(moment) = m {
                state.insert(format!("m_{}", i), moment.clone_data().unwrap());
            }
        }

        for (i, v) in self.v.iter().enumerate() {
            if let Some(moment) = v {
                state.insert(format!("v_{}", i), moment.clone_data().unwrap());
            }
        }

        if self.amsgrad {
            for (i, v_max) in self.v_max.iter().enumerate() {
                if let Some(moment) = v_max {
                    state.insert(format!("v_max_{}", i), moment.clone_data().unwrap());
                }
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        for (key, tensor) in state {
            if let Some(index_str) = key.strip_prefix("m_") {
                if let Ok(index) = index_str.parse::<usize>() {
                    if index < self.m.len() {
                        self.m[index] = Some(tensor);
                    }
                }
            } else if let Some(index_str) = key.strip_prefix("v_") {
                if let Ok(index) = index_str.parse::<usize>() {
                    if index < self.v.len() {
                        self.v[index] = Some(tensor);
                    }
                }
            } else if let Some(index_str) = key.strip_prefix("v_max_") {
                if let Ok(index) = index_str.parse::<usize>() {
                    if index < self.v_max.len() {
                        self.v_max[index] = Some(tensor);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "Adam"
    }
}

impl Adam {
    fn element_wise_sqrt(&self, tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.to_vec()?;
        let sqrt_data: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Tensor::from_slice_on_device(&sqrt_data, tensor.shape(), tensor.device().clone())
    }
}

/// AdaGrad optimizer implementation
#[derive(Debug)]
pub struct AdaGrad {
    learning_rate: f32,
    epsilon: f32,
    weight_decay: Option<f32>,
    sum_squared_gradients: Vec<Option<Tensor>>,
}

impl AdaGrad {
    pub fn new(config: &OptimizerConfig) -> Result<Self> {
        match config {
            OptimizerConfig::AdaGrad {
                learning_rate,
                epsilon,
                weight_decay,
            } => Ok(Self {
                learning_rate: *learning_rate,
                epsilon: *epsilon,
                weight_decay: *weight_decay,
                sum_squared_gradients: Vec::new(),
            }),
            _ => Err(RnnError::config("Invalid config for AdaGrad optimizer")),
        }
    }

    fn ensure_sum_squared_initialized(&mut self, parameters: &[Tensor]) -> Result<()> {
        if self.sum_squared_gradients.len() != parameters.len() {
            self.sum_squared_gradients = parameters
                .iter()
                .map(|param| {
                    Some(Tensor::zeros_on_device(param.shape(), param.device().clone()).unwrap())
                })
                .collect();
        }
        Ok(())
    }
}

impl Optimizer for AdaGrad {
    fn step(&mut self, parameters: &mut [Tensor], gradients: &[Tensor]) -> Result<()> {
        if parameters.len() != gradients.len() {
            return Err(RnnError::config("Parameters and gradients length mismatch"));
        }

        self.ensure_sum_squared_initialized(parameters)?;

        for (i, (param, grad)) in parameters.iter_mut().zip(gradients.iter()).enumerate() {
            let mut grad_with_decay = grad.clone_data()?;

            // Apply weight decay
            if let Some(decay) = self.weight_decay {
                let weight_penalty = param.mul_scalar(decay)?;
                grad_with_decay = grad_with_decay.add(&weight_penalty)?;
            }

            // Update sum of squared gradients
            if let Some(ref mut sum_sq) = self.sum_squared_gradients[i] {
                let grad_squared = grad_with_decay.mul(&grad_with_decay)?;
                *sum_sq = sum_sq.add(&grad_squared)?;

                // Compute adaptive learning rate
                let sum_sq_clone = sum_sq.clone();
                let sum_sq_sqrt = self.element_wise_sqrt(&sum_sq_clone)?;
                let denominator = sum_sq_sqrt.add_scalar(self.epsilon)?;
                let adaptive_lr = grad_with_decay
                    .div(&denominator)?
                    .mul_scalar(self.learning_rate)?;

                // Update parameter
                *param = param.sub(&adaptive_lr)?;
            }
        }

        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn set_learning_rate(&mut self, lr: f32) {
        self.learning_rate = lr;
    }

    fn zero_grad(&mut self) {
        // AdaGrad doesn't accumulate gradients, so nothing to zero
    }

    fn state_dict(&self) -> HashMap<String, Tensor> {
        let mut state = HashMap::new();

        for (i, sum_sq) in self.sum_squared_gradients.iter().enumerate() {
            if let Some(tensor) = sum_sq {
                state.insert(format!("sum_squared_{}", i), tensor.clone_data().unwrap());
            }
        }

        state
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        for (key, tensor) in state {
            if let Some(index_str) = key.strip_prefix("sum_squared_") {
                if let Ok(index) = index_str.parse::<usize>() {
                    if index < self.sum_squared_gradients.len() {
                        self.sum_squared_gradients[index] = Some(tensor);
                    }
                }
            }
        }
        Ok(())
    }

    fn name(&self) -> &str {
        "AdaGrad"
    }
}

impl AdaGrad {
    fn element_wise_sqrt(&self, tensor: &Tensor) -> Result<Tensor> {
        let data = tensor.to_vec()?;
        let sqrt_data: Vec<f32> = data.iter().map(|&x| x.sqrt()).collect();
        Tensor::from_slice_on_device(&sqrt_data, tensor.shape(), tensor.device().clone())
    }
}

/// Factory function to create optimizers from configuration
pub fn create_optimizer(config: OptimizerConfig) -> Result<Box<dyn Optimizer>> {
    match config {
        OptimizerConfig::SGD { .. } => Ok(Box::new(SGD::new(&config)?)),
        OptimizerConfig::Adam { .. } => Ok(Box::new(Adam::new(&config)?)),
        OptimizerConfig::AdaGrad { .. } => Ok(Box::new(AdaGrad::new(&config)?)),
        _ => Err(RnnError::unsupported("Optimizer not yet implemented")),
    }
}

impl fmt::Display for OptimizerConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            OptimizerConfig::SGD {
                learning_rate,
                momentum,
                ..
            } => {
                write!(f, "SGD(lr={}", learning_rate)?;
                if let Some(m) = momentum {
                    write!(f, ", momentum={}", m)?;
                }
                write!(f, ")")
            }
            OptimizerConfig::Adam {
                learning_rate,
                beta1,
                beta2,
                ..
            } => {
                write!(f, "Adam(lr={}, β₁={}, β₂={})", learning_rate, beta1, beta2)
            }
            OptimizerConfig::AdaGrad { learning_rate, .. } => {
                write!(f, "AdaGrad(lr={})", learning_rate)
            }
            OptimizerConfig::RMSprop {
                learning_rate,
                alpha,
                ..
            } => {
                write!(f, "RMSprop(lr={}, α={})", learning_rate, alpha)
            }
            OptimizerConfig::AdamW {
                learning_rate,
                weight_decay,
                ..
            } => {
                write!(f, "AdamW(lr={}, decay={})", learning_rate, weight_decay)
            }
            OptimizerConfig::LBFGS {
                learning_rate,
                max_iter,
                ..
            } => {
                write!(f, "LBFGS(lr={}, max_iter={})", learning_rate, max_iter)
            }
            OptimizerConfig::AdaBound {
                learning_rate,
                final_lr,
                ..
            } => {
                write!(f, "AdaBound(lr={}, final_lr={})", learning_rate, final_lr)
            }
            OptimizerConfig::Lookahead {
                base_optimizer,
                k,
                alpha,
            } => {
                write!(f, "Lookahead({}, k={}, α={})", base_optimizer, k, alpha)
            }
        }
    }
}

/// Common optimizer presets
impl OptimizerConfig {
    /// Standard SGD
    pub fn sgd(learning_rate: f32) -> Self {
        OptimizerConfig::SGD {
            learning_rate,
            momentum: None,
            weight_decay: None,
            nesterov: false,
        }
    }

    /// SGD with momentum
    pub fn sgd_momentum(learning_rate: f32, momentum: f32) -> Self {
        OptimizerConfig::SGD {
            learning_rate,
            momentum: Some(momentum),
            weight_decay: None,
            nesterov: false,
        }
    }

    /// Adam with default parameters
    pub fn adam(learning_rate: f32) -> Self {
        OptimizerConfig::Adam {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: None,
            amsgrad: false,
        }
    }

    /// AdaGrad with default parameters
    pub fn adagrad(learning_rate: f32) -> Self {
        OptimizerConfig::AdaGrad {
            learning_rate,
            epsilon: 1e-10,
            weight_decay: None,
        }
    }

    /// RMSprop with default parameters
    pub fn rmsprop(learning_rate: f32) -> Self {
        OptimizerConfig::RMSprop {
            learning_rate,
            alpha: 0.99,
            epsilon: 1e-8,
            weight_decay: None,
            momentum: None,
            centered: false,
        }
    }

    /// AdamW with default parameters
    pub fn adamw(learning_rate: f32, weight_decay: f32) -> Self {
        OptimizerConfig::AdamW {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay,
        }
    }

    /// LBFGS with default parameters
    pub fn lbfgs(learning_rate: f32) -> Self {
        OptimizerConfig::LBFGS {
            learning_rate,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 100,
            line_search_fn: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_sgd_optimizer() {
        let config = OptimizerConfig::sgd(0.1);
        let mut optimizer = SGD::new(&config).unwrap();

        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap()];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap()];

        optimizer.step(&mut params, &grads).unwrap();

        let updated_data = params[0].to_vec().unwrap();
        assert_eq!(updated_data, vec![0.99, 1.98]); // 1.0 - 0.1*0.1, 2.0 - 0.1*0.2
    }

    #[test]
    fn test_adam_optimizer() {
        let config = OptimizerConfig::adam(0.001);
        let mut optimizer = Adam::new(&config).unwrap();

        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap()];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap()];

        // First step
        optimizer.step(&mut params, &grads).unwrap();

        let updated_data = params[0].to_vec().unwrap();
        // Adam should make smaller updates initially due to bias correction
        assert!(updated_data[0] < 1.0);
        assert!(updated_data[1] < 2.0);
        assert!(updated_data[0] > 0.99);
        assert!(updated_data[1] > 1.99);
    }

    #[test]
    fn test_adagrad_optimizer() {
        let config = OptimizerConfig::adagrad(0.1);
        let mut optimizer = AdaGrad::new(&config).unwrap();

        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap()];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap()];

        optimizer.step(&mut params, &grads).unwrap();

        let updated_data = params[0].to_vec().unwrap();
        // AdaGrad should adapt learning rate based on gradient history
        assert!(updated_data[0] < 1.0);
        assert!(updated_data[1] < 2.0);
    }

    #[test]
    fn test_optimizer_state_serialization() {
        let config = OptimizerConfig::adam(0.001);
        let mut optimizer = Adam::new(&config).unwrap();

        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap()];
        let grads = vec![Tensor::from_slice(&[0.1, 0.2], &[2]).unwrap()];

        // Take a step to initialize state
        optimizer.step(&mut params, &grads).unwrap();

        // Save and load state
        let state = optimizer.state_dict();
        let mut new_optimizer = Adam::new(&config).unwrap();
        new_optimizer.load_state_dict(state).unwrap();

        // Both optimizers should behave identically now
        let mut params1 = params.clone();
        let mut params2 = params.clone();

        optimizer.step(&mut params1, &grads).unwrap();
        new_optimizer.step(&mut params2, &grads).unwrap();

        let data1 = params1[0].to_vec().unwrap();
        let data2 = params2[0].to_vec().unwrap();

        for (a, b) in data1.iter().zip(data2.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_learning_rate_scheduling() {
        let config = OptimizerConfig::sgd(0.1);
        let mut optimizer = SGD::new(&config).unwrap();

        assert_eq!(optimizer.learning_rate(), 0.1);

        optimizer.set_learning_rate(0.01);
        assert_eq!(optimizer.learning_rate(), 0.01);
    }

    #[test]
    fn test_optimizer_display() {
        let sgd = OptimizerConfig::sgd(0.1);
        assert!(format!("{}", sgd).contains("SGD"));

        let adam = OptimizerConfig::adam(0.001);
        assert!(format!("{}", adam).contains("Adam"));

        let sgd_momentum = OptimizerConfig::sgd_momentum(0.1, 0.9);
        assert!(format!("{}", sgd_momentum).contains("momentum"));
    }

    #[test]
    fn test_optimizer_factory() {
        let sgd_config = OptimizerConfig::sgd(0.1);
        let sgd_optimizer = create_optimizer(sgd_config).unwrap();
        assert_eq!(sgd_optimizer.name(), "SGD");

        let adam_config = OptimizerConfig::adam(0.001);
        let adam_optimizer = create_optimizer(adam_config).unwrap();
        assert_eq!(adam_optimizer.name(), "Adam");
    }

    #[test]
    fn test_weight_decay() {
        let config = OptimizerConfig::SGD {
            learning_rate: 0.1,
            momentum: None,
            weight_decay: Some(0.01),
            nesterov: false,
        };
        let mut optimizer = SGD::new(&config).unwrap();

        let mut params = vec![Tensor::from_slice(&[1.0, 2.0], &[2]).unwrap()];
        let grads = vec![Tensor::from_slice(&[0.0, 0.0], &[2]).unwrap()]; // Zero gradients

        let original_data = params[0].to_vec().unwrap();
        optimizer.step(&mut params, &grads).unwrap();
        let updated_data = params[0].to_vec().unwrap();

        // With weight decay and zero gradients, parameters should decrease
        assert!(updated_data[0] < original_data[0]);
        assert!(updated_data[1] < original_data[1]);
    }
}
