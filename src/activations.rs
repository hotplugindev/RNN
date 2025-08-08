//! Activation functions for neural networks
//!
//! This module provides a comprehensive set of activation functions commonly
//! used in neural networks, including their forward and backward passes for
//! gradient computation.

use crate::error::{Result, RnnError};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Enumeration of available activation functions
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum Activation {
    /// Linear activation (identity function)
    Linear,
    /// Rectified Linear Unit
    ReLU,
    /// Leaky ReLU with configurable negative slope
    LeakyReLU(f32),
    /// Sigmoid activation
    Sigmoid,
    /// Hyperbolic tangent
    Tanh,
    /// Softmax activation (for multi-class classification)
    Softmax,
    /// Exponential Linear Unit
    ELU(f32),
    /// Swish activation
    Swish,
    /// GELU (Gaussian Error Linear Unit)
    GELU,
    /// Mish activation
    Mish,
    /// Parametric ReLU
    PReLU(f32),
    /// Scaled Exponential Linear Unit
    SELU,
}

impl Activation {
    /// Apply activation function to a single value
    pub fn forward(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => x,
            Activation::ReLU => x.max(0.0),
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Softmax => {
                // Single value softmax is just exp(x) (normalization happens at tensor level)
                x.exp()
            }
            Activation::ELU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * (x.exp() - 1.0)
                }
            }
            Activation::Swish => x * (1.0 / (1.0 + (-x).exp())),
            Activation::GELU => {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
            }
            Activation::Mish => x * (1.0 + x.exp()).ln().tanh(),
            Activation::PReLU(alpha) => {
                if x > 0.0 {
                    x
                } else {
                    alpha * x
                }
            }
            Activation::SELU => {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                if x > 0.0 {
                    scale * x
                } else {
                    scale * alpha * (x.exp() - 1.0)
                }
            }
        }
    }

    /// Compute derivative of activation function
    pub fn backward(&self, x: f32) -> f32 {
        match self {
            Activation::Linear => 1.0,
            Activation::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            Activation::LeakyReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Activation::Sigmoid => {
                let s = self.forward(x);
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Softmax => {
                // Softmax derivative is computed at the tensor level
                // For single values, this is just the softmax value
                let s = self.forward(x);
                s * (1.0 - s)
            }
            Activation::ELU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    alpha * x.exp()
                }
            }
            Activation::Swish => {
                let sigmoid = 1.0 / (1.0 + (-x).exp());
                sigmoid + x * sigmoid * (1.0 - sigmoid)
            }
            Activation::GELU => {
                let sqrt_2_pi = (2.0 / std::f32::consts::PI).sqrt();
                let x_3 = x.powi(3);
                let tanh_arg = sqrt_2_pi * (x + 0.044715 * x_3);
                let tanh_val = tanh_arg.tanh();
                let sech2 = 1.0 - tanh_val.powi(2);

                0.5 * (1.0 + tanh_val)
                    + 0.5 * x * sech2 * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
            }
            Activation::Mish => {
                let exp_x = x.exp();
                let ln_1_exp = (1.0 + exp_x).ln();
                let tanh_ln = ln_1_exp.tanh();
                let _omega = 4.0 * (x + 1.0) + 4.0 * exp_x * (x + 2.0) + 2.0 * exp_x;
                let delta = 2.0 * exp_x + exp_x.powi(2) + 2.0;

                tanh_ln + x * (1.0 - tanh_ln.powi(2)) * exp_x / delta
            }
            Activation::PReLU(alpha) => {
                if x > 0.0 {
                    1.0
                } else {
                    *alpha
                }
            }
            Activation::SELU => {
                let alpha = 1.6732632423543772848170429916717;
                let scale = 1.0507009873554804934193349852946;
                if x > 0.0 {
                    scale
                } else {
                    scale * alpha * x.exp()
                }
            }
        }
    }

    /// Apply activation function to a slice of values
    pub fn forward_slice(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.len() != output.len() {
            return Err(RnnError::shape_mismatch(&[input.len()], &[output.len()]));
        }

        match self {
            Activation::Softmax => self.softmax_forward(input, output),
            _ => {
                for (i, &x) in input.iter().enumerate() {
                    output[i] = self.forward(x);
                }
                Ok(())
            }
        }
    }

    /// Compute derivative for a slice of values
    pub fn backward_slice(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<()> {
        if input.len() != grad_output.len() || input.len() != grad_input.len() {
            return Err(RnnError::shape_mismatch(
                &[input.len()],
                &[grad_output.len()],
            ));
        }

        match self {
            Activation::Softmax => self.softmax_backward(input, grad_output, grad_input),
            _ => {
                for (i, &x) in input.iter().enumerate() {
                    grad_input[i] = self.backward(x) * grad_output[i];
                }
                Ok(())
            }
        }
    }

    /// Specialized softmax forward pass
    fn softmax_forward(&self, input: &[f32], output: &mut [f32]) -> Result<()> {
        if input.is_empty() {
            return Ok(());
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exponentials and sum
        let mut sum = 0.0;
        for (i, &x) in input.iter().enumerate() {
            output[i] = (x - max_val).exp();
            sum += output[i];
        }

        // Normalize
        if sum == 0.0 {
            return Err(RnnError::math("Softmax sum is zero"));
        }

        for output_val in output.iter_mut() {
            *output_val /= sum;
        }

        Ok(())
    }

    /// Specialized softmax backward pass
    fn softmax_backward(
        &self,
        input: &[f32],
        grad_output: &[f32],
        grad_input: &mut [f32],
    ) -> Result<()> {
        // First compute softmax forward pass
        let mut softmax_output = vec![0.0; input.len()];
        self.softmax_forward(input, &mut softmax_output)?;

        // Compute gradient using Jacobian
        for i in 0..input.len() {
            grad_input[i] = 0.0;
            for j in 0..input.len() {
                let jacobian = if i == j {
                    softmax_output[i] * (1.0 - softmax_output[i])
                } else {
                    -softmax_output[i] * softmax_output[j]
                };
                grad_input[i] += jacobian * grad_output[j];
            }
        }

        Ok(())
    }

    /// Get the name of the activation function
    pub fn name(&self) -> &'static str {
        match self {
            Activation::Linear => "linear",
            Activation::ReLU => "relu",
            Activation::LeakyReLU(_) => "leaky_relu",
            Activation::Sigmoid => "sigmoid",
            Activation::Tanh => "tanh",
            Activation::Softmax => "softmax",
            Activation::ELU(_) => "elu",
            Activation::Swish => "swish",
            Activation::GELU => "gelu",
            Activation::Mish => "mish",
            Activation::PReLU(_) => "prelu",
            Activation::SELU => "selu",
        }
    }

    /// Check if activation function has learnable parameters
    pub fn has_parameters(&self) -> bool {
        matches!(
            self,
            Activation::LeakyReLU(_) | Activation::ELU(_) | Activation::PReLU(_)
        )
    }

    /// Get activation function parameters
    pub fn parameters(&self) -> Vec<f32> {
        match self {
            Activation::LeakyReLU(alpha) => vec![*alpha],
            Activation::ELU(alpha) => vec![*alpha],
            Activation::PReLU(alpha) => vec![*alpha],
            _ => Vec::new(),
        }
    }

    /// Set activation function parameters
    pub fn set_parameters(&mut self, params: &[f32]) -> Result<()> {
        match self {
            Activation::LeakyReLU(alpha) => {
                if params.len() != 1 {
                    return Err(RnnError::config("LeakyReLU requires exactly 1 parameter"));
                }
                *alpha = params[0];
            }
            Activation::ELU(alpha) => {
                if params.len() != 1 {
                    return Err(RnnError::config("ELU requires exactly 1 parameter"));
                }
                *alpha = params[0];
            }
            Activation::PReLU(alpha) => {
                if params.len() != 1 {
                    return Err(RnnError::config("PReLU requires exactly 1 parameter"));
                }
                *alpha = params[0];
            }
            _ => {
                if !params.is_empty() {
                    return Err(RnnError::config(
                        "This activation function has no parameters",
                    ));
                }
            }
        }
        Ok(())
    }
}

impl fmt::Display for Activation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Activation::Linear => write!(f, "Linear"),
            Activation::ReLU => write!(f, "ReLU"),
            Activation::LeakyReLU(alpha) => write!(f, "LeakyReLU(α={})", alpha),
            Activation::Sigmoid => write!(f, "Sigmoid"),
            Activation::Tanh => write!(f, "Tanh"),
            Activation::Softmax => write!(f, "Softmax"),
            Activation::ELU(alpha) => write!(f, "ELU(α={})", alpha),
            Activation::Swish => write!(f, "Swish"),
            Activation::GELU => write!(f, "GELU"),
            Activation::Mish => write!(f, "Mish"),
            Activation::PReLU(alpha) => write!(f, "PReLU(α={})", alpha),
            Activation::SELU => write!(f, "SELU"),
        }
    }
}

impl Default for Activation {
    fn default() -> Self {
        Activation::ReLU
    }
}

/// Common activation function presets
impl Activation {
    /// Standard ReLU activation
    pub fn relu() -> Self {
        Activation::ReLU
    }

    /// Leaky ReLU with default slope of 0.01
    pub fn leaky_relu() -> Self {
        Activation::LeakyReLU(0.01)
    }

    /// Leaky ReLU with custom slope
    pub fn leaky_relu_with_slope(alpha: f32) -> Self {
        Activation::LeakyReLU(alpha)
    }

    /// Sigmoid activation
    pub fn sigmoid() -> Self {
        Activation::Sigmoid
    }

    /// Hyperbolic tangent activation
    pub fn tanh() -> Self {
        Activation::Tanh
    }

    /// Softmax activation
    pub fn softmax() -> Self {
        Activation::Softmax
    }

    /// ELU with default alpha of 1.0
    pub fn elu() -> Self {
        Activation::ELU(1.0)
    }

    /// ELU with custom alpha
    pub fn elu_with_alpha(alpha: f32) -> Self {
        Activation::ELU(alpha)
    }

    /// Swish activation
    pub fn swish() -> Self {
        Activation::Swish
    }

    /// GELU activation
    pub fn gelu() -> Self {
        Activation::GELU
    }

    /// Mish activation
    pub fn mish() -> Self {
        Activation::Mish
    }

    /// PReLU with default alpha of 0.25
    pub fn prelu() -> Self {
        Activation::PReLU(0.25)
    }

    /// PReLU with custom alpha
    pub fn prelu_with_alpha(alpha: f32) -> Self {
        Activation::PReLU(alpha)
    }

    /// SELU activation
    pub fn selu() -> Self {
        Activation::SELU
    }

    /// Linear/identity activation
    pub fn linear() -> Self {
        Activation::Linear
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_relu_forward() {
        let relu = Activation::ReLU;
        assert_eq!(relu.forward(-1.0), 0.0);
        assert_eq!(relu.forward(0.0), 0.0);
        assert_eq!(relu.forward(1.0), 1.0);
        assert_eq!(relu.forward(5.0), 5.0);
    }

    #[test]
    fn test_relu_backward() {
        let relu = Activation::ReLU;
        assert_eq!(relu.backward(-1.0), 0.0);
        assert_eq!(relu.backward(0.0), 0.0);
        assert_eq!(relu.backward(1.0), 1.0);
        assert_eq!(relu.backward(5.0), 1.0);
    }

    #[test]
    fn test_sigmoid_forward() {
        let sigmoid = Activation::Sigmoid;
        assert_relative_eq!(sigmoid.forward(0.0), 0.5, epsilon = 1e-6);
        assert!(sigmoid.forward(-10.0) < 0.01);
        assert!(sigmoid.forward(10.0) > 0.99);
    }

    #[test]
    fn test_sigmoid_backward() {
        let sigmoid = Activation::Sigmoid;
        let x = 0.0;
        let y = sigmoid.forward(x);
        let expected_grad = y * (1.0 - y);
        assert_relative_eq!(sigmoid.backward(x), expected_grad, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_forward() {
        let tanh = Activation::Tanh;
        assert_relative_eq!(tanh.forward(0.0), 0.0, epsilon = 1e-6);
        assert!(tanh.forward(-10.0) > -1.0);
        assert!(tanh.forward(10.0) < 1.0);
    }

    #[test]
    fn test_leaky_relu() {
        let leaky_relu = Activation::LeakyReLU(0.01);
        assert_eq!(leaky_relu.forward(1.0), 1.0);
        assert_eq!(leaky_relu.forward(-1.0), -0.01);
        assert_eq!(leaky_relu.backward(1.0), 1.0);
        assert_eq!(leaky_relu.backward(-1.0), 0.01);
    }

    #[test]
    fn test_softmax_forward() {
        let softmax = Activation::Softmax;
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];

        softmax.forward_slice(&input, &mut output).unwrap();

        // Check that outputs sum to 1
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all outputs are positive
        assert!(output.iter().all(|&x| x > 0.0));

        // Check that larger inputs produce larger outputs
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_softmax_backward() {
        let softmax = Activation::Softmax;
        let input = vec![1.0, 2.0, 3.0];
        let grad_output = vec![1.0, 0.0, 0.0];
        let mut grad_input = vec![0.0; 3];

        softmax
            .backward_slice(&input, &grad_output, &mut grad_input)
            .unwrap();

        // Basic sanity check - gradients should sum to 0 for softmax
        let sum: f32 = grad_input.iter().sum();
        assert_relative_eq!(sum, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_gelu_forward() {
        let gelu = Activation::GELU;
        assert_relative_eq!(gelu.forward(0.0), 0.0, epsilon = 1e-6);
        assert!(gelu.forward(1.0) > 0.8); // GELU(1) ≈ 0.841
        assert!(gelu.forward(-1.0) < -0.2); // GELU(-1) ≈ -0.159
    }

    #[test]
    fn test_swish_forward() {
        let swish = Activation::Swish;
        assert_relative_eq!(swish.forward(0.0), 0.0, epsilon = 1e-6);
        assert!(swish.forward(1.0) > 0.7); // Swish(1) ≈ 0.731
        assert!(swish.forward(-1.0) > -0.3); // Swish(-1) ≈ -0.269
    }

    #[test]
    fn test_elu_forward() {
        let elu = Activation::ELU(1.0);
        assert_eq!(elu.forward(1.0), 1.0);
        assert_eq!(elu.forward(0.0), 0.0);
        assert!(elu.forward(-1.0) > -1.0);
    }

    #[test]
    fn test_activation_parameters() {
        let mut leaky_relu = Activation::LeakyReLU(0.01);
        assert!(leaky_relu.has_parameters());
        assert_eq!(leaky_relu.parameters(), vec![0.01]);

        leaky_relu.set_parameters(&[0.1]).unwrap();
        assert_eq!(leaky_relu.parameters(), vec![0.1]);

        let relu = Activation::ReLU;
        assert!(!relu.has_parameters());
        assert_eq!(relu.parameters(), Vec::<f32>::new());
    }

    #[test]
    fn test_activation_names() {
        assert_eq!(Activation::ReLU.name(), "relu");
        assert_eq!(Activation::Sigmoid.name(), "sigmoid");
        assert_eq!(Activation::Tanh.name(), "tanh");
        assert_eq!(Activation::Softmax.name(), "softmax");
        assert_eq!(Activation::LeakyReLU(0.01).name(), "leaky_relu");
    }

    #[test]
    fn test_activation_display() {
        assert_eq!(format!("{}", Activation::ReLU), "ReLU");
        assert_eq!(
            format!("{}", Activation::LeakyReLU(0.01)),
            "LeakyReLU(α=0.01)"
        );
        assert_eq!(format!("{}", Activation::ELU(1.0)), "ELU(α=1)");
    }

    #[test]
    fn test_selu_properties() {
        let selu = Activation::SELU;
        // SELU should preserve mean and variance for normalized inputs
        let x = 1.0;
        let y = selu.forward(x);
        assert!(y > 1.0); // SELU amplifies positive values

        let x = -1.0;
        let y = selu.forward(x);
        assert!(y < 0.0 && y > -2.0); // SELU has specific negative scaling
    }

    #[test]
    fn test_mish_forward() {
        let mish = Activation::Mish;
        assert_relative_eq!(mish.forward(0.0), 0.0, epsilon = 1e-6);
        assert!(mish.forward(1.0) > 0.8); // Mish(1) ≈ 0.865
        assert!(mish.forward(-1.0) < -0.2); // Mish has small negative values
    }

    #[test]
    fn test_prelu_vs_leaky_relu() {
        let alpha = 0.25;
        let prelu = Activation::PReLU(alpha);
        let leaky_relu = Activation::LeakyReLU(alpha);

        // PReLU and LeakyReLU should behave identically for the same alpha
        for x in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            assert_eq!(prelu.forward(*x), leaky_relu.forward(*x));
            assert_eq!(prelu.backward(*x), leaky_relu.backward(*x));
        }
    }
}
