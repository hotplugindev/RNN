//! Activation functions and their derivatives for neural networks.
//!
//! This module provides a comprehensive set of activation functions commonly used
//! in neural networks, along with their derivatives for backpropagation.

use crate::error::{NetworkError, Result};
use ndarray::{Array1, Array2, Zip};
use serde::{Deserialize, Serialize};
use std::f64::consts::{E, PI};

/// Enumeration of available activation functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationFunction {
    /// Linear activation: f(x) = x
    Linear,
    /// Sigmoid activation: f(x) = 1 / (1 + e^(-x))
    Sigmoid,
    /// Hyperbolic tangent: f(x) = tanh(x)
    Tanh,
    /// Rectified Linear Unit: f(x) = max(0, x)
    ReLU,
    /// Leaky ReLU: f(x) = max(αx, x) where α is typically 0.01
    LeakyReLU,
    /// Exponential Linear Unit: f(x) = x if x > 0, α(e^x - 1) if x ≤ 0
    ELU,
    /// Swish activation: f(x) = x * sigmoid(x)
    Swish,
    /// GELU (Gaussian Error Linear Unit): f(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
    GELU,
    /// Softmax activation (for output layers)
    Softmax,
    /// Softplus activation: f(x) = ln(1 + e^x)
    Softplus,
    /// Mish activation: f(x) = x * tanh(softplus(x))
    Mish,
    /// Hard Sigmoid: f(x) = max(0, min(1, (x + 1) / 2))
    HardSigmoid,
    /// Hard Tanh: f(x) = max(-1, min(1, x))
    HardTanh,
    /// Parametric ReLU: f(x) = max(αx, x)
    PReLU,
}

impl ActivationFunction {
    /// Apply the activation function to a single value.
    pub fn apply_scalar(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Linear => x,
            ActivationFunction::Sigmoid => sigmoid(x),
            ActivationFunction::Tanh => x.tanh(),
            ActivationFunction::ReLU => x.max(0.0),
            ActivationFunction::LeakyReLU => leaky_relu(x, 0.01),
            ActivationFunction::ELU => elu(x, 1.0),
            ActivationFunction::Swish => swish(x),
            ActivationFunction::GELU => gelu(x),
            ActivationFunction::Softmax => {
                // For scalar, softmax is just e^x (normalization happens at vector level)
                x.exp()
            }
            ActivationFunction::Softplus => softplus(x),
            ActivationFunction::Mish => mish(x),
            ActivationFunction::HardSigmoid => hard_sigmoid(x),
            ActivationFunction::HardTanh => hard_tanh(x),
            ActivationFunction::PReLU => leaky_relu(x, 0.01), // Default to 0.01
        }
    }

    /// Apply the activation function to a vector.
    pub fn apply_vector(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        match self {
            ActivationFunction::Softmax => Ok(softmax(x)?),
            _ => Ok(x.mapv(|val| self.apply_scalar(val))),
        }
    }

    /// Apply the activation function to a matrix (row-wise for Softmax).
    pub fn apply_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self {
            ActivationFunction::Softmax => {
                let mut result = Array2::zeros(x.raw_dim());
                for (i, mut row) in result.rows_mut().into_iter().enumerate() {
                    let input_row = x.row(i);
                    let softmax_row = softmax(&input_row.to_owned())?;
                    row.assign(&softmax_row);
                }
                Ok(result)
            }
            _ => Ok(x.mapv(|val| self.apply_scalar(val))),
        }
    }

    /// Compute the derivative of the activation function for a single value.
    pub fn derivative_scalar(&self, x: f64) -> f64 {
        match self {
            ActivationFunction::Linear => 1.0,
            ActivationFunction::Sigmoid => {
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            ActivationFunction::Tanh => 1.0 - x.tanh().powi(2),
            ActivationFunction::ReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::LeakyReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
            ActivationFunction::ELU => {
                if x > 0.0 {
                    1.0
                } else {
                    1.0 * (x.exp())
                }
            }
            ActivationFunction::Swish => {
                let s = sigmoid(x);
                s + x * s * (1.0 - s)
            }
            ActivationFunction::GELU => gelu_derivative(x),
            ActivationFunction::Softmax => {
                // Softmax derivative is computed differently (Jacobian matrix)
                // For individual elements, this is an approximation
                let s = sigmoid(x);
                s * (1.0 - s)
            }
            ActivationFunction::Softplus => sigmoid(x),
            ActivationFunction::Mish => mish_derivative(x),
            ActivationFunction::HardSigmoid => {
                if x >= -1.0 && x <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            ActivationFunction::HardTanh => {
                if x >= -1.0 && x <= 1.0 {
                    1.0
                } else {
                    0.0
                }
            }
            ActivationFunction::PReLU => {
                if x > 0.0 {
                    1.0
                } else {
                    0.01
                }
            }
        }
    }

    /// Compute the derivative of the activation function for a vector.
    pub fn derivative_vector(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        match self {
            ActivationFunction::Softmax => {
                // For softmax, return the softmax output (derivative is computed differently in backprop)
                softmax(x)
            }
            _ => Ok(x.mapv(|val| self.derivative_scalar(val))),
        }
    }

    /// Compute the derivative of the activation function for a matrix.
    pub fn derivative_matrix(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self {
            ActivationFunction::Softmax => {
                // For softmax, return the softmax output
                self.apply_matrix(x)
            }
            _ => Ok(x.mapv(|val| self.derivative_scalar(val))),
        }
    }

    /// Get the name of the activation function as a string.
    pub fn name(&self) -> &'static str {
        match self {
            ActivationFunction::Linear => "Linear",
            ActivationFunction::Sigmoid => "Sigmoid",
            ActivationFunction::Tanh => "Tanh",
            ActivationFunction::ReLU => "ReLU",
            ActivationFunction::LeakyReLU => "LeakyReLU",
            ActivationFunction::ELU => "ELU",
            ActivationFunction::Swish => "Swish",
            ActivationFunction::GELU => "GELU",
            ActivationFunction::Softmax => "Softmax",
            ActivationFunction::Softplus => "Softplus",
            ActivationFunction::Mish => "Mish",
            ActivationFunction::HardSigmoid => "HardSigmoid",
            ActivationFunction::HardTanh => "HardTanh",
            ActivationFunction::PReLU => "PReLU",
        }
    }

    /// Check if the activation function is suitable for output layers.
    pub fn is_output_suitable(&self) -> bool {
        matches!(
            self,
            ActivationFunction::Sigmoid
                | ActivationFunction::Softmax
                | ActivationFunction::Linear
                | ActivationFunction::Tanh
        )
    }

    /// Get the recommended range for this activation function.
    pub fn recommended_range(&self) -> (f64, f64) {
        match self {
            ActivationFunction::Linear => (f64::NEG_INFINITY, f64::INFINITY),
            ActivationFunction::Sigmoid => (-10.0, 10.0),
            ActivationFunction::Tanh => (-3.0, 3.0),
            ActivationFunction::ReLU => (0.0, f64::INFINITY),
            ActivationFunction::LeakyReLU => (f64::NEG_INFINITY, f64::INFINITY),
            ActivationFunction::ELU => (f64::NEG_INFINITY, f64::INFINITY),
            ActivationFunction::Swish => (-10.0, 10.0),
            ActivationFunction::GELU => (-3.0, 3.0),
            ActivationFunction::Softmax => (-10.0, 10.0),
            ActivationFunction::Softplus => (-10.0, 10.0),
            ActivationFunction::Mish => (-10.0, 10.0),
            ActivationFunction::HardSigmoid => (-2.0, 2.0),
            ActivationFunction::HardTanh => (-2.0, 2.0),
            ActivationFunction::PReLU => (f64::NEG_INFINITY, f64::INFINITY),
        }
    }
}

impl Default for ActivationFunction {
    fn default() -> Self {
        ActivationFunction::ReLU
    }
}

// Helper functions for individual activation functions

/// Sigmoid function: 1 / (1 + e^(-x))
#[inline]
fn sigmoid(x: f64) -> f64 {
    if x > 500.0 {
        1.0
    } else if x < -500.0 {
        0.0
    } else {
        1.0 / (1.0 + (-x).exp())
    }
}

/// Leaky ReLU function: max(αx, x)
#[inline]
fn leaky_relu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * x
    }
}

/// ELU function: x if x > 0, α(e^x - 1) if x ≤ 0
#[inline]
fn elu(x: f64, alpha: f64) -> f64 {
    if x > 0.0 {
        x
    } else {
        alpha * (x.exp() - 1.0)
    }
}

/// Swish function: x * sigmoid(x)
#[inline]
fn swish(x: f64) -> f64 {
    x * sigmoid(x)
}

/// GELU function: 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
#[inline]
fn gelu(x: f64) -> f64 {
    0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// GELU derivative
#[inline]
fn gelu_derivative(x: f64) -> f64 {
    let sqrt_2_pi = (2.0 / PI).sqrt();
    let x_cubed = x.powi(3);
    let inner = sqrt_2_pi * (x + 0.044715 * x_cubed);
    let tanh_inner = inner.tanh();
    let sech_inner = 1.0 - tanh_inner.powi(2);

    0.5 * (1.0 + tanh_inner) + 0.5 * x * sech_inner * sqrt_2_pi * (1.0 + 3.0 * 0.044715 * x.powi(2))
}

/// Softplus function: ln(1 + e^x)
#[inline]
fn softplus(x: f64) -> f64 {
    if x > 500.0 {
        x
    } else {
        (1.0 + x.exp()).ln()
    }
}

/// Mish function: x * tanh(softplus(x))
#[inline]
fn mish(x: f64) -> f64 {
    x * softplus(x).tanh()
}

/// Mish derivative
#[inline]
fn mish_derivative(x: f64) -> f64 {
    let sp = softplus(x);
    let tanh_sp = sp.tanh();
    let sech_sp = 1.0 - tanh_sp.powi(2);
    let sigmoid_x = sigmoid(x);

    tanh_sp + x * sech_sp * sigmoid_x
}

/// Hard Sigmoid function: max(0, min(1, (x + 1) / 2))
#[inline]
fn hard_sigmoid(x: f64) -> f64 {
    0.0_f64.max(1.0_f64.min((x + 1.0) / 2.0))
}

/// Hard Tanh function: max(-1, min(1, x))
#[inline]
fn hard_tanh(x: f64) -> f64 {
    (-1.0_f64).max(1.0_f64.min(x))
}

/// Softmax function for vectors
fn softmax(x: &Array1<f64>) -> Result<Array1<f64>> {
    if x.is_empty() {
        return Err(NetworkError::computation(
            "Cannot compute softmax of empty vector",
        ));
    }

    // Subtract max for numerical stability
    let max_val = x.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
    let shifted = x.mapv(|val| val - max_val);

    // Compute exponentials
    let exp_vals = shifted.mapv(|val| {
        if val > 500.0 {
            f64::INFINITY
        } else if val < -500.0 {
            0.0
        } else {
            val.exp()
        }
    });

    // Compute sum
    let sum = exp_vals.sum();

    if sum == 0.0 || !sum.is_finite() {
        return Err(NetworkError::numerical(
            "Softmax computation resulted in invalid sum",
        ));
    }

    // Normalize
    Ok(exp_vals / sum)
}

/// Parametric activation function with learnable parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParametricActivation {
    pub function_type: ActivationFunction,
    pub parameters: Vec<f64>,
}

impl ParametricActivation {
    /// Create a new parametric activation function
    pub fn new(function_type: ActivationFunction, parameters: Vec<f64>) -> Self {
        Self {
            function_type,
            parameters,
        }
    }

    /// Create a PReLU activation with learnable alpha parameter
    pub fn prelu(alpha: f64) -> Self {
        Self::new(ActivationFunction::PReLU, vec![alpha])
    }

    /// Create an ELU activation with learnable alpha parameter
    pub fn elu(alpha: f64) -> Self {
        Self::new(ActivationFunction::ELU, vec![alpha])
    }

    /// Apply the parametric activation function
    pub fn apply(&self, x: f64) -> f64 {
        match self.function_type {
            ActivationFunction::PReLU => {
                let alpha = self.parameters.get(0).copied().unwrap_or(0.01);
                leaky_relu(x, alpha)
            }
            ActivationFunction::ELU => {
                let alpha = self.parameters.get(0).copied().unwrap_or(1.0);
                elu(x, alpha)
            }
            _ => self.function_type.apply_scalar(x),
        }
    }

    /// Compute the derivative with respect to the input
    pub fn derivative(&self, x: f64) -> f64 {
        match self.function_type {
            ActivationFunction::PReLU => {
                let alpha = self.parameters.get(0).copied().unwrap_or(0.01);
                if x > 0.0 {
                    1.0
                } else {
                    alpha
                }
            }
            ActivationFunction::ELU => {
                let alpha = self.parameters.get(0).copied().unwrap_or(1.0);
                if x > 0.0 {
                    1.0
                } else {
                    alpha * x.exp()
                }
            }
            _ => self.function_type.derivative_scalar(x),
        }
    }

    /// Compute the derivative with respect to the parameters
    pub fn parameter_derivative(&self, x: f64) -> Vec<f64> {
        match self.function_type {
            ActivationFunction::PReLU => {
                if x > 0.0 {
                    vec![0.0]
                } else {
                    vec![x]
                }
            }
            ActivationFunction::ELU => {
                if x > 0.0 {
                    vec![0.0]
                } else {
                    vec![x.exp() - 1.0]
                }
            }
            _ => vec![],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_sigmoid() {
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.apply_scalar(0.0),
            0.5,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.apply_scalar(1.0),
            1.0 / (1.0 + E.powf(-1.0)),
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.apply_scalar(-1.0),
            1.0 / (1.0 + E),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_relu() {
        assert_eq!(ActivationFunction::ReLU.apply_scalar(5.0), 5.0);
        assert_eq!(ActivationFunction::ReLU.apply_scalar(-3.0), 0.0);
        assert_eq!(ActivationFunction::ReLU.apply_scalar(0.0), 0.0);
    }

    #[test]
    fn test_tanh() {
        assert_abs_diff_eq!(
            ActivationFunction::Tanh.apply_scalar(0.0),
            0.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ActivationFunction::Tanh.apply_scalar(1.0),
            1.0_f64.tanh(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_softmax() {
        let input = Array1::from(vec![1.0, 2.0, 3.0]);
        let result = ActivationFunction::Softmax.apply_vector(&input).unwrap();

        // Check that probabilities sum to 1
        let sum: f64 = result.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);

        // Check that all values are positive
        assert!(result.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_derivatives() {
        let x = 0.5;

        // Test sigmoid derivative
        let sig = ActivationFunction::Sigmoid.apply_scalar(x);
        let expected_derivative = sig * (1.0 - sig);
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.derivative_scalar(x),
            expected_derivative,
            epsilon = 1e-10
        );

        // Test ReLU derivative
        assert_eq!(ActivationFunction::ReLU.derivative_scalar(0.5), 1.0);
        assert_eq!(ActivationFunction::ReLU.derivative_scalar(-0.5), 0.0);
    }

    #[test]
    fn test_parametric_activation() {
        let prelu = ParametricActivation::prelu(0.1);
        assert_eq!(prelu.apply(2.0), 2.0);
        assert_eq!(prelu.apply(-2.0), -0.2);

        let param_grad = prelu.parameter_derivative(-2.0);
        assert_eq!(param_grad[0], -2.0);
    }

    #[test]
    fn test_gelu() {
        let x = 1.0;
        let result = ActivationFunction::GELU.apply_scalar(x);
        assert!(result > 0.8 && result < 1.0);

        let derivative = ActivationFunction::GELU.derivative_scalar(x);
        assert!(derivative > 0.0);
    }

    #[test]
    fn test_mish() {
        let x = 1.0;
        let result = ActivationFunction::Mish.apply_scalar(x);
        assert!(result > 0.8 && result < 1.0);

        let derivative = ActivationFunction::Mish.derivative_scalar(x);
        assert!(derivative > 0.0);
    }

    #[test]
    fn test_activation_properties() {
        assert!(ActivationFunction::Sigmoid.is_output_suitable());
        assert!(ActivationFunction::Softmax.is_output_suitable());
        assert!(!ActivationFunction::ReLU.is_output_suitable());

        assert_eq!(ActivationFunction::ReLU.name(), "ReLU");
        assert_eq!(ActivationFunction::Sigmoid.name(), "Sigmoid");
    }

    #[test]
    fn test_numerical_stability() {
        // Test sigmoid with extreme values
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.apply_scalar(1000.0),
            1.0,
            epsilon = 1e-10
        );
        assert_abs_diff_eq!(
            ActivationFunction::Sigmoid.apply_scalar(-1000.0),
            0.0,
            epsilon = 1e-10
        );

        // Test softmax with extreme values
        let extreme_input = Array1::from(vec![1000.0, 1001.0, 1002.0]);
        let result = ActivationFunction::Softmax
            .apply_vector(&extreme_input)
            .unwrap();
        let sum: f64 = result.sum();
        assert_abs_diff_eq!(sum, 1.0, epsilon = 1e-10);
    }
}
