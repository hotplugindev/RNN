//! Utility functions for the RNN library
//!
//! This module provides common utility functions used throughout the library
//! including numerical utilities, data preprocessing, and helper functions.

use crate::error::{Result, RnnError};
use crate::tensor::Tensor;
use rand::prelude::*;

/// Mathematical utility functions
pub mod math {
    use super::*;

    /// Calculate the softmax function for numerical stability
    pub fn stable_softmax(inputs: &[f32]) -> Result<Vec<f32>> {
        if inputs.is_empty() {
            return Ok(Vec::new());
        }

        // Find maximum for numerical stability
        let max_val = inputs.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Calculate exponentials
        let exp_values: Vec<f32> = inputs.iter().map(|&x| (x - max_val).exp()).collect();

        // Calculate sum
        let sum: f32 = exp_values.iter().sum();

        if sum == 0.0 {
            return Err(RnnError::math("Softmax sum is zero"));
        }

        // Normalize
        Ok(exp_values.iter().map(|&x| x / sum).collect())
    }

    /// Calculate cross entropy loss
    pub fn cross_entropy_loss(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        let mut loss = 0.0;
        let eps = 1e-15; // Small epsilon to prevent log(0)

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            if target > 0.0 {
                let pred_clamped = pred.max(eps).min(1.0 - eps);
                loss -= target * pred_clamped.ln();
            }
        }

        Ok(loss)
    }

    /// Calculate binary cross entropy loss
    pub fn binary_cross_entropy_loss(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        let mut loss = 0.0;
        let eps = 1e-15;

        for (&pred, &target) in predictions.iter().zip(targets.iter()) {
            let pred_clamped = pred.max(eps).min(1.0 - eps);
            loss -= target * pred_clamped.ln() + (1.0 - target) * (1.0 - pred_clamped).ln();
        }

        Ok(loss / predictions.len() as f32)
    }

    /// Calculate accuracy for classification
    pub fn classification_accuracy(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let correct = predictions
            .iter()
            .zip(targets.iter())
            .filter(|(&pred, &target)| {
                let predicted_class = if pred > 0.5 { 1.0 } else { 0.0 };
                (predicted_class - target).abs() < 0.5
            })
            .count();

        Ok(correct as f32 / predictions.len() as f32)
    }

    /// Calculate top-k accuracy for multi-class classification
    pub fn top_k_accuracy(predictions: &[Vec<f32>], targets: &[usize], k: usize) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        if predictions.is_empty() {
            return Ok(0.0);
        }

        let mut correct = 0;

        for (pred, &target) in predictions.iter().zip(targets.iter()) {
            // Get top-k indices
            let mut indexed_pred: Vec<(usize, f32)> =
                pred.iter().enumerate().map(|(i, &v)| (i, v)).collect();
            indexed_pred.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            let top_k_indices: Vec<usize> = indexed_pred.iter().take(k).map(|(i, _)| *i).collect();

            if top_k_indices.contains(&target) {
                correct += 1;
            }
        }

        Ok(correct as f32 / predictions.len() as f32)
    }

    /// Calculate mean squared error
    pub fn mean_squared_error(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        let mse = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).powi(2))
            .sum::<f32>()
            / predictions.len() as f32;

        Ok(mse)
    }

    /// Calculate root mean squared error
    pub fn root_mean_squared_error(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        Ok(mean_squared_error(predictions, targets)?.sqrt())
    }

    /// Calculate mean absolute error
    pub fn mean_absolute_error(predictions: &[f32], targets: &[f32]) -> Result<f32> {
        if predictions.len() != targets.len() {
            return Err(RnnError::shape_mismatch(
                &[predictions.len()],
                &[targets.len()],
            ));
        }

        let mae = predictions
            .iter()
            .zip(targets.iter())
            .map(|(&pred, &target)| (pred - target).abs())
            .sum::<f32>()
            / predictions.len() as f32;

        Ok(mae)
    }

    /// Clip gradients by value
    pub fn clip_gradients_by_value(gradients: &mut [f32], clip_value: f32) {
        for grad in gradients.iter_mut() {
            *grad = grad.max(-clip_value).min(clip_value);
        }
    }

    /// Clip gradients by global norm
    pub fn clip_gradients_by_norm(gradients: &mut [f32], max_norm: f32) -> f32 {
        let total_norm: f32 = gradients.iter().map(|&g| g * g).sum::<f32>().sqrt();

        if total_norm > max_norm {
            let clip_coeff = max_norm / total_norm;
            for grad in gradients.iter_mut() {
                *grad *= clip_coeff;
            }
        }

        total_norm
    }
}

/// Data preprocessing utilities
pub mod data {
    use super::*;

    /// Normalize data to zero mean and unit variance
    pub fn standardize(data: &mut [f32]) -> (f32, f32) {
        let mean = data.iter().sum::<f32>() / data.len() as f32;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev > 1e-8 {
            for x in data.iter_mut() {
                *x = (*x - mean) / std_dev;
            }
        }

        (mean, std_dev)
    }

    /// Normalize data to range [0, 1]
    pub fn min_max_normalize(data: &mut [f32]) -> (f32, f32) {
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_val - min_val;

        if range > 1e-8 {
            for x in data.iter_mut() {
                *x = (*x - min_val) / range;
            }
        }

        (min_val, max_val)
    }

    /// One-hot encode categorical labels
    pub fn one_hot_encode(labels: &[usize], num_classes: usize) -> Vec<Vec<f32>> {
        labels
            .iter()
            .map(|&label| {
                let mut encoded = vec![0.0; num_classes];
                if label < num_classes {
                    encoded[label] = 1.0;
                }
                encoded
            })
            .collect()
    }

    /// Convert one-hot encoded labels back to class indices
    pub fn one_hot_decode(one_hot: &[Vec<f32>]) -> Vec<usize> {
        one_hot
            .iter()
            .map(|encoded| {
                encoded
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(i, _)| i)
                    .unwrap_or(0)
            })
            .collect()
    }

    /// Shuffle data and labels together
    pub fn shuffle_data<T: Clone>(data: &mut Vec<T>, labels: &mut Vec<T>) -> Result<()> {
        if data.len() != labels.len() {
            return Err(RnnError::shape_mismatch(&[data.len()], &[labels.len()]));
        }

        let mut rng = thread_rng();
        let mut indices: Vec<usize> = (0..data.len()).collect();
        indices.shuffle(&mut rng);

        let shuffled_data: Vec<T> = indices.iter().map(|&i| data[i].clone()).collect();
        let shuffled_labels: Vec<T> = indices.iter().map(|&i| labels[i].clone()).collect();

        *data = shuffled_data;
        *labels = shuffled_labels;

        Ok(())
    }

    /// Split data into training and validation sets
    pub fn train_test_split<T: Clone>(
        data: Vec<T>,
        labels: Vec<T>,
        train_ratio: f32,
    ) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>)> {
        if data.len() != labels.len() {
            return Err(RnnError::shape_mismatch(&[data.len()], &[labels.len()]));
        }

        if !(0.0..=1.0).contains(&train_ratio) {
            return Err(RnnError::config("Train ratio must be between 0 and 1"));
        }

        let split_index = (data.len() as f32 * train_ratio) as usize;

        let (train_data, test_data) = data.split_at(split_index);
        let (train_labels, test_labels) = labels.split_at(split_index);

        Ok((
            train_data.to_vec(),
            train_labels.to_vec(),
            test_data.to_vec(),
            test_labels.to_vec(),
        ))
    }

    /// Create batches from data
    pub fn create_batches<T: Clone>(data: Vec<T>, batch_size: usize) -> Vec<Vec<T>> {
        data.chunks(batch_size)
            .map(|chunk| chunk.to_vec())
            .collect()
    }
}

/// Tensor utility functions
pub mod tensor_utils {
    use super::*;

    /// Convert tensor to image format (CHW to HWC)
    pub fn chw_to_hwc(tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(RnnError::tensor("Expected 3D tensor (CHW format)"));
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        let data = tensor.to_vec()?;
        let mut hwc_data = vec![0.0; data.len()];

        for y in 0..h {
            for x in 0..w {
                for channel in 0..c {
                    let chw_idx = channel * h * w + y * w + x;
                    let hwc_idx = y * w * c + x * c + channel;
                    hwc_data[hwc_idx] = data[chw_idx];
                }
            }
        }

        Tensor::from_slice_on_device(&hwc_data, &[h, w, c], tensor.device().clone())
    }

    /// Convert image format (HWC to CHW)
    pub fn hwc_to_chw(tensor: &Tensor) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(RnnError::tensor("Expected 3D tensor (HWC format)"));
        }

        let h = shape[0];
        let w = shape[1];
        let c = shape[2];

        let data = tensor.to_vec()?;
        let mut chw_data = vec![0.0; data.len()];

        for channel in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let hwc_idx = y * w * c + x * c + channel;
                    let chw_idx = channel * h * w + y * w + x;
                    chw_data[chw_idx] = data[hwc_idx];
                }
            }
        }

        Tensor::from_slice_on_device(&chw_data, &[c, h, w], tensor.device().clone())
    }

    /// Pad tensor with specified padding
    pub fn pad_2d(
        tensor: &Tensor,
        padding: (usize, usize, usize, usize), // (top, bottom, left, right)
        value: f32,
    ) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(RnnError::tensor("Expected 2D tensor for padding"));
        }

        let h = shape[0];
        let w = shape[1];
        let (pad_top, pad_bottom, pad_left, pad_right) = padding;

        let new_h = h + pad_top + pad_bottom;
        let new_w = w + pad_left + pad_right;

        let mut padded_data = vec![value; new_h * new_w];
        let data = tensor.to_vec()?;

        for y in 0..h {
            for x in 0..w {
                let old_idx = y * w + x;
                let new_idx = (y + pad_top) * new_w + (x + pad_left);
                padded_data[new_idx] = data[old_idx];
            }
        }

        Tensor::from_slice_on_device(&padded_data, &[new_h, new_w], tensor.device().clone())
    }

    /// Calculate convolution output size
    pub fn conv_output_size(
        input_size: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> usize {
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
    }
}

/// Random number generation utilities
pub mod random {
    use super::*;

    /// Generate random tensor with normal distribution
    pub fn randn_tensor(shape: &[usize], mean: f32, std: f32) -> Result<Tensor> {
        let size = shape.iter().product::<usize>();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..size)
            .map(|_| {
                use rand_distr::Normal;
                let normal = Normal::new(mean, std).unwrap();
                rng.sample(normal)
            })
            .collect();

        Tensor::from_slice(&data, shape)
    }

    /// Generate random tensor with uniform distribution
    pub fn rand_tensor(shape: &[usize], low: f32, high: f32) -> Result<Tensor> {
        let size = shape.iter().product::<usize>();
        let mut rng = thread_rng();
        let data: Vec<f32> = (0..size).map(|_| rng.gen_range(low..high)).collect();

        Tensor::from_slice(&data, shape)
    }

    /// Set random seed for reproducibility
    pub fn set_seed(seed: u64) {
        use rand::SeedableRng;
        let _rng: StdRng = StdRng::seed_from_u64(seed);
        // Note: This is a simplified implementation
        // In practice, you'd want to store the RNG state globally
    }
}

/// File I/O utilities
pub mod io {
    use super::*;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};
    use std::path::Path;

    /// Save data to CSV file
    pub fn save_to_csv<P: AsRef<Path>>(
        path: P,
        data: &[Vec<f32>],
        headers: Option<&[String]>,
    ) -> Result<()> {
        let mut file = File::create(path)?;

        // Write headers if provided
        if let Some(headers) = headers {
            writeln!(file, "{}", headers.join(","))?;
        }

        // Write data
        for row in data {
            let row_str: Vec<String> = row.iter().map(|x| x.to_string()).collect();
            writeln!(file, "{}", row_str.join(","))?;
        }

        Ok(())
    }

    /// Load data from CSV file
    pub fn load_from_csv<P: AsRef<Path>>(
        path: P,
        has_headers: bool,
    ) -> Result<(Vec<Vec<f32>>, Option<Vec<String>>)> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();

        let headers = if has_headers {
            Some(
                lines
                    .next()
                    .ok_or_else(|| {
                        RnnError::io(std::io::Error::new(
                            std::io::ErrorKind::UnexpectedEof,
                            "Empty file",
                        ))
                    })??
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            )
        } else {
            None
        };

        let mut data = Vec::new();
        for line in lines {
            let line = line?;
            let row: Result<Vec<f32>> = line
                .split(',')
                .map(|s| {
                    s.trim().parse().map_err(|e| {
                        RnnError::io(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            format!("Parse error: {}", e),
                        ))
                    })
                })
                .collect();
            data.push(row?);
        }

        Ok((data, headers))
    }
}

/// Performance monitoring utilities
pub mod perf {
    use std::collections::HashMap;
    use std::time::{Duration, Instant};

    /// Simple performance profiler
    #[derive(Debug, Default)]
    pub struct Profiler {
        timers: HashMap<String, Vec<Duration>>,
        active_timers: HashMap<String, Instant>,
    }

    impl Profiler {
        /// Create a new profiler instance
        pub fn new() -> Self {
            Self::default()
        }

        /// Start timing an operation
        pub fn start(&mut self, name: &str) {
            self.active_timers.insert(name.to_string(), Instant::now());
        }

        /// Stop timing an operation
        pub fn stop(&mut self, name: &str) {
            if let Some(start_time) = self.active_timers.remove(name) {
                let duration = start_time.elapsed();
                self.timers
                    .entry(name.to_string())
                    .or_insert_with(Vec::new)
                    .push(duration);
            }
        }

        /// Get average time for an operation
        pub fn average_time(&self, name: &str) -> Option<Duration> {
            self.timers.get(name).map(|times| {
                let total: Duration = times.iter().sum();
                total / times.len() as u32
            })
        }

        /// Get total time for an operation
        pub fn total_time(&self, name: &str) -> Option<Duration> {
            self.timers.get(name).map(|times| times.iter().sum())
        }

        /// Print summary of all timings
        pub fn print_summary(&self) {
            println!("Performance Summary:");
            println!("==================");
            for (name, times) in &self.timers {
                let avg = self.average_time(name).unwrap();
                let total = self.total_time(name).unwrap();
                println!(
                    "{}: {} calls, avg: {:.2}ms, total: {:.2}ms",
                    name,
                    times.len(),
                    avg.as_secs_f64() * 1000.0,
                    total.as_secs_f64() * 1000.0
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_stable_softmax() {
        let inputs = vec![1.0, 2.0, 3.0];
        let result = math::stable_softmax(&inputs).unwrap();

        // Check that probabilities sum to 1
        let sum: f32 = result.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that all probabilities are positive
        assert!(result.iter().all(|&x| x > 0.0));

        // Check that larger inputs produce larger probabilities
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    #[test]
    fn test_classification_accuracy() {
        let predictions = vec![0.8, 0.2, 0.9, 0.1];
        let targets = vec![1.0, 0.0, 1.0, 0.0];

        let accuracy = math::classification_accuracy(&predictions, &targets).unwrap();
        assert_relative_eq!(accuracy, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_mean_squared_error() {
        let predictions = vec![1.0, 2.0, 3.0];
        let targets = vec![1.5, 2.5, 2.5];

        let mse = math::mean_squared_error(&predictions, &targets).unwrap();
        let expected = ((0.5_f32).powi(2) + (0.5_f32).powi(2) + (0.5_f32).powi(2)) / 3.0;
        assert_relative_eq!(mse, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_standardize() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std_dev) = data::standardize(&mut data);

        assert_relative_eq!(mean, 3.0, epsilon = 1e-6);
        assert_relative_eq!(std_dev, (2.0_f32).sqrt(), epsilon = 1e-6);

        // Check that standardized data has zero mean
        let new_mean = data.iter().sum::<f32>() / data.len() as f32;
        assert_relative_eq!(new_mean, 0.0, epsilon = 1e-6);
    }

    #[test]
    fn test_one_hot_encode() {
        let labels = vec![0, 1, 2, 1];
        let encoded = data::one_hot_encode(&labels, 3);

        assert_eq!(encoded[0], vec![1.0, 0.0, 0.0]);
        assert_eq!(encoded[1], vec![0.0, 1.0, 0.0]);
        assert_eq!(encoded[2], vec![0.0, 0.0, 1.0]);
        assert_eq!(encoded[3], vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_one_hot_decode() {
        let encoded = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
        ];
        let decoded = data::one_hot_decode(&encoded);

        assert_eq!(decoded, vec![0, 1, 2, 1]);
    }

    #[test]
    fn test_train_test_split() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let labels = vec![1, 1, 1, 1, 1, 0, 0, 0, 0, 0];

        let (train_data, train_labels, test_data, test_labels) =
            data::train_test_split(data, labels, 0.7).unwrap();

        assert_eq!(train_data.len(), 7);
        assert_eq!(train_labels.len(), 7);
        assert_eq!(test_data.len(), 3);
        assert_eq!(test_labels.len(), 3);
    }

    #[test]
    fn test_conv_output_size() {
        // Standard convolution: input=32, kernel=3, stride=1, padding=1
        let output_size = tensor_utils::conv_output_size(32, 3, 1, 1, 1);
        assert_eq!(output_size, 32);

        // Stride 2 convolution: input=32, kernel=3, stride=2, padding=1
        let output_size = tensor_utils::conv_output_size(32, 3, 2, 1, 1);
        assert_eq!(output_size, 16);
    }

    #[test]
    fn test_gradient_clipping() {
        let mut gradients = vec![3.0, -5.0, 2.0, -8.0];
        math::clip_gradients_by_value(&mut gradients, 4.0);

        assert_eq!(gradients, vec![3.0, -4.0, 2.0, -4.0]);
    }

    #[test]
    fn test_gradient_norm_clipping() {
        let mut gradients = vec![3.0, 4.0]; // norm = 5.0
        let norm = math::clip_gradients_by_norm(&mut gradients, 2.0);

        assert_relative_eq!(norm, 5.0, epsilon = 1e-6);
        assert_relative_eq!(gradients[0], 1.2, epsilon = 1e-6); // 3.0 * 2.0/5.0
        assert_relative_eq!(gradients[1], 1.6, epsilon = 1e-6); // 4.0 * 2.0/5.0
    }
}
