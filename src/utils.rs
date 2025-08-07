//! Utility functions and helper operations for the RNN neural network library.
//!
//! This module provides various utility functions for data preprocessing, mathematical operations,
//! random number generation, and other common tasks used throughout the library.

use crate::error::{NetworkError, Result};
use ndarray::{Array1, Array2, Axis};
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal, Uniform};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Data preprocessing utilities.
pub struct DataPreprocessing;

impl DataPreprocessing {
    /// Normalize data using min-max scaling to range [0, 1].
    pub fn min_max_normalize(
        data: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        if data.is_empty() {
            return Err(NetworkError::data("Cannot normalize empty data"));
        }

        let mut normalized = data.clone();
        let mut min_vals = Array1::zeros(data.ncols());
        let mut max_vals = Array1::zeros(data.ncols());

        for (col_idx, mut column) in normalized.columns_mut().into_iter().enumerate() {
            let col_data = data.column(col_idx);
            let min_val = col_data.fold(f64::INFINITY, |acc, &x| acc.min(x));
            let max_val = col_data.fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));

            min_vals[col_idx] = min_val;
            max_vals[col_idx] = max_val;

            let range = max_val - min_val;
            if range != 0.0 {
                column.mapv_inplace(|x| (x - min_val) / range);
            }
        }

        Ok((normalized, min_vals, max_vals))
    }

    /// Apply min-max normalization using existing parameters.
    pub fn apply_min_max_normalize(
        data: &Array2<f64>,
        min_vals: &Array1<f64>,
        max_vals: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        if data.ncols() != min_vals.len() || data.ncols() != max_vals.len() {
            return Err(NetworkError::dimension_mismatch(
                format!("data columns: {}", data.ncols()),
                format!("normalization parameters: {}", min_vals.len()),
            ));
        }

        let mut normalized = data.clone();
        for (col_idx, mut column) in normalized.columns_mut().into_iter().enumerate() {
            let min_val = min_vals[col_idx];
            let max_val = max_vals[col_idx];
            let range = max_val - min_val;
            if range != 0.0 {
                column.mapv_inplace(|x| (x - min_val) / range);
            }
        }

        Ok(normalized)
    }

    /// Standardize data using z-score normalization (mean=0, std=1).
    pub fn standardize(data: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>, Array1<f64>)> {
        if data.is_empty() {
            return Err(NetworkError::data("Cannot standardize empty data"));
        }

        let mut standardized = data.clone();
        let mut means = Array1::zeros(data.ncols());
        let mut stds = Array1::zeros(data.ncols());

        for (col_idx, mut column) in standardized.columns_mut().into_iter().enumerate() {
            let col_data = data.column(col_idx);
            let mean = col_data.mean().unwrap();
            let variance = col_data.mapv(|x| (x - mean).powi(2)).mean().unwrap();
            let std = variance.sqrt();

            means[col_idx] = mean;
            stds[col_idx] = std;

            if std != 0.0 {
                column.mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok((standardized, means, stds))
    }

    /// Apply standardization using existing parameters.
    pub fn apply_standardize(
        data: &Array2<f64>,
        means: &Array1<f64>,
        stds: &Array1<f64>,
    ) -> Result<Array2<f64>> {
        if data.ncols() != means.len() || data.ncols() != stds.len() {
            return Err(NetworkError::dimension_mismatch(
                format!("data columns: {}", data.ncols()),
                format!("standardization parameters: {}", means.len()),
            ));
        }

        let mut standardized = data.clone();
        for (col_idx, mut column) in standardized.columns_mut().into_iter().enumerate() {
            let mean = means[col_idx];
            let std = stds[col_idx];
            if std != 0.0 {
                column.mapv_inplace(|x| (x - mean) / std);
            }
        }

        Ok(standardized)
    }

    /// Convert categorical labels to one-hot encoding.
    pub fn to_categorical(
        labels: &Array1<usize>,
        num_classes: Option<usize>,
    ) -> Result<Array2<f64>> {
        if labels.is_empty() {
            return Err(NetworkError::data(
                "Cannot convert empty labels to categorical",
            ));
        }

        let max_label = labels.fold(0, |acc, &x| acc.max(x));
        let num_classes = num_classes.unwrap_or(max_label + 1);

        if max_label >= num_classes {
            return Err(NetworkError::data(format!(
                "Label {} exceeds number of classes {}",
                max_label, num_classes
            )));
        }

        let mut one_hot = Array2::zeros((labels.len(), num_classes));
        for (i, &label) in labels.iter().enumerate() {
            one_hot[[i, label]] = 1.0;
        }

        Ok(one_hot)
    }

    /// Convert one-hot encoded labels back to categorical.
    pub fn from_categorical(one_hot: &Array2<f64>) -> Array1<usize> {
        let mut labels = Array1::zeros(one_hot.nrows());
        for (i, row) in one_hot.rows().into_iter().enumerate() {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            labels[i] = max_idx;
        }
        labels
    }

    /// Shuffle data and targets together.
    pub fn shuffle(
        data: &Array2<f64>,
        targets: &Array2<f64>,
        seed: Option<u64>,
    ) -> Result<(Array2<f64>, Array2<f64>)> {
        if data.nrows() != targets.nrows() {
            return Err(NetworkError::dimension_mismatch(
                format!("data rows: {}", data.nrows()),
                format!("target rows: {}", targets.nrows()),
            ));
        }

        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut indices: Vec<usize> = (0..data.nrows()).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);

        let shuffled_data = data.select(Axis(0), &indices);
        let shuffled_targets = targets.select(Axis(0), &indices);

        Ok((shuffled_data, shuffled_targets))
    }

    /// Split data into training and testing sets.
    pub fn train_test_split(
        data: &Array2<f64>,
        targets: &Array2<f64>,
        test_size: f64,
        shuffle: bool,
        seed: Option<u64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>)> {
        if test_size <= 0.0 || test_size >= 1.0 {
            return Err(NetworkError::invalid_parameter(
                "test_size",
                &test_size.to_string(),
                "must be between 0 and 1",
            ));
        }

        let (data_to_split, targets_to_split) = if shuffle {
            Self::shuffle(data, targets, seed)?
        } else {
            (data.clone(), targets.clone())
        };

        let n_samples = data_to_split.nrows();
        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = n_samples - n_test;

        let train_data = data_to_split.slice(ndarray::s![..n_train, ..]).to_owned();
        let train_targets = targets_to_split
            .slice(ndarray::s![..n_train, ..])
            .to_owned();
        let test_data = data_to_split.slice(ndarray::s![n_train.., ..]).to_owned();
        let test_targets = targets_to_split
            .slice(ndarray::s![n_train.., ..])
            .to_owned();

        Ok((train_data, test_data, train_targets, test_targets))
    }

    /// Add Gaussian noise to data for augmentation.
    pub fn add_noise(
        data: &Array2<f64>,
        noise_level: f64,
        seed: Option<u64>,
    ) -> Result<Array2<f64>> {
        if noise_level < 0.0 {
            return Err(NetworkError::invalid_parameter(
                "noise_level",
                &noise_level.to_string(),
                "must be non-negative",
            ));
        }

        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let normal = Normal::new(0.0, noise_level).map_err(|e| {
            NetworkError::configuration(format!("Invalid noise distribution: {}", e))
        })?;

        let noise = Array2::from_shape_fn(data.raw_dim(), |_| normal.sample(&mut rng));
        Ok(data + &noise)
    }
}

/// Mathematical utility functions.
pub struct MathUtils;

impl MathUtils {
    /// Compute the numerical gradient using finite differences.
    pub fn numerical_gradient<F>(func: F, x: &Array2<f64>, h: f64) -> Result<Array2<f64>>
    where
        F: Fn(&Array2<f64>) -> Result<f64>,
    {
        let mut gradient = Array2::zeros(x.raw_dim());

        for ((i, j), _) in x.indexed_iter() {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();

            x_plus[[i, j]] += h;
            x_minus[[i, j]] -= h;

            let f_plus = func(&x_plus)?;
            let f_minus = func(&x_minus)?;

            gradient[[i, j]] = (f_plus - f_minus) / (2.0 * h);
        }

        Ok(gradient)
    }

    /// Compute the Hessian matrix using finite differences.
    pub fn numerical_hessian<F>(func: F, x: &Array2<f64>, h: f64) -> Result<Array2<f64>>
    where
        F: Fn(&Array2<f64>) -> Result<f64>,
    {
        let n = x.len();
        let mut hessian = Array2::zeros((n, n));

        // Flatten for easier indexing
        let x_flat = x.clone().into_shape(n)?;

        for i in 0..n {
            for j in 0..n {
                let mut x_pp = x_flat.clone();
                let mut x_pm = x_flat.clone();
                let mut x_mp = x_flat.clone();
                let mut x_mm = x_flat.clone();

                x_pp[i] += h;
                x_pp[j] += h;

                x_pm[i] += h;
                x_pm[j] -= h;

                x_mp[i] -= h;
                x_mp[j] += h;

                x_mm[i] -= h;
                x_mm[j] -= h;

                let x_pp_2d = x_pp.into_shape(x.raw_dim())?;
                let x_pm_2d = x_pm.into_shape(x.raw_dim())?;
                let x_mp_2d = x_mp.into_shape(x.raw_dim())?;
                let x_mm_2d = x_mm.into_shape(x.raw_dim())?;

                let f_pp = func(&x_pp_2d)?;
                let f_pm = func(&x_pm_2d)?;
                let f_mp = func(&x_mp_2d)?;
                let f_mm = func(&x_mm_2d)?;

                hessian[[i, j]] = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
            }
        }

        Ok(hessian)
    }

    /// Clip values to a specified range.
    pub fn clip(x: &Array2<f64>, min_val: f64, max_val: f64) -> Array2<f64> {
        x.mapv(|val| val.max(min_val).min(max_val))
    }

    /// Compute the L2 norm of a matrix.
    pub fn l2_norm(x: &Array2<f64>) -> f64 {
        (x.mapv(|val| val * val).sum()).sqrt()
    }

    /// Compute the Frobenius norm of a matrix.
    pub fn frobenius_norm(x: &Array2<f64>) -> f64 {
        Self::l2_norm(x)
    }

    /// Check if a matrix contains NaN or infinite values.
    pub fn has_nan_or_inf(x: &Array2<f64>) -> bool {
        x.iter().any(|&val| !val.is_finite())
    }

    /// Replace NaN and infinite values with specified value.
    pub fn replace_nan_inf(x: &Array2<f64>, replacement: f64) -> Array2<f64> {
        x.mapv(|val| if val.is_finite() { val } else { replacement })
    }

    /// Compute element-wise absolute value.
    pub fn abs(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| val.abs())
    }

    /// Compute element-wise square.
    pub fn square(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| val * val)
    }

    /// Compute element-wise square root.
    pub fn sqrt(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| val.sqrt())
    }

    /// Compute element-wise exponential.
    pub fn exp(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| val.exp())
    }

    /// Compute element-wise natural logarithm.
    pub fn log(x: &Array2<f64>) -> Array2<f64> {
        x.mapv(|val| val.ln())
    }

    /// Safe division avoiding division by zero.
    pub fn safe_divide(
        numerator: &Array2<f64>,
        denominator: &Array2<f64>,
        epsilon: f64,
    ) -> Array2<f64> {
        numerator / &denominator.mapv(|val| if val.abs() < epsilon { epsilon } else { val })
    }
}

/// Random number generation utilities.
pub struct RandomUtils;

impl RandomUtils {
    /// Generate random matrix with uniform distribution.
    pub fn uniform_random(
        shape: (usize, usize),
        min: f64,
        max: f64,
        seed: Option<u64>,
    ) -> Result<Array2<f64>> {
        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let uniform = Uniform::new(min, max);
        let matrix = Array2::from_shape_fn(shape, |_| uniform.sample(&mut rng));
        Ok(matrix)
    }

    /// Generate random matrix with normal distribution.
    pub fn normal_random(
        shape: (usize, usize),
        mean: f64,
        std: f64,
        seed: Option<u64>,
    ) -> Result<Array2<f64>> {
        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let normal = Normal::new(mean, std).map_err(|e| {
            NetworkError::configuration(format!("Invalid normal distribution: {}", e))
        })?;

        let matrix = Array2::from_shape_fn(shape, |_| normal.sample(&mut rng));
        Ok(matrix)
    }

    /// Generate random indices for sampling.
    pub fn random_indices(n: usize, k: usize, seed: Option<u64>) -> Result<Vec<usize>> {
        if k > n {
            return Err(NetworkError::invalid_parameter(
                "k",
                &k.to_string(),
                "cannot be greater than n",
            ));
        }

        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        indices.truncate(k);
        Ok(indices)
    }

    /// Generate random permutation.
    pub fn permutation(n: usize, seed: Option<u64>) -> Vec<usize> {
        let mut rng = if let Some(seed) = seed {
            rand::rngs::StdRng::seed_from_u64(seed)
        } else {
            rand::rngs::StdRng::from_entropy()
        };

        let mut indices: Vec<usize> = (0..n).collect();
        use rand::seq::SliceRandom;
        indices.shuffle(&mut rng);
        indices
    }
}

/// Performance monitoring and profiling utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitor {
    pub times: HashMap<String, Vec<f64>>,
    pub memory_usage: HashMap<String, Vec<usize>>,
    pub counters: HashMap<String, usize>,
}

impl Default for PerformanceMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor.
    pub fn new() -> Self {
        Self {
            times: HashMap::new(),
            memory_usage: HashMap::new(),
            counters: HashMap::new(),
        }
    }

    /// Start timing an operation.
    pub fn start_timer(&mut self, name: &str) -> std::time::Instant {
        self.counters
            .entry(name.to_string())
            .and_modify(|e| *e += 1)
            .or_insert(1);
        std::time::Instant::now()
    }

    /// End timing an operation and record the duration.
    pub fn end_timer(&mut self, name: &str, start_time: std::time::Instant) {
        let duration = start_time.elapsed().as_secs_f64();
        self.times
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(duration);
    }

    /// Record memory usage.
    pub fn record_memory(&mut self, name: &str, bytes: usize) {
        self.memory_usage
            .entry(name.to_string())
            .or_insert_with(Vec::new)
            .push(bytes);
    }

    /// Get average time for an operation.
    pub fn average_time(&self, name: &str) -> Option<f64> {
        self.times
            .get(name)
            .map(|times| times.iter().sum::<f64>() / times.len() as f64)
    }

    /// Get total time for an operation.
    pub fn total_time(&self, name: &str) -> Option<f64> {
        self.times.get(name).map(|times| times.iter().sum())
    }

    /// Get average memory usage for an operation.
    pub fn average_memory(&self, name: &str) -> Option<f64> {
        self.memory_usage
            .get(name)
            .map(|usage| usage.iter().sum::<usize>() as f64 / usage.len() as f64)
    }

    /// Get peak memory usage for an operation.
    pub fn peak_memory(&self, name: &str) -> Option<usize> {
        self.memory_usage
            .get(name)
            .and_then(|usage| usage.iter().max())
            .copied()
    }

    /// Get operation count.
    pub fn count(&self, name: &str) -> Option<usize> {
        self.counters.get(name).copied()
    }

    /// Print performance summary.
    pub fn print_summary(&self) {
        println!("\n{}", "=".repeat(60));
        println!("Performance Summary");
        println!("{}", "=".repeat(60));

        println!("\nTiming (seconds):");
        println!(
            "{:<20} {:<10} {:<15} {:<15}",
            "Operation", "Count", "Total", "Average"
        );
        println!("{}", "-".repeat(60));
        for (name, times) in &self.times {
            let count = self.count(name).unwrap_or(0);
            let total = times.iter().sum::<f64>();
            let average = total / times.len() as f64;
            println!(
                "{:<20} {:<10} {:<15.6} {:<15.6}",
                name, count, total, average
            );
        }

        println!("\nMemory Usage (bytes):");
        println!("{:<20} {:<15} {:<15}", "Operation", "Average", "Peak");
        println!("{}", "-".repeat(50));
        for (name, usage) in &self.memory_usage {
            let average = usage.iter().sum::<usize>() as f64 / usage.len() as f64;
            let peak = usage.iter().max().unwrap_or(&0);
            println!("{:<20} {:<15.0} {:<15}", name, average, peak);
        }

        println!("{}", "=".repeat(60));
    }

    /// Reset all statistics.
    pub fn reset(&mut self) {
        self.times.clear();
        self.memory_usage.clear();
        self.counters.clear();
    }
}

/// Configuration management utilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfigManager {
    pub settings: HashMap<String, serde_json::Value>,
}

impl Default for ConfigManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ConfigManager {
    /// Create a new config manager.
    pub fn new() -> Self {
        Self {
            settings: HashMap::new(),
        }
    }

    /// Set a configuration value.
    pub fn set<T: Serialize>(&mut self, key: &str, value: T) -> Result<()> {
        let json_value = serde_json::to_value(value).map_err(NetworkError::from)?;
        self.settings.insert(key.to_string(), json_value);
        Ok(())
    }

    /// Get a configuration value.
    pub fn get<T>(&self, key: &str) -> Result<Option<T>>
    where
        T: for<'de> Deserialize<'de>,
    {
        if let Some(value) = self.settings.get(key) {
            let typed_value = serde_json::from_value(value.clone()).map_err(NetworkError::from)?;
            Ok(Some(typed_value))
        } else {
            Ok(None)
        }
    }

    /// Get a configuration value with a default.
    pub fn get_or_default<T>(&self, key: &str, default: T) -> T
    where
        T: for<'de> Deserialize<'de>,
    {
        self.get(key).unwrap_or(None).unwrap_or(default)
    }

    /// Load configuration from a JSON file.
    pub fn load_from_file<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let file = std::fs::File::open(path).map_err(NetworkError::from)?;
        let reader = std::io::BufReader::new(file);
        self.settings = serde_json::from_reader(reader).map_err(NetworkError::from)?;
        Ok(())
    }

    /// Save configuration to a JSON file.
    pub fn save_to_file<P: AsRef<std::path::Path>>(&self, path: P) -> Result<()> {
        let file = std::fs::File::create(path).map_err(NetworkError::from)?;
        let writer = std::io::BufWriter::new(file);
        serde_json::to_writer_pretty(writer, &self.settings).map_err(NetworkError::from)?;
        Ok(())
    }

    /// Check if a key exists.
    pub fn contains_key(&self, key: &str) -> bool {
        self.settings.contains_key(key)
    }

    /// Remove a configuration value.
    pub fn remove(&mut self, key: &str) -> Option<serde_json::Value> {
        self.settings.remove(key)
    }

    /// Get all keys.
    pub fn keys(&self) -> Vec<&String> {
        self.settings.keys().collect()
    }

    /// Clear all settings.
    pub fn clear(&mut self) {
        self.settings.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_min_max_normalize() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (normalized, min_vals, max_vals) = DataPreprocessing::min_max_normalize(&data).unwrap();

        // Check that values are in [0, 1] range
        assert!(normalized.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // Check min and max values
        assert_eq!(min_vals[0], 1.0);
        assert_eq!(max_vals[0], 5.0);
        assert_eq!(min_vals[1], 2.0);
        assert_eq!(max_vals[1], 6.0);

        // Check first column normalization
        assert_abs_diff_eq!(normalized[[0, 0]], 0.0, epsilon = 1e-10); // (1-1)/(5-1) = 0
        assert_abs_diff_eq!(normalized[[1, 0]], 0.5, epsilon = 1e-10); // (3-1)/(5-1) = 0.5
        assert_abs_diff_eq!(normalized[[2, 0]], 1.0, epsilon = 1e-10); // (5-1)/(5-1) = 1
    }

    #[test]
    fn test_standardize() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let (standardized, means, stds) = DataPreprocessing::standardize(&data).unwrap();

        // Check means
        assert_abs_diff_eq!(means[0], 3.0, epsilon = 1e-10); // (1+3+5)/3 = 3
        assert_abs_diff_eq!(means[1], 4.0, epsilon = 1e-10); // (2+4+6)/3 = 4

        // Check that standardized data has approximately zero mean
        let standardized_mean_0 = standardized.column(0).mean().unwrap();
        let standardized_mean_1 = standardized.column(1).mean().unwrap();
        assert_abs_diff_eq!(standardized_mean_0, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(standardized_mean_1, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_to_categorical() {
        let labels = Array1::from(vec![0, 1, 2, 1]);
        let one_hot = DataPreprocessing::to_categorical(&labels, Some(3)).unwrap();

        assert_eq!(one_hot.shape(), &[4, 3]);
        assert_eq!(one_hot[[0, 0]], 1.0);
        assert_eq!(one_hot[[1, 1]], 1.0);
        assert_eq!(one_hot[[2, 2]], 1.0);
        assert_eq!(one_hot[[3, 1]], 1.0);

        // Check that other positions are zero
        assert_eq!(one_hot[[0, 1]], 0.0);
        assert_eq!(one_hot[[0, 2]], 0.0);
    }

    #[test]
    fn test_from_categorical() {
        let one_hot =
            Array2::from_shape_vec((3, 3), vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
                .unwrap();
        let labels = DataPreprocessing::from_categorical(&one_hot);

        assert_eq!(labels[0], 0);
        assert_eq!(labels[1], 1);
        assert_eq!(labels[2], 2);
    }

    #[test]
    fn test_train_test_split() {
        let data = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let targets = Array2::from_shape_vec((10, 1), (0..10).map(|x| x as f64).collect()).unwrap();

        let (train_data, test_data, train_targets, test_targets) =
            DataPreprocessing::train_test_split(&data, &targets, 0.2, false, None).unwrap();

        assert_eq!(train_data.nrows(), 8);
        assert_eq!(test_data.nrows(), 2);
        assert_eq!(train_targets.nrows(), 8);
        assert_eq!(test_targets.nrows(), 2);
    }

    #[test]
    fn test_math_utils_clip() {
        let data = Array2::from_shape_vec((2, 2), vec![-2.0, -1.0, 1.0, 2.0]).unwrap();
        let clipped = MathUtils::clip(&data, -1.5, 1.5);

        assert_eq!(clipped[[0, 0]], -1.5); // -2.0 clipped to -1.5
        assert_eq!(clipped[[0, 1]], -1.0); // -1.0 unchanged
        assert_eq!(clipped[[1, 0]], 1.0); // 1.0 unchanged
        assert_eq!(clipped[[1, 1]], 1.5); // 2.0 clipped to 1.5
    }

    #[test]
    fn test_math_utils_l2_norm() {
        let data = Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).unwrap();
        let norm = MathUtils::l2_norm(&data);
        assert_abs_diff_eq!(norm, 5.0, epsilon = 1e-10); // sqrt(3^2 + 4^2) = 5
    }

    #[test]
    fn test_random_utils_uniform() {
        let matrix = RandomUtils::uniform_random((3, 3), 0.0, 1.0, Some(42)).unwrap();

        // Check shape
        assert_eq!(matrix.shape(), &[3, 3]);

        // Check that all values are in [0, 1] range
        assert!(matrix.iter().all(|&x| x >= 0.0 && x <= 1.0));

        // With fixed seed, results should be reproducible
        let matrix2 = RandomUtils::uniform_random((3, 3), 0.0, 1.0, Some(42)).unwrap();
        assert_eq!(matrix, matrix2);
    }

    #[test]
    fn test_random_utils_normal() {
        let matrix = RandomUtils::normal_random((2, 2), 0.0, 1.0, Some(42)).unwrap();

        // Check shape
        assert_eq!(matrix.shape(), &[2, 2]);

        // With fixed seed, results should be reproducible
        let matrix2 = RandomUtils::normal_random((2, 2), 0.0, 1.0, Some(42)).unwrap();
        assert_eq!(matrix, matrix2);
    }

    #[test]
    fn test_performance_monitor() {
        let mut monitor = PerformanceMonitor::new();

        let start = monitor.start_timer("test_op");
        std::thread::sleep(std::time::Duration::from_millis(10));
        monitor.end_timer("test_op", start);

        assert!(monitor.total_time("test_op").is_some());
        assert!(monitor.average_time("test_op").unwrap() > 0.0);
        assert_eq!(monitor.count("test_op"), Some(1));

        monitor.record_memory("test_op", 1024);
        assert_eq!(monitor.peak_memory("test_op"), Some(1024));
    }

    #[test]
    fn test_config_manager() {
        let mut config = ConfigManager::new();

        config.set("learning_rate", 0.001).unwrap();
        config.set("batch_size", 32).unwrap();
        config.set("verbose", true).unwrap();

        assert_eq!(config.get::<f64>("learning_rate").unwrap(), Some(0.001));
        assert_eq!(config.get::<i32>("batch_size").unwrap(), Some(32));
        assert_eq!(config.get::<bool>("verbose").unwrap(), Some(true));
        assert_eq!(config.get::<f64>("nonexistent").unwrap(), None);

        assert_eq!(config.get_or_default("learning_rate", 0.01), 0.001);
        assert_eq!(config.get_or_default("nonexistent", 0.01), 0.01);

        assert!(config.contains_key("learning_rate"));
        assert!(!config.contains_key("nonexistent"));

        config.remove("learning_rate");
        assert!(!config.contains_key("learning_rate"));
    }

    #[test]
    fn test_add_noise() {
        let data = Array2::zeros((3, 3));
        let noisy_data = DataPreprocessing::add_noise(&data, 0.1, Some(42)).unwrap();

        // Original data was all zeros, so noisy data should have some non-zero values
        assert!(noisy_data.iter().any(|&x| x != 0.0));

        // With fixed seed, results should be reproducible
        let noisy_data2 = DataPreprocessing::add_noise(&data, 0.1, Some(42)).unwrap();
        assert_eq!(noisy_data, noisy_data2);
    }

    #[test]
    fn test_shuffle() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let targets = Array2::from_shape_vec((4, 1), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        let (shuffled_data, shuffled_targets) =
            DataPreprocessing::shuffle(&data, &targets, Some(42)).unwrap();

        // Shape should be preserved
        assert_eq!(shuffled_data.shape(), data.shape());
        assert_eq!(shuffled_targets.shape(), targets.shape());

        // With fixed seed, results should be reproducible
        let (shuffled_data2, shuffled_targets2) =
            DataPreprocessing::shuffle(&data, &targets, Some(42)).unwrap();
        assert_eq!(shuffled_data, shuffled_data2);
        assert_eq!(shuffled_targets, shuffled_targets2);
    }

    #[test]
    fn test_error_cases() {
        // Test empty data normalization
        let empty_data = Array2::zeros((0, 0));
        assert!(DataPreprocessing::min_max_normalize(&empty_data).is_err());
        assert!(DataPreprocessing::standardize(&empty_data).is_err());

        // Test invalid test_size
        let data = Array2::zeros((10, 2));
        let targets = Array2::zeros((10, 1));
        assert!(DataPreprocessing::train_test_split(&data, &targets, 1.5, false, None).is_err());

        // Test negative noise level
        assert!(DataPreprocessing::add_noise(&data, -0.1, None).is_err());

        // Test invalid random indices
        assert!(RandomUtils::random_indices(5, 10, None).is_err());
    }
}
