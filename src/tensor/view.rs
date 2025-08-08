//! Tensor view module for memory-efficient operations
//!
//! This module provides views into tensor data that allow for operations
//! without copying the underlying data, enabling efficient slicing,
//! indexing, and sub-tensor operations.

use crate::error::{Result, RnnError};
use crate::tensor::Tensor;
use std::ops::{Index, Range};

/// A view into a tensor's data
#[derive(Debug, Clone)]
pub struct TensorView<'a> {
    tensor: &'a Tensor,
    data: Vec<f32>,
    offset: usize,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl<'a> TensorView<'a> {
    /// Create a new tensor view
    pub fn new(tensor: &'a Tensor) -> Self {
        let shape = tensor.shape().to_vec();
        let strides = compute_strides(&shape);
        let data = tensor.to_vec().unwrap_or_default();

        Self {
            tensor,
            data,
            offset: 0,
            shape,
            strides,
        }
    }

    /// Create a view with custom offset, shape, and strides
    pub fn new_with_params(
        tensor: &'a Tensor,
        offset: usize,
        shape: Vec<usize>,
        strides: Vec<usize>,
    ) -> Result<Self> {
        if shape.len() != strides.len() {
            return Err(RnnError::tensor(
                "Shape and strides must have the same length",
            ));
        }

        // Validate that the view doesn't exceed tensor bounds
        let max_index = if shape.is_empty() {
            0
        } else {
            offset
                + shape
                    .iter()
                    .zip(strides.iter())
                    .map(|(&dim, &stride)| (dim.saturating_sub(1)) * stride)
                    .max()
                    .unwrap_or(0)
        };

        if max_index >= tensor.size() {
            return Err(RnnError::tensor("View exceeds tensor bounds"));
        }

        let data = tensor.to_vec().unwrap_or_default();

        Ok(Self {
            tensor,
            data,
            offset,
            shape,
            strides,
        })
    }

    /// Get the shape of this view
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Get the strides of this view
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    /// Get the offset of this view
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Get the total number of elements in this view
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    /// Get the underlying tensor
    pub fn tensor(&self) -> &Tensor {
        self.tensor
    }

    /// Check if this view is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_empty() {
            return true;
        }

        let expected_strides = compute_strides(&self.shape);
        self.strides == expected_strides
    }

    /// Create a slice of this view along a specific dimension
    pub fn slice(&self, dim: usize, range: Range<usize>) -> Result<TensorView<'a>> {
        if dim >= self.ndim() {
            return Err(RnnError::tensor("Dimension index out of bounds"));
        }

        if range.end > self.shape[dim] {
            return Err(RnnError::tensor("Slice range exceeds dimension size"));
        }

        let mut new_shape = self.shape.clone();
        new_shape[dim] = range.end - range.start;

        let new_offset = self.offset + range.start * self.strides[dim];

        Self::new_with_params(self.tensor, new_offset, new_shape, self.strides.clone())
    }

    /// Create a view of a specific index along a dimension (reduces dimensionality by 1)
    pub fn select(&self, dim: usize, index: usize) -> Result<TensorView<'a>> {
        if dim >= self.ndim() {
            return Err(RnnError::tensor("Dimension index out of bounds"));
        }

        if index >= self.shape[dim] {
            return Err(RnnError::tensor("Index exceeds dimension size"));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.remove(dim);
        new_strides.remove(dim);

        let new_offset = self.offset + index * self.strides[dim];

        Self::new_with_params(self.tensor, new_offset, new_shape, new_strides)
    }

    /// Transpose the view (swap last two dimensions)
    pub fn transpose(&self) -> Result<TensorView<'a>> {
        if self.ndim() < 2 {
            return Err(RnnError::tensor(
                "Cannot transpose view with less than 2 dimensions",
            ));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        let last_idx = new_shape.len() - 1;
        new_shape.swap(last_idx - 1, last_idx);
        new_strides.swap(last_idx - 1, last_idx);

        Self::new_with_params(self.tensor, self.offset, new_shape, new_strides)
    }

    /// Reshape the view (only works if contiguous)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<TensorView<'a>> {
        let new_size = new_shape.iter().product::<usize>();
        if new_size != self.size() {
            return Err(RnnError::shape_mismatch(&[self.size()], &[new_size]));
        }

        if !self.is_contiguous() {
            return Err(RnnError::tensor("Cannot reshape non-contiguous view"));
        }

        let new_strides = compute_strides(new_shape);
        Self::new_with_params(self.tensor, self.offset, new_shape.to_vec(), new_strides)
    }

    /// Create a view with squeezed dimensions (remove dimensions of size 1)
    pub fn squeeze(&self) -> TensorView<'a> {
        let new_shape: Vec<usize> = self
            .shape
            .iter()
            .filter(|&&dim| dim != 1)
            .copied()
            .collect();
        let new_strides: Vec<usize> = self
            .shape
            .iter()
            .zip(self.strides.iter())
            .filter(|&(&dim, _)| dim != 1)
            .map(|(_, &stride)| stride)
            .collect();

        Self {
            tensor: self.tensor,
            data: self.data.clone(),
            offset: self.offset,
            shape: new_shape,
            strides: new_strides,
        }
    }

    /// Create a view with an added dimension of size 1
    pub fn unsqueeze(&self, dim: usize) -> Result<TensorView<'a>> {
        if dim > self.ndim() {
            return Err(RnnError::tensor("Dimension index out of bounds"));
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.insert(dim, 1);
        new_strides.insert(
            dim,
            if dim == 0 || self.strides.is_empty() {
                self.strides.first().copied().unwrap_or(1)
            } else {
                self.strides[dim - 1]
            },
        );

        Self::new_with_params(self.tensor, self.offset, new_shape, new_strides)
    }

    /// Convert linear index to multi-dimensional indices
    pub fn unravel_index(&self, index: usize) -> Result<Vec<usize>> {
        if index >= self.size() {
            return Err(RnnError::tensor("Index out of bounds"));
        }

        let mut indices = vec![0; self.ndim()];
        let mut remaining = index;

        for (i, &dim_size) in self.shape.iter().enumerate() {
            indices[i] = remaining % dim_size;
            remaining /= dim_size;
        }

        Ok(indices)
    }

    /// Convert multi-dimensional indices to linear index in the view
    pub fn ravel_index(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(RnnError::tensor(
                "Number of indices must match tensor dimensions",
            ));
        }

        for (i, (&idx, &dim_size)) in indices.iter().zip(self.shape.iter()).enumerate() {
            if idx >= dim_size {
                return Err(RnnError::tensor(format!(
                    "Index {} out of bounds for dimension {} (size {})",
                    idx, i, dim_size
                )));
            }
        }

        let linear_index = indices
            .iter()
            .zip(self.strides.iter())
            .map(|(&idx, &stride)| idx * stride)
            .sum::<usize>();

        Ok(self.offset + linear_index)
    }

    /// Get a scalar value at the given indices
    pub fn get(&self, indices: &[usize]) -> Result<f32> {
        let tensor_index = self.ravel_index(indices)?;
        let tensor_data = self.tensor.to_vec()?;

        if tensor_index >= tensor_data.len() {
            return Err(RnnError::tensor("Computed index exceeds tensor size"));
        }

        Ok(tensor_data[tensor_index])
    }

    /// Convert view to a new tensor (copies data)
    pub fn to_tensor(&self) -> Result<Tensor> {
        if self.is_contiguous() {
            // For contiguous views, we can copy a slice directly
            let tensor_data = self.tensor.to_vec()?;
            let start = self.offset;
            let end = start + self.size();

            if end > tensor_data.len() {
                return Err(RnnError::tensor("View exceeds tensor bounds"));
            }

            Tensor::from_slice_on_device(
                &tensor_data[start..end],
                &self.shape,
                self.tensor.device().clone(),
            )
        } else {
            // For non-contiguous views, we need to copy element by element
            let mut data = Vec::with_capacity(self.size());
            let tensor_data = self.tensor.to_vec()?;

            for i in 0..self.size() {
                let indices = self.unravel_index(i)?;
                let tensor_index = self.ravel_index(&indices)?;
                data.push(tensor_data[tensor_index]);
            }

            Tensor::from_slice_on_device(&data, &self.shape, self.tensor.device().clone())
        }
    }

    /// Iterate over all elements in the view
    pub fn iter(&'a self) -> TensorViewIterator<'a> {
        TensorViewIterator {
            view: self,
            current_index: 0,
        }
    }

    /// Create a view that iterates along a specific axis
    pub fn axis_iter(&'a self, axis: usize) -> Result<AxisIterator<'a>> {
        if axis >= self.ndim() {
            return Err(RnnError::tensor("Axis index out of bounds"));
        }

        Ok(AxisIterator {
            view: self,
            axis,
            current_index: 0,
        })
    }
}

/// Iterator over tensor view elements
pub struct TensorViewIterator<'a> {
    view: &'a TensorView<'a>,
    current_index: usize,
}

impl<'a> Iterator for TensorViewIterator<'a> {
    type Item = Result<f32>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.view.size() {
            return None;
        }

        let result = self
            .view
            .unravel_index(self.current_index)
            .and_then(|indices| self.view.get(&indices));

        self.current_index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.size() - self.current_index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for TensorViewIterator<'a> {}

/// Iterator over views along a specific axis
pub struct AxisIterator<'a> {
    view: &'a TensorView<'a>,
    axis: usize,
    current_index: usize,
}

impl<'a> Iterator for AxisIterator<'a> {
    type Item = Result<TensorView<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_index >= self.view.shape[self.axis] {
            return None;
        }

        let result = self.view.select(self.axis, self.current_index);
        self.current_index += 1;
        Some(result)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.shape[self.axis] - self.current_index;
        (remaining, Some(remaining))
    }
}

impl<'a> ExactSizeIterator for AxisIterator<'a> {}

/// Compute strides for a given shape (row-major order)
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    if shape.is_empty() {
        return Vec::new();
    }

    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// Indexing implementations for convenient access
impl<'a> Index<usize> for TensorView<'a> {
    type Output = f32;

    fn index(&self, index: usize) -> &Self::Output {
        // Return reference to the data stored in the view
        &self.data[self.offset + index]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_view_creation() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        assert_eq!(view.shape(), &[2, 3]);
        assert_eq!(view.strides(), &[3, 1]);
        assert_eq!(view.size(), 6);
        assert!(view.is_contiguous());
    }

    #[test]
    fn test_view_slicing() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        // Slice first row
        let row_view = view.slice(0, 0..1).unwrap();
        assert_eq!(row_view.shape(), &[1, 3]);

        // Slice first two columns
        let col_view = view.slice(1, 0..2).unwrap();
        assert_eq!(col_view.shape(), &[2, 2]);
    }

    #[test]
    fn test_view_selection() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        // Select first row
        let row_view = view.select(0, 0).unwrap();
        assert_eq!(row_view.shape(), &[3]);

        // Select first column
        let col_view = view.select(1, 0).unwrap();
        assert_eq!(col_view.shape(), &[2]);
    }

    #[test]
    fn test_view_transpose() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = TensorView::new(&tensor);

        let transposed = view.transpose().unwrap();
        assert_eq!(transposed.shape(), &[2, 2]);
        assert_eq!(transposed.strides(), &[1, 2]);
    }

    #[test]
    fn test_view_reshape() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        let reshaped = view.reshape(&[3, 2]).unwrap();
        assert_eq!(reshaped.shape(), &[3, 2]);
        assert_eq!(reshaped.strides(), &[2, 1]);
    }

    #[test]
    fn test_view_squeeze_unsqueeze() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]).unwrap();
        let view = TensorView::new(&tensor);

        let squeezed = view.squeeze();
        assert_eq!(squeezed.shape(), &[2, 2]);

        let unsqueezed = squeezed.unsqueeze(1).unwrap();
        assert_eq!(unsqueezed.shape(), &[2, 1, 2]);
    }

    #[test]
    fn test_view_indexing() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = TensorView::new(&tensor);

        assert_eq!(view.get(&[0, 0]).unwrap(), 1.0);
        assert_eq!(view.get(&[0, 1]).unwrap(), 2.0);
        assert_eq!(view.get(&[1, 0]).unwrap(), 3.0);
        assert_eq!(view.get(&[1, 1]).unwrap(), 4.0);
    }

    #[test]
    fn test_view_to_tensor() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        let row_view = view.select(0, 0).unwrap();
        let row_tensor = row_view.to_tensor().unwrap();

        assert_eq!(row_tensor.shape(), &[3]);
        assert_eq!(row_tensor.to_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_view_iterator() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let view = TensorView::new(&tensor);

        let values: Result<Vec<f32>> = view.iter().collect();
        assert_eq!(values.unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_compute_strides() {
        assert_eq!(compute_strides(&[2, 3, 4]), vec![12usize, 4usize, 1usize]);
        assert_eq!(compute_strides(&[5]), vec![1usize]);
        assert_eq!(compute_strides(&[]), Vec::<usize>::new());
    }

    #[test]
    fn test_axis_iterator() {
        let tensor = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let view = TensorView::new(&tensor);

        let mut axis_iter = view.axis_iter(0).unwrap();

        let first_row = axis_iter.next().unwrap().unwrap();
        assert_eq!(first_row.shape(), &[3]);

        let second_row = axis_iter.next().unwrap().unwrap();
        assert_eq!(second_row.shape(), &[3]);

        assert!(axis_iter.next().is_none());
    }
}
