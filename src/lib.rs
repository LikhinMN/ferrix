use rayon::prelude::*;
use std::ops::{Add, Mul};
pub struct NDArray<T> {
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    offset: usize,
}

pub struct NDArrayView<'a, T> {
    pub data: &'a [T],
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub offset: usize,
}

impl<T> NDArray<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>) -> Self {
        let expected: usize = shape.iter().product();
        if data.len() != expected {
            panic!(
                "data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                expected
            );
        }

        let mut strides = vec![0; shape.len()];
        let mut current_stride = 1;

        for i in (0..shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= shape[i];
        }

        NDArray {
            data,
            shape,
            strides,
            offset:0,
        }
    }
}
impl<T> NDArray<T> {
    pub fn fancy_index(&self, indices: &[usize]) -> NDArray<T>
    where
        T: Clone,
    {
        if self.shape.len() != 1 {
            panic!(
                "fancy_index is only valid for 1D arrays, got {}D",
                self.shape.len()
            );
        }

        let mut result_data = Vec::with_capacity(indices.len());
        for &idx in indices {
            result_data.push(self.get(&[idx]).clone());
        }

        NDArray::new(result_data, vec![indices.len()])
    }

    pub fn get(&self, index: &[usize]) -> &T {
        if index.len() != self.shape.len() {
            panic!("length of the given index and shape must match.")
        }
        let mut flatten = self.offset;
        for i in 0..index.len() {
            if index[i] >= self.shape[i] {
                panic!(
                    "index out of bounds for dimension {}: index {}, shape {}",
                    i, index[i], self.shape[i]
                );
            }
            flatten += index[i] * self.strides[i];
        }
        &self.data[flatten]
    }

    pub fn get_mut(&mut self, index: &[usize]) -> &mut T {
        if index.len() != self.shape.len() {
            panic!("length of the given index and shape must match.")
        }
        let mut flatten = self.offset;
        for i in 0..index.len() {
            if index[i] >= self.shape[i] {
                panic!(
                    "index out of bounds for dimension {}: index {}, shape {}",
                    i, index[i], self.shape[i]
                );
            }
            flatten += index[i] * self.strides[i];
        }
        &mut self.data[flatten]
    }

    pub fn slice_row(&self, row: usize) -> NDArrayView<'_, T> {
        if self.shape.len() != 2 {
            panic!("slice_row requires a 2D array, got {}D", self.shape.len());
        }
        if row >= self.shape[0] {
            panic!("index out of bounds for dimension 0: index {}, shape {}", row, self.shape[0]);
        }
        NDArrayView {
            data: &self.data,
            shape: vec![self.shape[1]],
            strides: vec![self.strides[1]],
            offset: self.offset + row * self.strides[0],
        }
    }

    pub fn slice_col(&self,col:usize)->NDArrayView<'_,T>{
        if self.shape.len() != 2 {
            panic!("slice_col requires a 2D array, got {}D", self.shape.len());
        }
        if col >= self.shape[1] {
            panic!("index out of bounds for dimension 1: index {}, shape {}", col, self.shape[1]);
        }
        NDArrayView {
            data: &self.data,
            shape: vec![self.shape[0]],
            strides: vec![self.strides[0]],
            offset: self.offset + col * self.strides[1]
        }
    }

    pub fn gather(&self, axis: usize, indices: &[usize]) -> NDArray<T>
    where
        T: Clone,
    {
        if axis >= self.shape.len() {
            panic!(
                "axis {} is out of bounds for NDArray with {} dimensions",
                axis,
                self.shape.len()
            );
        }
        for &idx in indices {
            if idx >= self.shape[axis] {
                panic!(
                    "index {} out of bounds for axis {} with shape {}",
                    idx, axis, self.shape[axis]
                );
            }
        }

        let mut out_shape = self.shape.clone();
        out_shape[axis] = indices.len();
        let out_size: usize = out_shape.iter().product();

        let mut result_data = Vec::with_capacity(out_size);
        let mut current_index = vec![0; out_shape.len()];

        for _ in 0..out_size {
            let mut in_index = current_index.clone();
            in_index[axis] = indices[current_index[axis]];
            result_data.push(self.get(&in_index).clone());

            for i in (0..out_shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < out_shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }

        NDArray::new(result_data, out_shape)
    }

    pub fn slice_range(&self, axis: usize, start: usize, end: usize) -> NDArrayView<'_, T> {
        if axis >= self.shape.len() {
            panic!(
                "axis {} is out of bounds for NDArray with {} dimensions",
                axis,
                self.shape.len()
            );
        }
        if start >= end {
            panic!(
                "invalid range: start {} must be less than end {}",
                start, end
            );
        }
        if end > self.shape[axis] {
            panic!(
                "range end {} is out of bounds for axis {} with shape {}",
                end, axis, self.shape[axis]
            );
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        NDArrayView {
            data: &self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[axis],
        }
    }

    pub fn reshape(&self,new_shape:Vec<usize>)->NDArrayView<'_,T>{
        let expected: usize = new_shape.iter().product();
        if self.data.len() != expected {
            panic!(
                "data length {} does not match shape {:?} (expected {})",
                self.data.len(),
                new_shape,
                expected
            );
        }

        let mut strides = vec![0; new_shape.len()];
        let mut current_stride = 1;

        for i in (0..new_shape.len()).rev() {
            strides[i] = current_stride;
            current_stride *= new_shape[i];
        }
        NDArrayView {
            data: &self.data,
            shape: new_shape,
            strides,
            offset: 0,
        }
    }

    pub fn transpose(&self) -> NDArrayView<'_, T>{
        if self.shape.len() != 2 {
            panic!("transpose requires a 2D array, got {}D", self.shape.len());
        }
        let new_shape=[self.shape[1],self.shape[0]].to_vec();
        let new_stride=[self.strides[1],self.strides[0]].to_vec();
        NDArrayView {
            data: &self.data,
            shape: new_shape,
            strides: new_stride,
            offset: 0
        }
    }

    pub fn add(&self, other: &NDArray<T>) -> NDArray<T>
    where
        T: Add<Output = T> + Copy + Send + Sync,
    {
        if self.shape != other.shape {
            panic!(
                "shape mismatch for addition: {:?} vs {:?}",
                self.shape, other.shape
            );
        }
        let data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(&a, &b)| a + b)
            .collect();
        NDArray::new(data, self.shape.clone())
    }
    pub fn mul(&self, other: &NDArray<T>) -> NDArray<T>
    where
        T: Mul<Output = T> + Copy + Send + Sync,
    {
        if self.shape != other.shape {
            panic!(
                "shape mismatch for multiplication: {:?} vs {:?}",
                self.shape, other.shape
            );
        }
        let data = self
            .data
            .par_iter()
            .zip(other.data.par_iter())
            .map(|(&a, &b)| a * b)
            .collect();
        NDArray::new(data, self.shape.clone())
    }

    pub fn scale(&self, scalar: T) -> NDArray<T>
    where
        T: Mul<Output = T> + Copy + Send + Sync,
    {
        let data = self.data.par_iter().map(|&a| a * scalar).collect();
        NDArray::new(data, self.shape.clone())
    }

    pub fn matmul(&self, other: &NDArray<T>) -> NDArray<T>
    where
        T: Add<Output = T> + Mul<Output = T> + Copy + Default,
    {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            panic!(
                "matmul requires 2D arrays, got {}D and {}D",
                self.shape.len(),
                other.shape.len()
            );
        }
        if self.shape[1] != other.shape[0] {
            panic!(
                "shape mismatch for matmul: self.shape[1] ({}) must equal other.shape[0] ({})",
                self.shape[1], other.shape[0]
            );
        }

        let m = self.shape[0];
        let k_limit = self.shape[1];
        let n = other.shape[1];
        let mut result_data = vec![T::default(); m * n];

        for i in 0..m {
            for j in 0..n {
                let mut sum = T::default();
                for k in 0..k_limit {
                    sum = sum + *self.get(&[i, k]) * *other.get(&[k, j]);
                }
                result_data[i * n + j] = sum;
            }
        }

        NDArray::new(result_data, vec![m, n])
    }
}

impl NDArray<f64> {
    pub fn set_slice(&mut self, axis: usize, start: usize, end: usize, value: f64) {
        if axis >= self.shape.len() {
            panic!(
                "axis {} is out of bounds for NDArray with {} dimensions",
                axis,
                self.shape.len()
            );
        }
        if start >= end {
            panic!(
                "invalid range: start {} must be less than end {}",
                start, end
            );
        }
        if end > self.shape[axis] {
            panic!(
                "range end {} is out of bounds for axis {} with shape {}",
                end, axis, self.shape[axis]
            );
        }

        let size: usize = (end - start) * self.shape.iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &s)| s)
            .product::<usize>();
        let mut current_index = vec![0; self.shape.len()];

        for i in 0..self.shape.len() {
            current_index[i] = if i == axis { start } else { 0 };
        }

        for _ in 0..size {
            *self.get_mut(&current_index) = value;

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }
    }

    pub fn cumsum(&self) -> NDArray<f64> {
        let size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(size);
        let mut current_index = vec![0; self.shape.len()];
        let mut sum = 0.0;

        for _ in 0..size {
            sum += *self.get(&current_index);
            result_data.push(sum);

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }

        NDArray::new(result_data, vec![size])
    }

    pub fn relu(&self) -> NDArray<f64> {
        let data = self.data.par_iter().map(|&x| x.max(0.0)).collect();
        NDArray::new(data, self.shape.clone())
    }

    pub fn sum(&self) -> f64 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.sum() / self.data.len() as f64
    }

    pub fn argmax(&self) -> usize {
        if self.data.is_empty() {
            panic!("argmax on empty array");
        }
        let mut max_idx = 0;
        let mut max_val = self.data[0];
        for (i, &val) in self.data.iter().enumerate().skip(1) {
            if val > max_val {
                max_val = val;
                max_idx = i;
            }
        }
        max_idx
    }

    pub fn sigmoid(&self) -> NDArray<f64> {
        let data = self.data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        NDArray::new(data, self.shape.clone())
    }

    pub fn softmax(&self) -> NDArray<f64> {
        let exp_data: Vec<f64> = self.data.iter().map(|&x| x.exp()).collect();
        let sum_exp: f64 = exp_data.iter().sum();
        let data = exp_data.into_iter().map(|x| x / sum_exp).collect();
        NDArray::new(data, self.shape.clone())
    }


    pub fn boolean_mask(&self, mask: &NDArray<bool>) -> NDArray<f64> {
        if self.shape != mask.shape {
            panic!(
                "mask shape {:?} must equal self shape {:?}",
                mask.shape, self.shape
            );
        }

        let mut result_data = Vec::new();
        // Use the same row-major iteration as to_owned() or similar logic
        // But since both self and mask are NDArray (not views in this signature), 
        // and NDArray stores data in row-major order flattened in .data, 
        // we can simply iterate through both datas if they are not offset/strided.
        // Wait, NDArray has strides and offset too! (lines 8-9)
        
        // However, NDArray constructor always creates a contiguous row-major buffer.
        // If we want to support any NDArray (including potentially strided ones if we allow them),
        // we should use a general iterator or index-based access.
        
        // Looking at current NDArray::new, it always builds contiguous.
        // But the struct has offset/strides, so they might be used by methods returning NDArray.
        // Currently slice_range/transpose etc return NDArrayView.
        
        let size: usize = self.shape.iter().product();
        let mut current_index = vec![0; self.shape.len()];

        for _ in 0..size {
            if *mask.get(&current_index) {
                result_data.push(*self.get(&current_index));
            }

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }

        let count = result_data.len();
        NDArray::new(result_data, vec![count])
    }

    pub fn masked_fill(&mut self, mask: &NDArray<bool>, value: f64) {
        if self.shape != mask.shape {
            panic!(
                "mask shape {:?} must equal self shape {:?}",
                mask.shape, self.shape
            );
        }

        let size: usize = self.shape.iter().product();
        let mut current_index = vec![0; self.shape.len()];

        for _ in 0..size {
            if *mask.get(&current_index) {
                *self.get_mut(&current_index) = value;
            }

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }
    }

    pub fn where_(&self, condition: &NDArray<bool>, other: &NDArray<f64>) -> NDArray<f64> {
        if self.shape != condition.shape || self.shape != other.shape {
            panic!(
                "shape mismatch: self.shape {:?}, condition.shape {:?}, other.shape {:?}",
                self.shape, condition.shape, other.shape
            );
        }

        let size: usize = self.shape.iter().product();
        let mut result_data = Vec::with_capacity(size);
        let mut current_index = vec![0; self.shape.len()];

        for _ in 0..size {
            if *condition.get(&current_index) {
                result_data.push(*self.get(&current_index));
            } else {
                result_data.push(*other.get(&current_index));
            }

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }

        NDArray::new(result_data, self.shape.clone())
    }
}

impl<'a,T> NDArrayView<'a,T>{
    pub fn get(&self, index: &[usize]) -> &T {
        if index.len() != self.shape.len() {
            panic!("length of the given index and shape must match.")
        }
        let mut flatten = self.offset;
        for i in 0..index.len() {
            if index[i] >= self.shape[i] {
                panic!(
                    "index out of bounds for dimension {}: index {}, shape {}",
                    i, index[i], self.shape[i]
                );
            }
            flatten += index[i] * self.strides[i];
        }
        &self.data[flatten]
    }

    pub fn slice_row(&self, row: usize) -> NDArrayView<'_, T> {
        if self.shape.len() != 2 {
            panic!("slice_row requires a 2D array, got {}D", self.shape.len());
        }
        if row >= self.shape[0] {
            panic!("index out of bounds for dimension 0: index {}, shape {}", row, self.shape[0]);
        }
        NDArrayView {
            data: self.data,
            shape: vec![self.shape[1]],
            strides: vec![self.strides[1]],
            offset: self.offset + row * self.strides[0],
        }
    }

    pub fn slice_col(&self,col:usize)->NDArrayView<'_,T>{
        if self.shape.len() != 2 {
            panic!("slice_col requires a 2D array, got {}D", self.shape.len());
        }
        if col >= self.shape[1] {
            panic!("index out of bounds for dimension 1: index {}, shape {}", col, self.shape[1]);
        }
        NDArrayView {
            data: self.data,
            shape: vec![self.shape[0]],
            strides: vec![self.strides[0]],
            offset: self.offset + col * self.strides[1]
        }
    }

    pub fn slice_range(&self, axis: usize, start: usize, end: usize) -> NDArrayView<'_, T> {
        if axis >= self.shape.len() {
            panic!(
                "axis {} is out of bounds for NDArrayView with {} dimensions",
                axis,
                self.shape.len()
            );
        }
        if start >= end {
            panic!(
                "invalid range: start {} must be less than end {}",
                start, end
            );
        }
        if end > self.shape[axis] {
            panic!(
                "range end {} is out of bounds for axis {} with shape {}",
                end, axis, self.shape[axis]
            );
        }

        let mut new_shape = self.shape.clone();
        new_shape[axis] = end - start;

        NDArrayView {
            data: self.data,
            shape: new_shape,
            strides: self.strides.clone(),
            offset: self.offset + start * self.strides[axis],
        }
    }

    pub fn transpose(&self) -> NDArrayView<'_, T>{
        if self.shape.len() != 2 {
            panic!("transpose requires a 2D array, got {}D", self.shape.len());
        }
        let new_shape=[self.shape[1],self.shape[0]].to_vec();
        let new_stride=[self.strides[1],self.strides[0]].to_vec();
        NDArrayView {
            data: self.data,
            shape: new_shape,
            strides: new_stride,
            offset: self.offset
        }
    }

    pub fn to_owned(&self) -> NDArray<T>
    where
        T: Clone,
    {
        let size: usize = self.shape.iter().product();
        let mut owned_data = Vec::with_capacity(size);
        let mut current_index = vec![0; self.shape.len()];

        for _ in 0..size {
            owned_data.push(self.get(&current_index).clone());

            for i in (0..self.shape.len()).rev() {
                current_index[i] += 1;
                if current_index[i] < self.shape[i] {
                    break;
                } else {
                    current_index[i] = 0;
                }
            }
        }

        NDArray::new(owned_data, self.shape.clone())
    }
}


use pyo3::prelude::*;
use crate::python::PyNDArray;

#[pymodule]
fn ferrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNDArray>()?;
    m.add_class::<python::PyNDArrayBool>()?;
    Ok(())
}

#[cfg(test)]
mod tests;

pub mod python;
