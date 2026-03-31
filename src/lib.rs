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
}

#[cfg(test)]
mod tests;
