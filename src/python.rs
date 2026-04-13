use pyo3::prelude::*;
use crate::NDArray;

#[pyclass]
pub struct PyNDArray {
    pub inner: NDArray<f64>,
}

#[pymethods]
impl PyNDArray {
    #[new]
    fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        PyNDArray {
            inner: NDArray::new(data, shape),
        }
    }

    fn get(&self, index: Vec<usize>) -> f64 {
        *self.inner.get(&index)
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }

    fn sum(&self) -> f64 {
        self.inner.sum()
    }

    fn mean(&self) -> f64 {
        self.inner.mean()
    }

    fn relu(&self) -> PyNDArray {
        PyNDArray { inner: self.inner.relu() }
    }

    fn sigmoid(&self) -> PyNDArray {
        PyNDArray { inner: self.inner.sigmoid() }
    }

    fn softmax(&self) -> PyNDArray {
        PyNDArray { inner: self.inner.softmax() }
    }

    fn transpose(&self) -> PyNDArray {
        PyNDArray { inner: self.inner.transpose().to_owned() }
    }

    fn argmax(&self) -> usize {
        self.inner.argmax()
    }

    fn scale(&self, scalar: f64) -> PyNDArray {
        PyNDArray { inner: self.inner.scale(scalar) }
    }

    fn add(&self, other: &PyNDArray) -> PyNDArray {
        PyNDArray { inner: self.inner.add(&other.inner) }
    }

    fn mul(&self, other: &PyNDArray) -> PyNDArray {
        PyNDArray { inner: self.inner.mul(&other.inner) }
    }

    fn matmul(&self, other: &PyNDArray) -> PyNDArray {
        PyNDArray { inner: self.inner.matmul(&other.inner) }
    }

    fn matmul_blas(&self, other: &PyNDArray) -> PyNDArray {
        // BLAS backend was removed; keep API and fall back to pure Rust matmul.
        PyNDArray { inner: self.inner.matmul(&other.inner) }
    }

    fn boolean_mask(&self, mask: &PyNDArrayBool) -> PyNDArray {
        PyNDArray { inner: self.inner.boolean_mask(&mask.inner) }
    }

    fn masked_fill(&mut self, mask: &PyNDArrayBool, value: f64) {
        self.inner.masked_fill(&mask.inner, value);
    }

    fn where_(&self, condition: &PyNDArrayBool, other: &PyNDArray) -> PyNDArray {
        PyNDArray { inner: self.inner.where_(&condition.inner, &other.inner) }
    }

    fn fancy_index(&self, indices: Vec<usize>) -> PyNDArray {
        PyNDArray { inner: self.inner.fancy_index(&indices) }
    }

    fn slice_row(&self, row: usize) -> PyNDArray {
        PyNDArray { inner: self.inner.slice_row(row).to_owned() }
    }

    fn slice_col(&self, col: usize) -> PyNDArray {
        PyNDArray { inner: self.inner.slice_col(col).to_owned() }
    }

    fn slice_range(&self, axis: usize, start: usize, end: usize) -> PyNDArray {
        PyNDArray { inner: self.inner.slice_range(axis, start, end).to_owned() }
    }

    fn reshape(&self, new_shape: Vec<usize>) -> PyNDArray {
        PyNDArray { inner: self.inner.reshape(new_shape).to_owned() }
    }

    fn gather(&self, axis: usize, indices: Vec<usize>) -> PyNDArray {
        PyNDArray { inner: self.inner.gather(axis, &indices) }
    }

    fn set_slice(&mut self, axis: usize, start: usize, end: usize, value: f64) {
        self.inner.set_slice(axis, start, end, value);
    }

    fn cumsum(&self) -> PyNDArray {
        PyNDArray { inner: self.inner.cumsum() }
    }
}

#[pyclass(name = "PyBoolArray")]
pub struct PyNDArrayBool {
    pub inner: NDArray<bool>,
}

#[pymethods]
impl PyNDArrayBool {
    #[new]
    fn new(data: Vec<bool>, shape: Vec<usize>) -> Self {
        PyNDArrayBool {
            inner: NDArray::new(data, shape),
        }
    }

    fn shape(&self) -> Vec<usize> {
        self.inner.shape.clone()
    }
}
