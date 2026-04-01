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
        PyNDArray { inner: self.inner.matmul_blas(&other.inner) }
    }
}

#[pymodule]
fn ferrix(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNDArray>()?;
    Ok(())
}
