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
}

#[cfg(test)]
mod tests;
