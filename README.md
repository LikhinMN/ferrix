# ferrix

ferrix is a NumPy-like N-dimensional tensor library in Rust, designed for performance and efficiency, and exposed to Python using PyO3. It leverages `rayon` for parallelized element-wise operations and `cblas` for high-performance matrix multiplications.

## Features

- **NDArray<T>**: core N-dimensional tensor implementation with row-major strides.
- **Zero-copy operations**: Slicing (`slice_row`, `slice_col`), `reshape`, and `transpose` are implemented as zero-copy views.
- **Parallel processing**: Element-wise operations (`add`, `mul`, `scale`, `relu`) are parallelized using `rayon`.
- **Fast matrix multiplication**: Both pure Rust and BLAS-accelerated (`matmul_blas`) implementations.
- **Python Bindings**: Seamless integration with Python via PyO3.

## Installation

To install and build `ferrix` in your Python environment:

```bash
# Clone the repository
git clone https://github.com/your-username/ferrix.git
cd ferrix

# Install dependencies and build the package
maturin develop --release
```

## Python Usage

```python
import ferrix
import numpy as np

# Create a new array
a = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
b = ferrix.PyNDArray([5.0, 6.0, 7.0, 8.0], [2, 2])

# Perform operations
c = a.add(b)
d = a.matmul_blas(b)

# Views
a_t = a.transpose()  # Zero-copy
print(f"Shape: {a.shape()}")
print(f"Max index: {a.argmax()}")

# Element-wise activations
relu_a = a.relu()
sigmoid_a = a.sigmoid()
```

## Benchmark Results

Benchmarks were performed on a machine with 1M elements for ReLU and various matrix sizes for Matmul. Comparison with NumPy:

### Matrix Multiplication (ms)

| Size | Pure Rust | ferrix (BLAS) | NumPy |
| :--- | :--- | :--- | :--- |
| 128x128 | 4.33 | 0.14 | 6.61 |
| 256x256 | 72.60 | 0.98 | 8.12 |
| 512x512 | 563.28 | 8.00 | 15.49 |

### ReLU (1M elements)

| Library | Time (ms) |
| :--- | :--- |
| ferrix | 0.79 |
| NumPy | 1.06 |
