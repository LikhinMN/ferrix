# ferrix

ferrix is a Rust-first N-dimensional array library with Python bindings.

It currently focuses on:

- A generic Rust `NDArray<T>` core with row-major indexing.
- Fast element-wise operations using `rayon` for parallelism.
- Matrix multiplication in both pure Rust and BLAS-backed forms.
- A PyO3 extension module exposing `f64` and `bool` arrays to Python.

## Table Of Contents

- [What ferrix is](#what-ferrix-is)
- [Current capabilities](#current-capabilities)
- [Installation](#installation)
- [Quickstart (Python)](#quickstart-python)
- [Rust API reference](#rust-api-reference)
- [Python API reference](#python-api-reference)
- [Data model and indexing](#data-model-and-indexing)
- [Error behavior](#error-behavior)
- [Running tests](#running-tests)
- [Benchmarking](#benchmarking)
- [Project layout](#project-layout)
- [Limitations and roadmap](#limitations-and-roadmap)

## What ferrix is

ferrix provides a contiguous, row-major tensor container:

- Storage: `Vec<T>`
- Shape: `Vec<usize>`
- Strides: computed row-major strides
- Offset: used by view/slicing machinery

The crate is built as both:

- `rlib` for Rust usage and testing
- `cdylib` for Python extension loading via `maturin`/PyO3

## Current capabilities

### Rust core (`NDArray<T>`)

- Construction and validation: `NDArray::new`
- Element access: `get`, `get_mut`
- Advanced indexing: `fancy_index` (1D only)
- Views: `slice_row`, `slice_col`, `slice_range`, `reshape`, `transpose`
- Element-wise ops: `add`, `mul`, `scale`
- Matrix multiplication: `matmul`
- `f64`-specific ops: `relu`, `sum`, `mean`, `argmax`, `sigmoid`, `softmax`, `matmul_blas`, `boolean_mask`, `masked_fill`, `where_`

### Python module (`ferrix`)

- `PyNDArray` (`f64`)
- `PyNDArrayBool` (`bool` mask container)
- Python-callable methods for creation, indexing, shape, arithmetic, reductions, activations, and matrix multiplication

## Installation

### Prerequisites

- Rust toolchain (`cargo`, `rustc`)
- Python 3.8+
- `maturin`
- A BLAS backend discoverable by `openblas-src` (`system` feature is enabled in this project)

### Build and install extension into active Python environment

```bash
maturin develop --release
```

If you use a virtual environment:

```bash
source .venv/bin/activate
maturin develop --release
```

## Quickstart (Python)

```python
import ferrix

a = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
b = ferrix.PyNDArray([5.0, 6.0, 7.0, 8.0], [2, 2])

c_add = a.add(b)
c_mul = a.mul(b)
c_mm = a.matmul_blas(b)

print(a.shape())          # [2, 2]
print(a.get([1, 0]))      # 3.0
print(a.sum())            # 10.0
print(a.mean())           # 2.5
print(a.argmax())         # 3

relu_a = a.relu()
sigmoid_a = a.sigmoid()
softmax_a = a.softmax()

t = a.transpose()         # Python wrapper returns owned array
```

Boolean mask example:

```python
import ferrix

x = ferrix.PyNDArray([-2.0, 3.0, -1.0, 4.0], [2, 2])
m = ferrix.PyNDArrayBool([False, True, False, True], [2, 2])
selected = x.boolean_mask(m)

print(selected.shape())   # [2]
print(selected.get([0]))  # 3.0
print(selected.get([1]))  # 4.0
```

## Rust API reference

### Generic API

`NDArray<T>` methods:

- `new(data, shape)`
- `get(index) -> &T`
- `get_mut(index) -> &mut T`
- `fancy_index(indices) -> NDArray<T>` (1D only)
- `slice_row(row) -> NDArrayView<T>` (2D only)
- `slice_col(col) -> NDArrayView<T>` (2D only)
- `slice_range(axis, start, end) -> NDArrayView<T>`
- `reshape(new_shape) -> NDArrayView<T>`
- `transpose() -> NDArrayView<T>` (2D only)
- `add(other) -> NDArray<T>`
- `mul(other) -> NDArray<T>`
- `scale(scalar) -> NDArray<T>`
- `matmul(other) -> NDArray<T>` (2D only)

`NDArrayView<'a, T>` methods:

- `get(index) -> &T`
- `slice_row`, `slice_col`, `slice_range`, `transpose`
- `to_owned() -> NDArray<T>`

### `f64`-specialized API

`impl NDArray<f64>` methods:

- `relu()`
- `sum()`
- `mean()`
- `argmax()`
- `sigmoid()`
- `softmax()`
- `matmul_blas(other)`
- `boolean_mask(mask: &NDArray<bool>)`
- `masked_fill(mask: &NDArray<bool>, value)`
- `where_(condition: &NDArray<bool>, other: &NDArray<f64>)`

## Python API reference

### `PyNDArray`

Constructor:

- `PyNDArray(data: list[float], shape: list[int])`

Methods:

- `get(index: list[int]) -> float`
- `shape() -> list[int]`
- `sum() -> float`
- `mean() -> float`
- `argmax() -> int`
- `relu() -> PyNDArray`
- `sigmoid() -> PyNDArray`
- `softmax() -> PyNDArray`
- `transpose() -> PyNDArray`
- `scale(scalar: float) -> PyNDArray`
- `add(other: PyNDArray) -> PyNDArray`
- `mul(other: PyNDArray) -> PyNDArray`
- `matmul(other: PyNDArray) -> PyNDArray`
- `matmul_blas(other: PyNDArray) -> PyNDArray`
- `boolean_mask(mask: PyNDArrayBool) -> PyNDArray`

### `PyNDArrayBool`

Constructor:

- `PyNDArrayBool(data: list[bool], shape: list[int])`

Methods:

- `shape() -> list[int]`

## Data model and indexing

ferrix uses row-major flattening. For shape `[d0, d1, ..., dn]`, strides are computed right-to-left.

Example:

- Shape `[2, 3, 4]`
- Strides `[12, 4, 1]`
- Flat index for `[i, j, k]` is `offset + i*12 + j*4 + k`

Views (`NDArrayView`) preserve borrowed storage and adjust shape/stride/offset instead of copying.

## Error behavior

Most validation errors currently `panic!` with detailed messages, including:

- Data/shape length mismatch on construction and reshape
- Wrong dimensionality for 2D-only operations (`slice_row`, `slice_col`, `transpose`, `matmul`, `matmul_blas`)
- Shape mismatch in element-wise and mask operations
- Out-of-bounds indices

This is intentional in the current stage and is validated by tests.

## Running tests

Rust tests:

```bash
cargo test
```

Library-only tests:

```bash
cargo test --lib
```

Python smoke tests (after `maturin develop`):

```bash
python test_ferrix.py
```

## Benchmarking

Run included benchmark script:

```bash
python bench.py
```

The script compares:

- `matmul` (pure Rust)
- `matmul_blas` (BLAS via `cblas`)
- NumPy matrix multiplication
- ReLU against NumPy

Example output (from previous run):

| Size | Pure Rust (ms) | ferrix BLAS (ms) | NumPy (ms) |
| :--- | :--- | :--- | :--- |
| 128x128 | 4.33 | 0.14 | 6.61 |
| 256x256 | 72.60 | 0.98 | 8.12 |
| 512x512 | 563.28 | 8.00 | 15.49 |

ReLU on 1M elements:

| Library | Time (ms) |
| :--- | :--- |
| ferrix | 0.79 |
| NumPy | 1.06 |

Your numbers will vary by hardware, BLAS setup, and build mode.

## Project layout

```text
ferrix/
	Cargo.toml
	pyproject.toml
	README.md
	src/
		lib.rs            # NDArray, NDArrayView, module init
		python.rs         # PyO3 bindings
		tests/            # Rust unit tests by feature
	bench.py            # Python benchmark harness
	test_ferrix.py      # Python smoke/regression checks
```

## Limitations and roadmap

Current limitations:

- No broadcasting semantics for arithmetic.
- No in-place Python mutating ops exposed.
- Error handling is panic-based in Rust core.
- Python API currently focuses on `f64` arrays and mask arrays.

Near-term improvements you can contribute:

- Add broadcasting and reduction-by-axis operations.
- Expose more view/slicing operations directly to Python.
- Introduce `Result`-based fallible APIs in Rust core.
- Add benchmarks under Criterion for Rust-native performance tracking.
