# ferrix

A fast, memory-safe tensor library implemented in Rust and exposed as a Python package.

`ferrix` provides an `NDArray` core with stride-based indexing, vectorized math operations, slicing/reshaping utilities, and Python bindings via `pyo3`.

## Why ferrix

- Rust core for predictable performance and memory safety.
- Python-friendly API for quick experimentation.
- Multi-dimensional arrays with row-major strides.
- Useful tensor operations for ML and numerical workloads.

## Installation

Install from PyPI:

```bash
python -m pip install ferrix
```

## Quick Start

```python
import ferrix

# Create two 2x2 arrays
a = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
b = ferrix.PyNDArray([5.0, 6.0, 7.0, 8.0], [2, 2])

# Core arithmetic
print(a.add(b).get([0, 0]))      # 6.0
print(a.mul(b).get([1, 1]))      # 32.0
print(a.scale(0.5).get([1, 0]))  # 1.5

# Matrix multiplication
print(a.matmul(b).get([0, 0]))       # 19.0
print(a.matmul_blas(b).get([0, 0]))  # Compatibility API; currently same as matmul

# Activations and reductions
print(a.relu().get([0, 0]))
print(a.softmax().sum())
print(a.sum(), a.mean(), a.argmax())

# Shape transforms
print(a.transpose().shape())
print(a.reshape([4]).shape())
```

## Complete API Reference (Python)

`ferrix` exposes two classes:

- `ferrix.PyNDArray` for numeric tensors (`f64`)
- `ferrix.PyBoolArray` for boolean masks

### `PyNDArray`

Constructor:

- `PyNDArray(data: list[float], shape: list[int]) -> PyNDArray`

```python
x = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
```

Introspection and element access:

- `shape() -> list[int]`
- `get(index: list[int]) -> float`

```python
x = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
print(x.shape())      # [2, 2]
print(x.get([1, 0]))  # 3.0
```

Reductions:

- `sum() -> float`
- `mean() -> float`
- `argmax() -> int` (index in flattened row-major order)

```python
x = ferrix.PyNDArray([1.0, 5.0, 3.0, 4.0], [2, 2])
print(x.sum())     # 13.0
print(x.mean())    # 3.25
print(x.argmax())  # 1
```

Element-wise math:

- `add(other: PyNDArray) -> PyNDArray`
- `mul(other: PyNDArray) -> PyNDArray`
- `scale(scalar: float) -> PyNDArray`

```python
a = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
b = ferrix.PyNDArray([5.0, 6.0, 7.0, 8.0], [2, 2])

print(a.add(b).get([0, 1]))   # 8.0
print(a.mul(b).get([1, 0]))   # 21.0
print(a.scale(10).get([1, 1]))  # 40.0
```

Activation functions:

- `relu() -> PyNDArray`
- `sigmoid() -> PyNDArray`
- `softmax() -> PyNDArray`

```python
x = ferrix.PyNDArray([-1.0, 0.0, 1.0], [1, 3])
print(x.relu().get([0, 0]))
print(x.sigmoid().get([0, 2]))
print(x.softmax().sum())
```

Matrix operations:

- `matmul(other: PyNDArray) -> PyNDArray`
- `matmul_blas(other: PyNDArray) -> PyNDArray`

```python
a = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
b = ferrix.PyNDArray([5.0, 6.0, 7.0, 8.0], [2, 2])

print(a.matmul(b).get([0, 0]))
print(a.matmul_blas(b).get([0, 0]))
```

Note: `matmul_blas` is currently a compatibility API that uses the same backend behavior as `matmul`.

Reshape and transpose:

- `reshape(new_shape: list[int]) -> PyNDArray`
- `transpose() -> PyNDArray` (2D transpose)

```python
x = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
print(x.reshape([4]).shape())
print(x.transpose().shape())
```

Slicing and indexing:

- `slice_row(row: int) -> PyNDArray` (2D)
- `slice_col(col: int) -> PyNDArray` (2D)
- `slice_range(axis: int, start: int, end: int) -> PyNDArray`
- `fancy_index(indices: list[int]) -> PyNDArray` (1D input)
- `gather(axis: int, indices: list[int]) -> PyNDArray`

```python
m = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
v = ferrix.PyNDArray([10.0, 20.0, 30.0, 40.0], [4])

print(m.slice_row(1).shape())
print(m.slice_col(2).shape())
print(m.slice_range(1, 0, 2).shape())
print(v.fancy_index([3, 1, 1]).shape())
print(m.gather(1, [2, 0]).shape())
```

Masking and conditional operations:

- `boolean_mask(mask: PyBoolArray) -> PyNDArray`
- `masked_fill(mask: PyBoolArray, value: float) -> None` (in-place)
- `where_(condition: PyBoolArray, other: PyNDArray) -> PyNDArray`

```python
x = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
y = ferrix.PyNDArray([9.0, 9.0, 9.0, 9.0], [2, 2])
mask = ferrix.PyBoolArray([True, False, True, False], [2, 2])

print(x.boolean_mask(mask).shape())
x.masked_fill(mask, -1.0)
print(x.where_(mask, y).shape())
```

Mutation and cumulative operations:

- `set_slice(axis: int, start: int, end: int, value: float) -> None` (in-place)
- `cumsum() -> PyNDArray` (flattened cumulative sum)

```python
x = ferrix.PyNDArray([1.0, 2.0, 3.0, 4.0], [2, 2])
x.set_slice(0, 0, 1, 0.0)
print(x.get([0, 1]))
print(x.cumsum().shape())
```

### `PyBoolArray`

Constructor and methods:

- `PyBoolArray(data: list[bool], shape: list[int]) -> PyBoolArray`
- `shape() -> list[int]`

```python
mask = ferrix.PyBoolArray([True, False, True, False], [2, 2])
print(mask.shape())
```

## Error Behavior

- Invalid shapes, indices, or axis values raise Python exceptions backed by Rust panics.
- Most binary operations require shape compatibility.
- `fancy_index` is for 1D arrays.
- `transpose` and `slice_row`/`slice_col` require 2D arrays.

## Feature Notes

- Arrays are row-major and stride-aware.
- Shape checks and index checks panic on invalid inputs in the Rust core.
- `matmul_blas` is currently a compatibility method that falls back to the same implementation as `matmul`.
- Parallel execution is used for selected element-wise operations through `rayon`.

## Benchmarks (Indicative)

The following numbers are from repository examples and should be treated as indicative (hardware and build mode dependent):

| Operation | ferrix | NumPy |
|---|---:|---:|
| matmul 512x512 | 6.5 ms | 2.1 ms |
| relu 1M elements | 0.52 ms | 0.76 ms |

## Build From Source (Optional)

This project uses `maturin` to build the Python extension from Rust.

### Prerequisites

- Rust toolchain (`cargo`, `rustc`)
- Python `>=3.8`
- `maturin`

### Local development build

```bash
python -m pip install --upgrade pip maturin
maturin develop
```

After this, `import ferrix` uses your local build in the active virtual environment.

### Build wheel/sdist

```bash
rm -rf dist
maturin build --release --out dist
maturin sdist --out dist
```

## Release and Publishing

For a complete PyPI release runbook, see:

- `docs/pypi-guide.md`

It includes versioning rules, artifact validation, upload options, and post-release checks.

## Repository Layout

Key files and directories:

- `src/lib.rs`: Rust tensor core (`NDArray` and `NDArrayView`)
- `src/python.rs`: Python bindings (`PyNDArray`, `PyBoolArray`)
- `src/tests/`: Rust unit tests
- `test_ferrix.py`: Python API tests
- `pyproject.toml`: Python packaging metadata and build backend
- `Cargo.toml`: Rust crate configuration

## Contributing

Contributions are welcome.

For contributions and release process details, start from `docs/pypi-guide.md` and open a PR with clear change notes.
