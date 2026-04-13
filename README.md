# ferrix

A fast, memory-safe tensor library in Rust — exposed as a Python package.

## Install

```bash
pip install ferrix
```

## Usage

```python
import ferrix

a = ferrix.PyNDArray([1.0,2.0,3.0,4.0], [2,2])
b = ferrix.PyNDArray([5.0,6.0,7.0,8.0], [2,2])

c = a.matmul_blas(b)
print(c.get([0,0]))  # 19.0

print(a.relu().get([0,0]))
print(a.softmax().sum())  # 1.0
```

## Benchmarks

| Operation | ferrix | NumPy |
|-----------|--------|-------|
| matmul 512×512 (BLAS) | 6.5ms | 2.1ms |
| relu 1M elements | 0.52ms | 0.76ms |

## Features

- N-dimensional arrays with stride-based indexing
- Zero-copy slicing, reshape, transpose
- Element-wise ops: add, mul, scale, relu, sigmoid, softmax
- Matrix multiply via OpenBLAS FFI
- Parallel ops via rayon
- Boolean masking, fancy indexing, gather
