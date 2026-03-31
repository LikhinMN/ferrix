import ferrix
import numpy as np
import time

SIZE = 512

# ferrix
a = ferrix.PyNDArray(list(range(SIZE*SIZE)), [SIZE, SIZE])
b = ferrix.PyNDArray(list(range(SIZE*SIZE)), [SIZE, SIZE])

start = time.perf_counter()
for _ in range(10):
    c = a.matmul(b)
ferrix_time = (time.perf_counter() - start) / 10

# numpy
an = np.arange(SIZE*SIZE, dtype=np.float64).reshape(SIZE, SIZE)
bn = np.arange(SIZE*SIZE, dtype=np.float64).reshape(SIZE, SIZE)

start = time.perf_counter()
for _ in range(10):
    cn = an @ bn
numpy_time = (time.perf_counter() - start) / 10

print(f"Matrix size:     {SIZE}x{SIZE}")
print(f"ferrix matmul:   {ferrix_time*1000:.2f} ms")
print(f"numpy  matmul:   {numpy_time*1000:.2f} ms")
print(f"ratio:           {ferrix_time/numpy_time:.1f}x slower than numpy")