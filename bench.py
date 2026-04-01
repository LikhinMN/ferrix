import ferrix
import numpy as np
import time

def benchmark_matmul(size, iterations=5):
    a_list = np.random.randn(size * size).tolist()
    b_list = np.random.randn(size * size).tolist()
    
    a_ferrix = ferrix.PyNDArray(a_list, [size, size])
    b_ferrix = ferrix.PyNDArray(b_list, [size, size])
    
    a_numpy = np.array(a_list, dtype=np.float64).reshape(size, size)
    b_numpy = np.array(b_list, dtype=np.float64).reshape(size, size)
    
    # Pure Rust matmul
    start = time.perf_counter()
    for _ in range(iterations):
        _ = a_ferrix.matmul(b_ferrix)
    ferrix_pure_time = (time.perf_counter() - start) / iterations
    
    # BLAS matmul
    start = time.perf_counter()
    for _ in range(iterations):
        _ = a_ferrix.matmul_blas(b_ferrix)
    ferrix_blas_time = (time.perf_counter() - start) / iterations
    
    # NumPy matmul
    start = time.perf_counter()
    for _ in range(iterations):
        _ = a_numpy @ b_numpy
    numpy_time = (time.perf_counter() - start) / iterations
    
    return ferrix_pure_time, ferrix_blas_time, numpy_time

def benchmark_relu(size, iterations=50):
    data = np.random.randn(size).tolist()
    a_ferrix = ferrix.PyNDArray(data, [size])
    a_numpy = np.array(data, dtype=np.float64)
    
    # Ferrix ReLU
    start = time.perf_counter()
    for _ in range(iterations):
        _ = a_ferrix.relu()
    ferrix_time = (time.perf_counter() - start) / iterations
    
    # NumPy ReLU
    start = time.perf_counter()
    for _ in range(iterations):
        _ = np.maximum(a_numpy, 0)
    numpy_time = (time.perf_counter() - start) / iterations
    
    return ferrix_time, numpy_time

if __name__ == "__main__":
    print("Benchmarking Matmul...")
    print(f"{'Size':<10} | {'Pure (ms)':<12} | {'BLAS (ms)':<12} | {'NumPy (ms)':<12}")
    print("-" * 55)
    for size in [128, 256, 512]:
        pure, blas, np_time = benchmark_matmul(size)
        print(f"{size:<10} | {pure*1000:>12.2f} | {blas*1000:>12.2f} | {np_time*1000:>12.2f}")

    print("\nBenchmarking ReLU (1M elements)...")
    f_relu, n_relu = benchmark_relu(1_000_000)
    print(f"{'Library':<10} | {'Time (ms)':<12}")
    print("-" * 25)
    print(f"{'ferrix':<10} | {f_relu*1000:>12.2f}")
    print(f"{'NumPy':<10} | {n_relu*1000:>12.2f}")