import ferrix
import numpy as np

def test_creation_and_get():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape = [2, 3]
    arr = ferrix.PyNDArray(data, shape)
    assert arr.shape() == shape
    assert arr.get([0, 0]) == 1.0
    assert arr.get([0, 1]) == 2.0
    assert arr.get([0, 2]) == 3.0
    assert arr.get([1, 0]) == 4.0
    assert arr.get([1, 1]) == 5.0
    assert arr.get([1, 2]) == 6.0

def test_add():
    data1 = [1.0, 2.0, 3.0, 4.0]
    data2 = [5.0, 6.0, 7.0, 8.0]
    shape = [2, 2]
    arr1 = ferrix.PyNDArray(data1, shape)
    arr2 = ferrix.PyNDArray(data2, shape)
    res = arr1.add(arr2)
    
    np_res = np.array(data1).reshape(shape) + np.array(data2).reshape(shape)
    for i in range(2):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_mul():
    data1 = [1.0, 2.0, 3.0, 4.0]
    data2 = [5.0, 6.0, 7.0, 8.0]
    shape = [2, 2]
    arr1 = ferrix.PyNDArray(data1, shape)
    arr2 = ferrix.PyNDArray(data2, shape)
    res = arr1.mul(arr2)
    
    np_res = np.array(data1).reshape(shape) * np.array(data2).reshape(shape)
    for i in range(2):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_matmul():
    data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape1 = [2, 3]
    data2 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    shape2 = [3, 2]
    arr1 = ferrix.PyNDArray(data1, shape1)
    arr2 = ferrix.PyNDArray(data2, shape2)
    res = arr1.matmul(arr2)
    
    np_res = np.dot(np.array(data1).reshape(shape1), np.array(data2).reshape(shape2))
    assert res.shape() == [2, 2]
    for i in range(2):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_matmul_blas():
    data1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape1 = [2, 3]
    data2 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    shape2 = [3, 2]
    arr1 = ferrix.PyNDArray(data1, shape1)
    arr2 = ferrix.PyNDArray(data2, shape2)
    res = arr1.matmul_blas(arr2)
    
    np_res = np.dot(np.array(data1).reshape(shape1), np.array(data2).reshape(shape2))
    assert res.shape() == [2, 2]
    for i in range(2):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_relu():
    data = [-1.0, 0.0, 1.0, -2.0]
    shape = [2, 2]
    arr = ferrix.PyNDArray(data, shape)
    res = arr.relu()
    
    np_res = np.maximum(0, np.array(data).reshape(shape))
    for i in range(2):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_sum_mean():
    data = [1.0, 2.0, 3.0, 4.0]
    shape = [2, 2]
    arr = ferrix.PyNDArray(data, shape)
    assert arr.sum() == 10.0
    assert arr.mean() == 2.5

def test_transpose():
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    shape = [2, 3]
    arr = ferrix.PyNDArray(data, shape)
    res = arr.transpose()
    
    assert res.shape() == [3, 2]
    np_res = np.array(data).reshape(shape).T
    for i in range(3):
        for j in range(2):
            assert res.get([i, j]) == np_res[i, j]

def test_softmax():
    data = [1.0, 2.0, 3.0]
    shape = [1, 3]
    arr = ferrix.PyNDArray(data, shape)
    res = arr.softmax()
    
    np_data = np.array(data)
    e_x = np.exp(np_data - np.max(np_data))
    np_res = e_x / e_x.sum()
    
    for i in range(3):
        assert abs(res.get([0, i]) - np_res[i]) < 1e-6

if __name__ == "__main__":
    test_creation_and_get()
    test_add()
    test_mul()
    test_matmul()
    test_matmul_blas()
    test_relu()
    test_sum_mean()
    test_transpose()
    test_softmax()
    print("All tests passed!")
