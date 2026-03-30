use super::*;

#[test]
fn test_ndarray_strides() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    assert_eq!(array.strides, vec![3, 1]);
}

#[test]
fn test_3d_strides() {
    let array = NDArray::new(vec![0; 24], vec![2, 3, 4]);
    assert_eq!(array.strides, vec![12, 4, 1]);
}

#[test]
#[should_panic(expected = "does not match shape")]
fn test_invalid() {
    let _array = NDArray::new(vec![1, 2, 3], vec![2, 3]);
}

#[test]
fn test_get_2d() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    assert_eq!(*array.get(&[0, 0]), 1);
    assert_eq!(*array.get(&[0, 2]), 3);
    assert_eq!(*array.get(&[1, 0]), 4);
    assert_eq!(*array.get(&[1, 2]), 6);
}

#[test]
#[should_panic(expected = "index out of bounds for dimension 0: index 2, shape 2")]
fn test_get_panic_oob() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let _ = array.get(&[2, 0]);
}

#[test]
fn test_slice_row() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let row1 = array.slice_row(1);
    assert_eq!(row1.data, vec![4, 5, 6]);
    assert_eq!(row1.shape, vec![3]);
    // Test 1: slice_row(1) on [[1,2,3],[4,5,6]]
    // → get(&[0]) == 4, get(&[1]) == 5, get(&[2]) == 6
    assert_eq!(*row1.get(&[0]), 4);
    assert_eq!(*row1.get(&[1]), 5);
    assert_eq!(*row1.get(&[2]), 6);
}

#[test]
#[should_panic(expected = "index out of bounds for dimension 0: index 5, shape 2")]
fn test_slice_row_panic_5() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let _ = array.slice_row(5);
}

#[test]
#[should_panic(expected = "index out of bounds for dimension 0: index 2, shape 2")]
fn test_slice_row_panic() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let _ = array.slice_row(2);
}
