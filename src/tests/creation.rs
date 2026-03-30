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
