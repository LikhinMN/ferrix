use super::*;

#[test]
fn test_transpose() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let t = a.transpose();
    assert_eq!(t.shape, vec![3, 2]);

    assert_eq!(*t.get(&[0, 0]), 1);
    assert_eq!(*t.get(&[1, 0]), 2);
    assert_eq!(*t.get(&[2, 0]), 3);
    assert_eq!(*t.get(&[0, 1]), 4);
    assert_eq!(*t.get(&[1, 1]), 5);
    assert_eq!(*t.get(&[2, 1]), 6);
}

#[test]
#[should_panic(expected = "transpose requires a 2D array, got 1D")]
fn test_transpose_panic_1d() {
    let a = NDArray::new(vec![1, 2, 3], vec![3]);
    let _ = a.transpose();
}
