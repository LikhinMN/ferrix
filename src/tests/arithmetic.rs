use super::*;

#[test]
fn test_add() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = NDArray::new(vec![6, 5, 4, 3, 2, 1], vec![2, 3]);
    let c = a.add(&b);

    assert_eq!(*c.get(&[0, 0]), 7);
    assert_eq!(*c.get(&[0, 1]), 7);
    assert_eq!(*c.get(&[0, 2]), 7);
    assert_eq!(*c.get(&[1, 0]), 7);
    assert_eq!(*c.get(&[1, 1]), 7);
    assert_eq!(*c.get(&[1, 2]), 7);
}

#[test]
#[should_panic(expected = "shape mismatch for addition: [2, 3] vs [3, 2]")]
fn test_add_shape_mismatch() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![3, 2]);
    let _ = a.add(&b);
}
