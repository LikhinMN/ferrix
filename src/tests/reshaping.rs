use super::*;

#[test]
fn test_reshape() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = a.reshape(vec![3, 2]);
    assert_eq!(*b.get(&[0, 0]), 1);
    assert_eq!(*b.get(&[0, 1]), 2);
    assert_eq!(*b.get(&[1, 0]), 3);
    assert_eq!(*b.get(&[2, 1]), 6);

    let c = a.reshape(vec![6]);
    assert_eq!(*c.get(&[0]), 1);
    assert_eq!(*c.get(&[5]), 6);
}
