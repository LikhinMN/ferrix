use super::*;

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
fn test_get_3d() {
    let array = NDArray::new((0..24).collect(), vec![2, 3, 4]);
    assert_eq!(*array.get(&[0, 0, 0]), 0);
    assert_eq!(*array.get(&[1, 2, 3]), 23);
    assert_eq!(*array.get(&[0, 2, 1]), 9); // 0*12 + 2*4 + 1*1 = 9
}
