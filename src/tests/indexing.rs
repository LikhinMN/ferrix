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

#[test]
fn test_fancy_index() {
    let a = NDArray::new(vec![10, 20, 30, 40, 50], vec![5]);
    let r = a.fancy_index(&[0, 2, 4]);
    assert_eq!(r.shape, vec![3]);
    assert_eq!(*r.get(&[0]), 10);
    assert_eq!(*r.get(&[1]), 30);
    assert_eq!(*r.get(&[2]), 50);

    let r2 = a.fancy_index(&[4, 0, 2]);
    assert_eq!(r2.shape, vec![3]);
    assert_eq!(*r2.get(&[0]), 50);
    assert_eq!(*r2.get(&[1]), 10);
    assert_eq!(*r2.get(&[2]), 30);
}

#[test]
#[should_panic(expected = "fancy_index is only valid for 1D arrays")]
fn test_fancy_index_not_1d() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let _ = a.fancy_index(&[0, 1]);
}

#[test]
#[should_panic(expected = "index out of bounds for dimension 0: index 5, shape 5")]
fn test_fancy_index_oob() {
    let a = NDArray::new(vec![10, 20, 30, 40, 50], vec![5]);
    let _ = a.fancy_index(&[0, 5]);
}
