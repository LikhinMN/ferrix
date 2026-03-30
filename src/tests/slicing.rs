use super::*;

#[test]
fn test_slice_row() {
    let array = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let row1 = array.slice_row(1);
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

#[test]
fn test_view_slice_row() {
    let array = NDArray::new((0..12).collect(), vec![4, 3]);
    let view = array.slice_row(1); // row 1 is [3, 4, 5], 1D
    assert_eq!(view.shape, vec![3]);
}

#[test]
fn test_slice_of_view() {
    let array = NDArray::new((0..12).collect(), vec![4, 3]);
    // [[0, 1, 2],
    //  [3, 4, 5],
    //  [6, 7, 8],
    //  [9, 10, 11]]
    
    let view = array.slice_row(1); // row 1 is [3, 4, 5], shape [3]
    assert_eq!(view.shape, vec![3]);
    assert_eq!(*view.get(&[1]), 4);
    
    // slice_row on view requires view.shape.len() == 2.
    // Our view is 1D, so slice_row should panic.
}

#[test]
#[should_panic(expected = "slice_row requires a 2D array, got 1D")]
fn test_slice_row_1d_panic() {
    let array = NDArray::new(vec![1, 2, 3], vec![3]);
    let _ = array.slice_row(0);
}
