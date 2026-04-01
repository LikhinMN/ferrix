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

// Added per challenge: slice_row on NDArrayView (nested slice) for 3×3 verifying values 7,8,9
#[test]
fn test_slice_row_on_view_values_3x3() {
    let a = NDArray::new(vec![1,2,3,4,5,6,7,8,9], vec![3,3]);
    let view = a.slice_row(2);
    assert_eq!(*view.get(&[0]), 7);
    assert_eq!(*view.get(&[1]), 8);
    assert_eq!(*view.get(&[2]), 9);
}

#[test]
fn test_slice_col() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);

    let col0 = a.slice_col(0);
    assert_eq!(*col0.get(&[0]), 1);
    assert_eq!(*col0.get(&[1]), 4);

    let col1 = a.slice_col(1);
    assert_eq!(*col1.get(&[0]), 2);
    assert_eq!(*col1.get(&[1]), 5);

    let col2 = a.slice_col(2);
    assert_eq!(*col2.get(&[0]), 3);
    assert_eq!(*col2.get(&[1]), 6);
}

#[test]
#[should_panic(expected = "slice_col requires a 2D array, got 1D")]
fn test_nested_slice_row_col_panic() {
    let a = NDArray::new((1..=9).collect(), vec![3, 3]);
    let row = a.slice_row(1); // NDArrayView of row 1, shape [3]
    row.slice_col(0); // should panic - row view is 1D not 2D
}

#[test]
#[should_panic(expected = "index out of bounds for dimension 1: index 5, shape 3")]
fn test_slice_col_oob_panic() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    a.slice_col(5);
}

#[test]
#[should_panic(expected = "slice_col requires a 2D array, got 1D")]
fn test_slice_col_not_2d_panic() {
    let a = NDArray::new(vec![1, 2, 3], vec![3]);
    a.slice_col(0);
}

#[test]
fn test_slice_range() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6, 7, 8, 9], vec![3, 3]);

    let s = a.slice_range(0, 1, 3); // rows 1..3
    assert_eq!(s.shape, vec![2, 3]);
    assert_eq!(*s.get(&[0, 0]), 4);
    assert_eq!(*s.get(&[1, 2]), 9);

    let s2 = a.slice_range(1, 0, 2); // cols 0..2
    assert_eq!(s2.shape, vec![3, 2]);
    assert_eq!(*s2.get(&[0, 0]), 1);
    assert_eq!(*s2.get(&[2, 1]), 8);
}

#[test]
fn test_view_slice_range() {
    let a = NDArray::new((1..=24).collect(), vec![2, 3, 4]);
    let v = a.slice_range(1, 1, 3); // axis 1, range 1..3
    assert_eq!(v.shape, vec![2, 2, 4]);

    let v2 = v.slice_range(0, 1, 2); // axis 0, range 1..2
    assert_eq!(v2.shape, vec![1, 2, 4]);
}

#[test]
#[should_panic(expected = "axis 3 is out of bounds for NDArray with 2 dimensions")]
fn test_slice_range_axis_oob() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    a.slice_range(3, 0, 1);
}

#[test]
#[should_panic(expected = "invalid range: start 2 must be less than end 1")]
fn test_slice_range_invalid_range() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    a.slice_range(0, 2, 1);
}

#[test]
#[should_panic(expected = "range end 3 is out of bounds for axis 0 with shape 2")]
fn test_slice_range_end_oob() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    a.slice_range(0, 0, 3);
}
