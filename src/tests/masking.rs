use super::*;

#[test]
fn test_boolean_mask() {
    let a = NDArray::new(vec![-2.0, 3.0, -1.0, 4.0, 0.0, 5.0], vec![2, 3]);
    let mask = NDArray::new(vec![false, true, false, true, false, true], vec![2, 3]);
    let result = a.boolean_mask(&mask);
    
    assert_eq!(result.shape, vec![3]);
    assert_eq!(*result.get(&[0]), 3.0);
    assert_eq!(*result.get(&[1]), 4.0);
    assert_eq!(*result.get(&[2]), 5.0);
}

#[test]
#[should_panic(expected = "mask shape [2, 2] must equal self shape [2, 3]")]
fn test_boolean_mask_shape_mismatch() {
    let a = NDArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mask = NDArray::new(vec![true, false, true, false], vec![2, 2]);
    let _ = a.boolean_mask(&mask);
}

#[test]
fn test_boolean_mask_all_false() {
    let a = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mask = NDArray::new(vec![false, false, false, false], vec![2, 2]);
    let result = a.boolean_mask(&mask);
    
    assert_eq!(result.shape, vec![0]);
    assert_eq!(result.data.len(), 0);
}

#[test]
fn test_boolean_mask_all_true() {
    let a = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let mask = NDArray::new(vec![true, true, true, true], vec![2, 2]);
    let result = a.boolean_mask(&mask);
    
    assert_eq!(result.shape, vec![4]);
    assert_eq!(result.data, vec![1.0, 2.0, 3.0, 4.0]);
}
