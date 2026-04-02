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

#[test]
fn test_masked_fill() {
    let mut a = NDArray::new(vec![-2.0, 3.0, -1.0, 4.0], vec![2, 2]);
    let mask = NDArray::new(vec![false, true, false, true], vec![2, 2]);
    a.masked_fill(&mask, 0.0);
    assert_eq!(*a.get(&[0, 0]), -2.0);
    assert_eq!(*a.get(&[0, 1]), 0.0);
    assert_eq!(*a.get(&[1, 0]), -1.0);
    assert_eq!(*a.get(&[1, 1]), 0.0);
}

#[test]
#[should_panic(expected = "mask shape [2, 2] must equal self shape [2, 3]")]
fn test_masked_fill_shape_mismatch() {
    let mut a = NDArray::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    let mask = NDArray::new(vec![true, false, true, false], vec![2, 2]);
    a.masked_fill(&mask, 0.0);
}

#[test]
fn test_where() {
    let a    = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b    = NDArray::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]);
    let cond = NDArray::new(vec![true, false, true, false], vec![2, 2]);
    
    let r = a.where_(&cond, &b);
    assert_eq!(r.shape, vec![2, 2]);
    assert_eq!(*r.get(&[0, 0]), 1.0);   // cond true  → take from a
    assert_eq!(*r.get(&[0, 1]), 0.0);   // cond false → take from b
    assert_eq!(*r.get(&[1, 0]), 3.0);   // cond true  → take from a
    assert_eq!(*r.get(&[1, 1]), 0.0);   // cond false → take from b
}

#[test]
#[should_panic(expected = "shape mismatch")]
fn test_where_shape_mismatch() {
    let a    = NDArray::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
    let b    = NDArray::new(vec![0.0; 6], vec![2, 3]);
    let cond = NDArray::new(vec![true, false, true, false], vec![2, 2]);
    let _ = a.where_(&cond, &b);
}
