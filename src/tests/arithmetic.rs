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

#[test]
fn test_mul() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let b = a.mul(&a);

    assert_eq!(*b.get(&[0, 0]), 1);
    assert_eq!(*b.get(&[0, 1]), 4);
    assert_eq!(*b.get(&[1, 0]), 9);
    assert_eq!(*b.get(&[1, 1]), 16);
}

#[test]
#[should_panic(expected = "shape mismatch for multiplication: [2, 2] vs [2, 3]")]
fn test_mul_shape_mismatch() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let b = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let _ = a.mul(&b);
}

#[test]
fn test_scale() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let b = a.scale(2);

    assert_eq!(*b.get(&[0, 0]), 2);
    assert_eq!(*b.get(&[0, 1]), 4);
    assert_eq!(*b.get(&[1, 0]), 6);
    assert_eq!(*b.get(&[1, 1]), 8);
}

#[test]
fn test_matmul() {
    let a = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let b = NDArray::new(vec![5, 6, 7, 8], vec![2, 2]);
    let c = a.matmul(&b);

    assert_eq!(*c.get(&[0, 0]), 19); // 1*5 + 2*7
    assert_eq!(*c.get(&[0, 1]), 22); // 1*6 + 2*8
    assert_eq!(*c.get(&[1, 0]), 43); // 3*5 + 4*7
    assert_eq!(*c.get(&[1, 1]), 50); // 3*6 + 4*8
}

#[test]
fn test_matmul_different_shapes() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = NDArray::new(vec![7, 8, 9, 10, 11, 12], vec![3, 2]);
    let c = a.matmul(&b);

    assert_eq!(c.shape, vec![2, 2]);
    // [1 2 3] * [7 8]  = [1*7 + 2*9 + 3*11,  1*8 + 2*10 + 3*12]
    // [4 5 6]   [9 10]   [4*7 + 5*9 + 6*11,  4*8 + 5*10 + 6*12]
    //           [11 12]
    // C[0,0] = 7 + 18 + 33 = 58
    // C[0,1] = 8 + 20 + 36 = 64
    // C[1,0] = 28 + 45 + 66 = 139
    // C[1,1] = 32 + 50 + 72 = 154
    assert_eq!(*c.get(&[0, 0]), 58);
    assert_eq!(*c.get(&[0, 1]), 64);
    assert_eq!(*c.get(&[1, 0]), 139);
    assert_eq!(*c.get(&[1, 1]), 154);
}

#[test]
#[should_panic(expected = "shape mismatch for matmul: self.shape[1] (3) must equal other.shape[0] (2)")]
fn test_matmul_shape_mismatch() {
    let a = NDArray::new(vec![1, 2, 3, 4, 5, 6], vec![2, 3]);
    let b = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let _ = a.matmul(&b);
}

#[test]
#[should_panic(expected = "matmul requires 2D arrays, got 1D and 2D")]
fn test_matmul_not_2d() {
    let a = NDArray::new(vec![1, 2], vec![2]);
    let b = NDArray::new(vec![1, 2, 3, 4], vec![2, 2]);
    let _ = a.matmul(&b);
}

#[test]
fn test_f64_ops() {
    let a = NDArray::new(vec![-2.0, 0.0, 3.0, -1.0], vec![2, 2]);

    let r = a.relu();
    assert_eq!(*r.get(&[0, 0]), 0.0);
    assert_eq!(*r.get(&[0, 1]), 0.0);
    assert_eq!(*r.get(&[1, 0]), 3.0);
    assert_eq!(*r.get(&[1, 1]), 0.0);

    assert_eq!(a.sum(), 0.0);
    assert_eq!(a.mean(), 0.0);
    assert_eq!(a.argmax(), 2);

    let b = NDArray::new(vec![2.0, 1.0, 0.5], vec![3]);
    let sig = b.sigmoid();
    assert!((sig.get(&[0]) - 0.880797).abs() < 1e-5);

    let soft = b.softmax();
    assert!((soft.get(&[0]) - 0.628532).abs() < 1e-5);
    assert!((soft.get(&[1]) - 0.231223).abs() < 1e-5);
    assert!((soft.get(&[2]) - 0.140244).abs() < 1e-5);
    assert!((soft.sum() - 1.0).abs() < 1e-10);
}
