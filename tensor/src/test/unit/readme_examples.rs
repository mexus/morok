use ndarray::array;

use crate::Tensor;

/// Tests that mirror README.md Quick Example exactly.
/// If these tests break, the README examples are wrong.

#[test]
fn test_readme_quick_example() {
    let a = Tensor::from_ndarray(&array![[1.0f32, 2.0], [3.0, 4.0]]);
    let b = Tensor::from_ndarray(&array![[5.0f32, 6.0], [7.0, 8.0]]);
    let mut c = &a + &b;
    c.realize().unwrap();
    let result = c.as_ndarray::<f32>().unwrap();
    assert_eq!(result, array![[6.0, 8.0], [10.0, 12.0]].into_dyn());
    let flat = c.as_vec::<f32>().unwrap();
    assert_eq!(flat, vec![6.0, 8.0, 10.0, 12.0]);
}
