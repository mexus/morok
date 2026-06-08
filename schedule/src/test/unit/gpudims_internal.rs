use super::*;

/// Build a Vec<Arc<UOp>> of concrete `index_const` dims for tests that only
/// exercise the numeric grouping/splitting logic.
fn d(vals: &[usize]) -> Vec<Arc<UOp>> {
    vals.iter().map(|&v| UOp::index_const(v as i64)).collect()
}

/// Extract dim_max from a slice — round-trips numeric-only test inputs back
/// through the sint abstraction.
fn dmax(vs: &[Arc<UOp>]) -> Vec<usize> {
    vs.iter().map(dim_max).collect()
}

#[test]
fn test_group_dims_already_fits() {
    // Dims already fit, no grouping needed.
    let result = group_dims(&d(&[4, 4]), &[16, 16, 16]);
    assert_eq!(dmax(&result.unwrap()), vec![4, 4]);
}

#[test]
fn test_group_dims_needs_grouping() {
    // 4 dims need to be grouped to fit into 3 max_sizes:
    // [4, 4, 4, 4] → [16, 4, 4].
    let result = group_dims(&d(&[4, 4, 4, 4]), &[256, 256, 256]);
    let result = result.unwrap();
    assert!(result.len() <= 3);
    assert_eq!(dmax(&result), vec![16, 4, 4]);
}

#[test]
fn test_group_dims_no_change() {
    // Dims already fit.
    let result = group_dims(&d(&[8, 8, 8]), &[256, 256, 256]);
    assert_eq!(dmax(&result.unwrap()), vec![8, 8, 8]);
}

#[test]
fn test_group_dims_impossible() {
    // Can't fit 1000 into max 10.
    let result = group_dims(&d(&[1000]), &[10]);
    assert!(result.is_none());
}

#[test]
fn test_non_cubic_local_dims_fit_product_cap() {
    // Regression: a 1024-product local shape with per-axis caps = product
    // ([1024;3]) fits [32,2,2] unchanged. The old cube-root cap (10 each)
    // made axis 0 (32) unfittable and panicked at split_dims.
    let result = group_dims(&d(&[32, 2, 2]), &[1024, 1024, 1024]);
    assert_eq!(dmax(&result.unwrap()), vec![32, 2, 2]);
}

#[test]
fn test_split_dims_simple() {
    // 100 exceeds 64, should split.
    let result = split_dims(&d(&[100]), &[64, 64, 64]).unwrap();
    assert!(result.iter().all(|x| dim_max(x) <= 64));
}

#[test]
fn test_split_dims_symbolic_too_big_returns_none() {
    // Symbolic dim with vmax > limit and no concrete factor — must report
    // failure rather than emit a malformed split.
    let v = UOp::define_var("n".to_string(), 0, 200);
    let result = split_dims(&[v], &[64, 64, 64]);
    assert!(result.is_none());
}

#[test]
fn test_find_smallest_divisor() {
    assert_eq!(find_smallest_divisor(1), 1);
    assert_eq!(find_smallest_divisor(2), 2);
    assert_eq!(find_smallest_divisor(3), 1); // prime
    assert_eq!(find_smallest_divisor(4), 2);
    assert_eq!(find_smallest_divisor(9), 3);
    assert_eq!(find_smallest_divisor(100), 2);
}

#[test]
fn test_get_contraction_non_consecutive() {
    // [2, 5, 2] → [10, 2]: dims 0,1 fuse to 10; dim 2 stays as 2.
    // acc_old must hash-cons to match acc_new at positions [1, 2].
    let two = UOp::index_const(2);
    let five = UOp::index_const(5);
    let ten = two.mul(&five); // exactly the limited[0] UOp
    let result = get_contraction(&[two.clone(), five.clone(), two.clone()], &[ten, two.clone()]);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}

#[test]
fn test_get_contraction_identity() {
    // [4, 4, 4] → [4, 4, 4]: no grouping (Arc-identical limited matches acc_old).
    let four = UOp::index_const(4);
    let dims = vec![four.clone(), four.clone(), four.clone()];
    let result = get_contraction(&dims, &dims);
    assert_eq!(result, Some(vec![vec![0], vec![1], vec![2]]));
}

#[test]
fn test_get_contraction_all_fused() {
    // [2, 3, 4] → [24]: all dims fuse to one.
    let two = UOp::index_const(2);
    let three = UOp::index_const(3);
    let four = UOp::index_const(4);
    // 2 * 3 = 6, 6 * 4 = 24 — must use the exact same hash-cons chain.
    let twenty_four = two.mul(&three).mul(&four);
    let result = get_contraction(&[two, three, four], &[twenty_four]);
    assert_eq!(result, Some(vec![vec![0, 1, 2]]));
}

#[test]
fn test_get_contraction_empty() {
    let result = get_contraction(&[], &[]);
    assert_eq!(result, Some(vec![]));
}

#[test]
fn test_get_contraction_invalid() {
    // [2, 3, 4] → [5, 4]: 2*3 = 6 != 5.
    let two = UOp::index_const(2);
    let three = UOp::index_const(3);
    let four = UOp::index_const(4);
    let five = UOp::index_const(5);
    let result = get_contraction(&[two, three, four.clone()], &[five, four]);
    assert_eq!(result, None);
}

#[test]
fn test_get_contraction_partial() {
    // [2, 4, 3] → [8, 3]: dims 0,1 fuse to 8; dim 2 stays as 3.
    let two = UOp::index_const(2);
    let four = UOp::index_const(4);
    let three = UOp::index_const(3);
    let eight = two.mul(&four);
    let result = get_contraction(&[two, four, three.clone()], &[eight, three]);
    assert_eq!(result, Some(vec![vec![0, 1], vec![2]]));
}

#[test]
fn test_group_dims_symbolic_fits_under_vmax() {
    // Symbolic dim with vmax=100 plus 3 concrete dims gets grouped down to 3
    // since 100*4 = 400 ≤ 65535 (typical y-axis cap).
    let v = UOp::define_var("n".to_string(), 0, 100);
    let dims = vec![v.clone(), UOp::index_const(4), UOp::index_const(8), UOp::index_const(8)];
    let result = group_dims(&dims, &[2147483647, 65535, 65535]).unwrap();
    assert_eq!(result.len(), 3);
    // First slot should hold the merged symbolic*4; vmax 400.
    assert_eq!(dim_max(&result[0]), 400);
    assert_eq!(dim_max(&result[1]), 8);
    assert_eq!(dim_max(&result[2]), 8);
}
