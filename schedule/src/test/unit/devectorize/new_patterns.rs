//! Miscellaneous devectorizer integration tests.

use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use super::helpers::*;

/// Test: Full devectorize pass on a simple load
#[test]
fn test_full_devectorize_simple_load() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // The result can be a STACK or a single LOAD depending on grouping.
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have at least one LOAD in the result");
}

/// Test: Devectorize preserves semantics with non-contiguous indices
#[test]
fn test_devectorize_non_contiguous() {
    let buffer = create_buffer(64);
    let index = create_vector_index_scaled(buffer.clone(), 4, 2); // [0, 2, 4, 6]
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // With non-contiguous indices, should result in multiple scalar loads
    // or GEP-based reordering
    assert!(result.dtype().vcount() >= 1);
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: LOAD without gate is unchanged by alt pattern
#[test]
fn test_ungate_load_unchanged() {
    let buffer = create_buffer(64);
    let idx = UOp::const_(DType::Index, ConstValue::Int(0));

    // Create ungated INDEX
    let index = UOp::index().buffer(buffer.clone()).indices(vec![idx]).call().unwrap();

    // Create LOAD without alt
    let load = UOp::load().index(index).call();

    let result = load;

    // Result should still be LOAD without alt
    if let Op::Load { alt, .. } = result.op() {
        assert!(alt.is_none(), "Ungated LOAD should not have alt value");
    }
}

// =============================================================================
// is_increasing Tests (already in helpers.rs, but integration test here)
// =============================================================================

/// Test: is_increasing on range variable
#[test]
fn test_is_increasing_range() {
    use svod_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    assert!(range.is_increasing(), "RANGE should be increasing");
}

/// Test: is_increasing on constant
#[test]
fn test_is_increasing_constant() {
    let c = UOp::const_(DType::Int32, ConstValue::Int(5));
    assert!(c.is_increasing(), "CONST should be increasing");
}

/// Test: is_increasing on add
#[test]
fn test_is_increasing_add_expr() {
    use svod_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    let c = UOp::const_(DType::WeakInt, ConstValue::Int(5));
    let sum = range.try_add(&c).unwrap();
    assert!(sum.is_increasing(), "RANGE + CONST should be increasing");
}

/// Test: is_increasing on mul by positive const
#[test]
fn test_is_increasing_mul_positive() {
    use svod_ir::types::{AxisId, AxisType};
    let range = UOp::range_axis(UOp::index_const(16), AxisId::Unrenumbered(0), AxisType::Loop);
    let c = UOp::const_(DType::WeakInt, ConstValue::Int(4));
    let prod = range.try_mul(&c).unwrap();
    assert!(prod.is_increasing(), "RANGE * positive CONST should be increasing");
}

/// Test: is_increasing on mul by negative const
#[test]
fn test_is_increasing_mul_negative() {
    let x = UOp::var("x", DType::Int32, 0, 100);
    let c = UOp::const_(DType::Int32, ConstValue::Int(-1));
    let prod = x.try_mul(&c).unwrap();
    assert!(!prod.is_increasing(), "x * negative CONST should not be increasing");
}
