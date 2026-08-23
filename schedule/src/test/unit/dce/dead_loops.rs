//! Tests for dead loop / trivial range symbolic simplifications.
//!
//! - `Range(Const)` with `vmin == vmax` folds to `Const(vmin)`.
//! - `Range(_)` with `vmax < 0` folds to `Const(0)` (dead loop).
//!
//! END/REDUCE empty-ranges folds were removed: they conflated trivial-Range
//! Const(0,Index) with dead-Range markers, breaking `Range(Unroll, end=1)`
//! inside REDUCE/END. Downstream `reduce_to_acc` handles dead/empty ranges.

use smallvec::smallvec;
use std::sync::Arc;
use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{AxisId, AxisType, Op, UOp};

use crate::rewrite::graph_rewrite;

use super::helpers::{assert_const_value, get_matcher};

// ----------------------------------------------------------------------------
// RANGE Elimination Tests
// ----------------------------------------------------------------------------

#[test]
fn test_range_zero_to_const() {
    // RANGE(0) → Const(0)
    let zero = UOp::native_const(0i32);
    let range = UOp::range(zero, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_range_negative_to_const() {
    // RANGE(-5) → Const(0)
    let neg_five = UOp::native_const(-5i32);
    let range = UOp::range(neg_five, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_range_symbolic_dead() {
    // size ∈ [0,5], RANGE(size - 10) → Const(0)
    // vmax(size - 10) = 5 - 10 = -5 ≤ 0, so dead
    let size = UOp::variable("size".into(), 0, 5, DType::Int32);
    let ten = UOp::native_const(10i32);
    let count = size.try_sub(&ten).expect("SUB should succeed");
    let range = UOp::new(
        Op::Range { end: count.clone(), axis_id: AxisId::Renumbered(0), axis_type: AxisType::Loop, deps: smallvec![] },
        count.dtype(),
    );

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

#[test]
fn test_range_boundary_vmax_zero() {
    // max(-10, 0) = 0, so RANGE has vmax = 0 (boundary)
    // RANGE(max(-10, 0)) → Const(0)
    let neg_ten = UOp::native_const(-10i32);
    let zero = UOp::native_const(0i32);
    let max_val = neg_ten.try_max(&zero).unwrap();
    let range = UOp::range(max_val, 0);

    let matcher = get_matcher();
    let result = graph_rewrite(&matcher, range, &mut ());

    assert_const_value(&result, ConstValue::Int(0));
}

// ----------------------------------------------------------------------------
// END constructor
// ----------------------------------------------------------------------------

#[test]
fn test_end_empty_ranges_returns_self() {
    // UOp::end(empty) returns self.
    let store = UOp::noop();
    let end = Arc::clone(&store).end(smallvec![]);

    // Empty ranges: end() should return the computation directly
    assert!(Arc::ptr_eq(&end, &store), "end(empty) should return self");
}
