//! Tests for range flattening and canonicalization.
//!
//! Validates that flatten_range canonicalizes ended RANGE dependencies.

use std::sync::Arc;

use svod_ir::UOp;

use crate::rangeify::transforms::{flatten_range_impl, flatten_ranges};

#[test]
fn test_flatten_range_impl_non_supported_op() {
    // Operations that don't support flattening should return None
    let const_op = UOp::native_const(1.0f32);

    let result = flatten_range_impl(&const_op);
    assert!(result.is_none());
}

#[test]
fn test_flatten_range_impl_no_ranges() {
    // STORE operation with no ranges should return None
    let index = UOp::index_const(0);
    let value = UOp::native_const(1.0f32);
    let store = index.store(value);

    let result = flatten_range_impl(&store);
    assert!(result.is_none());
}

#[test]
fn test_flatten_ranges_identity() {
    // Graph with no nested ranges should return unchanged
    let computation = UOp::native_const(1.0f32);
    let flattened = flatten_ranges(&computation);

    // Should return identical graph (same pointer)
    assert!(Arc::ptr_eq(&flattened, &computation));
}

// ===== Nesting Tests =====

#[test]
fn test_flatten_range_does_not_unwrap_nested_end_counterexample() {
    // Tinygrad only flattens the explicit ended-range sources, not computation ENDs.
    use smallvec::smallvec;
    use svod_ir::Op;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    // Create nested END: END(END(computation, [r1]), [r2])
    let inner_end = computation.clone().end(smallvec![r1.clone()]);
    let outer_end = inner_end.end(smallvec![r2.clone()]);

    let flattened = flatten_range_impl(&outer_end);
    assert!(flattened.is_none());
    assert!(
        matches!(outer_end.op(), Op::End { computation, ranges } if Arc::ptr_eq(computation, &inner_end) && ranges.len() == 1)
    );
}

#[test]
fn test_flatten_range_does_not_unwrap_deeply_nested_end_counterexample() {
    use smallvec::smallvec;
    use svod_ir::Op;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);
    let r3 = UOp::range(UOp::index_const(30), 2);

    // Create 3-level nesting
    let end1 = computation.clone().end(smallvec![r1.clone()]);
    let end2 = end1.end(smallvec![r2.clone()]);
    let end3 = end2.end(smallvec![r3.clone()]);

    assert!(flatten_range_impl(&end3).is_none());
    assert!(matches!(end3.op(), Op::End { ranges, .. } if ranges.len() == 1));
}

#[test]
fn test_flatten_range_flattens_explicit_range_expression_and_preserves_computation() {
    use smallvec::smallvec;
    use svod_ir::Op;

    // Create a binary computation: 1.0 + 2.0
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();

    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    let combined_range = r1.add(&r2);
    let outer_end = add.clone().end(smallvec![combined_range]);

    let flattened = flatten_range_impl(&outer_end);

    assert!(flattened.is_some());
    let flattened = flattened.unwrap();

    if let Op::End { computation, ranges } = flattened.op() {
        assert!(Arc::ptr_eq(computation, &add));
        assert_eq!(ranges.len(), 2);
    } else {
        panic!("Expected END operation");
    }
}

#[test]
fn test_flatten_ranges_nested_end_graph_is_identity() {
    use smallvec::smallvec;
    use svod_ir::Op;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);
    let r2 = UOp::range(UOp::index_const(20), 1);

    // Create nested structure
    let inner_end = computation.clone().end(smallvec![r1.clone()]);
    let outer_end = inner_end.end(smallvec![r2.clone()]);

    let flattened = flatten_ranges(&outer_end);
    assert!(Arc::ptr_eq(&flattened, &outer_end));
    assert!(matches!(flattened.op(), Op::End { .. }));
}

#[test]
fn test_flatten_range_single_range() {
    // END with single range that's already flat returns None (no change needed)
    // This is important for the rewrite engine to avoid infinite loops
    use smallvec::smallvec;

    let computation = UOp::native_const(1.0f32);
    let r1 = UOp::range(UOp::index_const(10), 0);

    let end = computation.clone().end(smallvec![r1.clone()]);

    let flattened = flatten_range_impl(&end);

    // Should return None because nothing changed (single range, already flat)
    // Returning Some with unchanged value would cause infinite loops in rewrite engine
    assert!(flattened.is_none());
}
