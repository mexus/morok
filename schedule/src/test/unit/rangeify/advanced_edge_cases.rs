//! Advanced edge case tests for rangeify.
//!
//! Tests for IR-level edge cases that aren't covered by basic tests:
//! - Symbolic (variable-sized) ranges
//! - Nested STAGE operations
//! - Multi-consumer patterns
//! - Complex indexing scenarios

use svod_ir::{DType, Op, UOp};

use crate::rangeify::transforms::rangeify;

use super::helpers::{create_bufferize, create_const, create_range, create_range_symbolic};

// ============================================================================
// Symbolic Range Size Tests
// ============================================================================

#[test]
fn test_symbolic_range_size() {
    // Test STAGE with symbolic (variable) range size
    // This tests that rangeify doesn't crash on non-constant range sizes

    let size_var = UOp::var("size", DType::Index, 0, 1024);
    let compute = UOp::native_const(1.0f32);

    // Create range with symbolic size
    let range = create_range_symbolic(size_var, 0);
    let bufferized = create_bufferize(compute, vec![range]);

    // Symbolic ranges work correctly and create kernels
    let (_result, _ctx) = rangeify(bufferized).unwrap();

    // Note: Dead-axis optimization now works for provably-dead symbolic ranges
    // (uses vmax analysis - see test_is_dead_axis_symbolic_bounded test)
}

#[test]
fn test_symbolic_range_multiple() {
    // Test multiple symbolic ranges
    let size1 = UOp::var("size1", DType::Index, 0, 1024);
    let size2 = UOp::var("size2", DType::Index, 0, 1024);

    let compute = UOp::native_const(2.0f32);

    let range1 = create_range_symbolic(size1, 0);
    let range2 = create_range_symbolic(size2, 1);

    let bufferized = create_bufferize(compute.clone(), vec![range1, range2]);

    // Symbolic ranges work correctly with multiple dimensions
    let (_result, _ctx) = rangeify(bufferized).unwrap();

    // Note: Dead-axis optimization is skipped for symbolic ranges
    // TODO: Enhance dead-axis detection to handle provably-dead symbolic ranges
}

#[test]
fn test_symbolic_range_with_arithmetic() {
    // Test symbolic range size with arithmetic expression
    let n = UOp::variable("n".into(), 0, 512, DType::Int32);
    let size = n.try_mul(&create_const(2)).unwrap();

    let compute = UOp::native_const(3.0f32);
    let range = create_range_symbolic(size, 0);
    let bufferized = create_bufferize(compute, vec![range]);

    // Symbolic arithmetic expressions work correctly as range sizes
    let (_result, _ctx) = rangeify(bufferized).unwrap();

    // Note: Dead-axis optimization is skipped for symbolic ranges
    // TODO: Enhance dead-axis detection to handle provably-dead symbolic ranges
}

// ============================================================================
// Nested STAGE Tests
// ============================================================================

#[test]
fn test_nested_bufferize_different_ranges() {
    // Test STAGE(STAGE(x, R1), R2) where R1 != R2
    // This tests multi-level buffering with different iteration spaces

    let inner_compute = UOp::native_const(1.0f32);

    // Inner stage with range [0, 10)
    let inner_range = create_range(10, 0);
    let inner_buf = create_bufferize(inner_compute, vec![inner_range]);

    // Outer stage with different range [0, 20)
    let outer_range = create_range(20, 1);
    let outer_buf = create_bufferize(inner_buf, vec![outer_range]);

    // Should handle nested bufferization without crashing
    let (_result, _ctx) = rangeify(outer_buf).unwrap();

    // Note: Tests robustness - nested STAGE operations should be handled gracefully
}

#[test]
fn test_deeply_nested_bufferize() {
    // Test 3-level nesting: STAGE(STAGE(STAGE(x)))
    let compute = UOp::native_const(1.0f32);

    let r1 = create_range(5, 0);
    let buf1 = create_bufferize(compute, vec![r1]);

    let r2 = create_range(10, 1);
    let buf2 = create_bufferize(buf1, vec![r2]);

    let r3 = create_range(15, 2);
    let buf3 = create_bufferize(buf2, vec![r3]);

    // Should handle deep nesting without crashing
    let (_result, _ctx) = rangeify(buf3).unwrap();

    // Note: Tests that deeply nested STAGE operations don't cause stack overflow or panics
}

// ============================================================================
// Multi-Consumer Pattern Tests
// ============================================================================

#[test]
fn test_bufferize_multiple_consumers() {
    use svod_ir::SInt;
    use svod_ir::shape::Shape;

    // Test single STAGE with multiple consumers
    // Pattern: buf = STAGE(x); y = f(buf); z = g(buf)

    let compute = UOp::native_const(1.0f32);
    let range = create_range(10, 0);
    let buf = create_bufferize(compute, vec![range]);

    // Get STAGE shape and broadcast constants to match
    // STAGE now has shape [10], so we need to reshape [] -> [1] -> expand [10]
    let buf_shape = buf.shape().unwrap().unwrap();
    let ones_shape: Shape = buf_shape.iter().map(|_| SInt::Const(1)).collect();

    // Two independent consumers of the same buffer
    let const2 = UOp::native_const(2.0f32).try_reshape(&ones_shape).unwrap().try_expand(buf_shape).unwrap();
    let consumer1 = buf.try_add(&const2).unwrap();

    let const3 = UOp::native_const(3.0f32).try_reshape(&ones_shape).unwrap().try_expand(buf_shape).unwrap();
    let consumer2 = buf.try_mul(&const3).unwrap();

    // Combine consumers with SINK
    let sink = UOp::sink(vec![consumer1, consumer2]);

    // Should handle multi-consumer pattern without crashing
    let (_result, _ctx) = rangeify(sink).unwrap();

    // Note: Tests that multiple consumers of the same STAGE don't cause issues
}

#[test]
fn test_operation_with_multiple_uses() {
    // Test intermediate operation used multiple times
    // Pattern: x = CONST; buf1 = STAGE(x); buf2 = STAGE(x)

    let compute = UOp::native_const(1.0f32);

    let r1 = create_range(10, 0);
    let buf1 = create_bufferize(compute.clone(), vec![r1]);

    let r2 = create_range(20, 1);
    let buf2 = create_bufferize(compute.clone(), vec![r2]);

    // Both stage the same compute
    let sink = UOp::sink(vec![buf1, buf2]);

    // Should handle same operation bufferized with different ranges
    let (_result, _ctx) = rangeify(sink).unwrap();

    // Note: Tests that same compute can be buffered with different iteration spaces
}

// ============================================================================
// Complex Indexing Tests
// ============================================================================

#[test]
fn test_index_with_multiple_ranges() {
    // Test INDEX operation with multiple range dimensions
    let compute = UOp::native_const(1.0f32);
    let r1 = create_range(10, 0);
    let r2 = create_range(20, 1);
    let r3 = create_range(5, 2);

    let bufferized = create_bufferize(compute, vec![r1.clone(), r2.clone(), r3.clone()]);

    // Create INDEX with all three ranges
    let index_op = UOp::new(Op::Index { buffer: bufferized.clone(), indices: vec![r1, r2, r3].into() }, DType::Float32);

    let (_result, _ctx) = rangeify(index_op).unwrap();
}

#[test]
fn test_range_size_mismatch() {
    // Test STAGE with mixed constant and symbolic range sizes
    let const_range = create_range(10, 0);
    let sym_size = UOp::param(0, 1, DType::Index, None);
    let sym_range = create_range_symbolic(sym_size, 1);

    let compute = UOp::native_const(1.0f32);
    let bufferized = create_bufferize(compute, vec![const_range, sym_range]);

    // Mixed constant and symbolic ranges work correctly
    let (_result, _ctx) = rangeify(bufferized).unwrap();
}

// ============================================================================
// Dead Axis Detection Tests (is_dead_axis with vmax analysis)
// ============================================================================

#[test]
fn test_is_dead_axis_constant_ranges() {
    use crate::rangeify::indexing::is_dead_axis;

    // Dead: RANGE(0) - vmax = -1
    let range_0 = create_range(0, 0);
    assert!(is_dead_axis(&range_0));

    // Dead: RANGE(1) - vmax = 0
    let range_1 = create_range(1, 0);
    assert!(is_dead_axis(&range_1));

    // Live: RANGE(2) - vmax = 1
    let range_2 = create_range(2, 0);
    assert!(!is_dead_axis(&range_2));

    // Live: RANGE(10) - vmax = 9
    let range_10 = create_range(10, 0);
    assert!(!is_dead_axis(&range_10));
}

#[test]
fn test_is_dead_axis_non_range() {
    use crate::rangeify::indexing::is_dead_axis;

    // Non-RANGE operations should return false
    let const_op = UOp::index_const(0);
    assert!(!is_dead_axis(&const_op));

    let add_op = const_op.try_add(&const_op).unwrap();
    assert!(!is_dead_axis(&add_op));
}
