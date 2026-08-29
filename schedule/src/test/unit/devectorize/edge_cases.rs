//! Edge cases and regression tests.
//!
//! Tests for corner cases, boundary conditions, and regressions
//! in the devectorize pass.

use std::sync::Arc;

use smallvec::SmallVec;
use svod_dtype::{AddrSpace, DType, ScalarDType};
use svod_ir::types::ConstValue;
use svod_ir::{Op, UOp};

use super::helpers::*;

#[test]
fn register_reads_merge_only_their_shared_range_ends() {
    let range = UOp::range_const(4, 0);
    let make_register_end = |slot, value| {
        let buffer = UOp::buffer(slot, 1, DType::Int32, AddrSpace::Reg, None);
        let index = UOp::index().buffer(buffer.clone()).indices(vec![UOp::index_const(0)]).call().unwrap();
        let end = index.store(UOp::native_const(value)).end(smallvec::smallvec![range.clone()]);
        let load_index = UOp::index()
            .buffer(buffer.after(smallvec::smallvec![end.clone()]))
            .indices(vec![UOp::index_const(0)])
            .call()
            .unwrap();
        (UOp::load().index(load_index).call(), end)
    };
    let (left, left_end) = make_register_end(0, 1i32);
    let (right, right_end) = make_register_end(1, 2i32);

    let unrelated = create_index(create_buffer_typed(1, ScalarDType::Int32), 0)
        .store(UOp::native_const(3i32))
        .end(smallvec::smallvec![range.clone()]);
    let root = UOp::sink(vec![left, right, unrelated.clone()]);
    let result = crate::devectorize::merge_register_read_ends(root);

    let matching: Vec<_> = result
        .toposort()
        .into_iter()
        .filter(
            |node| matches!(node.op(), Op::End { ranges, .. } if ranges.len() == 1 && Arc::ptr_eq(&ranges[0], &range)),
        )
        .collect();
    assert_eq!(matching.len(), 2);
    assert!(matching.iter().any(|node| Arc::ptr_eq(node, &unrelated)));
    assert!(matching.iter().any(|node| matches!(node.op(), Op::End { computation, .. }
        if matches!(computation.op(), Op::Group { sources } if sources.len() == 2))));
    assert!(!result.toposort().iter().any(|node| Arc::ptr_eq(node, &left_end) || Arc::ptr_eq(node, &right_end)));
}

// =============================================================================
// Scalar Passthrough Tests
// =============================================================================

/// Test: Scalar operations pass through unchanged.
#[test]
fn test_devectorize_scalar_passthrough() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 0);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // Scalar LOAD should pass through
    assert_is_load(&result);
    assert_eq!(result.dtype().vcount(), 1);
}

/// Test: Scalar INDEX passes through.
#[test]
fn test_devectorize_scalar_index_passthrough() {
    let buffer = create_buffer(64);
    let index = create_index(buffer.clone(), 5);

    let result = apply_devectorize(&index);

    // Scalar INDEX should remain unchanged
    assert_is_index(&result);
}

// =============================================================================
// Empty/Trivial Tests
// =============================================================================

/// Test: Empty SINK passes through.
#[test]
fn test_devectorize_empty_sink() {
    let sink = UOp::sink(vec![]);

    let result = apply_devectorize(&sink);

    // Empty SINK should remain as SINK
    match result.op() {
        Op::Sink { sources, .. } => {
            assert!(sources.is_empty());
        }
        other => panic!("Expected SINK, got {:?}", other),
    }
}

/// Devect runs Tinygrad's symbolic_simple tier, which does not flatten SINK.
#[test]
fn test_devectorize_sink_noop() {
    let noop = UOp::noop();
    let sink = UOp::sink(vec![noop]);

    let result = apply_devectorize(&sink);

    match result.op() {
        Op::Sink { sources, .. } => {
            assert_eq!(sources.len(), 1, "devectorize must not run the larger sym cleanup tier");
            assert!(matches!(sources[0].op(), Op::Noop));
        }
        Op::Noop => {}
        other => panic!("Expected empty SINK or Noop, got {:?}", other),
    }
}

// =============================================================================
// Precision Tests
// =============================================================================

/// Test: Half precision (f16) buffer handling.
#[test]
fn test_devectorize_half_precision() {
    let buffer = create_buffer_typed(64, ScalarDType::Float16);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // Should preserve f16 dtype and produce 4 elements total
    assert_eq!(result.dtype().base(), ScalarDType::Float16, "Base dtype should be f16");
    assert_vcount(&result, 4);
    assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
}

/// Test: Int8 buffer handling.
#[test]
fn test_devectorize_int8() {
    let buffer = create_buffer_typed(64, ScalarDType::Int8);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().base(), ScalarDType::Int8, "Base dtype should be i8");
    assert_vcount(&result, 4);
}

/// Test: UInt8 buffer handling.
#[test]
fn test_devectorize_uint8() {
    let buffer = create_buffer_typed(64, ScalarDType::UInt8);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    assert_eq!(result.dtype().base(), ScalarDType::UInt8, "Base dtype should be u8");
    assert_vcount(&result, 4);
}

// =============================================================================
// Mixed Dtype Tests
// =============================================================================

/// Test: Multiple dtypes in same kernel.
#[test]
fn test_devectorize_mixed_dtypes() {
    let buffer_f32 = create_buffer_typed(64, ScalarDType::Float32);
    let buffer_i32 = create_buffer_typed(64, ScalarDType::Int32);

    // Load f32
    let index_f32 = create_vector_index_iota(buffer_f32.clone(), 4);
    let load_f32 = UOp::load().index(index_f32).call();

    // Load i32
    let index_i32 = create_vector_index_iota(buffer_i32.clone(), 4);
    let load_i32 = UOp::load().index(index_i32).call();

    // Process both
    let result_f32 = apply_devectorize(&load_f32);
    let result_i32 = apply_devectorize(&load_i32);

    assert_eq!(result_f32.dtype().base(), ScalarDType::Float32);
    assert_eq!(result_i32.dtype().base(), ScalarDType::Int32);
}

// =============================================================================
// Address Space Tests
// =============================================================================

/// Test: Local (shared) memory handling.
#[test]
fn test_devectorize_local_memory() {
    let buffer = create_buffer_local(64, ScalarDType::Float32);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // Should work with local memory, preserving vcount and dtype
    assert_vcount(&result, 4);
    assert_eq!(result.dtype().base(), ScalarDType::Float32, "Base dtype should be f32");
    assert!(count_loads(&result) >= 1, "Should have at least one LOAD");
}

/// Test: Different address spaces in same kernel.
#[test]
fn test_devectorize_mixed_addrspaces() {
    let buffer_global = create_buffer(64);
    let buffer_local = create_buffer_local(64, ScalarDType::Float32);

    let index_global = create_vector_index_iota(buffer_global.clone(), 4);
    let load_global = UOp::load().index(index_global).call();

    let index_local = create_vector_index_iota(buffer_local.clone(), 4);
    let load_local = UOp::load().index(index_local).call();

    let result_global = apply_devectorize(&load_global);
    let result_local = apply_devectorize(&load_local);

    // Both should produce valid results with preserved vcount
    assert_vcount(&result_global, 4);
    assert_vcount(&result_local, 4);
    assert!(count_loads(&result_global) >= 1, "Global should have LOADs");
    assert!(count_loads(&result_local) >= 1, "Local should have LOADs");
}

// =============================================================================
// Large Vector Tests
// =============================================================================

/// Test: Very large vector (vec64).
#[test]
fn test_devectorize_very_large_vector() {
    let buffer = create_buffer(512);

    // Create codegen PARAM to match the stacked INDEX rule.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(10000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 512, buffer.dtype(), None);

    // Create vec64 index
    let indices: SmallVec<[Arc<UOp>; 4]> = (0..64).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::stack(indices);

    let index = UOp::new(Op::Index { buffer: define, indices: smallvec::smallvec![vec_idx] }, DType::Float32);

    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // Should handle vec64 (will be split into smaller chunks)
    assert_vcount(&result, 64);
    let load_count = count_loads(&result);
    assert!(load_count >= 1, "Should have LOADs");
}

/// Test: Vec32 access.
#[test]
fn test_devectorize_vec32() {
    let buffer = create_buffer(256);

    // Create codegen PARAM to match the stacked INDEX rule.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(11000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 256, buffer.dtype(), None);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..32).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::stack(indices);

    let index = UOp::new(Op::Index { buffer: define, indices: smallvec::smallvec![vec_idx] }, DType::Float32);

    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    assert_vcount(&result, 32);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Unaligned Access Tests
// =============================================================================

/// Test: Non-power-of-2 offset.
#[test]
fn test_devectorize_unaligned_access() {
    let buffer = create_buffer(64);

    // Index starting at 3 (not aligned to 4)
    let index = create_vector_index_offset(buffer.clone(), 4, 3);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // Should still produce valid result with preserved vcount
    assert_vcount(&result, 4);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Odd vector size (vec3).
#[test]
fn test_devectorize_vec3() {
    let buffer = create_buffer(64);

    // Create codegen PARAM to match the stacked INDEX rule.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(12000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 64, buffer.dtype(), None);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..3).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::stack(indices);

    let index = UOp::new(Op::Index { buffer: define, indices: smallvec::smallvec![vec_idx] }, DType::Float32);

    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // vec3 should be handled (split into smaller pieces)
    assert_vcount(&result, 3);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Vec5 (non-power-of-2, larger than 4).
#[test]
fn test_devectorize_vec5() {
    let buffer = create_buffer(64);

    // Create codegen PARAM to match the stacked INDEX rule.
    static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(13000);
    let def_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let define = UOp::param(def_id, 64, buffer.dtype(), None);

    let indices: SmallVec<[Arc<UOp>; 4]> = (0..5).map(|i| UOp::const_(DType::Index, ConstValue::Int(i))).collect();
    let vec_idx = UOp::stack(indices);

    let index = UOp::new(Op::Index { buffer: define, indices: smallvec::smallvec![vec_idx] }, DType::Float32);

    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    // vec5 = 4 + 1, should be split but total vcount preserved
    assert_vcount(&result, 5);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Constants and Special Values
// =============================================================================

/// Test: Zero index.
#[test]
fn test_devectorize_zero_index() {
    let buffer = create_buffer(64);
    let index = create_vector_index_offset(buffer.clone(), 4, 0);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    assert_vcount(&result, 4);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

/// Test: Large constant offset.
#[test]
fn test_devectorize_large_offset() {
    let buffer = create_buffer(10000);
    let index = create_vector_index_offset(buffer.clone(), 4, 9000);
    let load = UOp::load().index(index).call();

    let result = apply_devectorize(&load);

    assert_vcount(&result, 4);
    assert!(count_loads(&result) >= 1, "Should have LOADs");
}

// =============================================================================
// Idempotency Tests
// =============================================================================

/// Test: Applying devectorize twice produces same result.
#[test]
fn test_devectorize_idempotent() {
    let buffer = create_buffer(64);
    let index = create_vector_index_iota(buffer.clone(), 4);
    let load = UOp::load().index(index).call();

    let result1 = apply_devectorize(&load);
    let result2 = apply_devectorize(&result1);

    // Second application should not change the result
    // (structure should be equivalent even if IDs differ)
    assert_eq!(result1.dtype(), result2.dtype());
    assert_eq!(count_loads(&result1), count_loads(&result2));
}

// =============================================================================
// Regression Tests
// =============================================================================

/// Regression: shaped INDEX positions should be preserved.
#[test]
fn test_regression_index_positions_preserved() {
    let vec = create_vector_float_iota(8);
    let indexed = vec.index_axes(vec![1, 3, 5, 7]);
    let Op::Index { indices, .. } = indexed.op() else { panic!("expected INDEX") };
    let Op::Stack { sources } = indices[0].op() else { panic!("expected shaped index") };
    assert_eq!(
        sources
            .iter()
            .map(|source| match source.op() {
                Op::Const(value) => match value.0 {
                    ConstValue::Int(value) => value,
                    ConstValue::UInt(value) => value as i64,
                    _ => panic!("expected integer position"),
                },
                _ => panic!("expected constant position"),
            })
            .collect::<Vec<_>>(),
        vec![1, 3, 5, 7]
    );
}

// =============================================================================
// fold_expanded_index: contiguous grouping
// =============================================================================

// =============================================================================
// ScatterND regression
// =============================================================================

/// A zero-sized trailing dimension has no elements to chunk: reject instead of panicking.
#[test_case::test_case(0, &[4, 0], false; "trailing zero dim")]
#[test_case::test_case(4, &[2, 2], true; "square")]
#[test_case::test_case(4, &[4, 0], false; "zero dim with elements")]
#[test_case::test_case(3, &[2, 2], false; "not divisible")]
fn stack_with_shape_rejects_unchunkable_shapes(count: usize, dims: &[usize], expected: bool) {
    let elements: Vec<_> = (0..count).map(|i| UOp::native_const(i as i32)).collect();
    let shape: Vec<_> = dims.iter().map(|&d| svod_ir::SInt::Const(d)).collect();
    assert_eq!(crate::devectorize::stack_with_shape(elements, &shape).is_some(), expected);
}
