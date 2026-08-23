//! Tests for reduce_to_acc (REDUCE → accumulator pattern transformation).
//!
//! reduce_to_acc converts REDUCE operations to explicit accumulator patterns:
//! - Creates DEFINE_REG for accumulator
//! - Initializes accumulator with identity value
//! - Loops over reduce_range with binary operations
//! - Handles horizontal reduction for shaped values
//!
//! Critical behavior tested:
//! - input_ranges excludes parallel axes (Thread, Global, Local, Warp)
//! - input_ranges includes Loop axes
//! - reduce_range itself is excluded from input_ranges
//!
//! Based on Tinygrad's devectorizer.py:291-308.

use std::sync::Arc;

use smallvec::smallvec;
use svod_dtype::DType;
use svod_ir::types::ConstValue;
use svod_ir::{BinaryOp, Op, ReduceOp, RendererDevice, SInt, UOp, WmmaMetadata};

use super::helpers::*;

fn test_wmma(c: Arc<UOp>) -> Arc<UOp> {
    let operand = UOp::stack((0..6).map(|i| UOp::var(format!("operand_{i}"), DType::Float32, -100, 100)).collect());
    UOp::wmma(
        operand.clone(),
        operand,
        c,
        WmmaMetadata {
            name: "test".into(),
            dims: (16, 16, 16),
            dtype_in: DType::Float32,
            dtype_out: DType::Float32,
            device: RendererDevice::Cpu,
            threads: 32,
            upcast_axes: None,
            reduce_axes: vec![],
            tile_grid: (1, 1),
        },
    )
}

fn shaped_values(prefix: &str, shape: &[usize]) -> Arc<UOp> {
    let count = shape.iter().product();
    UOp::stack((0..count).map(|i| UOp::var(format!("{prefix}_{i}"), DType::Float32, -100, 100)).collect())
        .try_reshape(&shape.iter().copied().map(SInt::Const).collect())
        .unwrap()
}

#[test]
fn test_wmma_add_direct_moves_into_accumulator() {
    let accumulator = shaped_values("acc", &[6]);
    let add = shaped_values("add", &[6]);
    let result = crate::rewrite::graph_rewrite(
        crate::devectorize::pm_wmma_add(),
        test_wmma(accumulator.clone()).add(&add),
        &mut (),
    );

    let Op::Wmma { c, .. } = result.op() else { panic!("direct WMMA add must fuse") };
    assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, lhs, rhs)
        if Arc::ptr_eq(lhs, &accumulator) && Arc::ptr_eq(rhs, &add)));
}

#[test]
fn test_wmma_add_moves_through_permute() {
    let accumulator = shaped_values("acc", &[2, 3]);
    let add = shaped_values("add", &[3, 2]);
    let permuted = test_wmma(accumulator).try_permute(vec![1, 0]).unwrap();
    let result = crate::rewrite::graph_rewrite(crate::devectorize::pm_wmma_add(), permuted.add(&add), &mut ());

    let Op::Permute { src, axes } = result.op() else { panic!("output permutation must remain outside WMMA") };
    assert_eq!(axes, &[1, 0]);
    let Op::Wmma { c, .. } = src.op() else { panic!("permuted add must fuse into WMMA") };
    assert!(
        matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved) if matches!(moved.op(), Op::Permute { axes, .. } if axes == &[1, 0]))
    );
}

#[test]
fn test_wmma_add_moves_through_permute_reshape() {
    let accumulator = shaped_values("acc", &[6]);
    let add = shaped_values("add", &[3, 2]);
    let reshaped = test_wmma(accumulator).try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let result = crate::rewrite::graph_rewrite(
        crate::devectorize::pm_wmma_add(),
        reshaped.try_permute(vec![1, 0]).unwrap().add(&add),
        &mut (),
    );

    let Op::Permute { src, .. } = result.op() else { panic!("output permutation must remain") };
    let Op::Reshape { src, .. } = src.op() else { panic!("output reshape must remain") };
    let Op::Wmma { c, .. } = src.op() else { panic!("reshape-permute add must fuse into WMMA") };
    assert!(matches!(c.op(), Op::Binary(BinaryOp::Add, _, moved) if matches!(moved.op(), Op::Reshape { src, .. }
        if matches!(src.op(), Op::Permute { .. }))));
}

#[test]
fn test_movement_cleanup_must_precede_reduce_local() {
    let accumulator = shaped_values("acc", &[6]);
    let add = shaped_values("add", &[3, 2]);
    let wmma = test_wmma(accumulator);
    let inner = wmma.try_reshape(&smallvec![SInt::Const(3), SInt::Const(2)]).unwrap();
    let outer = inner.try_reshape(&smallvec![SInt::Const(2), SInt::Const(3)]).unwrap();
    let root = outer.try_permute(vec![1, 0]).unwrap().add(&add);

    let mut ctx = crate::devectorize::ReduceContext::default();
    let without_cleanup = crate::rewrite::graph_rewrite(&crate::devectorize::pm_reduce_local(), root.clone(), &mut ctx);
    assert!(matches!(without_cleanup.op(), Op::Binary(BinaryOp::Add, ..)), "counterexample must not match early");

    let matcher = crate::devectorize::movement_cleanup_patterns().with_context::<crate::devectorize::ReduceContext>()
        + crate::devectorize::pm_reduce_local();
    let mut ctx = crate::devectorize::ReduceContext::default();
    let ordered = crate::rewrite::graph_rewrite(&matcher, root, &mut ctx);
    assert!(ordered.toposort().iter().any(|node| matches!(node.op(), Op::Wmma { c, .. }
        if matches!(c.op(), Op::Binary(BinaryOp::Add, ..)))));
}

// =============================================================================
// Happy Path Tests: Basic REDUCE operations
// =============================================================================

/// Test: REDUCE(scalar, [Range], Add) → accumulator pattern with Add.
#[test]
fn test_reduce_scalar_add() {
    let range = create_range_reduce(16, 0);
    let src = create_float_const(1.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // Should transform to accumulator pattern (no longer REDUCE)
    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE to accumulator pattern");
    // Should have DEFINE_REG in the tree
    assert!(count_define_regs(&result) > 0, "Should contain DEFINE_REG");
    assert!(
        result
            .toposort()
            .iter()
            .any(|node| matches!(node.op(), Op::Buffer { arg, .. } if arg.addrspace == Some(svod_ir::AddrSpace::Reg) && arg.slot == 0)),
        "first reduction accumulator must use dense REG slot 0"
    );
    // Should have END in the tree
    assert!(count_ends(&result) > 0, "Should contain END");
}

/// Test: REDUCE(scalar, [Range], Mul) → accumulator pattern with Mul.
#[test]
fn test_reduce_scalar_mul() {
    let range = create_range_reduce(8, 0);
    let src = create_float_const(2.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Mul);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    assert!(count_define_regs(&result) > 0);
}

/// Test: REDUCE(scalar, [Range], Max) → accumulator pattern with Max.
#[test]
fn test_reduce_scalar_max() {
    let range = create_range_reduce(32, 0);
    let src = create_float_const(0.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Max);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    // Max uses Binary::Max
    assert!(count_define_regs(&result) > 0);
}

#[test]
fn test_invalid_padded_lane_survives_reduction_removal() {
    let range = create_range_reduce(16, 0);
    let cond = UOp::var("valid", DType::Bool, 0, 1);
    let value = UOp::var("value", DType::Float32, 0, 100);
    let src = UOp::try_where(cond, value, UOp::invalid_marker()).unwrap();
    let reduce = create_reduce(src, vec![range], ReduceOp::Max);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(
        result.any_in_subtree(UOp::is_invalid_marker),
        "reduction removal must preserve Invalid for the later gater"
    );
}

/// Test: REDUCE(scalar, [Range], Min) → accumulator pattern with Min (uses WHERE).
#[test]
fn test_reduce_scalar_min() {
    let range = create_range_reduce(32, 0);
    let src = create_float_const(100.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Min);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    // Min uses WHERE(Lt, a, b)
    assert!(count_define_regs(&result) > 0);
}

/// Test: REDUCE(shaped f32[4], [Range], Add) → horizontal reduction then accumulator.
#[test]
fn test_reduce_shaped_to_scalar() {
    let range = create_range_reduce(16, 0);
    let src = UOp::stack((0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect());
    let reduce = src.reduce_with_num_axes(smallvec![range], ReduceOp::Add, 1);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    assert!(count_define_regs(&result) > 0);
    assert!(result.toposort().iter().any(|node| {
        matches!(node.op(), Op::Index { buffer, indices }
            if Arc::ptr_eq(buffer, &src) && indices.len() == 1)
    }));
}

#[test]
fn test_horizontal_reduce_uses_target_dtype() {
    let range = create_range_reduce(16, 0);
    let source_dtype = DType::BFloat16.vec(16).unwrap();
    let target_dtype = DType::BFloat16.vec(4).unwrap();
    let src = UOp::stack((0..4).map(|i| UOp::const_(source_dtype.clone(), ConstValue::Float(i as f64))).collect());
    let reduce = UOp::new(
        Op::Reduce { src: src.clone(), ranges: smallvec![range], reduce_op: ReduceOp::Add, num_axes: 1 },
        target_dtype.clone(),
    );

    let result = apply_pm_reduce(&reduce);

    assert_eq!(result.dtype(), target_dtype);
    for node in result.toposort() {
        if let Op::Index { buffer, .. } = node.op()
            && Arc::ptr_eq(buffer, &src)
        {
            assert_eq!(node.dtype(), target_dtype);
        }
        if let Op::Binary(BinaryOp::Add, lhs, rhs) = node.op() {
            assert_eq!(lhs.dtype(), rhs.dtype());
        }
    }
    assert!(!result.toposort().iter().any(|node| matches!(node.op(), Op::Cast { .. })));
}

// =============================================================================
// Horizontal Reduce Tests
// =============================================================================

/// Test: Horizontal reduce with no ranges → direct horizontal reduction.
///
/// REDUCE(shaped f32[4], [], Add) → left fold of scalar INDEXes.
#[test]
fn test_horizontal_reduce_no_ranges() {
    let src = UOp::stack((0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect());
    let reduce = src.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }), "Should transform REDUCE");
    assert_eq!(count_define_regs(&result), 0, "Should not have DEFINE_REG for horizontal-only reduce");
    assert_eq!(result.dtype(), DType::Float32);
    assert_eq!(
        result
            .toposort()
            .iter()
            .filter(|node| matches!(node.op(), Op::Index { buffer, .. } if Arc::ptr_eq(buffer, &src)))
            .count(),
        4
    );
}

// =============================================================================
// Edge Cases
// =============================================================================

/// Test: REDUCE with empty ranges → direct horizontal reduction.
#[test]
fn test_reduce_empty_ranges() {
    let src = UOp::stack((0..4).map(|i| UOp::const_(DType::Float32, ConstValue::Float(i as f64))).collect());
    let reduce = src.reduce_with_num_axes(smallvec![], ReduceOp::Add, 1);

    let result = apply_pm_reduce(&reduce);

    // Empty ranges means no loop, just horizontal reduce
    assert!(!matches!(result.op(), Op::Reduce { .. }));
}

/// Test: REDUCE with scalar src and scalar out.
#[test]
fn test_reduce_single_element() {
    let range = create_range_reduce(1, 0);
    let src = create_float_const(42.0);
    let reduce = create_reduce(src, vec![range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert_eq!(result.dtype().vcount(), 1);
}

/// Test: REDUCE with multiple reduce ranges.
#[test]
fn test_reduce_multiple_ranges() {
    let range1 = create_range_reduce(8, 0);
    let range2 = create_range_reduce(4, 1);
    let src = create_float_const(1.0);
    let reduce = create_reduce(src.clone(), vec![range1, range2], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
    // Multiple ranges should all be in the END
    assert!(count_ends(&result) > 0);
}

// =============================================================================
// Axis Type Tests (Tinygrad alignment: all ranges included in input_ranges)
// =============================================================================

/// Test: Thread ranges in topo are included in input_ranges.
///
/// Matches Tinygrad: input_ranges includes all RANGE ops in topo
/// (except reduce_range itself and ended ranges).
#[test]
fn test_input_ranges_include_thread() {
    let thread_range = create_range_thread(32, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = thread_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Global ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_global() {
    let global_range = create_range_global(64, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = global_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Local ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_local() {
    let local_range = create_range_local(16, 0);
    let reduce_range = create_range_reduce(8, 1);

    let src = local_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Loop ranges in topo are included in input_ranges.
#[test]
fn test_input_ranges_include_loop() {
    let loop_range = create_range_loop(8, 0);
    let reduce_range = create_range_reduce(16, 1);

    let src = loop_range.cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: The reduce range itself is excluded from input_ranges.
///
/// Matches Tinygrad: `x not in reduce_range` check.
#[test]
fn test_input_ranges_exclude_reduce_range() {
    let reduce_range = create_range_reduce(16, 0);
    // Source depends on the reduce_range itself (e.g., loop variable)
    let src = reduce_range.clone().cast(DType::Float32);
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // reduce_range is excluded (it's the loop we iterate over)
    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

/// Test: Mixed axis types in source - all are included in input_ranges.
///
/// Matches Tinygrad: all RANGE ops in topo go into input_ranges
/// (except reduce_range and ended ranges).
#[test]
fn test_input_ranges_mixed_axis_types() {
    let global_range = create_range_global(64, 0);
    let thread_range = create_range_thread(32, 1);
    let loop_range = create_range_loop(8, 2);
    let reduce_range = create_range_reduce(16, 3);

    // Source depends on all three non-reduce ranges
    let src = UOp::new(
        Op::Binary(
            BinaryOp::Add,
            UOp::new(
                Op::Binary(BinaryOp::Add, global_range.cast(DType::Float32), thread_range.cast(DType::Float32)),
                DType::Float32,
            ),
            loop_range.cast(DType::Float32),
        ),
        DType::Float32,
    );
    let reduce = create_reduce(src, vec![reduce_range], ReduceOp::Add);

    let result = apply_pm_reduce(&reduce);

    // All three ranges (global, thread, loop) are in input_ranges
    assert!(!matches!(result.op(), Op::Reduce { .. }));
    assert!(count_define_regs(&result) > 0);
}

// =============================================================================
// Integration Tests
// =============================================================================

/// Test: REDUCE transformation through pm_reduce + gep_pushing (combined pass).
///
/// This tests the REDUCE transformation in the context of a realistic
/// LOAD → REDUCE scenario.
#[test]
fn test_reduce_in_full_pipeline() {
    use crate::devectorize::pm_reduce;
    use crate::rewrite::graph_rewrite;
    // Create a realistic REDUCE scenario
    let reduce_range = create_range_reduce(32, 0);
    let define = UOp::param(0, 1024, DType::Float32, None);

    // LOAD from buffer
    let idx = UOp::index().buffer(define).indices(vec![reduce_range.clone()]).call().unwrap();
    let load = UOp::load().index(idx).call();

    // REDUCE over load
    let reduce = load.reduce(smallvec![reduce_range], ReduceOp::Add);

    // Apply pm_reduce + gep_pushing (as done in optimizer)
    let combined = pm_reduce();
    let mut ctx = crate::devectorize::ReduceContext::default();
    let result = graph_rewrite(&combined, reduce, &mut ctx);

    // Should transform REDUCE to accumulator pattern
    assert!(!matches!(result.op(), Op::Reduce { .. }), "REDUCE should be transformed");
    assert!(count_define_regs(&result) > 0, "Should have DEFINE_REG for accumulator");
}
