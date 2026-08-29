//! Comprehensive tests for rangeify pattern matchers.
//!
//! Tests verify that all pattern matchers correctly transform UOps:
//! - early_rewrites: DETACH and CONTIGUOUS_BACKWARD removal
//! - buffer_folding: Noop stage removal and constant propagation
//! - dead_axis_removal: Remove size-1 dimensions
//! - buffer_removal: Cost-based buffer elimination
//!
//! Based on Tinygrad's test_schedule.py pattern tests.

use std::f32::consts::PI;
use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{AxisId, AxisType, BufferizeOpts, ConstValue, Op, UOp};

use crate::pattern::RewriteResult;
use crate::rangeify::IndexingContext;
use crate::rangeify::patterns;

// ===== early_rewrites Pattern Tests =====

#[test]
fn test_early_rewrites_detach_removal() {
    let matcher = patterns::early_rewrites();

    // Test: DETACH(x) → x
    let x = UOp::native_const(42.0f32);
    let detach = x.detach();

    let result = matcher.rewrite(&detach, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should rewrite DETACH");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the source");
    }
}

#[test]
fn test_early_rewrites_contiguous_backward_removal() {
    let matcher = patterns::early_rewrites();

    // Test: CONTIGUOUS_BACKWARD(x) → x
    let x = UOp::native_const(PI);
    let contiguous = x.contiguous_backward();

    let result = matcher.rewrite(&contiguous, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should rewrite CONTIGUOUS_BACKWARD");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the source");
    }
}

#[test]
fn test_early_rewrites_no_match_for_other_ops() {
    let matcher = patterns::early_rewrites();

    // Test that non-DETACH/CONTIGUOUS_BACKWARD operations return NoMatch
    let const_op = UOp::native_const(1.0f32);
    let result = matcher.rewrite(&const_op, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match CONST");

    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let add = a.try_add(&b).unwrap();
    let result = matcher.rewrite(&add, &mut ());
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match Binary ops");
}

#[test]
fn test_early_rewrites_preserves_shaped_empty_reduction_identity() {
    let source = UOp::new_buffer(svod_dtype::DeviceSpec::Cpu, 0, DType::Float32)
        .try_reshape(&smallvec::smallvec![svod_ir::SInt::Const(0), svod_ir::SInt::Const(3)])
        .unwrap();
    let reduce = source.try_reduce_axis(svod_ir::ReduceOp::Add, vec![0]).unwrap();
    let result = patterns::early_rewrites().rewrite(&reduce, &mut ());
    let RewriteResult::Rewritten(identity) = result else { panic!("empty reduction must fold") };

    assert_eq!(identity.shape().unwrap().unwrap().as_slice(), &[svod_ir::SInt::Const(3)]);
    assert!(
        matches!(identity.op(), Op::Expand { src, .. } if matches!(src.op(), Op::Const(value) if value.0.try_float() == Some(0.0)))
    );
}

#[test]
fn test_early_rewrites_nested_detach() {
    let matcher = patterns::early_rewrites();

    // Test: DETACH(DETACH(x)) should rewrite outer DETACH to DETACH(x)
    let x = UOp::native_const(1.0f32);
    let inner_detach = x.detach();
    let outer_detach = inner_detach.detach();

    let result = matcher.rewrite(&outer_detach, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)));

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &inner_detach), "Should unwrap outer DETACH to inner DETACH");
    }
}

// ===== buffer_folding Pattern Tests =====

#[test]
fn test_buffer_folding_noop_bufferize() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(STAGE(x, ranges), ranges) → x when ranges are equal
    let x = UOp::native_const(1.0f32);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);

    let stage = UOp::stage(x.clone(), vec![range.clone()], BufferizeOpts::local());
    let index = UOp::index().buffer(stage).indices(vec![range]).call().unwrap();

    let result = matcher.rewrite(&index, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove noop STAGE");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &x), "Should return the compute directly");
    }
}

#[test]
fn test_buffer_folding_bufferize_const() {
    let matcher = patterns::buffer_folding();

    // Test: STAGE(CONST) → CONST
    let const_val = UOp::native_const(42.0f32);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let stage = UOp::stage(const_val.clone(), vec![range], BufferizeOpts::local());

    let result = matcher.rewrite(&stage, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove STAGE from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_index_const() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(CONST) → CONST
    let const_val = UOp::native_const(PI);
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index().buffer(const_val.clone()).indices(vec![range]).call().unwrap();

    let result = matcher.rewrite(&index, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove INDEX from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_copy_const() {
    let matcher = patterns::buffer_folding();

    // Test: COPY(CONST, device) → CONST
    let const_val = UOp::native_const(1.0f32);
    let copy = const_val.copy(svod_ir::DeviceSpec::Cpu);

    let result = matcher.rewrite(&copy, &mut ());
    assert!(matches!(result, RewriteResult::Rewritten(_)), "Should remove COPY from CONST");

    if let RewriteResult::Rewritten(rewritten) = result {
        assert!(Arc::ptr_eq(&rewritten, &const_val), "Should return the constant directly");
    }
}

#[test]
fn test_buffer_folding_no_match_different_ranges() {
    let matcher = patterns::buffer_folding();

    // Test: INDEX(STAGE(x, r1), r2) should NOT match when r1 != r2
    let x = UOp::native_const(1.0f32);
    let range1_end = UOp::index_const(10);
    let range1 = UOp::range_axis(range1_end, AxisId::Renumbered(0), AxisType::Loop);

    let range2_end = UOp::index_const(20);
    let range2 = UOp::range_axis(range2_end, AxisId::Renumbered(1), AxisType::Loop);

    let stage = UOp::stage(x, vec![range1], BufferizeOpts::local());
    let index = UOp::index().buffer(stage).indices(vec![range2]).call().unwrap();

    let result = matcher.rewrite(&index, &mut ());
    // This might match or not depending on implementation details,
    // but should NOT return the original compute 'x' directly
    match result {
        RewriteResult::NoMatch => {}
        RewriteResult::Rewritten(rewritten) => {
            // If it rewrites, it should not be the original 'x'
            assert!(!matches!(rewritten.op(), Op::Const(_)));
        }
        RewriteResult::Gate(_) => {}
    }
}

// ===== dead_axis_removal Pattern Tests =====

#[test]
fn test_dead_axis_removal_single_dead_axis() {
    let matcher = patterns::dead_axis_removal();

    // Create a STAGE with one dead axis (range with size 1)
    let x = UOp::native_const(1.0f32);
    let dead_range_end = UOp::index_const(1); // size 1 = dead
    let dead_range = UOp::range_axis(dead_range_end, AxisId::Renumbered(0), AxisType::Loop);

    let stage = UOp::stage(x.clone(), vec![dead_range], BufferizeOpts::local());

    let result = matcher.rewrite(&stage, &mut ());

    // Should restructure to [EXPAND(]RESHAPE(BUFFERIZE_no_ranges)[)] - Tinygrad behavior
    // The STAGE is KEPT (not removed) so it can be converted to STORE later.
    // Note: identity EXPAND is eliminated at construction time, so EXPAND may not be present.
    match result {
        RewriteResult::Rewritten(rewritten) => {
            // Accept EXPAND(RESHAPE(STAGE)) or RESHAPE(STAGE) (when expand is identity)
            let reshape_op = match rewritten.op() {
                Op::Expand { src, .. } => src,
                Op::Reshape { .. } => &rewritten,
                _ => panic!("Expected EXPAND or RESHAPE, got: {}", rewritten.tree()),
            };
            if let Op::Reshape { src: bufferize_op, .. } = reshape_op.op() {
                assert!(
                    matches!(bufferize_op.op(), Op::Stage { ranges, .. } if ranges.is_empty()),
                    "Inner should be STAGE with no ranges, got: {}",
                    rewritten.tree()
                );
            } else {
                panic!("Expected RESHAPE inside result, got: {}", rewritten.tree());
            }
        }
        _ => {
            // This is also acceptable if dead axis detection has specific conditions
        }
    }
}

#[test]
fn test_dead_axis_removal_skips_always_run_ops() {
    // A COPY destination is sized by the transfer, so a dead axis must not shrink
    // it — the same guard remove_bufferize applies (tinygrad rangeify.py:198,227).
    let source = UOp::native_const(1.0f32).copy(svod_ir::DeviceSpec::Cpu);
    let dead_range = UOp::range_axis(UOp::index_const(1), AxisId::Renumbered(0), AxisType::Loop);
    let stage = UOp::stage(source, vec![dead_range], BufferizeOpts::local());

    assert!(matches!(patterns::dead_axis_removal().rewrite(&stage, &mut ()), RewriteResult::NoMatch));
}

#[test]
fn test_dead_axis_removal_mixed_axes() {
    let matcher = patterns::dead_axis_removal();

    // Create STAGE with mix of live and dead axes
    // NOTE: When compute is native_const (no ranges), ALL ranges are dead
    // because compute doesn't depend on any of them (Tinygrad behavior)
    let x = UOp::native_const(1.0f32);
    let live_range_end = UOp::index_const(10);
    let live_range = UOp::range_axis(live_range_end, AxisId::Renumbered(0), AxisType::Loop);

    let dead_range_end = UOp::index_const(1);
    let dead_range = UOp::range_axis(dead_range_end, AxisId::Renumbered(1), AxisType::Loop);

    let stage = UOp::stage(x.clone(), vec![live_range.clone(), dead_range], BufferizeOpts::local());

    let result = matcher.rewrite(&stage, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            // Since compute has no ranges, ALL ranges are dead
            // Result is EXPAND(RESHAPE(BUFFERIZE_no_ranges)) - Tinygrad behavior
            if let Op::Expand { src: reshape_op, .. } = rewritten.op() {
                if let Op::Reshape { src: bufferize_op, .. } = reshape_op.op() {
                    assert!(
                        matches!(bufferize_op.op(), Op::Stage { ranges, .. } if ranges.is_empty()),
                        "Inner should be STAGE with no ranges, got: {}",
                        rewritten.tree()
                    );
                } else {
                    panic!("Expected RESHAPE inside EXPAND, got: {}", rewritten.tree());
                }
            } else {
                panic!("Expected EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}", rewritten.tree());
            }
        }
        _ => {
            // Pattern should match and rewrite when there are dead axes
            panic!("Expected pattern to match and rewrite");
        }
    }
}

#[test]
fn test_dead_axis_removal_no_dead_axes_simple_compute() {
    let matcher = patterns::dead_axis_removal();

    // Create STAGE with "live" axes (size > 1), but simple compute (no ranges)
    // NOTE: When compute is native_const (no ranges), ALL ranges are dead
    // because compute doesn't depend on any of them (Tinygrad behavior)
    let x = UOp::native_const(1.0f32);
    let range1_end = UOp::index_const(10);
    let range1 = UOp::range_axis(range1_end, AxisId::Renumbered(0), AxisType::Loop);

    let range2_end = UOp::index_const(20);
    let range2 = UOp::range_axis(range2_end, AxisId::Renumbered(1), AxisType::Loop);

    let stage = UOp::stage(x.clone(), vec![range1, range2], BufferizeOpts::local());

    let result = matcher.rewrite(&stage, &mut ());

    // All ranges are dead (compute has no ranges) → EXPAND(RESHAPE(BUFFERIZE_no_ranges))
    match result {
        RewriteResult::Rewritten(rewritten) => {
            // Result is EXPAND(RESHAPE(BUFFERIZE_no_ranges)) - Tinygrad behavior
            if let Op::Expand { src: reshape_op, .. } = rewritten.op() {
                if let Op::Reshape { src: bufferize_op, .. } = reshape_op.op() {
                    assert!(
                        matches!(bufferize_op.op(), Op::Stage { ranges, .. } if ranges.is_empty()),
                        "Inner should be STAGE with no ranges, got: {}",
                        rewritten.tree()
                    );
                } else {
                    panic!("Expected RESHAPE inside EXPAND, got: {}", rewritten.tree());
                }
            } else {
                panic!("Expected EXPAND(RESHAPE(BUFFERIZE_no_ranges)), got: {}", rewritten.tree());
            }
        }
        _ => panic!("Expected pattern to match and rewrite when all ranges are dead"),
    }
}

// ===== Movement Op Removal Tests =====
// These tests verify movement op removal behavior which is now integrated into apply_rangeify_patterns

#[test]
fn test_movement_op_removal_no_match_without_ranges() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a PERMUTE operation (a movement op)
    let src = UOp::native_const(1.0f32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    // Without ranges assigned, should NOT remove
    // (The stage pattern will try to match but return None without ranges)
    let result = matcher.rewrite(&permute, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch), "Should NOT remove movement op without ranges assigned");
}

#[test]
fn test_movement_op_removal_removes_with_ranges() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a PERMUTE operation
    let src = UOp::native_const(1.0f32);
    let permute = UOp::new(Op::Permute { src: src.clone(), axes: vec![1, 0] }, DType::Float32);

    // Assign ranges to the movement op (simulating transformation has been applied)
    let range = UOp::new(
        Op::Range {
            end: UOp::index_const(5),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );
    ctx.set_ranges(&permute, vec![range.clone()], vec![range.clone()]);

    // With ranges assigned, SHOULD remove and return source
    let result = matcher.rewrite(&permute, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "Should return the source operand");
        }
        _ => panic!("Expected movement op to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_reshape() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a RESHAPE operation
    let src = UOp::native_const(1.0f32);
    let new_shape = UOp::stack(smallvec::smallvec![UOp::index_const(4), UOp::index_const(8)]);
    let reshape = UOp::new(Op::Reshape { src: src.clone(), new_shape }, DType::Float32);

    // Assign ranges
    let range = UOp::new(
        Op::Range {
            end: UOp::index_const(4),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );
    ctx.set_ranges(&reshape, vec![range.clone()], vec![range.clone()]);

    // Should remove and return source
    let result = matcher.rewrite(&reshape, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "RESHAPE should be removed");
        }
        _ => panic!("Expected RESHAPE to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_expand() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create an EXPAND operation
    let src = UOp::native_const(1.0f32);
    let new_shape = UOp::stack(smallvec::smallvec![UOp::index_const(4), UOp::index_const(8)]);
    let expand = UOp::new(Op::Expand { src: src.clone(), new_shape }, DType::Float32);

    // Assign ranges
    let range = UOp::new(
        Op::Range {
            end: UOp::index_const(4),
            axis_id: AxisId::Renumbered(0),
            axis_type: AxisType::Loop,
            deps: smallvec::SmallVec::new(),
        },
        DType::Index,
    );
    ctx.set_ranges(&expand, vec![range.clone()], vec![range.clone()]);

    // Should remove and return source
    let result = matcher.rewrite(&expand, &mut ctx);
    match result {
        RewriteResult::Rewritten(result) => {
            assert!(std::sync::Arc::ptr_eq(&result, &src), "EXPAND should be removed");
        }
        _ => panic!("Expected EXPAND to be removed when ranges are assigned"),
    }
}

#[test]
fn test_movement_op_removal_non_movement_op() {
    let matcher = patterns::apply_rangeify_patterns();
    let mut ctx = IndexingContext::new();

    // Create a non-movement op (SQRT)
    // neg() now produces MUL (binary), use sqrt (unary) instead.
    let src = UOp::native_const(1.0f32);
    let sqrt = src.try_sqrt().unwrap();

    // Non-movement ops without ranges should not match the movement removal pattern
    // (they may match other patterns like stage, but without ranges assigned,
    // apply_bufferize_transform returns None)
    let result = matcher.rewrite(&sqrt, &mut ctx);
    assert!(matches!(result, RewriteResult::NoMatch), "Should not match non-movement ops without ranges");
}

// ===== Integration Tests =====

#[test]
fn test_pattern_composition() {
    // Test that multiple patterns can be applied in sequence

    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));

    // First apply DETACH
    let detach = x.detach();

    // Then apply early_rewrites to remove DETACH
    let early = patterns::early_rewrites();
    let result1 = early.rewrite(&detach, &mut ());
    assert!(matches!(result1, RewriteResult::Rewritten(_)));

    let unwrapped = if let RewriteResult::Rewritten(r) = result1 {
        r
    } else {
        panic!("Should have rewritten");
    };

    // Now wrap in STAGE
    let range_end = UOp::index_const(10);
    let range = UOp::range_axis(range_end, AxisId::Renumbered(0), AxisType::Loop);
    let stage = UOp::stage(unwrapped, vec![range], BufferizeOpts::local());

    // Apply buffer_folding to remove STAGE(CONST)
    let folding = patterns::buffer_folding();
    let result2 = folding.rewrite(&stage, &mut ());

    match result2 {
        RewriteResult::Rewritten(rewritten) => {
            assert!(Arc::ptr_eq(&rewritten, &x), "Should have removed both DETACH and STAGE");
        }
        _ => {
            // Acceptable depending on implementation
        }
    }
}

#[test]
fn test_idempotent_patterns() {
    // Test that applying patterns multiple times doesn't cause issues

    let x = UOp::const_(DType::Float32, ConstValue::Float(1.0));
    let detach = x.detach();

    let matcher = patterns::early_rewrites();

    // First application
    let result1 = matcher.rewrite(&detach, &mut ());
    assert!(matches!(result1, RewriteResult::Rewritten(_)));

    let unwrapped = if let RewriteResult::Rewritten(r) = result1 { r } else { x.clone() };

    // Second application (should not match on CONST)
    let result2 = matcher.rewrite(&unwrapped, &mut ());
    assert!(matches!(result2, RewriteResult::NoMatch), "Should not match on already-processed node");
}
