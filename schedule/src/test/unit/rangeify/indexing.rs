//! Tests for IndexingContext and range assignment.
//!
//! Validates:
//! - Range creation and ID assignment
//! - Realize map tracking
//! - Range map operations
//! - Symbolic size handling
//! - Axis types (Loop vs Reduce)

use std::sync::Arc;

use svod_ir::{AxisId, AxisType, DType, Op, SInt, UOp};

use crate::rangeify::{
    IndexingContext,
    indexing::{broadcast_ranges, data_sources},
};

// ===== Index Linearization =====

#[test]
fn test_image_buffers_keep_two_index_addresses() {
    use svod_ir::ParamArg;

    let ranges = [
        UOp::range_axis(UOp::index_const(2), AxisId::Renumbered(0), AxisType::Loop),
        UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(1), AxisType::Loop),
    ];
    let shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![2usize.into(), 8usize.into()]);

    for (dtype, expected_indices) in
        [(DType::Image { kind: svod_dtype::ImageKind::Float, shape: vec![2, 8, 4] }, 2), (DType::Float32, 1)]
    {
        let arg = ParamArg::buffer(0, dtype.clone(), svod_dtype::AddrSpace::Global, Some(svod_ir::DeviceSpec::Cpu));
        let buffer = UOp::new(Op::Buffer { shape: shape.clone(), arg }, dtype);
        let indexed = crate::rangeify::transforms::transform_single_source(
            &UOp::sink(vec![]),
            &buffer,
            &ranges,
            &mut IndexingContext::new(),
        );
        assert!(matches!(indexed.op(), Op::Index { indices, .. } if indices.len() == expected_indices));
    }
}

// ===== Basic Range Creation =====

#[test]
fn test_indexing_context_new_range() {
    let mut ctx = IndexingContext::new();

    // Test constant size - ranges are created with AxisId::Unrenumbered
    let r1 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    assert!(matches!(r1.op(), Op::Range { axis_id, .. } if *axis_id == AxisId::Unrenumbered(0)));

    let r2 = ctx.new_range(&SInt::Const(20), AxisType::Loop);
    assert!(matches!(r2.op(), Op::Range { axis_id, .. } if *axis_id == AxisId::Unrenumbered(1)));

    // Test size 1 optimization (returns const 0)
    let r3 = ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert!(matches!(r3.op(), Op::Const(_)));
}

#[test]
fn test_indexing_context_realize_map() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);

    assert!(!ctx.should_realize(&x));

    ctx.mark_realize_all(&x).unwrap();
    assert!(ctx.should_realize(&x));
}

#[test]
fn test_data_sources_exclude_index_metadata_and_after_dependencies() {
    let buffer = UOp::new_buffer(svod_device::DeviceSpec::Cpu, 8, DType::Float32);
    let range = UOp::range_axis(UOp::index_const(8), AxisId::Renumbered(0), AxisType::Loop);
    let index = UOp::index().buffer(buffer.clone()).indices(vec![range]).call().unwrap();
    let dep = UOp::noop();
    let after = buffer.after(smallvec::smallvec![dep]);

    let index_sources = data_sources(&index);
    assert_eq!(index_sources.len(), 1);
    assert!(Arc::ptr_eq(&index_sources[0], &buffer));

    let after_sources = data_sources(&after);
    assert_eq!(after_sources.len(), 1);
    assert!(Arc::ptr_eq(&after_sources[0], &buffer));
}

#[test]
fn test_broadcast_ranges_preserves_extra_range_for_zero_rank_shapes() {
    let source = UOp::var("source", DType::Float32, 0, 4);
    let consumer = source.try_add(&UOp::var("other", DType::Float32, 0, 4)).unwrap();
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);

    let mapped = broadcast_ranges(&consumer, &source, std::slice::from_ref(&range));

    assert_eq!(mapped.len(), 1);
    assert!(Arc::ptr_eq(&mapped[0], &range));
}

#[test]
fn test_broadcast_ranges_zeroes_expanded_singleton_axis() {
    let source = UOp::const_(DType::Float32, 1.0f32.into()).try_reshape(&smallvec::smallvec![SInt::Const(1)]).unwrap();
    let expanded = source.try_expand(&smallvec::smallvec![SInt::Const(4)]).unwrap();
    let consumer = expanded.try_add(&expanded).unwrap();
    let range = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(0), AxisType::Loop);

    let mapped = broadcast_ranges(&consumer, &source, &[range]);

    assert_eq!(mapped.len(), 1);
    assert!(matches!(mapped[0].op(), Op::Const(_)));
}

// ===== Range Counter =====

#[test]
fn test_range_counter_increments() {
    let mut ctx = IndexingContext::new();

    assert_eq!(ctx.range_counter(), 0);

    ctx.new_range(&SInt::Const(10), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 1);

    ctx.new_range(&SInt::Const(20), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 2);

    // Size 1 should NOT increment counter (returns const 0)
    ctx.new_range(&SInt::Const(1), AxisType::Loop);
    assert_eq!(ctx.range_counter(), 2);
}

// ===== Axis Types =====

#[test]
fn test_range_axis_types() {
    let mut ctx = IndexingContext::new();

    // Loop axis
    let loop_range = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    if let Op::Range { axis_type, .. } = loop_range.op() {
        assert_eq!(*axis_type, AxisType::Loop);
    } else {
        panic!("Expected Range op");
    }

    // Reduce axis
    let reduce_range = ctx.new_range(&SInt::Const(10), AxisType::Reduce);
    if let Op::Range { axis_type, .. } = reduce_range.op() {
        assert_eq!(*axis_type, AxisType::Reduce);
    } else {
        panic!("Expected Range op");
    }
}

// ===== Symbolic Sizes =====

#[test]
fn test_symbolic_size_range() {
    let mut ctx = IndexingContext::new();

    // Create symbolic size
    let n = UOp::define_var("n".to_string(), 0, i64::MAX);
    let symbolic_size = SInt::Symbolic(n.clone());

    let range = ctx.new_range(&symbolic_size, AxisType::Loop);

    // Should create range with symbolic end
    if let Op::Range { end, .. } = range.op() {
        assert!(Arc::ptr_eq(end, &n));
    } else {
        panic!("Expected Range op");
    }
}

// ===== Range Map Operations =====

#[test]
fn test_set_get_ranges() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);

    let r0 = ctx.new_range(&SInt::Const(10), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(20), AxisType::Loop);

    // Initially no ranges
    assert!(ctx.get_ranges(&x).is_none());

    // Set ranges
    let input_ranges = vec![r0.clone(), r1.clone()];
    let output_ranges = vec![r0.clone()];
    ctx.set_ranges(&x, input_ranges.clone(), output_ranges.clone());

    // Get ranges
    let ranges = ctx.get_ranges(&x);
    assert!(ranges.is_some());

    let (inp, out) = ranges.unwrap();
    assert_eq!(inp.len(), 2);
    assert_eq!(out.len(), 1);
    assert!(Arc::ptr_eq(&inp[0], &r0));
    assert!(Arc::ptr_eq(&inp[1], &r1));
    assert!(Arc::ptr_eq(&out[0], &r0));
}

// ===== Realize Axes =====

#[test]
fn test_mark_realize_specific_axes() {
    let mut ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);

    // Mark specific axes
    ctx.mark_realize(&x, vec![0, 2]);

    assert!(ctx.should_realize(&x));

    let axes = ctx.get_realize_axes(&x);
    assert!(axes.is_some());
    assert_eq!(axes.unwrap(), &[0, 2]);
}

#[test]
fn test_get_realize_axes_none() {
    let ctx = IndexingContext::new();
    let x = UOp::var("x", DType::Float32, 0, i64::MAX);

    // Not in realize map
    assert!(ctx.get_realize_axes(&x).is_none());
}

// ===== Multi-Dimensional =====

#[test]
fn test_multi_dimensional_ranges() {
    let mut ctx = IndexingContext::new();

    // Create 3D ranges
    let r0 = ctx.new_range(&SInt::Const(32), AxisType::Loop);
    let r1 = ctx.new_range(&SInt::Const(64), AxisType::Loop);
    let r2 = ctx.new_range(&SInt::Const(128), AxisType::Loop);

    // Verify sequential IDs
    assert!(matches!(r0.op(), Op::Range { axis_id: AxisId::Unrenumbered(0), .. }));
    assert!(matches!(r1.op(), Op::Range { axis_id: AxisId::Unrenumbered(1), .. }));
    assert!(matches!(r2.op(), Op::Range { axis_id: AxisId::Unrenumbered(2), .. }));

    // Verify sizes (ConstValueHash is a tuple struct wrapping ConstValue)
    use svod_ir::ConstValue;
    if let Op::Range { end, .. } = r0.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(32))));
    }
    if let Op::Range { end, .. } = r1.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(64))));
    }
    if let Op::Range { end, .. } = r2.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(128))));
    }
}

// ===== Edge Cases =====

#[test]
fn test_zero_size_range() {
    let mut ctx = IndexingContext::new();

    // Size 0 should still create a range (not optimized like size 1)
    let range = ctx.new_range(&SInt::Const(0), AxisType::Loop);
    assert!(matches!(range.op(), Op::Range { .. }));
}

#[test]
fn test_large_size_range() {
    let mut ctx = IndexingContext::new();

    // Very large size
    let range = ctx.new_range(&SInt::Const(1 << 30), AxisType::Loop);

    use svod_ir::ConstValue;
    if let Op::Range { end, .. } = range.op() {
        assert!(matches!(end.op(), Op::Const(c) if matches!(c.0, ConstValue::Int(v) if v == 1 << 30)));
    }
}

#[test]
fn test_multiple_contexts_independent() {
    // Two separate contexts should be independent
    let mut ctx1 = IndexingContext::new();
    let mut ctx2 = IndexingContext::new();

    ctx1.new_range(&SInt::Const(10), AxisType::Loop);
    ctx1.new_range(&SInt::Const(20), AxisType::Loop);

    // ctx2 starts fresh
    assert_eq!(ctx2.range_counter(), 0);

    let r = ctx2.new_range(&SInt::Const(30), AxisType::Loop);
    assert!(matches!(r.op(), Op::Range { axis_id: AxisId::Unrenumbered(0), .. }));
}

/// `apply_movement_op` and `_apply_reshape` are `@functools.cache` upstream
/// (tinygrad/schedule/indexing.py:158,171): process-global and keyed on the inputs, so
/// a second call with the same op, input shape and range tuple never rebuilds the
/// index chain — it hands back the very nodes the first call produced.
#[test]
fn equal_movement_inputs_reuse_the_cached_index_chain() {
    // Prime extents so no other test shares these inputs in the process-global cache.
    let rngs = vec![UOp::range_const(13, 0), UOp::range_const(11, 1)];
    let in_shape = [SInt::Const(11), SInt::Const(13)];
    let out_shape = svod_ir::shape::shape_to_uop(&smallvec::smallvec![SInt::Const(13), SInt::Const(11)]);
    let reshape = UOp::new(Op::Reshape { src: UOp::index_const(0), new_shape: out_shape }, DType::Float32);
    let holds = || crate::rangeify::indexing::movement_cache_holds(reshape.op(), &in_shape, &rngs);

    assert!(!holds(), "these inputs must be new");
    let first = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);
    assert!(holds(), "the first call memoises the inputs");
    let second = crate::rangeify::apply_movement_op(reshape.op(), &in_shape, &rngs);

    assert_eq!(first.len(), in_shape.len());
    assert!(first.iter().zip(&second).all(|(a, b)| Arc::ptr_eq(a, b)), "a hit returns the cached nodes");
}
