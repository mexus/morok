//! `RangeifyBufferContext`: three independent slot counters, a buffer map, and
//! the bound-variable table.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::UOp;

use crate::rangeify::RangeifyBufferContext;

#[test]
fn each_counter_starts_at_zero_and_advances_independently() {
    let mut ctx = RangeifyBufferContext::new();
    assert_eq!((ctx.global_counter, ctx.local_counter, ctx.range_counter), (0, 0, 0));

    assert_eq!([ctx.next_global(), ctx.next_global(), ctx.next_global()], [0, 1, 2]);
    assert_eq!([ctx.next_local(), ctx.next_local()], [0, 1]);
    assert_eq!([ctx.next_range()], [0]);
    assert_eq!((ctx.global_counter, ctx.local_counter, ctx.range_counter), (3, 2, 1));
}

#[test]
fn a_mapped_buffer_reads_back_by_uop_identity() {
    let mut ctx = RangeifyBufferContext::new();
    let original = UOp::native_const(1.0f32);
    let replacement = UOp::param(0, 1, DType::Float32, None);

    assert!(!ctx.has_buffer(&original));
    ctx.map_buffer(original.clone(), replacement.clone());

    assert!(Arc::ptr_eq(ctx.get_buffer(&original).expect("mapped"), &replacement));
}

#[test]
fn a_tracked_var_keeps_its_uop_and_bound_value() {
    let mut ctx = RangeifyBufferContext::new();
    let var = UOp::define_var("test_var".to_string(), 0, 10);

    assert!(ctx.vars.is_empty());
    ctx.add_var(var.clone(), Some(5));

    let (stored_uop, stored_val) = ctx.vars.get("test_var").expect("tracked");
    assert_eq!(stored_uop.id, var.id);
    assert_eq!(*stored_val, Some(5));
}
