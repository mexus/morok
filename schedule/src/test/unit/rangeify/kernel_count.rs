//! One CALL per global STAGE, each wrapping exactly one STORE.
//!
//! Fusion-level counts over realistic graphs live in `fusion.rs`.

use std::sync::Arc;

use svod_ir::UOp;
use test_case::test_case;

use crate::rangeify::{RangeifyBufferContext, transforms::bufferize_to_store, try_get_kernel_graph};

use super::helpers::{count_kernels, count_stores};

fn one_stage() -> Arc<UOp> {
    UOp::stage_global(UOp::native_const(1.0f32), vec![UOp::range_const(10, 0)])
}

fn two_stages() -> Arc<UOp> {
    UOp::sink(vec![
        UOp::stage_global(UOp::native_const(1.0f32), vec![UOp::range_const(10, 0)]),
        UOp::stage_global(UOp::native_const(2.0f32), vec![UOp::range_const(20, 1)]),
    ])
}

#[test_case(super::one_stage, 1 ; "one stage")]
#[test_case(super::two_stages, 2 ; "two independent stages")]
fn each_global_stage_becomes_one_call_with_one_store(build: fn() -> Arc<UOp>, expected: usize) {
    let (result, _ctx) = try_get_kernel_graph(build()).expect("kernel split");
    assert_eq!(count_kernels(&result), expected);
    assert_eq!(count_stores(&result), expected, "each CALL body owns exactly one STORE");
}

/// The buffer a STAGE is lowered to is memoised, so a second lowering of the
/// same STAGE reuses it while a different STAGE gets its own.
#[test]
fn stage_identity_decides_buffer_reuse() {
    let mut ctx = RangeifyBufferContext::new();
    let range = UOp::range_const(5, 0);
    let first = UOp::stage_global(UOp::native_const(42i32), vec![range.clone()]);
    let second = UOp::stage_global(UOp::native_const(43i32), vec![range]);

    for stage in [&first, &first, &second] {
        bufferize_to_store(stage, &mut ctx);
    }

    assert!(Arc::ptr_eq(ctx.get_buffer(&first).expect("first"), ctx.get_buffer(&first).expect("first again")));
    assert!(!Arc::ptr_eq(ctx.get_buffer(&first).expect("first"), ctx.get_buffer(&second).expect("second")));
}
