//! Unit tests for [`pm_remove_bufferize`].
//!
//! Covered behavior:
//! - `INDEX(BUFFERIZE)` inlining:
//!     - always-run sources (Contiguous / Copy / Noop) are kept,
//!     - non-removable bufferizes are kept,
//!     - the >3 accessed-buffer threshold,
//!     - the `buffer_in_reduce` rejection,
//!     - successful substitution with CONST range keys skipped.
//! - `STORE(x, x) → NOOP`.
//! - `END(NOOP, ..) → NOOP`.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, BufferizeOpts, Op, ReduceOp, UOp};

use crate::pattern::RewriteResult;
use crate::rangeify::patterns::pm_remove_bufferize;

// ============================================================================
// Helper builders
// ============================================================================

fn range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(axis_id), AxisType::Loop)
}

fn param(slot: usize, size: usize) -> Arc<UOp> {
    let dev = UOp::device(DeviceSpec::Cpu);
    UOp::param(slot, size, DType::Float32, Some(dev))
}

fn removable_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::bufferize(compute, ranges, BufferizeOpts::local())
}

fn non_removable_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::bufferize(compute, ranges, BufferizeOpts { device: None, addrspace: AddrSpace::Local, removable: false })
}

/// Build `INDEX(BUFFERIZE(compute, buf_ranges), idx_ranges)`.
fn index_bufferize(compute: Arc<UOp>, buf_ranges: Vec<Arc<UOp>>, idx_ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    let buf = removable_bufferize(compute, buf_ranges);
    UOp::index().buffer(buf).indices(idx_ranges).call().expect("INDEX construction must succeed")
}

// ============================================================================
// Rule 1: INDEX(BUFFERIZE) — `remove_bufferize` cost heuristic
// ============================================================================

#[test]
fn always_run_contiguous_is_kept() {
    let x = UOp::native_const(1.0f32);
    let contig = x.contiguous();
    let r = range(8, 0);

    let idx = index_bufferize(contig, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::NoMatch), "BUFFERIZE(CONTIGUOUS) must not be inlined");
}

#[test]
fn always_run_copy_is_kept() {
    let x = UOp::native_const(1.0f32);
    let cp = x.copy_to_device(DeviceSpec::Cpu);
    let r = range(8, 0);

    let idx = index_bufferize(cp, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::NoMatch), "BUFFERIZE(COPY) must not be inlined");
}

#[test]
fn always_run_noop_is_kept() {
    let r = range(8, 0);
    let idx = index_bufferize(UOp::noop(), vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::NoMatch), "BUFFERIZE(NOOP) must not be inlined");
}

#[test]
fn non_removable_bufferize_is_kept() {
    // Multi-consumer realize boundary — inlining would duplicate compute into
    // every consumer's kernel.
    let a = UOp::native_const(1.0f32);
    let b = UOp::native_const(2.0f32);
    let r = range(8, 0);
    let compute = a.try_add(&b).expect("add");

    let buf = non_removable_bufferize(compute, vec![r.clone()]);
    let idx = UOp::index().buffer(buf).indices(vec![r]).call().expect("INDEX");
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::NoMatch), "non-removable BUFFERIZE must not be inlined");
}

#[test]
fn three_accessed_buffers_inlines() {
    // At the threshold (3 distinct Param/Bufferize/MStack accesses) — inline.
    let r = range(8, 0);
    let p1 = UOp::index().buffer(param(0, 8)).indices(vec![r.clone()]).call().expect("idx");
    let p2 = UOp::index().buffer(param(1, 8)).indices(vec![r.clone()]).call().expect("idx");
    let p3 = UOp::index().buffer(param(2, 8)).indices(vec![r.clone()]).call().expect("idx");
    let compute = p1.try_add(&p2).expect("add").try_add(&p3).expect("add");

    let idx = index_bufferize(compute, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(
        matches!(result, RewriteResult::Rewritten(_)),
        "BUFFERIZE accessing 3 Params must be inlined (threshold is `> 3`)"
    );
}

#[test]
fn four_accessed_buffers_is_kept() {
    let r = range(8, 0);
    let p1 = UOp::index().buffer(param(0, 8)).indices(vec![r.clone()]).call().expect("idx");
    let p2 = UOp::index().buffer(param(1, 8)).indices(vec![r.clone()]).call().expect("idx");
    let p3 = UOp::index().buffer(param(2, 8)).indices(vec![r.clone()]).call().expect("idx");
    let p4 = UOp::index().buffer(param(3, 8)).indices(vec![r.clone()]).call().expect("idx");
    let compute = p1.try_add(&p2).expect("add").try_add(&p3).expect("add").try_add(&p4).expect("add");

    let idx = index_bufferize(compute, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::NoMatch), "BUFFERIZE accessing 4 Params must be kept (threshold is `> 3`)");
}

#[test]
fn buffer_in_reduce_is_kept() {
    // Reduce body reads a Param — would compound buffer reads inside the loop.
    let r_loop = range(8, 0);
    let r_red = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(1), AxisType::Reduce);
    let p = UOp::index().buffer(param(0, 32)).indices(vec![r_loop.clone(), r_red.clone()]).call().expect("idx");
    let reduced = p.reduce(smallvec![r_red], ReduceOp::Add);

    let idx = index_bufferize(reduced, vec![r_loop.clone()], vec![r_loop]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(
        matches!(result, RewriteResult::NoMatch),
        "BUFFERIZE whose reduce body reads a Param must be kept (`buffer_in_reduce`)"
    );
}

#[test]
fn reduce_without_buffer_access_inlines() {
    // Reduce body has only ranges + constants — `buffer_in_reduce` is false.
    let r_loop = range(8, 0);
    let r_red = UOp::range_axis(UOp::index_const(4), AxisId::Renumbered(1), AxisType::Reduce);
    let two = UOp::native_const(2.0f32);
    let reduced = two.reduce(smallvec![r_red], ReduceOp::Add);

    let idx = index_bufferize(reduced, vec![r_loop.clone()], vec![r_loop]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(
        matches!(result, RewriteResult::Rewritten(_)),
        "BUFFERIZE(REDUCE(const)) must inline — `buffer_in_reduce` is false"
    );
}

#[test]
fn const_keys_skipped_during_substitution() {
    // Mix CONST and RANGE in buf_ranges. The CONST slot is a broadcast dim
    // (not a real range key) and must be skipped during substitution; the
    // inlined result must reference the live `idx_r` from the index side.
    let buf_const = UOp::index_const(0);
    let buf_r = range(8, 0);
    let idx_const = UOp::index_const(0);
    let idx_r = range(8, 1);

    // Use the range as the compute (Index dtype) — `compute` directly references
    // `buf_r`. After substitution it must reference `idx_r`.
    let compute = buf_r.clone();
    let idx = index_bufferize(compute, vec![buf_const, buf_r.clone()], vec![idx_const, idx_r.clone()]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            // After substitution the inlined value must equal idx_r — the
            // (CONST, CONST) pair is skipped and (buf_r → idx_r) substituted.
            assert!(Arc::ptr_eq(&rewritten, &idx_r), "substitution must replace buf_r with idx_r (CONST keys skipped)");
        }
        other => panic!("expected Rewritten, got {other:?}"),
    }
}

#[test]
fn invalid_index_value_is_not_substituted() {
    // A dead-load `Invalid` index value must NOT be substituted into the
    // inlined compute — doing so would poison the expression. The buffer range
    // paired with Invalid is kept verbatim.
    let buf_r0 = range(8, 0);
    let buf_r1 = range(8, 1);
    let idx0 = range(8, 2);
    let invalid = UOp::invalid_marker();

    // compute references both buffer ranges so we can observe each substitution.
    let compute = buf_r0.try_add(&buf_r1).expect("add");
    let buf = removable_bufferize(compute, vec![buf_r0.clone(), buf_r1.clone()]);
    let idx = UOp::index().buffer(buf).indices(vec![idx0.clone(), invalid]).call().expect("INDEX");

    let result = pm_remove_bufferize().rewrite(&idx, &mut ());
    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(
                !rewritten.any_in_subtree(|n| matches!(n.op(), Op::Invalid)),
                "Invalid index value must not be substituted into the inlined compute"
            );
            assert!(
                rewritten.any_in_subtree(|n| Arc::ptr_eq(n, &buf_r1)),
                "the buffer range paired with Invalid must be kept (not substituted)"
            );
            assert!(
                rewritten.any_in_subtree(|n| Arc::ptr_eq(n, &idx0)),
                "the non-Invalid index (idx0) must be substituted in"
            );
        }
        other => panic!("expected Rewritten, got {other:?}"),
    }
}

// ============================================================================
// Rule 2: STORE(x, x) → NOOP
// ============================================================================

#[test]
fn store_to_self_rewrites_to_noop() {
    // Fires when remove_bufferize substitution leaves a STORE whose value
    // equals its destination INDEX.
    let p = param(0, 8);
    let r = range(8, 0);
    let idx = UOp::index().buffer(p).indices(vec![r]).call().expect("idx");
    let store = idx.store(idx.clone());

    let result = pm_remove_bufferize().rewrite(&store, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(matches!(rewritten.op(), Op::Noop), "STORE(x, x) must rewrite to NOOP");
        }
        other => panic!("expected Rewritten(NOOP), got {other:?}"),
    }
}

// ============================================================================
// Rule 3: END(NOOP, ..) → NOOP
// ============================================================================

#[test]
fn end_of_noop_rewrites_to_noop() {
    let r = range(8, 0);
    let end = UOp::noop().end(smallvec![r]);

    let result = pm_remove_bufferize().rewrite(&end, &mut ());

    match result {
        RewriteResult::Rewritten(rewritten) => {
            assert!(matches!(rewritten.op(), Op::Noop), "END(NOOP, ..) must rewrite to NOOP");
        }
        other => panic!("expected Rewritten(NOOP), got {other:?}"),
    }
}
