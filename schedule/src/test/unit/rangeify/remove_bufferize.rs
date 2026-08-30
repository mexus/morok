//! `pm_remove_bufferize`: inline `INDEX(STAGE)` when the compute is cheaper to
//! recompute than to buffer, plus the two NOOP collapses that fall out of it.

use std::sync::Arc;

use smallvec::smallvec;
use svod_device::DeviceSpec;
use svod_dtype::{AddrSpace, DType};
use svod_ir::{AxisId, AxisType, BufferizeOpts, Op, ReduceOp, UOp};
use test_case::test_case;

use crate::pattern::RewriteResult;
use crate::rangeify::patterns::pm_remove_bufferize;

// ============================================================================
// Helper builders
// ============================================================================

fn range(end: i64, axis_id: usize) -> Arc<UOp> {
    UOp::range_axis(UOp::index_const(end), AxisId::Renumbered(axis_id), AxisType::Loop)
}

fn param(slot: usize, size: usize) -> Arc<UOp> {
    UOp::param(slot, size, DType::Float32, Some(DeviceSpec::Cpu))
}

fn removable_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::stage(compute, ranges, BufferizeOpts::local())
}

fn non_removable_bufferize(compute: Arc<UOp>, ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    UOp::stage(
        compute,
        ranges,
        BufferizeOpts { device: None, local_axis: None, addrspace: AddrSpace::Local, removable: false },
    )
}

/// Build `INDEX(STAGE(compute, buf_ranges), idx_ranges)`.
fn index_bufferize(compute: Arc<UOp>, buf_ranges: Vec<Arc<UOp>>, idx_ranges: Vec<Arc<UOp>>) -> Arc<UOp> {
    let buf = removable_bufferize(compute, buf_ranges);
    UOp::index().buffer(buf).indices(idx_ranges).call().expect("INDEX construction must succeed")
}

// ============================================================================
// Rule 1: INDEX(STAGE) — `remove_bufferize` cost heuristic
// ============================================================================

fn contiguous() -> Arc<UOp> {
    UOp::native_const(1.0f32).contiguous()
}

fn copy() -> Arc<UOp> {
    UOp::native_const(1.0f32).copy_to_device(DeviceSpec::Cpu)
}

/// An always-run source has effects (or a transfer-sized destination) that
/// inlining would duplicate or resize.
#[test_case(super::contiguous ; "contiguous")]
#[test_case(super::copy ; "copy")]
#[test_case(UOp::noop ; "noop")]
fn always_run_sources_are_kept(build: fn() -> Arc<UOp>) {
    let r = range(8, 0);
    let idx = index_bufferize(build(), vec![r.clone()], vec![r]);

    assert!(matches!(pm_remove_bufferize().rewrite(&idx, &mut ()), RewriteResult::NoMatch));
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

    assert!(matches!(result, RewriteResult::NoMatch), "non-removable STAGE must not be inlined");
}

/// Inlining is worth it up to three distinct Param/Stage/MStack accesses; the
/// cutoff is `> 3`.
#[test_case(3, true ; "at the threshold")]
#[test_case(4, false ; "over the threshold")]
fn the_accessed_buffer_count_decides_inlining(params: usize, inlines: bool) {
    let r = range(8, 0);
    let read = |slot| UOp::index().buffer(param(slot, 8)).indices(vec![r.clone()]).call().expect("idx");
    let compute = (1..params).fold(read(0), |acc, slot| acc.try_add(&read(slot)).expect("add"));

    let idx = index_bufferize(compute, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert_eq!(matches!(result, RewriteResult::Rewritten(_)), inlines);
}

#[test]
fn after_stops_the_accessed_buffer_walk() {
    // Three params sit behind an AFTER's ordering dep; this compute reads only
    // the buffer the AFTER passes through. Walking into the dep would count 4
    // buffers and keep the bufferize; the AFTER costs its own buffer, once.
    let r = range(8, 0);
    let reads: Vec<Arc<UOp>> =
        (1..4).map(|slot| UOp::index().buffer(param(slot, 8)).indices(vec![r.clone()]).call().expect("idx")).collect();
    let ordered = reads[0].try_add(&reads[1]).expect("add").try_add(&reads[2]).expect("add");
    let after = param(0, 8).after(smallvec![ordered]);
    let compute = UOp::index().buffer(after).indices(vec![r.clone()]).call().expect("idx");

    let idx = index_bufferize(compute, vec![r.clone()], vec![r]);
    let result = pm_remove_bufferize().rewrite(&idx, &mut ());

    assert!(matches!(result, RewriteResult::Rewritten(_)), "the AFTER's deps must not count toward the >3 cutoff");
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
        "STAGE whose reduce body reads a Param must be kept (`buffer_in_reduce`)"
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
        "STAGE(REDUCE(const)) must inline — `buffer_in_reduce` is false"
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
                !rewritten.any_in_subtree(UOp::is_invalid_marker),
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
