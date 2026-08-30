//! Grid → tile L2 / chiplet swizzle.
//!
//! Maps a *flattened* 1-D workgroup id to a `(pid_m, pid_n)` block coordinate so
//! co-scheduled workgroups share an XCD / L2 slice — more A/B re-use across the
//! CDNA3's 8 XCDs (gfx942). Pure `Index` arithmetic on `block_idx[0]`: no data movement,
//! just a permutation of *which* workgroup computes *which* output block (so the
//! computed C is unchanged). Port of HipKittens `GEMM:50-65` + `util.cuh:90`.

use std::sync::Arc;

use svod_ir::UOp;

use crate::index::cidx;

/// CDNA3 XCD (chiplet) count (gfx942; 8 on multi-XCC). See HK `util.cuh:124`.
pub const NUM_XCDS: i64 = 8;
/// Grouped-M L2 swizzle group width, in blocks (HK `GEMM:48` `WGM`).
pub const WGM: i64 = 4;

/// `min(a, b)` on `Index` UOps (the IR has `max` but no `min`): `a < b ? a : b`.
fn imin(a: &Arc<UOp>, b: &Arc<UOp>) -> Arc<UOp> {
    UOp::try_where(a.lt(b), a.clone(), b.clone()).expect("imin: where")
}

/// HK `chiplet_transform_chunked` (`util.cuh:90`): reorder `wgid` so each run of
/// `chunk_size` consecutive ids lands on the same XCD (round-robin → chunked).
/// Identity above the last full `num_xcds * chunk_size` block (`wgid > limit`).
/// A bijection over `0..num_wgs`.
fn chiplet_transform_chunked(wgid: &Arc<UOp>, num_wgs: i64, num_xcds: i64, chunk_size: i64) -> Arc<UOp> {
    let block = num_xcds * chunk_size;
    let limit = (num_wgs / block) * block;
    let xcd = wgid.mod_(&cidx(num_xcds));
    let local_pid = wgid.floor_div(&cidx(num_xcds));
    let chunk_idx = local_pid.floor_div(&cidx(chunk_size));
    let pos_in_chunk = local_pid.mod_(&cidx(chunk_size));
    let transformed = chunk_idx.mul(&cidx(block)).add(&xcd.mul(&cidx(chunk_size))).add(&pos_in_chunk);
    // `wgid > limit ? wgid : transformed`  (`wgid > limit` ⇔ `limit < wgid`).
    UOp::try_where(cidx(limit).lt(wgid), wgid.clone(), transformed).expect("chiplet: keep-or-transform")
}

/// Grid / chiplet L2 swizzle: map flattened workgroup id `wgid` to a
/// `(pid_m, pid_n)` block coordinate (in block units), reordering so
/// co-scheduled workgroups hit the same XCD/L2 slice (HK `GEMM:50-65`). A
/// bijection over `0..num_wgs` for any `grid_m × grid_n` grid (`grid_m`/`grid_n`
/// kept separate so rectangular grids never generate out-of-bounds columns).
pub fn l2_swizzle(wgid: Arc<UOp>, num_wgs: i64, grid_m: i64, grid_n: i64) -> (Arc<UOp>, Arc<UOp>) {
    let wgid = chiplet_transform_chunked(&wgid, num_wgs, NUM_XCDS, WGM * WGM);
    let in_group = WGM * grid_n;
    let group_id = wgid.floor_div(&cidx(in_group));
    let first_pid_m = group_id.mul(&cidx(WGM));
    // group_size_m = min(grid_m - first_pid_m, WGM) — clamps the final M-group of
    // a non-`WGM`-multiple grid so its rows stay in range.
    let gsize_m = imin(&cidx(grid_m).sub(&first_pid_m), &cidx(WGM));
    let local = wgid.mod_(&cidx(in_group));
    let pid_m = first_pid_m.add(&local.mod_(&gsize_m));
    let pid_n = local.floor_div(&gsize_m);
    (pid_m, pid_n)
}

// A pure-`i64` reference of `l2_swizzle` (same integer math, no UOps) — the
// oracle that documents the exact algorithm — lives next to the bijection test
// it backs, in `crate::test::unit::grid` (`l2_swizzle_ref`).
