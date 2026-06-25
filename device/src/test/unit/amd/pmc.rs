//! Unit tests for PMC PM4 stream construction (GPU-free).
//!
//! `build_streams` resolves every perf-counter register and chooses SET_SH vs
//! SET_UCONFIG by absolute address — so this exercises the register table and
//! address windows without a GPU (it would have caught a missing `_HI` register).

use crate::amd::pmc::{PmcGrid, build_streams, readback_bytes};
use crate::profile::PmcCounter;

#[test]
fn readback_sizing() {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    assert_eq!(grid.instances(), 20);
    assert_eq!(readback_bytes(3, &grid), 3 * 20 * 4);
}

#[test]
fn build_streams_resolves_all_registers() {
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    let counters = PmcCounter::all();
    // Must not panic on any register name/window resolution.
    let (start, read) = build_streams(&counters, &grid, 0x1_0000);
    assert!(!start.is_empty(), "start stream programs SELECTs + CTRL");
    assert!(!read.is_empty(), "read stream copies counters out");
}

#[test]
fn build_streams_fits_dispatch_budget() {
    // gfx1151-scale grid + all counters must stay well under the 1024-dword
    // single-dispatch ring budget (the readback is the dominant contributor).
    let grid = PmcGrid { se: 2, sa: 2, wgp: 5 };
    let (start, read) = build_streams(&PmcCounter::all(), &grid, 0x1_0000);
    assert!(start.len() + read.len() < 900, "pmc streams = {} dwords", start.len() + read.len());
}
