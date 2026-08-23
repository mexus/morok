//! Unit tests for PMC PM4 stream construction (GPU-free).
//!
//! `build_streams` resolves every perf-counter register and chooses SET_SH vs
//! SET_UCONFIG by absolute address — so this exercises the register table and
//! address windows without a GPU (it would have caught a missing `_HI` register).

use super::test_support::MockAmdIface;
use crate::allocator::RawBuffer;
use crate::amd::AmdAllocator;
use crate::amd::connector::SubmissionFinalizer;
use crate::amd::pmc::{PmcGrid, PmcHandle, build_streams, readback_bytes};
use crate::error::Error;
use crate::profile::PmcCounter;
use std::sync::Arc;

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

fn mock_pmc_handle(
    iface: &Arc<MockAmdIface>,
) -> (Arc<crate::amd::signal::SignalPool>, PmcHandle, Arc<crate::amd::signal::AmdSignal>) {
    let dev = iface.device();
    let allocator = AmdAllocator { dev, device_id: 0 };
    let pool = crate::amd::signal::SignalPool::new(&allocator, 64).unwrap();
    let signal = Arc::new(pool.acquire().unwrap());
    let finalizer = SubmissionFinalizer::timeline(Arc::clone(&signal), 1, None);
    let buffer = allocator.alloc_uncached(64).unwrap();
    let host = match &buffer {
        RawBuffer::AmdDevice { host_ptr: Some(host), .. } => *host,
        other => panic!("unexpected readback buffer: {other:?}"),
    };
    let handle = PmcHandle::new(Arc::clone(&signal), finalizer, buffer, host, Vec::new(), 0);
    (pool, handle, signal)
}

#[test]
fn mock_pmc_readback_frees_after_retirement() {
    let iface = Arc::new(MockAmdIface::default());
    let (pool, handle, signal) = mock_pmc_handle(&iface);
    signal.reset(1);
    drop(handle);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (2, 1, 1));
    drop(signal);
    drop(pool);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (2, 0));
    assert!(iface.free_issues().is_empty());
}

#[test]
fn mock_pmc_failed_drain_poisons_and_quarantines_readback() {
    let iface = Arc::new(MockAmdIface::default());
    let (pool, handle, signal) = mock_pmc_handle(&iface);
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "mock PMC drain", errno: 5 }));
    drop(handle);
    assert!(signal.wait_signal_value(1, 1).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (2, 0, 2));
    drop(signal);
    drop(pool);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (0, 2));
}
