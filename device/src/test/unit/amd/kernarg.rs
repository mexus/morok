use super::test_support::{MockAmdIface, amd_alloc_or_skip};
use crate::amd::AmdAllocator;
use crate::amd::kernarg::*;
use crate::error::Error;
use std::sync::Arc;

#[test]
fn kernarg_bump_wraps_when_full() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let core = Arc::clone(alloc.dev.core());
    let arena = KernargArena::new(&alloc, &core).expect("arena");
    let half = arena.size / 2;
    let a = arena.bump(half, 16).expect("first");
    assert_eq!(a, 0);
    let b = arena.bump(half / 2, 16).expect("second");
    assert!(b > a && b < arena.size);
    // Wrap path: requests something that would overflow. The wrap drains
    // every live connector via the core (no-op on an idle device) and
    // resets the cursor.
    let c = arena.bump(arena.size - 16, 16).expect("third (wrap)");
    assert_eq!(c, 0, "expected wrap to start of arena");
}

#[test]
fn mock_kernarg_arena_construction_balances_and_scripts_allocation_failure() {
    let iface = Arc::new(MockAmdIface::default());
    let dev = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&dev), device_id: 0 };
    iface.script_alloc(Err(Error::Runtime { message: "scripted kernarg allocation".into() }));
    assert!(KernargArena::new(&allocator, dev.core()).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));

    let arena = KernargArena::new(&allocator, dev.core()).expect("arena");
    assert_eq!(iface.allocation_count(), 1);
    drop(arena);
    assert_eq!((iface.free_count(), iface.live_handle_count()), (1, 0));
    assert!(iface.free_issues().is_empty());
}

#[test]
fn mock_kernarg_wrap_failed_drain_does_not_reset_or_free() {
    let iface = Arc::new(MockAmdIface::default());
    let dev = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&dev), device_id: 0 };
    dev.core().install_signal_pool(crate::amd::signal::SignalPool::new(&allocator, 64).unwrap());
    let pool = crate::amd::connector::PoolQueue::new_with_resources(Arc::clone(dev.core()), &allocator).unwrap();
    pool.arena().bump(pool.arena().size, 1).unwrap();
    pool.next_pm4();
    iface.script_wait(Err(Error::Runtime { message: "scripted kernarg wrap drain".into() }));
    let allocations = iface.allocation_count();

    assert!(pool.arena().bump(16, 16).is_err());
    assert!(dev.is_poisoned());
    assert_eq!(iface.allocation_count(), allocations);
    assert_eq!(iface.free_count(), 0);
    drop(pool);
    assert_eq!(iface.free_count(), 0, "all hardware-referenced queue storage must remain quarantined");
}
