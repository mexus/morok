use super::test_support::{MockAmdCall, MockAmdIface, amd_alloc_or_skip};
use crate::amd::AmdAllocator;
use crate::amd::signal::*;
use crate::error::Error;
use crate::sync::TimelineSignal;
use std::sync::Arc;

fn mock_signal() -> (Arc<MockAmdIface>, Arc<crate::amd::AmdDevice>, Arc<SignalPool>, Arc<AmdSignal>) {
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: Arc::clone(&device), device_id: 0 };
    let pool = SignalPool::new(&allocator, 1).expect("mock signal pool");
    let signal = Arc::new(pool.acquire().expect("mock signal"));
    (iface, device, pool, signal)
}

/// Live pool round-trip on real hardware (skipped when no supported AMD
/// GPU is present).
#[test]
fn signal_pool_acquire_release_roundtrip() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let pool = SignalPool::new(&alloc, 64).expect("create pool");
    let s1 = pool.acquire().expect("acquire 1");
    let s2 = pool.acquire().expect("acquire 2");
    assert_ne!(s1.value_addr(), s2.value_addr());
    assert_eq!(s1.value(), 0);
    s1.set(7);
    assert_eq!(s1.value(), 7);
    drop(s1);
    // After drop, slot is back in the pool; acquiring should give it back
    // (slot count restored).
    let s3 = pool.acquire().expect("acquire 3");
    let _ = s3;
    let _ = s2;
}

#[test]
fn signal_pool_exhaustion_is_clean_err() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    const N: usize = 64;
    let pool = SignalPool::new(&alloc, N).expect("create pool");
    let mut sigs = Vec::new();
    for _ in 0..N {
        sigs.push(pool.acquire().expect("ack"));
    }
    let err = pool.acquire().expect_err("pool must be exhausted");
    assert!(matches!(err, Error::AmdAllocFailed { .. }));
}

#[test]
fn signal_slot_releases_once_after_last_finalizer_clone_drops() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let pool = SignalPool::new(&alloc, 64).expect("create pool");
    let free = pool.free();
    let signal = Arc::new(pool.acquire().expect("acquire"));
    signal.reset(0);
    let finalizer = crate::amd::connector::SubmissionFinalizer::timeline(signal, 1, None);
    let clone = Arc::clone(&finalizer);
    assert_eq!(pool.free(), free - 1);
    drop(finalizer);
    assert_eq!(pool.free(), free - 1, "a retained finalizer must keep its completion slot");
    drop(clone);
    assert_eq!(pool.free(), free, "the finalizer's last drop releases exactly once");
}

#[test]
fn released_signal_slot_is_reset_before_reuse() {
    let Some(alloc) = amd_alloc_or_skip() else { return };
    let pool = SignalPool::new(&alloc, 64).expect("create pool");
    let slot = {
        let signal = pool.acquire().expect("first acquire");
        signal.set(99);
        signal.slot()
    };
    let reused = pool.acquire().expect("reacquire");
    assert_eq!(reused.slot(), slot);
    assert_eq!(reused.value(), 0);
}

#[test]
fn wait_events_error_stays_typed_and_poisons_owner() {
    let (iface, device, _pool, signal) = mock_signal();
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 5 }));

    let error = signal.wait(1, 10_000).expect_err("scripted wait failure");
    assert!(matches!(error, Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 5 }));
    assert!(device.is_poisoned());
    assert!(matches!(device.poison_error(), Some(Error::Runtime { message }) if message.contains("WAIT_EVENTS")));
    assert_eq!(iface.transcript().iter().filter(|call| matches!(call, MockAmdCall::WaitEvents { .. })).count(), 1);
}

#[test]
fn poisoned_owner_wakes_active_signal_waiter() {
    let (iface, device, _pool, signal) = mock_signal();
    let (tx, rx) = std::sync::mpsc::channel();
    let waiter = std::thread::spawn(move || tx.send(signal.wait(1, 60_000)).unwrap());

    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(1);
    while !iface.transcript().iter().any(|call| matches!(call, MockAmdCall::WaitEvents { .. })) {
        assert!(std::time::Instant::now() < deadline, "waiter never entered event polling");
        std::thread::yield_now();
    }
    device.poison("concurrent synthetic fault");
    let error = rx
        .recv_timeout(std::time::Duration::from_secs(1))
        .expect("poisoned waiter did not wake")
        .expect_err("poisoned waiter must fail");
    assert!(matches!(error, Error::Runtime { message } if message == "concurrent synthetic fault"));
    waiter.join().unwrap();
}

#[test]
fn future_signal_wait_fails_without_another_backend_wait() {
    let (iface, _device, _pool, signal) = mock_signal();
    iface.script_wait(Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 19 }));
    assert!(matches!(signal.wait(1, 10_000), Err(Error::AmdIoctl { ioctl: "AMDKFD_IOC_WAIT_EVENTS", errno: 19 })));
    let waits_before = iface.transcript().iter().filter(|call| matches!(call, MockAmdCall::WaitEvents { .. })).count();

    let error = signal.wait(1, 10_000).expect_err("poison must fail before polling");
    assert!(matches!(error, Error::Runtime { message } if message.contains("WAIT_EVENTS")));
    let waits_after = iface.transcript().iter().filter(|call| matches!(call, MockAmdCall::WaitEvents { .. })).count();
    assert_eq!(waits_after, waits_before, "future wait reached the backend despite device poison");
}

#[test]
fn mock_signal_pool_construction_failure_and_drop_balance_backing() {
    let iface = Arc::new(MockAmdIface::default());
    let device = iface.device();
    let allocator = AmdAllocator { dev: device, device_id: 0 };
    iface.script_alloc(Err(Error::Runtime { message: "scripted signal allocation".into() }));
    assert!(SignalPool::new(&allocator, 64).is_err());
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (0, 0, 0));

    let pool = SignalPool::new(&allocator, 64).unwrap();
    assert_eq!(iface.live_handle_count(), 1);
    drop(pool);
    assert_eq!((iface.allocation_count(), iface.free_count(), iface.live_handle_count()), (1, 1, 0));
    assert!(iface.free_issues().is_empty());
}
