use super::test_support::amd_alloc_or_skip;
use crate::amd::signal::*;
use crate::error::Error;
use crate::sync::TimelineSignal;
use std::sync::Arc;

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
