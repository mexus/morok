use super::test_support::amd_alloc_or_skip;
use crate::amd::signal::*;
use crate::error::Error;
use crate::sync::TimelineSignal;

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
