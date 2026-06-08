//! Unit + property tests for the VA → allocation registry classifier
//! (`crate::amd::va_registry`). Pure bookkeeping — no GPU required.

use proptest::prelude::*;

use crate::amd::va_registry::{AllocTag, FREED_HISTORY, VaClass, VaRegistry};

/// A VA strictly inside a live allocation classifies as `Live` with the right
/// interval, offset, handle, and tag.
#[test]
fn inside_live_alloc() {
    let reg = VaRegistry::default();
    reg.insert(0x1000, 0x400, 0x42, AllocTag::Scratch);
    match reg.classify(0x1040) {
        VaClass::Live { base, end, offset, handle, tag } => {
            assert_eq!(base, 0x1000);
            assert_eq!(end, 0x1400);
            assert_eq!(offset, 0x40);
            assert_eq!(handle, 0x42);
            assert_eq!(tag, AllocTag::Scratch);
        }
        other => panic!("expected Live, got {other:?}"),
    }
}

/// `[base, base+size)` is half-open: the base byte is in, the end byte is out.
#[test]
fn live_interval_is_half_open() {
    let reg = VaRegistry::default();
    reg.insert(0x1000, 0x400, 1, AllocTag::Vram);

    // First byte: offset 0.
    assert!(matches!(reg.classify(0x1000), VaClass::Live { offset: 0, .. }));
    // Last byte: offset size-1.
    assert!(matches!(reg.classify(0x13ff), VaClass::Live { offset: 0x3ff, .. }));
    // One past the end: NOT live (the classic off-by-one overrun).
    assert!(matches!(reg.classify(0x1400), VaClass::Unmapped { .. }));
    // One before the base: NOT live.
    assert!(matches!(reg.classify(0x0fff), VaClass::Unmapped { .. }));
}

/// After a free, a VA in the freed region classifies as `Freed` — the
/// use-after-free signal.
#[test]
fn freed_region_is_use_after_free() {
    let reg = VaRegistry::default();
    reg.insert(0x2000, 0x1000, 7, AllocTag::Scratch);
    reg.remove(0x2000);
    match reg.classify(0x2500) {
        VaClass::Freed { base, end, handle, tag } => {
            assert_eq!(base, 0x2000);
            assert_eq!(end, 0x3000);
            assert_eq!(handle, 7);
            assert_eq!(tag, AllocTag::Scratch);
        }
        other => panic!("expected Freed, got {other:?}"),
    }
}

/// A re-allocation at a previously-freed base shadows the stale freed record:
/// the live mapping wins, so we don't mis-report a valid VA as use-after-free.
#[test]
fn live_shadows_freed() {
    let reg = VaRegistry::default();
    reg.insert(0x4000, 0x1000, 1, AllocTag::Vram);
    reg.remove(0x4000);
    // Re-allocate a *smaller* buffer at the same base with a new handle.
    reg.insert(0x4000, 0x100, 2, AllocTag::Vram);

    // Inside the new live region → Live (new handle), not the stale Freed.
    assert!(matches!(reg.classify(0x4040), VaClass::Live { handle: 2, .. }));
    // Inside the old freed region but past the new (smaller) live one → Freed.
    assert!(matches!(reg.classify(0x4500), VaClass::Freed { handle: 1, .. }));
}

/// A VA between two live allocations reports both nearest neighbours with the
/// correct gaps; below-/above-all report a single neighbour.
#[test]
fn unmapped_reports_nearest_neighbours() {
    let reg = VaRegistry::default();
    reg.insert(0x1000, 0x1000, 1, AllocTag::Vram); // A: [0x1000, 0x2000)
    reg.insert(0x5000, 0x1000, 2, AllocTag::Gtt); // B: [0x5000, 0x6000)

    // In the gap between A and B.
    match reg.classify(0x3000) {
        VaClass::Unmapped { below: Some(b), above: Some(a) } => {
            assert_eq!((b.base, b.end, b.tag), (0x1000, 0x2000, AllocTag::Vram));
            assert_eq!(b.gap, 0x3000 - 0x2000); // bytes past A's end
            assert_eq!((a.base, a.end, a.tag), (0x5000, 0x6000, AllocTag::Gtt));
            assert_eq!(a.gap, 0x5000 - 0x3000); // bytes before B's start
        }
        other => panic!("expected Unmapped with both neighbours, got {other:?}"),
    }

    // Below everything: only an above-neighbour.
    assert!(matches!(reg.classify(0x500), VaClass::Unmapped { below: None, above: Some(_) }));
    // Above everything: only a below-neighbour.
    assert!(matches!(reg.classify(0x9000), VaClass::Unmapped { below: Some(_), above: None }));
}

/// An overrun just past an allocation's end reports that allocation as the
/// below-neighbour with the exact byte distance.
#[test]
fn overrun_gap_is_exact() {
    let reg = VaRegistry::default();
    reg.insert(0x1000, 0x1000, 1, AllocTag::Vram);

    // Exactly at the end: gap 0 (the boundary byte).
    match reg.classify(0x2000) {
        VaClass::Unmapped { below: Some(b), .. } => assert_eq!(b.gap, 0),
        other => panic!("expected below-neighbour, got {other:?}"),
    }
    // Eight bytes past the end.
    match reg.classify(0x2008) {
        VaClass::Unmapped { below: Some(b), .. } => assert_eq!(b.gap, 8),
        other => panic!("expected below-neighbour, got {other:?}"),
    }
}

/// Removing a base that was never inserted (double-free / untracked VA such as
/// the event page) is a no-op: no panic, nothing recorded as freed.
#[test]
fn remove_untracked_is_noop() {
    let reg = VaRegistry::default();
    reg.remove(0xdead_0000);
    assert!(matches!(reg.classify(0xdead_0000), VaClass::Unmapped { below: None, above: None }));
}

/// Empty registry classifies everything as fully-unmapped.
#[test]
fn empty_registry() {
    let reg = VaRegistry::default();
    assert!(matches!(reg.classify(0x1234), VaClass::Unmapped { below: None, above: None }));
}

/// The freed-history ring is bounded: after freeing more than `FREED_HISTORY`
/// distinct regions, the oldest fall off (classify back to `Unmapped`) while
/// the most-recent are still reported as `Freed`.
#[test]
fn freed_history_is_bounded() {
    let reg = VaRegistry::default();
    let n = FREED_HISTORY + 8;
    // Each region is 0x1000 wide at base i*0x1000 (disjoint), allocated then freed.
    for i in 0..n as u64 {
        let base = (i + 1) * 0x1000;
        reg.insert(base, 0x1000, i, AllocTag::Scratch);
        reg.remove(base);
    }
    // The 8 oldest frees were evicted → no longer tracked.
    for i in 0..8u64 {
        let base = (i + 1) * 0x1000;
        assert!(
            matches!(reg.classify(base + 0x10), VaClass::Unmapped { .. }),
            "oldest freed region #{i} should have been evicted",
        );
    }
    // The most-recent free is still retained.
    let newest = n as u64 * 0x1000;
    assert!(matches!(reg.classify(newest + 0x10), VaClass::Freed { .. }));
}

/// The `Display` rendering carries the keywords the human-facing fault message
/// relies on, for each class.
#[test]
fn display_keywords() {
    let reg = VaRegistry::default();
    reg.insert(0x1000, 0x1000, 1, AllocTag::Scratch);
    assert!(reg.classify(0x1010).to_string().contains("LIVE scratch"));

    reg.remove(0x1000);
    assert!(reg.classify(0x1010).to_string().contains("RECENTLY-FREED scratch"));

    assert!(reg.classify(0x9_0000).to_string().contains("NO tracked allocation"));
}

proptest! {
    /// Core invariant: a VA classifies as `Live` **iff** it lies in some live
    /// interval, and the returned interval/offset are exact. With no frees, the
    /// only other outcome is `Unmapped`.
    #[test]
    fn classify_live_iff_in_some_live_interval(
        layout in prop::collection::vec((1u64..0x1000, 1u64..0x1000), 0..24),
        probe in 0u64..0x60000,
    ) {
        let reg = VaRegistry::default();
        let mut intervals: Vec<(u64, u64)> = Vec::new();
        // Lay out strictly-disjoint intervals: advance by `gap (>=1)` before
        // each base and by `size` after, so consecutive ranges never touch.
        let mut cursor = 0x1000u64;
        for (gap, size) in layout {
            cursor += gap;
            let base = cursor;
            reg.insert(base, size as usize, base, AllocTag::Vram);
            intervals.push((base, base + size));
            cursor += size;
        }

        let containing = intervals.iter().find(|(b, e)| *b <= probe && probe < *e).copied();
        match (reg.classify(probe), containing) {
            (VaClass::Live { base, end, offset, .. }, Some((b, e))) => {
                prop_assert_eq!(base, b);
                prop_assert_eq!(end, e);
                prop_assert_eq!(offset, probe - b);
            }
            (VaClass::Unmapped { .. }, None) => {}
            (got, want) => prop_assert!(false, "classify/containment mismatch: got {:?}, containing {:?}", got, want),
        }
    }
}
