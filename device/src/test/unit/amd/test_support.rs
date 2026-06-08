//! Shared gating helpers for the AMD hardware tests.
//!
//! Sibling modules under `crate::test::unit::amd` reach these via
//! `super::test_support`. Centralising the probe keeps the per-test boilerplate
//! to a single `let … else { return }` and gives every hardware test identical
//! skip semantics on hosts without a supported GPU.

use crate::amd::AmdAllocator;

/// Open the device-0 AMD allocator, or `None` (with a skip note) on any host
/// that lacks a supported AMD GPU — no `/dev/kfd`, unsupported arch, or missing
/// permissions. Hardware tests early-return on `None`:
///
/// ```ignore
/// let Some(alloc) = amd_alloc_or_skip() else { return };
/// ```
pub(crate) fn amd_alloc_or_skip() -> Option<AmdAllocator> {
    match AmdAllocator::new(0) {
        Ok(alloc) => Some(alloc),
        Err(_) => {
            eprintln!("skipping: no supported AMD GPU on this host");
            None
        }
    }
}

/// `true` if `alloc` drives a multi-XCC (CDNA SPX) device. The native-completion
/// and AQL-scratch probes are meaningless on a single-XCC part, so they gate on
/// this and skip (with a note) otherwise.
pub(crate) fn require_multi_xcc(alloc: &AmdAllocator) -> bool {
    if alloc.dev.node.num_xcc.max(1) > 1 {
        return true;
    }
    eprintln!("PROBE skipped: single-XCC device (multi-XCC AQL only)");
    false
}
