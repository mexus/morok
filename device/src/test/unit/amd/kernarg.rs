use super::test_support::amd_alloc_or_skip;
use crate::amd::kernarg::*;
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
