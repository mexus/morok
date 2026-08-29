//! `LaneClaims` bitset: lane reservations are exclusive, released lanes are
//! reusable, and lanes beyond the initialized prefix are never handed out.

use crate::amd::connector::LaneClaims;
#[test]
fn lane_claims_are_exclusive_and_reusable() {
    let claims = LaneClaims::new(3);
    assert_eq!(claims.try_claim(0), None);
    assert_eq!(claims.try_claim(3), Some(0));
    assert_eq!(claims.try_claim(3), Some(1));
    assert_eq!(claims.try_claim(3), Some(2));
    assert_eq!(claims.try_claim(3), None);
    claims.release(1);
    assert_eq!(claims.try_claim(3), Some(1));
}

#[test]
fn lane_claims_never_expose_uninitialized_slots() {
    let claims = LaneClaims::new(4);
    claims.claim_new(0);
    assert_eq!(claims.try_claim(1), None);
    assert_eq!(claims.try_claim(2), Some(1));
}
