//! `LaneClaims` bitset: lane reservations are exclusive, released lanes are
//! reusable, and lanes beyond the initialized prefix are never handed out.

use crate::amd::connector::LaneClaims;

#[test]
fn lane_claims_are_exclusive_reusable_and_bounded_by_the_initialized_prefix() {
    let claims = LaneClaims::new(3);
    assert_eq!(claims.try_claim(0), None, "no lane is handed out below the initialized prefix");
    assert_eq!([claims.try_claim(3), claims.try_claim(3), claims.try_claim(3)], [Some(0), Some(1), Some(2)]);
    assert_eq!(claims.try_claim(3), None, "a full pool hands out nothing");
    claims.release(1);
    assert_eq!(claims.try_claim(3), Some(1));

    let claims = LaneClaims::new(4);
    claims.claim_new(0);
    assert_eq!(claims.try_claim(1), None);
    assert_eq!(claims.try_claim(2), Some(1));
}
