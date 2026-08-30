//! `RangeifyContext`: a range-id counter and an original → rangeified map.

use std::sync::Arc;

use svod_dtype::DType;
use svod_ir::{ConstValue, UOp};

#[test]
fn a_fresh_context_hands_out_ids_from_zero() {
    for mut ctx in [crate::rangeify::RangeifyContext::new(), crate::rangeify::RangeifyContext::default()] {
        assert_eq!(ctx.range_counter, 0);
        assert!(ctx.range_map.is_empty());

        assert_eq!((0..1000).map(|_| ctx.next_range_id()).collect::<Vec<_>>(), (0..1000).collect::<Vec<_>>());
        assert_eq!(ctx.range_counter, 1000);
    }
}

/// Recording is keyed on UOp identity: a re-record overwrites, and an unrecorded
/// key reads back as `None`.
#[test]
fn the_transform_map_is_a_last_write_wins_lookup() {
    let mut ctx = crate::rangeify::RangeifyContext::new();
    let original = UOp::native_const(1.0f32);
    let (first, second) = (UOp::native_const(2.0f32), UOp::native_const(3.0f32));

    assert!(ctx.get_rangeified(&original).is_none());

    ctx.record_transform(original.clone(), first);
    ctx.record_transform(original.clone(), second.clone());

    assert!(Arc::ptr_eq(ctx.get_rangeified(&original).expect("recorded"), &second));
    assert_eq!(ctx.range_map.len(), 1, "the second record overwrites rather than appends");
}

/// The counter and the map are independent pieces of state.
#[test]
fn recording_a_transform_does_not_consume_a_range_id() {
    let mut ctx = crate::rangeify::RangeifyContext::new();

    for i in 0..10 {
        let original = UOp::const_(DType::Int32, ConstValue::Int(i));
        let rangeified = UOp::const_(DType::Int32, ConstValue::Int(i * 2));
        ctx.record_transform(original.clone(), rangeified.clone());
        assert!(Arc::ptr_eq(ctx.get_rangeified(&original).expect("recorded"), &rangeified));
    }

    assert_eq!(ctx.range_map.len(), 10);
    assert_eq!(ctx.range_counter, 0);
    assert_eq!(ctx.next_range_id(), 0);
}
