use crate::yolo::{YoloScale, make_depth, make_divisible, scale_channels};

#[test]
fn make_divisible_rounds_up_to_multiple_of_8() {
    assert_eq!(make_divisible(16, 8), 16);
    assert_eq!(make_divisible(17, 8), 24);
    assert_eq!(make_divisible(64, 8), 64);
}

#[test]
fn nano_scale_channels() {
    let s = YoloScale::Nano;
    assert_eq!(scale_channels(64, s), 16);
    assert_eq!(scale_channels(128, s), 32);
    assert_eq!(scale_channels(256, s), 64);
    assert_eq!(scale_channels(512, s), 128);
    assert_eq!(scale_channels(1024, s), 256);
}

#[test]
fn depth_halves_repeats_for_nano() {
    let s = YoloScale::Nano;
    assert_eq!(make_depth(2, s), 1);
    assert_eq!(make_depth(1, s), 1);
}

#[test]
fn depth_preserved_for_large() {
    let s = YoloScale::Large;
    assert_eq!(make_depth(2, s), 2);
    assert_eq!(make_depth(1, s), 1);
}
