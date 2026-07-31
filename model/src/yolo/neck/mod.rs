//! YOLO v26 FPN+PAN neck variants.
//!
//! - [`YoloNeck`] — standard P3-P5 (layers 11-22)
//! - [`YoloNeckP2`] — extended P2-P5 (layers 11-28)
//! - [`YoloNeckP6`] — extended P3-P6 (layers 13-30)

pub(crate) mod p2;
pub(crate) mod p6;
pub(crate) mod standard;

pub use p2::YoloNeckP2;
pub use p6::YoloNeckP6;
pub use standard::YoloNeck;
