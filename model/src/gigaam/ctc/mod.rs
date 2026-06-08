//! CTC head for GigaAM.
//!
//! Submodules:
//! - [`head`] — `CTCHead` (Conv1d + log-softmax projection).
//! - [`jit`] — `GigaAmCtcJit` (encoder + head fused into one plan; the RN-T
//!   path instead reuses the shared [`crate::gigaam::GigaAmEncoderJit`]).
//!
//! The unified model wrapper itself lives in [`crate::gigaam::model`].

mod head;
mod jit;

pub use head::*;
pub use jit::*;
