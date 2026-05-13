//! CTC head for GigaAM.
//!
//! Submodules:
//! - [`head`] — `CTCHead` (Conv1d + log-softmax projection).
//! - [`model`] — `GigaAm`: encoder + head + loaders + forward.
//! - [`jit`] — `GigaAmJit`, `GigaAmBatchedJit`.

mod head;
mod jit;
mod model;

pub use head::*;
pub use jit::*;
pub use model::*;
