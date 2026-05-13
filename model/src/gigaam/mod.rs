//! GigaAM ASR model wrappers.
//!
//! Two head variants share the same Conformer encoder:
//! - [`ctc`] — `GigaAm` + `CTCHead`, decoded via `morok_arch::ctc`.
//! - [`rnnt`] — `GigaAmRnnt` + `RnntHead`, decoded via `morok_arch::rnnt`.
//!
//! Shared infrastructure:
//! - [`config`] — `GigaAmConfig` JSON parsing.
//! - [`encoder`] — `Encoder` + Conformer building blocks.
//! - [`rope`] — RoPE cache.
//! - [`remap`] — PyTorch state-dict key remapping.
//! - [`error`] — `Error` / `Result`.

mod config;
mod ctc;
mod encoder;
mod error;
pub(crate) mod remap;
mod rnnt;
mod rope;

pub use config::*;
pub use ctc::*;
pub use encoder::*;
pub use error::{Error, Result};
pub use rnnt::*;
pub use rope::*;
