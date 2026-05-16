//! WeSpeaker ResNet34 speaker-embedding model
//! (`pyannote/wespeaker-voxceleb-resnet34-LM`).
//!
//! 1-channel mel-spectrogram input (`[B, T=1598, F=80]`) + per-frame attention
//! weights (`[B, T_w=799]`) → 256-d L2-normalisable speaker embedding
//! (`[B, 256]`).
//!
//! Reuses [`crate::blocks::BasicBlock`] / [`crate::blocks::ResidualStage`]; the
//! WeSpeaker variant differs only in stem (3×3 stride 1, no maxpool), width
//! schedule (32→64→128→256), input modality, and head (TSTP weighted-stats
//! pooling + `Linear(5120 → 256)`).

mod error;
mod jit;
mod model;
pub mod pickle;
mod tstp;

pub use error::{Error, Result};
pub use jit::WeSpeakerResNet34Jit;
pub use model::{EMBED_DIM, M_CHANNELS, NUM_BLOCKS, NUM_MEL_BINS, WeSpeakerConfig, WeSpeakerResNet34};
