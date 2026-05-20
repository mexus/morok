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
//!
//! # Loader gotchas — pyannote checkpoint format
//!
//! Two non-obvious things the pyannote-side `WeSpeakerResNet34` wrapper does
//! to its checkpoint that we have to undo on load (see [`pickle`] and
//! `model::rename_shortcut_to_downsample`):
//!
//! - **Nested pickle.** `torch.save({"state_dict": OrderedDict(...),
//!   "pyannote.audio": ..., "pytorch-lightning_version": ...})` does *not*
//!   surface to `repugnant-pickle::torch::RepugnantTorchTensors::new_from_file`
//!   as a flat tensor dict — the latter only handles a single top-level
//!   OrderedDict/Dict and skips entries that aren't `_rebuild_tensor_v2`
//!   calls. We use `parse_ops` + `evaluate` directly and walk the `Value`
//!   tree to descend into the `state_dict` key first.
//!
//! - **`shortcut.{0,1}` naming.** pyannote's `BasicBlock` calls the
//!   downsample sub-module `shortcut` rather than torchvision's `downsample`.
//!   The svod [`crate::blocks::BasicBlock`] uses the torchvision keys, so
//!   the loader renames `.shortcut.` → `.downsample.` in every key on the
//!   way in.
//!
//! # TSTP / interpolation gotcha
//!
//! pyannote's `StatsPool` `F.interpolate(weights, size=T, mode="nearest")`
//! is implemented in [`tstp`] as a precomputed one-hot matmul instead of via
//! [`svod_tensor::Tensor::resize`]. Reason: `resize()` (and its siblings
//! `gather` / `index_select`) call `to_vec_usize(full_shape)` and abort on
//! any symbolic dim, even when only the spatial dims are actually consumed.
//! Once that's relaxed in the tensor crate (search `TODO(symbolic-batch)`)
//! we can move back to `resize().mode(Nearest).nearest_mode(Floor).coordinate_transformation_mode(Asymmetric).axes(&[3])`.

mod error;
mod jit;
mod model;
pub mod pickle;
mod tstp;

pub use error::{Error, Result};
pub use jit::WeSpeakerResNet34Jit;
pub use model::{EMBED_DIM, M_CHANNELS, NUM_BLOCKS, NUM_MEL_BINS, WeSpeakerConfig, WeSpeakerResNet34};
