//! Shared ResNet-family building blocks.
//!
//! Originally part of `model::resnet`, factored out so sibling models
//! (`model::wespeaker`) can compose the same primitives without depending on
//! resnet's wrapping types. All conv layers are bias-less; biases live only in
//! the BN affine parameters. BatchNorm is stored with the running variance
//! already folded into `invstd = 1 / sqrt(var + eps)` so the forward path is a
//! pure affine-and-shift — the fold happens once at load time in
//! [`remap::fold_batchnorm`].

mod basic_block;
mod batchnorm;
mod bottleneck;
mod conv;
pub mod error;
pub mod remap;
mod stage;

pub use basic_block::{BasicBlock, BlockKind};
pub use batchnorm::BatchNormWeights;
pub use bottleneck::Bottleneck;
pub use conv::Conv2dWeights;
pub use error::{Error, Result};
pub use stage::{Block, ResidualStage};
