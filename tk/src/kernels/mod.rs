//! Kernel implementations authored on top of the tile DSL: the bf16→f32
//! [`matmul`] (M1 multi-wave + M7 size-adaptive + M2 pipeline) and the
//! [`fa`] flash-attention forward (single-warp, multi-wave, double-buffered).
//!
//! The DSL tooling lives in the crate-root modules ([`kernel`](crate::kernel),
//! [`group`](crate::group), [`tile`](crate::tile), …); this module is the place
//! for concrete kernels built from those primitives.

pub mod fa;
pub mod matmul;
