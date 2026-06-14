//! `svod-tk` — a ThunderKittens-style tile DSL for authoring hand-tuned GPU
//! kernels (matmul, flash-attention) as raw UOp graphs.
//!
//! This is a thin eager builder, not a backend: tiles wrap UOp buffers and emit
//! the same lowered-kernel IR (`Range` + `index().store(..).end(..)`) that the
//! normal renderer consumes. It is a port of tinygrad's `extra/thunder/tiny/tk`.
//!
//! Kernels execute through the *direct* program-pipeline path ([`launch`] /
//! [`run_kernel`]) — build a SINK, compile it, and dispatch against concrete
//! buffers — mirroring tinygrad's `sink.call(bufs)` + `run_linear`, bypassing the
//! tensor scheduler entirely.
//!
//! # Supported configuration (the only validated target)
//! - GPU: **gfx942** (AMD MI300, CDNA3)
//! - Wave width: **64** ([`WARP_THREADS`])
//! - WMMA: **bf16 inputs, f32 accumulation, K=16** → `mfma.f32.16x16x16bf16.1k`
//!
//! Other arches / dtypes / K values are intentionally out of scope.

pub mod arch;
pub mod asm;
pub mod fingerprint;
pub mod grid;
pub mod group;
pub mod index;
pub mod kernel;
pub mod kernels;
pub mod launch;
pub mod loop_scope;
pub mod math;
pub mod ops;
pub mod scaffold;
pub mod sched;
pub mod swizzle;
pub mod target;
pub mod tile;
pub mod tiles;

/// Threads per warp/wave the **register-tile fragment-layout tables**
/// ([`tiles`] strides, [`group`]'s per-lane WMMA upcast counts) are calibrated
/// for — gfx942 wave64. The *runtime* lane count flows through
/// [`ArchCaps::wave_size`](arch::ArchCaps::wave_size); this constant is only the
/// layout-table calibration, pinned to the canonical arch by the assert below.
pub const WARP_THREADS: usize = 64;
const _: () = assert!(WARP_THREADS == svod_dtype::AmdArch::Gfx942.wave_size() as usize);

pub use arch::ArchCaps;
pub use fingerprint::{KernelFingerprint, kernel_fingerprint};
pub use group::{Group, LoadInto, MoveIdx, StoreInto, SwapDir};
pub use kernel::Kernel;
pub use kernels::fa::{
    FaOpts, flash_attention, flash_attention_forward, flash_attention_forward_mw, flash_attention_forward_mw_db,
    flash_attention_with,
};
pub use kernels::matmul::matmul;
pub use launch::{
    CompiledLaunch, Error as LaunchError, Result as LaunchResult, compile, compile_kernel, graph_launch, launch,
    run_kernel,
};
pub use loop_scope::Loop;
pub use scaffold::GlSpec;
pub use swizzle::Swizzle;
pub use tile::{GL, RT, RV, RegTile, ST};
pub use tiles::{
    BaseShape, RT_16X16, RT_16X32, RT_32X16, RT_32X32, RTBaseShape, ST_16X16, ST_16X16_SWIZZLED, ST_16X32, ST_32X16,
    ST_32X32, STBaseShape, TileLayout, VecLayout,
};

#[cfg(test)]
mod test;
