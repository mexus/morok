//! LLVM IR code generation.
//!
//! This module generates LLVM IR code from optimized UOp graphs.
//!
//! # Module Structure
//!
//! - `common/`: Shared utilities (types, ctx, target enum) for CPU and AMD
//! - `cpu/`: CPU-specific rendering (host x86/AArch64 via clang)
//! - `amd/`: AMD GPU rendering (amdgcn LLVM IR via clang)
//! - `text/`: Main entry point that orchestrates target-aware rendering

pub mod amd;
pub mod common;
pub mod cpu;
pub mod sched;
pub mod text;

pub use common::LlvmTarget;
pub use cpu::render_uop as cpu_render_uop;
pub use text::LlvmTextRenderer;
