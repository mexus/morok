//! AMD GPU runtime support.
//!
//! Splits cleanly into:
//! - [`compile`]: invokes the host `clang` with `--target=amdgcn-amd-amdhsa`
//!   to lower AMDLLVMRenderer output (Phase 1) to an AMDGPU code object.
//! - (later phases) device, allocator, queue, signal, program — KFD-direct
//!   runtime mirroring tinygrad's `runtime/ops_amd.py`.
//!
//! All entry points are *infallible to compile* on non-Linux hosts; runtime
//! calls return clean `Err(NoAmdGpu)` when there's no AMD GPU.

pub mod compile;

pub use compile::{compile_ir_to_amd_object, has_amdgpu_target};
