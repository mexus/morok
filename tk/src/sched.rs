//! Instruction-scheduling intent for the tile DSL.
//!
//! A kernel marks its pipeline loop with [`pipeline`] (placed at the loop top,
//! threaded through the in-loop buffers the body reads). The lowering pass lives in
//! `svod-codegen` and runs after linearization, splicing the gfx9 machine
//! scheduling controls in by instruction class. See [`svod_codegen::llvm::sched`].

pub use svod_codegen::llvm::sched::{SchedKind, pipeline};
