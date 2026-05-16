//! THREEFRY-backed deterministic random tensor generation.
//!
//! The IR-level `BinaryOp::Threefry` and its decomposition pattern
//! (`pm_threefry_decomp` in `morok-schedule`) already exist. This module
//! provides the Tensor-layer wrapper:
//!
//! - `manual_seed(seed)` resets per-device RNG state.
//! - `Tensor::rand(shape)` returns a uniform `[0, 1)` float tensor.
//! - Distribution helpers (`uniform`, `kaiming_uniform`, `glorot_uniform`, …)
//!   build on the primitive.
//!
//! Determinism contract: for a fixed seed and a fixed sequence of `Tensor::rand`
//! calls, outputs are bit-identical across runs and bit-identical against
//! `jax.extend.random.threefry_2x32` for matching seeds.

pub(crate) mod distributions;
pub(crate) mod like;
pub(crate) mod primitive;
pub(crate) mod state;

pub use state::manual_seed;
