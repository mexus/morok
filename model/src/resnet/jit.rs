//! JIT wrapper for [`ResNet`]. Compiles the full forward graph once at
//! `prepare()` time and replays it per call with the batch dimension bound to
//! the live value of `b` through
//! [`execute_with_vars`](ResNetJit::execute_with_vars).
//!
//! Shape contract:
//! - `prepare(InputSpec::f32(&[max_b, 3, H, W]))` bakes `H`/`W` into the plan.
//! - `b` is rebindable on every call; values are clamped to
//!   `[1, max_batch_size]` by the macro-generated setters.
//! - For a different image resolution, prepare a fresh wrapper or call
//!   `prepare` again on this one with a new `InputSpec`.
//!
//! See `website/docs/architecture/jit-graphs.md` for the wrapper contract.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::ResNet;

jit_wrapper! {
    ResNetJit(ResNet) {
        images: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(images, b) {
            model.forward(images, &b)
        }
    }
}
