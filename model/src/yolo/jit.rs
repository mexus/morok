//! JIT wrappers for YOLO v26 models.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::detect::Yolo26Detect;

jit_wrapper! {
    Yolo26DetectJit(Yolo26Detect) {
        images: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(images, b) {
            model.forward(images, &b)
        }
    }
}
