//! JIT wrapper for [`super::model::XlmRobertaModel`].

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::XlmRobertaModel;

jit_wrapper! {
    XlmRobertaJit(XlmRobertaModel) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.forward_batch(input_ids, Some(attention_mask), &b)
        }
    }
}
