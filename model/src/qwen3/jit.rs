//! JIT wrapper for [`Qwen3Embedding`]. Bakes the `input_ids` /
//! `attention_mask` shapes into the plan and exposes `b` as the rebindable
//! batch variable. The entire pipeline (backbone + last-token pooling +
//! L2 normalize) runs in one JIT plan.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::embedder::Qwen3Embedding;
use super::reranker::Qwen3Reranker;

jit_wrapper! {
    Qwen3EmbeddingJit(Qwen3Embedding) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.encode_batch(input_ids, attention_mask, &b)
        }
    }
}

jit_wrapper! {
    Qwen3RerankerJit(Qwen3Reranker) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.forward_batch(input_ids, attention_mask, &b)
        }
    }
}
