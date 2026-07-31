//! JIT wrappers for BGE-M3 and BGE-reranker-v2-m3.

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::embedder::BgeM3;
use super::reranker::BgeRerankerV2M3;

jit_wrapper! {
    BgeM3DenseJit(BgeM3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.encode_dense_batch(input_ids, attention_mask, &b)
        }
    }
}

jit_wrapper! {
    BgeM3ColbertJit(BgeM3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.encode_colbert_batch(input_ids, attention_mask, &b)
        }
    }
}

jit_wrapper! {
    BgeRerankerJit(BgeRerankerV2M3) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            model.forward_batch(input_ids, Some(attention_mask), &b)
        }
    }
}
