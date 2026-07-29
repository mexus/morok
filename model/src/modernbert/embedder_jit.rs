//! JIT wrapper for the ModernBERT **embedder**: backbone forward (with the
//! padding mask) + fused masked mean-pool + L2-normalize, compiled as ONE JIT
//! plan. `input_ids` `(B, L)` int64 + `attention_mask` `(B, L)` bool → finished
//! `(B, D)` embeddings.
//!
//! Distinct from [`super::jit::ModernBertJit`] (the bare backbone JIT, which
//! returns the `(B, L, D)` last-hidden-state with `padding_mask = None`). The
//! mask is numerically load-bearing for embeddings — masked attention is what
//! keeps pad tokens out of real-token representations (see the backbone parity
//! test) — so the embedder takes it as a second JIT input. Fusing pool+norm in
//! the same plan keeps the `(B, L, D)` activations on-device; only the small
//! `(B, D)` embeddings are read back (same rationale as gigaam's fused
//! encoder+head JIT).

extern crate self as svod_model;

use snafu::ResultExt;
use svod_ir::SInt;
use svod_macros::jit_wrapper;

use super::error::TensorSnafu;
use super::model::ModernBert;
use super::pooling::pool_embed;

jit_wrapper! {
    ModernBertEmbedderJit(ModernBert) {
        input_ids: Tensor,
        attention_mask: Tensor,

        vars {
            b: (1, model.config.max_batch_size),
        }

        build(input_ids, attention_mask, b) {
            // The mask arrives as i64 (InputSpec has no bool constructor); cast
            // to bool where `true` = real token, the convention forward/pool expect.
            let mask = attention_mask.cast(svod_dtype::DType::Bool).context(TensorSnafu)?;
            // Shrink the mask's batch dim to the live `b` ourselves so the mask
            // passed to pool_embed matches the symbolic-batch hidden state that
            // forward_batch returns. (forward_batch shrinks its own internal
            // copy; without shrinking here, pool_embed would see a concrete
            // max_batch mask against a symbolic-b hidden → broadcast mismatch.)
            let mask = mask
                .try_shrink([Some((SInt::Const(0), b.as_sint())), None])
                .context(TensorSnafu)?;
            let hidden = model.forward_batch(input_ids, Some(&mask), &b)?;
            pool_embed(&hidden, &mask)
        }
    }
}
