//! JIT wrappers for Whisper.
//!
//! - `WhisperEncoderJit`: mel → encoder features. Fixed shape [1, n_mels, N_FRAMES].
//! - `WhisperDecoderJit`: encoder features + tokens → logits. Fixed shape
//!   [1, n_text_ctx] so the plan compiles once; the caller reads only the
//!   current position's logits each step.
#![allow(clippy::too_many_arguments)]

extern crate self as svod_model;

use svod_macros::jit_wrapper;

use super::model::Whisper;

// Encoder-only JIT: mel `[B, n_mels, T]` → `[B, T/2, D]`.
jit_wrapper! {
    WhisperEncoderJit(Whisper) {
        mel: Tensor,

        build(mel) {
            model.encode(mel)
        }
    }
}

// Decoder-only JIT: audio features + tokens → logits.
// Prepared at `[1, n_audio_ctx, D]` × `[1, n_text_ctx]`.
// The caller writes the current token sequence (padded with EOT to n_text_ctx)
// and reads logits at the current position.
jit_wrapper! {
    WhisperDecoderJit(Whisper) {
        audio_features: Tensor,
        tokens: Tensor,

        build(audio_features, tokens) {
            model.decode(tokens, audio_features, 0)
        }
    }
}

// Prefill JIT: initial tokens [1, init_len] + audio features → logits + K/V caches.
// Outputs: logits [1, init_len, n_vocab], self_k/v per-layer, cross_k/v per-layer.
// Compiled once at fixed init_len; the plan owns all buffers, reused per window.
// No realize() needed — logits read via copyout, K/V copied to step JIT caches.
jit_wrapper! {
    WhisperPrefillJit(Whisper) {
        tokens: Tensor,
        audio_features: Tensor,

        outputs { logits, self_k, self_v, cross_k, cross_v }

        build(tokens, audio_features) {
            model.decode_prefill(tokens, audio_features, 0)
        }
    }
}

// KV-cached decoder step JIT: single-token forward with K/V cache recycling.
// Inputs: token [1,1], pos_emb [1,1,D], self/cross K/V caches, attention mask.
// Outputs: logits [1,n_vocab], new_self_k [1,1,n_layer*H,Dh], new_self_v [...].
// After execute: copy_output_to_self_k_cache/v_cache to append new K/V at pos.
jit_wrapper! {
    WhisperDecoderStepJit(Whisper) {
        token: Tensor,
        pos_emb: Tensor,
        self_k_cache: Tensor,
        self_v_cache: Tensor,
        cross_k: Tensor,
        cross_v: Tensor,
        self_mask: Tensor,

        outputs { logits, new_self_k, new_self_v }

        build(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_mask) {
            model.decode_step(token, pos_emb, self_k_cache, self_v_cache, cross_k, cross_v, self_mask)
        }
    }
}
