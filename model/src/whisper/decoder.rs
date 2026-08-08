//! Text decoder: token + positional embeddings + self/cross-attention transformer blocks.

use snafu::ResultExt;
use svod_dtype::DType;
use svod_ir::ConstValue;
use svod_tensor::Tensor;

use crate::init::fan_in_uniform;
use crate::state::{self, HasStateDict, StateDict, get_tensor, prefixed};

use super::attention::{MultiHeadAttention, causal_mask};
use super::blocks::LayerNormWeights;
use super::config::ModelDimensions;
use super::error::{Result, TensorSnafu};

/// Decoder transformer block: self-attn + cross-attn + MLP, all pre-norm.
#[derive(Clone)]
pub struct DecoderBlock {
    pub attn: MultiHeadAttention,
    pub attn_ln: LayerNormWeights,
    pub cross_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNormWeights,
    pub mlp0_w: Tensor,
    pub mlp0_b: Tensor,
    pub mlp1_w: Tensor,
    pub mlp1_b: Tensor,
    pub mlp_ln: LayerNormWeights,
    pub n_state: usize,
}

impl DecoderBlock {
    pub fn empty(n_state: usize, n_head: usize) -> Self {
        Self::empty_dtype(n_state, n_head, DType::Float32)
    }

    pub fn empty_dtype(n_state: usize, n_head: usize, dtype: DType) -> Self {
        let mlp = n_state * 4;
        Self {
            attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            attn_ln: LayerNormWeights::empty_dtype(n_state, dtype.clone()),
            cross_attn: MultiHeadAttention::empty_dtype(n_state, n_head, dtype.clone()),
            cross_attn_ln: LayerNormWeights::empty_dtype(n_state, dtype.clone()),
            mlp0_w: fan_in_uniform(&[mlp, n_state], n_state, dtype.clone()),
            mlp0_b: fan_in_uniform(&[mlp], n_state, dtype.clone()),
            mlp1_w: fan_in_uniform(&[n_state, mlp], mlp, dtype.clone()),
            mlp1_b: fan_in_uniform(&[n_state], mlp, dtype.clone()),
            mlp_ln: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
        }
    }

    /// Forward with SDPA (standard path). `xa` is the encoder output.
    pub fn forward(&self, x: &Tensor, xa: &Tensor, mask: &Tensor) -> Result<Tensor> {
        // Self-attention (causal)
        let h = self.attn_ln.apply(x)?;
        let attn_out = self.attn.forward(&h, None, Some(mask))?;
        let x = x.try_add(&attn_out).context(TensorSnafu)?;

        // Cross-attention
        let h = self.cross_attn_ln.apply(&x)?;
        let cross_out = self.cross_attn.forward(&h, Some(xa), None)?;
        let x = x.try_add(&cross_out).context(TensorSnafu)?;

        // MLP
        let h = self.mlp_ln.apply(&x)?;
        let h = h.linear().weight(&self.mlp0_w).bias(&self.mlp0_b).call().context(TensorSnafu)?;
        let h = h.gelu_exact().context(TensorSnafu)?;
        let h = h.linear().weight(&self.mlp1_w).bias(&self.mlp1_b).call().context(TensorSnafu)?;
        let x = x.try_add(&h).context(TensorSnafu)?;
        Ok(x)
    }
}

impl HasStateDict for DecoderBlock {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.extend(self.attn.state_dict(&prefixed(prefix, "attn")));
        sd.extend(self.attn_ln.state_dict(&prefixed(prefix, "attn_ln")));
        sd.extend(self.cross_attn.state_dict(&prefixed(prefix, "cross_attn")));
        sd.extend(self.cross_attn_ln.state_dict(&prefixed(prefix, "cross_attn_ln")));
        sd.insert(prefixed(prefix, "mlp.0.weight"), self.mlp0_w.clone());
        sd.insert(prefixed(prefix, "mlp.0.bias"), self.mlp0_b.clone());
        sd.insert(prefixed(prefix, "mlp.2.weight"), self.mlp1_w.clone());
        sd.insert(prefixed(prefix, "mlp.2.bias"), self.mlp1_b.clone());
        sd.extend(self.mlp_ln.state_dict(&prefixed(prefix, "mlp_ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.attn.load_state_dict(sd, &prefixed(prefix, "attn"))?;
        self.attn_ln.load_state_dict(sd, &prefixed(prefix, "attn_ln"))?;
        self.cross_attn.load_state_dict(sd, &prefixed(prefix, "cross_attn"))?;
        self.cross_attn_ln.load_state_dict(sd, &prefixed(prefix, "cross_attn_ln"))?;
        self.mlp0_w = get_tensor(sd, &prefixed(prefix, "mlp.0.weight"))?;
        self.mlp0_b = get_tensor(sd, &prefixed(prefix, "mlp.0.bias"))?;
        self.mlp1_w = get_tensor(sd, &prefixed(prefix, "mlp.2.weight"))?;
        self.mlp1_b = get_tensor(sd, &prefixed(prefix, "mlp.2.bias"))?;
        self.mlp_ln.load_state_dict(sd, &prefixed(prefix, "mlp_ln"))?;
        Ok(())
    }
}

/// Whisper text decoder: token embedding + learned positional embedding +
/// N × DecoderBlock + LayerNorm + tied output projection.
#[derive(Clone)]
pub struct TextDecoder {
    pub token_embedding: Tensor,      // [n_vocab, D]
    pub positional_embedding: Tensor, // [n_text_ctx, D]
    pub blocks: Vec<DecoderBlock>,
    pub ln: LayerNormWeights,
    pub n_state: usize,
    pub n_head: usize,
    pub n_text_ctx: usize,
}

impl TextDecoder {
    pub fn empty(dims: &ModelDimensions) -> Self {
        let n_state = dims.n_text_state;
        let dtype = dims.dtype.clone();
        Self {
            token_embedding: fan_in_uniform(&[dims.n_vocab, n_state], n_state, dtype.clone()),
            positional_embedding: Tensor::zeros(&[dims.n_text_ctx, n_state], dtype.clone())
                .expect("positional embedding"),
            blocks: (0..dims.n_text_layer)
                .map(|_| DecoderBlock::empty_dtype(n_state, dims.n_text_head, dtype.clone()))
                .collect(),
            ln: LayerNormWeights::empty_dtype(n_state, dtype),
            n_state,
            n_head: dims.n_text_head,
            n_text_ctx: dims.n_text_ctx,
        }
    }

    fn pack_kv(kvs: Vec<Tensor>) -> Result<Tensor> {
        let permuted: Vec<Tensor> = kvs
            .into_iter()
            .map(|tensor| tensor.try_permute(&[0, 2, 1, 3]).context(TensorSnafu))
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&permuted.iter().collect::<Vec<_>>(), 2).context(TensorSnafu)
    }

    fn validate_kv_dtypes(keys: &[Tensor], values: &[Tensor], cache: &str) -> Result<()> {
        let dtype = keys.first().map(|tensor| tensor.uop().dtype());
        if keys.len() != values.len()
            || dtype.is_none()
            || keys.iter().chain(values).any(|tensor| Some(tensor.uop().dtype()) != dtype)
        {
            return Err(super::error::Error::Decode { msg: format!("{cache} K/V producer dtypes are inconsistent") });
        }
        Ok(())
    }

    /// Packed cross-cache activation/output dtype. Dense projections currently
    /// produce their weight dtype; quantized projections must expose a separate
    /// output contract rather than treating low-bit weight storage as cache dtype.
    pub fn cross_cache_output_dtype(&self) -> Result<DType> {
        let first = self.blocks.first().ok_or_else(|| super::error::Error::Decode {
            msg: "decoder has no cross-attention projections".to_string(),
        })?;
        let dtype = first.cross_attn.key.weight.uop().dtype();
        for block in &self.blocks {
            let key = &block.cross_attn.key;
            let value = &block.cross_attn.value;
            let consistent = key.weight.uop().dtype() == dtype
                && value.weight.uop().dtype() == dtype
                && key.bias.as_ref().is_none_or(|bias| bias.uop().dtype() == dtype)
                && value.bias.as_ref().is_none_or(|bias| bias.uop().dtype() == dtype);
            if !consistent {
                return Err(super::error::Error::Decode {
                    msg: "cross-attention K/V projection output dtypes are inconsistent".to_string(),
                });
            }
        }
        Ok(dtype.clone())
    }

    /// Project encoder features into the fixed packed cross-attention cache.
    /// This graph is independent of decoder tokens and runs once per window.
    pub fn project_cross_kv(&self, xa: &Tensor) -> Result<(Tensor, Tensor)> {
        let output_dtype = self.cross_cache_output_dtype()?;
        let xa = xa.cast(output_dtype.clone()).context(TensorSnafu)?;
        let mut cross_ks = Vec::with_capacity(self.blocks.len());
        let mut cross_vs = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            // Keep each GEMM independent from the final layer/head packing.
            let k = block.cross_attn.key.forward(&xa)?.contiguous();
            let v = block.cross_attn.value.forward(&xa)?.contiguous();
            if k.uop().dtype() != output_dtype || v.uop().dtype() != output_dtype {
                return Err(super::error::Error::Decode {
                    msg: "cross-attention K/V producers violated their output dtype contract".to_string(),
                });
            }
            cross_ks.push(block.cross_attn.split_heads(&k)?);
            cross_vs.push(block.cross_attn.split_heads(&v)?);
        }
        Ok((Self::pack_kv(cross_ks)?, Self::pack_kv(cross_vs)?))
    }

    /// Forward pass producing logits for all positions.
    /// `tokens`: `[B, L]` int tensor. `xa`: `[B, T_enc, D]` encoder output.
    /// `offset`: positional embedding offset (for KV-cached incremental decoding).
    pub fn forward(&self, tokens: &Tensor, xa: &Tensor, offset: usize) -> Result<Tensor> {
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder forward seq_len".into(),
                }),
            })?;

        // Token embedding: [B, L, D]
        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;

        // Positional embedding slice: [L, D]
        let pos_emb = self
            .positional_embedding
            .try_shrink([Some((offset as isize, (offset + seq_len) as isize)), None])
            .context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(xa.uop().dtype()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, xa, &mask)?;
        }

        // Final LayerNorm
        let x = self.ln.apply(&x)?;

        // Tied output: logits = x @ token_embedding.T  → [B, L, n_vocab]
        let logits = x.linear().weight(&self.token_embedding).call().context(TensorSnafu)?;
        logits.cast(DType::Float32).context(TensorSnafu)
    }

    /// Teacher-forced decoder pass over packed cross K/V, returning raw scaled
    /// QK scores for the statically selected alignment heads.
    pub fn forward_alignment(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        alignment_heads: &[(usize, usize)],
    ) -> Result<Tensor> {
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder alignment seq_len".into(),
                }),
            })?;

        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;

        let pos_emb =
            self.positional_embedding.try_shrink([Some((0isize, seq_len as isize)), None]).context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(cross_k.uop().dtype()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        let mut selected_qk: Vec<Option<Tensor>> = (0..alignment_heads.len()).map(|_| None).collect();
        for (layer, block) in self.blocks.iter().enumerate() {
            let h = block.attn_ln.apply(&x)?;
            let attn_out = block.attn.forward(&h, None, Some(&mask))?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            let h = block.cross_attn_ln.apply(&x)?;
            let query = block.cross_attn.split_heads(&block.cross_attn.query.forward(&h)?)?;
            let head_start = layer * self.n_head;
            let head_end = head_start + self.n_head;
            let layer_ck = cross_k
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let layer_cv = cross_v
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let cross_out = query
                .scaled_dot_product_attention()
                .key(&layer_ck)
                .value(&layer_cv)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let cross_out = block.cross_attn.merge_heads(&cross_out)?;
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            let layer_heads: Vec<(usize, usize)> = alignment_heads
                .iter()
                .enumerate()
                .filter_map(|(selected, &(selected_layer, head))| (selected_layer == layer).then_some((selected, head)))
                .collect();
            for (selected, head) in layer_heads {
                let selected_q = query
                    .try_shrink([None, Some((head as isize, head as isize + 1)), None, None])
                    .context(TensorSnafu)?;
                let selected_k = layer_ck
                    .try_shrink([None, Some((head as isize, head as isize + 1)), None, None])
                    .context(TensorSnafu)?;
                let kt = selected_k.try_transpose(-1, -2).context(TensorSnafu)?;
                let scores = selected_q.matmul(&kt).context(TensorSnafu)?;
                let scale = Tensor::const_(
                    ConstValue::Float(1.0 / ((self.n_state / self.n_head) as f64).sqrt()),
                    scores.uop().dtype().clone(),
                );
                selected_qk[selected] = Some(scores.try_mul(&scale).context(TensorSnafu)?);
            }

            let h = block.mlp_ln.apply(&x)?;
            let h = h.linear().weight(&block.mlp0_w).bias(&block.mlp0_b).call().context(TensorSnafu)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = h.linear().weight(&block.mlp1_w).bias(&block.mlp1_b).call().context(TensorSnafu)?;
            x = x.try_add(&h).context(TensorSnafu)?;
        }
        let selected_qk = selected_qk
            .into_iter()
            .map(|qk| {
                qk.ok_or_else(|| super::error::Error::Tensor {
                    source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                        operation: "alignment head layer out of range".into(),
                    }),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Tensor::cat(&selected_qk.iter().collect::<Vec<_>>(), 1)
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)
    }

    /// Prefill consuming fixed packed cross-attention caches.
    /// Returns `(logits[1, init_len, n_vocab], self_k, self_v)`, where each packed
    /// self K/V is [1, seq_len, n_layer*H, Dh].
    #[allow(clippy::too_many_arguments)]
    pub fn forward_prefill(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        offset: usize,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let seq_len =
            tokens.shape().context(TensorSnafu)?[1].as_const().ok_or_else(|| super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "decoder prefill seq_len".into(),
                }),
            })?;

        let tok_emb = self.token_embedding.embedding(tokens).context(TensorSnafu)?;
        let pos_emb = self
            .positional_embedding
            .try_shrink([Some((offset as isize, (offset + seq_len) as isize)), None])
            .context(TensorSnafu)?;

        let x = tok_emb.try_add(&pos_emb).context(TensorSnafu)?;
        let x = x.cast(cross_k.uop().dtype()).context(TensorSnafu)?;

        let mask = causal_mask(seq_len, x.uop().dtype().clone())?;

        let mut x = x;
        let mut self_ks: Vec<Tensor> = Vec::with_capacity(self.blocks.len());
        let mut self_vs: Vec<Tensor> = Vec::with_capacity(self.blocks.len());

        for (layer, block) in self.blocks.iter().enumerate() {
            let h = block.attn_ln.apply(&x)?;
            let (attn_out, sk, sv) = block.attn.forward_return_kv(&h, None, Some(&mask))?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            self_ks.push(block.attn.split_heads(&sk)?);
            self_vs.push(block.attn.split_heads(&sv)?);

            let h = block.cross_attn_ln.apply(&x)?;
            let query = block.cross_attn.split_heads(&block.cross_attn.query.forward(&h)?)?;
            let head_start = layer * self.n_head;
            let head_end = head_start + self.n_head;
            let layer_ck = cross_k
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let layer_cv = cross_v
                .try_shrink([None, None, Some((head_start as isize, head_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;
            let cross_out = query
                .scaled_dot_product_attention()
                .key(&layer_ck)
                .value(&layer_cv)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let cross_out = block.cross_attn.merge_heads(&cross_out)?;
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            let h = block.mlp_ln.apply(&x)?;
            let h = h.linear().weight(&block.mlp0_w).bias(&block.mlp0_b).call().context(TensorSnafu)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = h.linear().weight(&block.mlp1_w).bias(&block.mlp1_b).call().context(TensorSnafu)?;
            x = x.try_add(&h).context(TensorSnafu)?;
        }

        let x = self.ln.apply(&x)?;
        let logits = x
            .linear()
            .weight(&self.token_embedding)
            .call()
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)?;

        Self::validate_kv_dtypes(&self_ks, &self_vs, "self-cache")?;
        Ok((logits, Self::pack_kv(self_ks)?, Self::pack_kv(self_vs)?))
    }

    /// Decoder logits using an already prepared cross-attention cache.
    pub fn forward_with_cross_kv(
        &self,
        tokens: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        offset: usize,
    ) -> Result<Tensor> {
        let (logits, _, _) = self.forward_prefill(tokens, cross_k, cross_v, offset)?;
        Ok(logits)
    }

    /// Single-token forward with KV cache. Used for incremental decoding.
    /// Works for any batch size B (B=1 for greedy, B=beam_size for beam search).
    ///
    /// - `token`: [B, 1] int32
    /// - `pos_emb`: [B, 1, D] positional embedding for this position
    /// - `self_k_cache`: [B, max_len, n_layer*H, Dh] self-attn K cache
    /// - `self_v_cache`: [B, max_len, n_layer*H, Dh] self-attn V cache
    /// - `cross_k`: [B, n_audio_ctx, n_layer*H, Dh] cross-attn K (fixed)
    /// - `cross_v`: [B, n_audio_ctx, n_layer*H, Dh] cross-attn V (fixed)
    /// - `self_mask`: [B(or 1), 1, 1, max_len+1] bool mask; true masks a position
    ///
    /// Returns `(logits[B, n_vocab], new_self_k[B, 1, n_layer*H, Dh], new_self_v[...])`.
    #[allow(clippy::too_many_arguments)]
    pub fn forward_step(
        &self,
        token: &Tensor,
        pos_emb: &Tensor,
        self_k_cache: &Tensor,
        self_v_cache: &Tensor,
        cross_k: &Tensor,
        cross_v: &Tensor,
        self_mask: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        let n_head = self.n_head;
        let n_layer = self.blocks.len();
        let d_head = self.n_state / n_head;

        // Infer batch from token shape
        let token_shape = token.shape().context(TensorSnafu)?;
        let batch = token_shape[0].as_const().ok_or_else(|| super::error::Error::Tensor {
            source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                operation: "forward_step batch".into(),
            }),
        })?;

        // Embed single token + positional embedding
        let tok_emb = self.token_embedding.embedding(token).context(TensorSnafu)?;
        let x = tok_emb.try_add(pos_emb).context(TensorSnafu)?;
        let x = x.cast(self_k_cache.uop().dtype()).context(TensorSnafu)?;

        let mut x = x;
        let mut new_ks: Vec<Tensor> = Vec::with_capacity(n_layer);
        let mut new_vs: Vec<Tensor> = Vec::with_capacity(n_layer);

        for (l, block) in self.blocks.iter().enumerate() {
            let lh_start = l * n_head;
            let lh_end = (l + 1) * n_head;

            // ── Self-attn with cache ─────────────────────────────────────
            let h = block.attn_ln.apply(&x)?;
            let q = block.attn.query.forward(&h)?;
            let new_k_raw = block.attn.key.forward(&h)?;
            let new_v_raw = block.attn.value.forward(&h)?;

            // Split heads: [B, 1, D] → [B, H, 1, Dh]
            let q_h = block.attn.split_heads(&q)?;
            let new_k_h = block.attn.split_heads(&new_k_raw)?;
            let new_v_h = block.attn.split_heads(&new_v_raw)?;

            // Slice this layer's cached K/V: [B, max_len, n_layer*H, Dh]
            // → [B, max_len, H, Dh] → permute → [B, H, max_len, Dh]
            let cached_k = self_k_cache
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?; // [B, H, max_len, Dh]
            let cached_v = self_v_cache
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;

            // Concatenate cached K/V with new K/V along seq dim:
            // [B, H, max_len, Dh] cat [B, H, 1, Dh] → [B, H, max_len+1, Dh]
            let full_k = Tensor::cat(&[&cached_k, &new_k_h], 2).context(TensorSnafu)?;
            let full_v = Tensor::cat(&[&cached_v, &new_v_h], 2).context(TensorSnafu)?;

            // Attention with a boolean mask, avoiding promotion of low-precision scores.
            let out = q_h
                .scaled_dot_product_attention()
                .key(&full_k)
                .value(&full_v)
                .attn_mask(self_mask)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let attn_out = block.attn.merge_heads(&out)?;
            let attn_out = block.attn.out.forward(&attn_out)?;
            x = x.try_add(&attn_out).context(TensorSnafu)?;

            // ── Cross-attn (fixed cache, no mask) ────────────────────────
            let h = block.cross_attn_ln.apply(&x)?;
            let cq = block.cross_attn.query.forward(&h)?;
            let cq_h = block.cross_attn.split_heads(&cq)?;

            let layer_ck = cross_k
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?; // [B, H, n_audio_ctx, Dh]
            let layer_cv = cross_v
                .try_shrink([None, None, Some((lh_start as isize, lh_end as isize)), None])
                .context(TensorSnafu)?
                .try_permute(&[0, 2, 1, 3])
                .context(TensorSnafu)?;

            let cross_out = cq_h
                .scaled_dot_product_attention()
                .key(&layer_ck)
                .value(&layer_cv)
                .is_causal(false)
                .call()
                .context(TensorSnafu)?;
            let cross_out = block.cross_attn.merge_heads(&cross_out)?;
            let cross_out = block.cross_attn.out.forward(&cross_out)?;
            x = x.try_add(&cross_out).context(TensorSnafu)?;

            // ── MLP ───────────────────────────────────────────────────────
            let h = block.mlp_ln.apply(&x)?;
            let h = h.linear().weight(&block.mlp0_w).bias(&block.mlp0_b).call().context(TensorSnafu)?;
            let h = h.gelu_exact().context(TensorSnafu)?;
            let h = h.linear().weight(&block.mlp1_w).bias(&block.mlp1_b).call().context(TensorSnafu)?;
            x = x.try_add(&h).context(TensorSnafu)?;

            // Collect new K/V for cache update: [B, H, 1, Dh]
            new_ks.push(new_k_h);
            new_vs.push(new_v_h);
        }

        // Permute each layer's K/V from [B, H, 1, Dh] to [B, 1, H, Dh],
        // then cat along dim 1 → [B, n_layer, H, Dh] → reshape [B, 1, n_layer*H, Dh].
        // Catting along dim 0 would interleave beams and layers for B > 1.
        let permuted_ks: Vec<Tensor> =
            new_ks.iter().map(|t| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)).collect::<Result<Vec<_>>>()?;
        let permuted_vs: Vec<Tensor> =
            new_vs.iter().map(|t| t.try_permute(&[0, 2, 1, 3]).context(TensorSnafu)).collect::<Result<Vec<_>>>()?;

        Self::validate_kv_dtypes(&permuted_ks, &permuted_vs, "step self-cache")?;

        let stacked_k = Tensor::cat(&permuted_ks.iter().collect::<Vec<_>>(), 1).context(TensorSnafu)?;
        let stacked_v = Tensor::cat(&permuted_vs.iter().collect::<Vec<_>>(), 1).context(TensorSnafu)?;

        let new_k_flat = stacked_k
            .try_reshape(&[
                svod_ir::SInt::Const(batch),
                svod_ir::SInt::Const(1usize),
                svod_ir::SInt::Const(n_layer * n_head),
                svod_ir::SInt::Const(d_head),
            ])
            .context(TensorSnafu)?;
        let new_v_flat = stacked_v
            .try_reshape(&[
                svod_ir::SInt::Const(batch),
                svod_ir::SInt::Const(1usize),
                svod_ir::SInt::Const(n_layer * n_head),
                svod_ir::SInt::Const(d_head),
            ])
            .context(TensorSnafu)?;
        let x = self.ln.apply(&x)?;
        let logits = x
            .linear()
            .weight(&self.token_embedding)
            .call()
            .context(TensorSnafu)?
            .cast(DType::Float32)
            .context(TensorSnafu)?;

        // logits is [B, 1, n_vocab] → reshape to [B, n_vocab]
        let n_vocab = self.token_embedding.shape().context(TensorSnafu)?[0].as_const().ok_or_else(|| {
            super::error::Error::Tensor {
                source: Box::new(svod_tensor::error::Error::SymbolicShapeUnsupported {
                    operation: "forward_step n_vocab".into(),
                }),
            }
        })?;
        let logits =
            logits.try_reshape(&[svod_ir::SInt::Const(batch), svod_ir::SInt::Const(n_vocab)]).context(TensorSnafu)?;

        Ok((logits, new_k_flat, new_v_flat))
    }
}

impl HasStateDict for TextDecoder {
    fn state_dict(&self, prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert(prefixed(prefix, "token_embedding.weight"), self.token_embedding.clone());
        sd.insert(prefixed(prefix, "positional_embedding"), self.positional_embedding.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            sd.extend(block.state_dict(&prefixed(prefix, &format!("blocks.{i}"))));
        }
        sd.extend(self.ln.state_dict(&prefixed(prefix, "ln")));
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, prefix: &str) -> std::result::Result<(), state::Error> {
        self.token_embedding = get_tensor(sd, &prefixed(prefix, "token_embedding.weight"))?;
        self.positional_embedding = get_tensor(sd, &prefixed(prefix, "positional_embedding"))?;
        for (i, block) in self.blocks.iter_mut().enumerate() {
            block.load_state_dict(sd, &prefixed(prefix, &format!("blocks.{i}")))?;
        }
        self.ln.load_state_dict(sd, &prefixed(prefix, "ln"))?;
        Ok(())
    }
}
