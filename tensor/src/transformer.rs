//! Transformer building blocks: embedding, attention, rotary position embeddings.

use crate::Tensor;
use bon::bon;
use snafu::{OptionExt, ensure};
use svod_dtype::DType;
use svod_ir::ConstValue;

use crate::error::{FloatDTypeRequiredSnafu, NdimMinimumSnafu, SymbolicShapeUnsupportedSnafu};

type Result<T> = crate::Result<T>;

impl Tensor {
    /// Embedding lookup: `self` is the weight table `[vocab_size, embed_dim]`.
    /// Returns `self[indices]` with shape `[*indices.shape, embed_dim]`.
    pub fn embedding(&self, indices: &Tensor) -> Result<Tensor> {
        let weight_shape = self.shape()?;
        let embed_dim =
            weight_shape[1].as_const().context(SymbolicShapeUnsupportedSnafu { operation: "embedding" })? as isize;
        let idx_shape = indices.shape()?;

        let flat = indices.try_reshape([-1])?;
        let expanded = flat.try_unsqueeze(-1)?.try_expand([-1, embed_dim])?;
        let gathered = self.gather(0, &expanded)?;

        let mut out_shape: Vec<isize> = idx_shape
            .iter()
            .map(|d| Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "embedding" })? as isize))
            .collect::<Result<_>>()?;
        out_shape.push(embed_dim);
        gathered.try_reshape(&out_shape)
    }

    /// Apply rotary position embedding rotation.
    /// `self`: `[..., rot_dim]` tensor to rotate.
    /// `cos`, `sin`: broadcastable to `self`'s shape `[..., rot_dim/2]`.
    /// If interleaved: pairs are (even, odd) indices.
    /// If not interleaved: pairs are (first_half, second_half).
    pub fn apply_rotary_emb(&self, cos: &Tensor, sin: &Tensor, interleaved: bool) -> Result<Tensor> {
        let shape = self.shape()?;
        let last_dim = shape
            .last()
            .context(NdimMinimumSnafu { op: "apply_rotary_emb", min: 1usize, actual: 0usize })?
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })?;
        let half = last_dim / 2;

        let (x1, x2) = if interleaved {
            let mut rs: Vec<isize> = shape
                .iter()
                .take(shape.len() - 1)
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            rs.push(half as isize);
            rs.push(2);
            let r = self.try_reshape(&rs)?;
            let p = r.split(&[1, 1], -1)?;
            (p[0].try_squeeze(Some(-1))?, p[1].try_squeeze(Some(-1))?)
        } else {
            let p = self.split(&[half, half], -1)?;
            (p[0].clone(), p[1].clone())
        };

        let real = x1.try_mul(cos)?.try_sub(&x2.try_mul(sin)?)?;
        let imag = x1.try_mul(sin)?.try_add(&x2.try_mul(cos)?)?;

        if interleaved {
            let stacked = Tensor::stack(&[&real, &imag], -1)?;
            let mut fs: Vec<isize> = shape
                .iter()
                .map(|d| {
                    Ok(d.as_const().context(SymbolicShapeUnsupportedSnafu { operation: "apply_rotary_emb" })? as isize)
                })
                .collect::<Result<_>>()?;
            // Last dim already correct from original shape
            let _ = fs.last_mut().map(|d| *d = last_dim as isize);
            stacked.try_reshape(&fs)
        } else {
            Tensor::cat(&[&real, &imag], -1)
        }
    }
}

#[bon]
impl Tensor {
    /// Scaled dot-product attention.
    /// `self` (Q): `[B, H, Sq, D]`, `key` (K): `[B, H, Sk, D]`, `value` (V): `[B, H, Sk, Dv]`.
    /// Returns `[B, H, Sq, Dv]`.
    ///
    /// `window = Some((left, right))` restricts each query `q` to keys in
    /// `[q - left, q + right]` (sliding-window / banded attention, as in
    /// ModernBERT's local layers). `None` = full (global) attention. The band is
    /// intersected with any causal mask and the boolean `attn_mask` (when the
    /// latter encodes padding).
    #[builder]
    pub fn scaled_dot_product_attention(
        &self,
        key: &Tensor,
        value: &Tensor,
        attn_mask: Option<&Tensor>,
        scale: Option<f64>,
        #[builder(default)] is_causal: bool,
        window: Option<(usize, usize)>,
        softcap: Option<f64>,
    ) -> Result<Tensor> {
        let q_dtype = self.uop().dtype();
        ensure!(
            q_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "query", dtype: q_dtype.clone() }
        );
        let k_dtype = key.uop().dtype();
        ensure!(
            k_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "key", dtype: k_dtype.clone() }
        );
        let v_dtype = value.uop().dtype();
        ensure!(
            v_dtype.is_float(),
            FloatDTypeRequiredSnafu { op: "scaled_dot_product_attention", arg: "value", dtype: v_dtype.clone() }
        );

        let q_shape = self.shape()?;
        let k_shape = key.shape()?;
        let head_dim = q_shape[q_shape.len() - 1]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let scale_val = scale.unwrap_or(1.0 / (head_dim as f64).sqrt());

        let scores_dtype = self.uop().dtype();

        // Q @ K^T
        let kt = key.try_transpose(-1, -2)?;
        let mut scores = self.matmul(&kt)?;

        // Scale
        let scale_t = Tensor::const_(scale_val, scores_dtype.clone());
        scores = scores.try_mul(&scale_t)?;

        // Build a boolean "keep" mask that ANDs together the causal constraint,
        // the optional sliding-window band, and the user-supplied `attn_mask`.
        // True = attend, False = masked out. The mask is applied additively
        // (mask_out → -large) before softmax, and the weights are also zeroed
        // post-softmax to guarantee exact-zero out-of-band columns even when a
        // full row is masked (softmax-of-all-equal → uniform, not zero).
        let q_len = q_shape[q_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;
        let k_len = k_shape[k_shape.len() - 2]
            .as_const()
            .context(SymbolicShapeUnsupportedSnafu { operation: "scaled_dot_product_attention" })?;

        let mut keep_mask: Option<Tensor> = None;

        // Causal: keep k ≤ q.
        if is_causal {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let causal = k_idx.try_le(&q_idx)?; // k <= q
            keep_mask = Some(causal);
        }

        // Sliding-window band: keep q - left ≤ k ≤ q + right.
        if let Some((left, right)) = window {
            let q_idx = Tensor::arange(0, Some(q_len as i64), None)?.try_unsqueeze(-1)?; // (Q, 1)
            let k_idx = Tensor::arange(0, Some(k_len as i64), None)?; // (K,)
            let lo = Tensor::const_(ConstValue::Int(left as i64), DType::Int32);
            let hi = Tensor::const_(ConstValue::Int(right as i64), DType::Int32);
            // q - left <= k  AND  k <= q + right
            let lower = q_idx.try_sub(&lo)?.try_le(&k_idx)?;
            let upper = k_idx.try_le(&q_idx.try_add(&hi)?)?;
            let band = lower.try_bitand(&upper)?;
            keep_mask = Some(match keep_mask {
                Some(prev) => prev.try_bitand(&band)?,
                None => band,
            });
        }

        // User-supplied attention mask. Bool: True = mask OUT (False = keep) —
        // invert before ANDing. Float: additive, applied separately below.
        let mut float_additive_mask: Option<&Tensor> = None;
        if let Some(mask) = attn_mask {
            if mask.uop().dtype() == DType::Bool {
                let keep = mask.logical_not()?; // True = keep
                keep_mask = Some(match keep_mask {
                    Some(prev) => prev.try_bitand(&keep)?,
                    None => keep,
                });
            } else {
                float_additive_mask = Some(mask);
            }
        }

        // Apply the boolean keep mask additively (out-of-band → -large).
        if let Some(keep) = keep_mask.as_ref() {
            let neg_large = Tensor::const_(ConstValue::min(scores_dtype.base()), scores_dtype.clone());
            scores = scores.where_(keep, &neg_large)?;
        }
        // Apply a float additive mask (e.g. a pre-computed -inf padding mask).
        if let Some(additive) = float_additive_mask {
            scores = scores.try_add(additive)?;
        }

        // Softcap
        if let Some(cap) = softcap
            && cap > 0.0
        {
            let cap_t = Tensor::const_(cap, scores_dtype.clone());
            scores = scores.try_div(&cap_t)?.tanh()?.try_mul(&cap_t)?;
        }

        // Softmax + output. Re-zero out-of-band weights so a fully-masked row
        // (whose softmax would otherwise be uniform over the masked keys)
        // produces exact zeros rather than `1/k_len` leakage.
        let mut attn_weights = scores.softmax(-1isize)?;
        if let Some(keep) = keep_mask.as_ref() {
            let zero = Tensor::const_(ConstValue::zero(scores_dtype.base()), scores_dtype);
            let masked_out = keep.logical_not()?;
            attn_weights = zero.where_(&masked_out, &attn_weights)?;
        }
        attn_weights.matmul(value)
    }
}
