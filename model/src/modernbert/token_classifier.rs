//! [`ModernBertTokenClassifier`] — token classification over the ModernBERT
//! backbone (NER, POS tagging, chunking, …): `input_ids` + `attention_mask` →
//! per-token raw logits `(seq_len, num_labels)` per chunk.
//!
//! Implements `svod_arch::pipelines::text::Recognize` so it drops straight into
//! an [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline). The
//! model owns the forward + fused token head (via
//! [`ModernBertTokenClassifierJit`]); the pipeline owns chunking, profile
//! assembly, and span decoding (`labels_for_tokens` / `group_spans`).
//!
//! The token head is HF's `ModernBertPredictionHead` + `classifier` applied to
//! the full `(B, L, D)` last-hidden-state — no pooling, so every position gets a
//! logits row. It reuses the sequence-classification head's weights and IR tail
//! ([`prediction_head_tail`]); the two heads differ only in pooling.

use snafu::ResultExt;
use svod_arch::pipelines::text::{
    ChunkTokenClassification, Encoder, Encoding, Recognize, RunProfile, TextChunk, TokenClassification,
};
use svod_tensor::{BoundVariable, PrepareConfig, Tensor};

use crate::jit::InputSpec;
use crate::modernbert::classifier::{ClassifierHead, prediction_head_tail};
use crate::modernbert::config::ModernBertConfig;
use crate::modernbert::error::{Result, StateSnafu};
use crate::modernbert::head_jit::{HeadError, JitSnafu, execute_head};
use crate::modernbert::model::ModernBert;
use crate::state::{HasStateDict, StateDict};

// ─── composite model (backbone + head) ─────────────────────────────────────

/// Backbone + token-classification head — the model type wrapped by the JIT.
/// `forward_batch` fuses backbone → [`prediction_head_tail`] over the full
/// `(B, L, D)` hidden state into one graph. Unlike
/// [`ModernBertClassificationModel`](super::classifier::ModernBertClassificationModel)
/// there is no pooling — every token position gets a logits row.
#[derive(Clone)]
pub(crate) struct ModernBertTokenClassificationModel {
    pub(crate) backbone: ModernBert,
    head: ClassifierHead,
}

impl ModernBertTokenClassificationModel {
    /// Deterministic-init model for testing (mirrors `ModernBert::empty`).
    #[cfg(test)]
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        Self { backbone: ModernBert::empty(config.clone()), head: ClassifierHead::empty(config) }
    }

    pub(crate) fn from_state_dict(sd: &StateDict, config: &ModernBertConfig) -> Result<Self> {
        let casted = crate::state::cast_all(sd, config.dtype.clone());

        let mut backbone = ModernBert::empty(config.clone());
        backbone.load_state_dict(&casted, "").context(StateSnafu)?;

        let mut head = ClassifierHead::empty(config);
        head.load_state_dict(&casted, "").context(StateSnafu)?;

        Ok(Self { backbone, head })
    }

    /// Fused forward: backbone → prediction head tail → logits `(B, L, num_labels)`.
    /// The mask is required (load-bearing for backbone attention); the head
    /// itself doesn't pool, so it consumes no mask.
    pub(crate) fn forward_batch(
        &self,
        input_ids: &Tensor,
        padding_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let mask = padding_mask.expect("token classification requires an attention mask");
        let hidden = self.backbone.forward_batch(input_ids, Some(mask), b)?;
        prediction_head_tail(&hidden, &self.head)
    }
}

// ─── runtime (owns JIT, impl Encoder + Recognize) ──────────────────────────

/// Finished token-classifier model. Build once (eager JIT prepare) and reuse
/// across calls. Implements [`Encoder`] (with [`Recognize`] fixing the output
/// kinds) for drop-in use with
/// [`EncoderPipeline`](svod_arch::pipelines::text::EncoderPipeline).
pub struct ModernBertTokenClassifier {
    jit: crate::modernbert::token_classifier_jit::ModernBertTokenClassifierJit,
    max_batch: usize,
    max_seq: usize,
    num_labels: usize,
}

impl ModernBertTokenClassifier {
    /// Prepare the token-classifier JIT at `[max_batch, max_seq]`.
    pub(crate) fn new(
        model: ModernBertTokenClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, HeadError> {
        let num_labels = model.head.num_labels();
        let mut jit =
            crate::modernbert::token_classifier_jit::ModernBertTokenClassifierJit::new(model).with_b_bound(max_batch);
        let ids_spec = InputSpec::i64(&[max_batch, max_seq]);
        let mask_spec = InputSpec::i64(&[max_batch, max_seq]);
        jit.prepare_with_config(ids_spec, mask_spec, &PrepareConfig::from_env()).context(JitSnafu)?;
        Ok(Self { jit, max_batch, max_seq, num_labels })
    }
}

impl Encoder for ModernBertTokenClassifier {
    type Output = TokenClassification;
    type ChunkOutput = ChunkTokenClassification;
    type Error = HeadError;

    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    fn run_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<TokenClassification>, Option<RunProfile>), HeadError> {
        let (b, flat, prof) = execute_head(&mut self.jit, batch, self.max_batch, self.max_seq, profile, "recognize")?;
        // Output is row-major `(B, max_seq, num_labels)`; each chunk occupies a
        // `max_seq`-wide row slab. Slice to each chunk's live token count so the
        // returned grid is `(seq_len, num_labels)` — padding positions dropped.
        let nl = self.num_labels;
        let stride = self.max_seq * nl;
        let results: Vec<TokenClassification> = (0..b)
            .map(|i| {
                let seq_len = batch[i].input_ids.len().min(self.max_seq);
                let start = i * stride;
                TokenClassification { logits: flat[start..start + seq_len * nl].to_vec(), num_labels: nl }
            })
            .collect();
        Ok((results, prof))
    }

    fn attach(chunk: &TextChunk, out: TokenClassification) -> ChunkTokenClassification {
        ChunkTokenClassification {
            byte_offset: chunk.byte_offset,
            token_offsets: chunk.encoding.offsets.clone(),
            special_tokens_mask: chunk.encoding.special_tokens_mask.clone(),
            logits: out.logits,
            num_labels: out.num_labels,
        }
    }
}

impl Recognize for ModernBertTokenClassifier {
    fn num_labels(&self) -> usize {
        self.num_labels
    }
}
