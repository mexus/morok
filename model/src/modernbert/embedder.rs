//! [`ModernBertEmbedder`] — finished embeddings from the ModernBERT backbone:
//! `input_ids` + `attention_mask` → mean-pooled, L2-normalized `(B, D)` vectors.
//!
//! Implements `svod_arch::pipelines::text::Embed` so it drops straight into an
//! [`EmbeddingsPipeline`](svod_arch::pipelines::text::EmbeddingsPipeline). The
//! model owns the forward + fused pooling/normalization (via
//! [`ModernBertEmbedderJit`]); the pipeline owns chunking and profile assembly.
//!
//! For the one-call hub loader that builds this embedder alongside a matching
//! [`HfTokenizer`](svod_arch::pipelines::text::HfTokenizer), see
//! [`from_hub`](crate::modernbert::from_hub).
//!
//! The JIT plan is sized once at construction (from `max_batch` + `max_seq`) and
//! runs at that size every call: inputs are padded to `[max_batch, max_seq]`,
//! `execute()` runs the full batch, and the live rows are sliced out of the
//! `[max_batch, D]` output — the same pad-and-slice shape gigaam's transcriber
//! uses.

use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::text::{Embed, Embedding, Encoding, RunProfile};
use svod_runtime::StageProfile;
use svod_tensor::PrepareConfig;

use crate::jit::InputSpec;
use crate::modernbert::ModernBert;
use crate::modernbert::embedder_jit::ModernBertEmbedderJit;

#[derive(Debug, Snafu)]
pub enum EmbedderError {
    #[snafu(display("JIT op failed: {source}"))]
    Jit { source: crate::jit::JitError },
    #[snafu(display("device op failed: {source}"))]
    Device { source: svod_device::error::Error },
    #[snafu(display("embedding batch of {got} exceeds prepared max_batch {max}"))]
    CapacityExceeded { got: usize, max: usize },
}

/// Finished-embeddings model over a [`ModernBert`] backbone. Build once (eager
/// JIT prepare) and reuse across calls.
pub struct ModernBertEmbedder {
    jit: ModernBertEmbedderJit,
    max_batch: usize,
    max_seq: usize,
    hidden_size: usize,
}

impl ModernBertEmbedder {
    /// Prepare the embedder JIT at `[max_batch, max_seq]`. `max_batch`/`max_seq`
    /// are caller-chosen and typically flow in from the pipeline (the chunker's
    /// `max_seq` at assembly). The model's config must already reflect the
    /// checkpoint (e.g. via [`ModernBert::from_hub`]).
    pub fn new(model: ModernBert, max_batch: usize, max_seq: usize) -> Result<Self, EmbedderError> {
        let hidden_size = model.config.hidden_size;
        // `b` is declared `vars { b: (1, model.config.max_batch_size) }` in the JIT
        // wrapper, but the caller-chosen `max_batch` (which sizes the input buffers
        // below) is what the plan must bake in. Override the upper bound so prepare
        // binds `b` to `max_batch` and the output buffer is sized `max_batch × D` —
        // rebinding `b` to a smaller live batch at execute only shrinks the live
        // region, never the allocation (the JIT batch-rebind contract).
        let mut jit = ModernBertEmbedderJit::new(model).with_b_bound(max_batch);
        // `b` binds to max_batch at prepare (see the jit_wrapper codegen), so the
        // plan runs at max_batch every execute. ids are i64 (the embedding-gather
        // convention); the mask is i64 0/1 here and cast to bool inside the build
        // closure (InputSpec has no bool constructor).
        let ids_spec = InputSpec::i64(&[max_batch, max_seq]);
        let mask_spec = InputSpec::i64(&[max_batch, max_seq]);
        jit.prepare_with_config(ids_spec, mask_spec, &PrepareConfig::from_env()).context(JitSnafu)?;
        Ok(Self { jit, max_batch, max_seq, hidden_size })
    }
}

impl Embed for ModernBertEmbedder {
    type Error = EmbedderError;

    fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    fn embed_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> Result<(Vec<Embedding>, Option<RunProfile>), EmbedderError> {
        let b = batch.len();
        if b == 0 {
            return Ok((Vec::new(), profile.then(RunProfile::default)));
        }
        if b > self.max_batch {
            return Err(CapacityExceededSnafu { got: b, max: self.max_batch }.build());
        }

        // Write the live rows into the max_batch-sized input buffers, then bind
        // `b` to the live batch at execute — one compiled plan serves every
        // batch size ≤ max_batch (the JIT batch-rebind contract; see
        // `jit_rebinds_batch_without_reprepare`). Rows beyond `b` are ignored:
        // the symbolic-batch graph only computes the first `b`.
        pack_ids_buffer(self.jit.input_ids_mut().context(JitSnafu)?, batch, self.max_seq).context(DeviceSnafu)?;
        pack_mask_buffer(self.jit.attention_mask_mut().context(JitSnafu)?, batch, self.max_seq).context(DeviceSnafu)?;

        let vars = &[("b", b as i64)];
        let mut prof = profile.then(RunProfile::default);
        if let Some(p) = &mut prof {
            let kernels = self.jit.execute_with_vars_profiled(vars).context(JitSnafu)?;
            // Fused backbone+pool+norm: one GPU stage (kernels carry the timing;
            // host wall is negligible relative to the GPU work, like gigaam).
            p.push(StageProfile::gpu("embed", std::time::Duration::ZERO, kernels));
        } else {
            self.jit.execute_with_vars(vars).context(JitSnafu)?;
        }

        // The output buffer is always max_batch-sized (the plan bakes the upper
        // bound); only the first `b` rows are live. Read [max_batch, hidden] and
        // take them.
        let out = self.jit.output().context(JitSnafu)?;
        let view = out.as_array::<f32>().context(DeviceSnafu)?;
        let flat = view.as_slice().expect("contiguous embedding buffer");
        let d = self.hidden_size;
        let embeddings: Vec<Embedding> =
            (0..b).map(|i| Embedding { values: flat[i * d..i * d + d].to_vec() }).collect();

        Ok((embeddings, prof))
    }
}

// ─── buffer packing ─────────────────────────────────────────────────────────

/// Pad each chunk's `input_ids` into the `[max_batch, max_seq]` i64 JIT buffer,
/// zero-filling past each chunk's length and over unused rows.
pub(crate) fn pack_ids_buffer(
    buf: &mut svod_device::Buffer,
    batch: &[&Encoding],
    max_seq: usize,
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<i64>()?;
    let slice = view.as_slice_mut().expect("contiguous ids buffer");
    slice.fill(0);
    for (i, enc) in batch.iter().enumerate() {
        let take = enc.input_ids.len().min(max_seq);
        for (j, &id) in enc.input_ids[..take].iter().enumerate() {
            slice[i * max_seq + j] = id as i64;
        }
    }
    Ok(())
}

/// Pad each chunk's `attention_mask` (1 = real token, 0 = pad) into the
/// `[max_batch, max_seq]` i64 JIT buffer. The mask follows the chunk's real
/// token count; trailing pad positions and unused rows stay 0.
pub(crate) fn pack_mask_buffer(
    buf: &mut svod_device::Buffer,
    batch: &[&Encoding],
    max_seq: usize,
) -> Result<(), svod_device::error::Error> {
    let mut view = buf.as_array_mut::<i64>()?;
    let slice = view.as_slice_mut().expect("contiguous mask buffer");
    slice.fill(0);
    for (i, enc) in batch.iter().enumerate() {
        let take = enc.attention_mask.len().min(max_seq);
        for (j, &m) in enc.attention_mask[..take].iter().enumerate() {
            slice[i * max_seq + j] = m as i64;
        }
    }
    Ok(())
}
