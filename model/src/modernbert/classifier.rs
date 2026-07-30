//! [`ModernBertClassifier`] — sequence classification over the ModernBERT
//! backbone: `input_ids` + `attention_mask` → raw class logits `(B, num_labels)`.
//!
//! Implements `svod_arch::pipelines::text::Classify` so it drops straight into a
//! [`ClassifyPipeline`](svod_arch::pipelines::text::ClassifyPipeline). The model
//! owns the forward + fused classification head (via
//! [`ModernBertClassifierJit`]); the pipeline owns chunking and profile
//! assembly.
//!
//! The classification head mirrors HF's `ModernBertForSequenceClassification`:
//! pool (cls or mean) → `head.dense` → GELU → `head.norm` → `classifier` linear.
//! Fused into one JIT plan so the `(B, L, D)` activations stay on-device.

use snafu::{ResultExt, Snafu};
use svod_arch::pipelines::text::{Classification, Classify, Encoding, RunProfile};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_runtime::StageProfile;
use svod_tensor::{BoundVariable, PrepareConfig, Tensor};

use crate::init::fan_in_uniform;
use crate::jit::InputSpec;
use crate::modernbert::config::{ClassifierPooling, ModernBertConfig};
use crate::modernbert::embedder::{pack_ids_buffer, pack_mask_buffer};
use crate::modernbert::error::{Result, StateSnafu, TensorSnafu};
use crate::modernbert::model::ModernBert;
use crate::modernbert::normalization::LayerNormWeights;
use crate::state::{self, HasStateDict, StateDict, get_tensor};

// ─── error ─────────────────────────────────────────────────────────────────

#[derive(Debug, Snafu)]
pub enum ClassifierError {
    #[snafu(display("JIT op failed: {source}"))]
    Jit { source: crate::jit::JitError },
    #[snafu(display("device op failed: {source}"))]
    Device { source: svod_device::error::Error },
    #[snafu(display("classification batch of {got} exceeds prepared max_batch {max}"))]
    CapacityExceeded { got: usize, max: usize },
}

// ─── head weights ──────────────────────────────────────────────────────────

/// Classification head weights: HF `head.dense` + `head.norm` + `classifier`.
/// `dense_bias` is `None` when `classifier_bias = false` (ModernBERT default).
#[derive(Clone)]
pub(crate) struct ClassifierHead {
    dense_weight: Tensor,
    dense_bias: Option<Tensor>,
    norm: LayerNormWeights,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
}

impl ClassifierHead {
    fn empty(config: &ModernBertConfig) -> Self {
        let d = config.hidden_size;
        let n = config.num_labels;
        let dt = config.dtype.clone();
        Self {
            dense_weight: fan_in_uniform(&[d, d], d, dt.clone()),
            dense_bias: None,
            norm: LayerNormWeights::with_eps(d, config.layer_norm_eps, dt.clone()),
            classifier_weight: fan_in_uniform(&[n, d], d, dt.clone()),
            classifier_bias: fan_in_uniform(&[n], d, dt),
        }
    }
}

impl HasStateDict for ClassifierHead {
    fn state_dict(&self, _prefix: &str) -> StateDict {
        let mut sd = StateDict::new();
        sd.insert("head.dense.weight".to_string(), self.dense_weight.clone());
        if let Some(b) = &self.dense_bias {
            sd.insert("head.dense.bias".to_string(), b.clone());
        }
        sd.extend(self.norm.state_dict("head.norm"));
        sd.insert("classifier.weight".to_string(), self.classifier_weight.clone());
        sd.insert("classifier.bias".to_string(), self.classifier_bias.clone());
        sd
    }

    fn load_state_dict(&mut self, sd: &StateDict, _prefix: &str) -> std::result::Result<(), state::Error> {
        self.dense_weight = get_tensor(sd, "head.dense.weight")?;
        self.dense_bias = sd.get("head.dense.bias").cloned();
        self.norm.load_state_dict(sd, "head.norm")?;
        self.classifier_weight = get_tensor(sd, "classifier.weight")?;
        self.classifier_bias = get_tensor(sd, "classifier.bias")?;
        Ok(())
    }
}

// ─── composite model (backbone + head) ─────────────────────────────────────

/// Backbone + classification head — the model type wrapped by the JIT.
/// `forward_batch` fuses backbone → pool → dense → GELU → norm → classifier
/// into a single graph.
#[derive(Clone)]
pub(crate) struct ModernBertClassificationModel {
    pub(crate) backbone: ModernBert,
    head: ClassifierHead,
    pooling: ClassifierPooling,
}

impl ModernBertClassificationModel {
    /// Deterministic-init model for testing (mirrors `ModernBert::empty`).
    #[cfg(test)]
    pub(crate) fn empty(config: &ModernBertConfig) -> Self {
        Self {
            backbone: ModernBert::empty(config.clone()),
            head: ClassifierHead::empty(config),
            pooling: config.classifier_pooling,
        }
    }

    pub(crate) fn from_state_dict(sd: &StateDict, config: &ModernBertConfig) -> Result<Self> {
        let dtype = config.dtype.clone();
        let casted: StateDict = sd
            .iter()
            .map(|(k, v)| {
                let t = if v.uop().dtype() == dtype {
                    v.clone()
                } else {
                    v.cast(dtype.clone()).unwrap_or_else(|_| v.clone())
                };
                (k.clone(), t)
            })
            .collect();

        let mut backbone = ModernBert::empty(config.clone());
        backbone.load_state_dict(&casted, "").context(StateSnafu)?;

        let mut head = ClassifierHead::empty(config);
        head.load_state_dict(&casted, "").context(StateSnafu)?;

        Ok(Self { backbone, head, pooling: config.classifier_pooling })
    }

    /// Fused forward: backbone → pool → head → classifier → logits `(B, num_labels)`.
    pub(crate) fn forward_batch(
        &self,
        input_ids: &Tensor,
        padding_mask: Option<&Tensor>,
        b: &BoundVariable,
    ) -> Result<Tensor> {
        let hidden = self.backbone.forward_batch(input_ids, padding_mask, b)?;
        let mask = padding_mask.expect("classification requires an attention mask");
        classify_head(&hidden, mask, &self.head, self.pooling)
    }
}

// ─── classify_head IR builder ──────────────────────────────────────────────

/// Numerical epsilon guarding the masked-mean denominator.
const EPS: f64 = 1e-12;

/// Pool → dense → GELU → LayerNorm → classifier linear. Pure IR builder — fused
/// into the JIT plan by the `build` closure in [`ModernBertClassifierJit`].
fn classify_head(hidden: &Tensor, mask: &Tensor, head: &ClassifierHead, pooling: ClassifierPooling) -> Result<Tensor> {
    let pooled = match pooling {
        ClassifierPooling::Cls => {
            let slice = hidden.try_shrink([None, Some((SInt::Const(0), SInt::Const(1))), None]).context(TensorSnafu)?;
            slice.try_squeeze(Some(1)).context(TensorSnafu)?
        }
        ClassifierPooling::Mean => masked_mean(hidden, mask)?,
    };

    // head.dense → GELU → head.norm
    let dense = if let Some(b) = &head.dense_bias {
        pooled.linear().weight(&head.dense_weight).bias(b).call().context(TensorSnafu)?
    } else {
        pooled.linear().weight(&head.dense_weight).call().context(TensorSnafu)?
    };
    let activated = dense.gelu_exact().context(TensorSnafu)?;
    let normed = head.norm.apply(&activated)?;

    // classifier: Linear(hidden → num_labels)
    let logits =
        normed.linear().weight(&head.classifier_weight).bias(&head.classifier_bias).call().context(TensorSnafu)?;

    logits.cast(DType::Float32).context(TensorSnafu)
}

/// Masked mean over the sequence axis (no L2-norm, unlike `pool_embed`).
/// `hidden` is `(B, L, D)`, `mask` is bool `(B, L)`. Returns `(B, D)`.
fn masked_mean(hidden: &Tensor, mask: &Tensor) -> Result<Tensor> {
    let dtype = hidden.uop().dtype();
    let eps = Tensor::const_(EPS, dtype.clone());

    let mask_f = mask.cast(dtype.clone()).context(TensorSnafu)?;
    let mask_f = mask_f.try_unsqueeze(2).context(TensorSnafu)?; // (B, L, 1)

    let xw_sum = hidden
        .try_mul(&mask_f)
        .context(TensorSnafu)?
        .sum_with()
        .axes(1isize)
        .keepdim(true)
        .call()
        .context(TensorSnafu)?;
    let denom = mask_f.sum_with().axes(1isize).keepdim(true).call().context(TensorSnafu)?;
    let denom = denom.try_add(&eps).context(TensorSnafu)?;

    let mean = xw_sum.try_div(&denom).context(TensorSnafu)?;
    mean.try_squeeze(Some(1)).context(TensorSnafu)
}

// ─── runtime (owns JIT, impl Classify) ─────────────────────────────────────

/// Finished-classifier model. Build once (eager JIT prepare) and reuse across
/// calls. Implements [`Classify`] for drop-in use with [`ClassifyPipeline`].
///
/// [`ClassifyPipeline`]: svod_arch::pipelines::text::ClassifyPipeline
pub struct ModernBertClassifier {
    jit: crate::modernbert::classifier_jit::ModernBertClassifierJit,
    max_batch: usize,
    max_seq: usize,
    num_classes: usize,
}

impl ModernBertClassifier {
    /// Prepare the classifier JIT at `[max_batch, max_seq]`.
    pub(crate) fn new(
        model: ModernBertClassificationModel,
        max_batch: usize,
        max_seq: usize,
    ) -> std::result::Result<Self, ClassifierError> {
        let num_classes = model.head.classifier_weight.shape().expect("classifier weight shape")[0]
            .as_const()
            .expect("classifier weight row count must be concrete");
        let mut jit = crate::modernbert::classifier_jit::ModernBertClassifierJit::new(model).with_b_bound(max_batch);
        let ids_spec = InputSpec::i64(&[max_batch, max_seq]);
        let mask_spec = InputSpec::i64(&[max_batch, max_seq]);
        jit.prepare_with_config(ids_spec, mask_spec, &PrepareConfig::from_env()).context(JitSnafu)?;
        Ok(Self { jit, max_batch, max_seq, num_classes })
    }
}

impl Classify for ModernBertClassifier {
    type Error = ClassifierError;

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn capacity(&self) -> (usize, usize) {
        (self.max_batch, self.max_seq)
    }

    fn classify_batch(
        &mut self,
        batch: &[&Encoding],
        profile: bool,
    ) -> std::result::Result<(Vec<Classification>, Option<RunProfile>), ClassifierError> {
        let b = batch.len();
        if b == 0 {
            return Ok((Vec::new(), profile.then(RunProfile::default)));
        }
        if b > self.max_batch {
            return Err(CapacityExceededSnafu { got: b, max: self.max_batch }.build());
        }

        pack_ids_buffer(self.jit.input_ids_mut().context(JitSnafu)?, batch, self.max_seq).context(DeviceSnafu)?;
        pack_mask_buffer(self.jit.attention_mask_mut().context(JitSnafu)?, batch, self.max_seq).context(DeviceSnafu)?;

        let vars = &[("b", b as i64)];
        let mut prof = profile.then(RunProfile::default);
        if let Some(p) = &mut prof {
            let kernels = self.jit.execute_with_vars_profiled(vars).context(JitSnafu)?;
            p.push(StageProfile::gpu("classify", std::time::Duration::ZERO, kernels));
        } else {
            self.jit.execute_with_vars(vars).context(JitSnafu)?;
        }

        let out = self.jit.output().context(JitSnafu)?;
        let view = out.as_array::<f32>().context(DeviceSnafu)?;
        let flat = view.as_slice().expect("contiguous classification buffer");
        let nc = self.num_classes;
        let classifications: Vec<Classification> =
            (0..b).map(|i| Classification { logits: flat[i * nc..i * nc + nc].to_vec() }).collect();

        Ok((classifications, prof))
    }
}
