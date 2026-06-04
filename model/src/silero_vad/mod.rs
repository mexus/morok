//! Silero V5 voice-activity detection.
//!
//! The forward pass mirrors the upstream Silero architecture: STFT via a
//! convolutional filterbank, four 1D conv blocks, an LSTM cell carrying
//! `(h, c)` between chunks, and a sigmoid head that produces a per-chunk
//! speech probability.
//!
//! [`VadInference::probs`] exposes the raw per-chunk probability array (one
//! entry per [`NUM_SAMPLES`] samples). [`VadInference::segment`] feeds those
//! into [`svod_arch::vad::chunks_from_probs`] to produce sample ranges
//! suitable for long-form ASR — see the `svod-arch::vad` module for
//! tunable knobs (min/max chunk duration, alignment, padding, etc.).

extern crate self as svod_model;

mod splitter;

pub use splitter::{SileroVadSplitter, SileroVadSplitterError};

use std::path::Path;

use snafu::{ResultExt, Snafu};
use svod_dtype::DType;
use svod_ir::SInt;
use svod_macros::jit_wrapper;
use svod_tensor::Tensor;
use svod_tensor::nn::{Conv1d, LSTMCell, Layer, PadMode};

use crate::init::fan_in_uniform;
use crate::state;

#[derive(Debug, Snafu)]
#[snafu(visibility(pub))]
pub enum Error {
    #[snafu(display("{source}"))]
    Tensor {
        #[snafu(source(from(svod_tensor::error::Error, Box::new)))]
        source: Box<svod_tensor::error::Error>,
    },
    #[snafu(display("{source}"))]
    State {
        #[snafu(source(from(crate::state::Error, Box::new)))]
        source: Box<crate::state::Error>,
    },
    #[snafu(display("hub error: {source}"))]
    Hub { source: hf_hub::api::sync::ApiError },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Number of input samples covered by one VAD probability entry. Exposed so
/// callers can build [`svod_arch::vad::ChunkerOpts`] with the right
/// `samples_per_prob`.
pub const NUM_SAMPLES: usize = 512;
pub(crate) const CONTEXT_SIZE: usize = 64;
const STFT_PAD: usize = 64;
const CUTOFF: usize = 128 + 1;
pub(crate) const HIDDEN: usize = 128;
const CHUNK_LEN: usize = CONTEXT_SIZE + NUM_SAMPLES;

pub struct SileroVad {
    stft_conv: Conv1d,
    conv1: Conv1d,
    conv2: Conv1d,
    conv3: Conv1d,
    conv4: Conv1d,
    lstm: LSTMCell,
    final_conv: Conv1d,
}

impl SileroVad {
    pub fn from_hub() -> Result<Self> {
        let api = hf_hub::api::sync::Api::new().context(HubSnafu)?;
        let repo =
            api.repo(hf_hub::Repo::with_revision("vpermilp/silero-vad".into(), hf_hub::RepoType::Model, "main".into()));
        let path = repo.get("silero_vad_16k.safetensors").context(HubSnafu)?;
        Self::from_safetensors(&path)
    }

    pub fn from_safetensors(path: &Path) -> Result<Self> {
        let sd = state::load_safetensors(path).context(StateSnafu)?;
        Ok(Self {
            stft_conv: Conv1d::new(get(&sd, "stft_conv.weight")?, None).with_stride(128),
            conv1: Conv1d::new(get(&sd, "conv1.weight")?, Some(get(&sd, "conv1.bias")?)).with_padding((1, 1)),
            conv2: Conv1d::new(get(&sd, "conv2.weight")?, Some(get(&sd, "conv2.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv3: Conv1d::new(get(&sd, "conv3.weight")?, Some(get(&sd, "conv3.bias")?))
                .with_stride(2)
                .with_padding((1, 1)),
            conv4: Conv1d::new(get(&sd, "conv4.weight")?, Some(get(&sd, "conv4.bias")?)).with_padding((1, 1)),
            lstm: LSTMCell::new(
                get(&sd, "lstm_cell.weight_ih")?,
                get(&sd, "lstm_cell.weight_hh")?,
                get(&sd, "lstm_cell.bias_ih")?,
                get(&sd, "lstm_cell.bias_hh")?,
            ),
            final_conv: Conv1d::new(get(&sd, "final_conv.weight")?, Some(get(&sd, "final_conv.bias")?)),
        })
    }

    /// Build with random weights matching the Silero V5 16 kHz layout. Strides
    /// and paddings mirror [`Self::from_safetensors`]; the lazy
    /// `fan_in_uniform` graphs keep the forward path from collapsing under
    /// const-folding so the JIT pipeline can be exercised without a checkpoint.
    pub fn with_random_weights() -> Self {
        let dt = DType::Float32;
        let mk_conv = |shape: [usize; 3], has_bias: bool, configure: fn(Conv1d) -> Conv1d| -> Conv1d {
            let fan_in = shape[1] * shape[2];
            let weight = fan_in_uniform(&shape, fan_in, dt.clone());
            let bias = has_bias.then(|| fan_in_uniform(&[shape[0]], fan_in, dt.clone()));
            configure(Conv1d::new(weight, bias))
        };

        Self {
            stft_conv: mk_conv([258, 1, 256], false, |c| c.with_stride(128)),
            conv1: mk_conv([128, 129, 3], true, |c| c.with_padding((1, 1))),
            conv2: mk_conv([64, 128, 3], true, |c| c.with_stride(2).with_padding((1, 1))),
            conv3: mk_conv([64, 64, 3], true, |c| c.with_stride(2).with_padding((1, 1))),
            conv4: mk_conv([128, 64, 3], true, |c| c.with_padding((1, 1))),
            lstm: LSTMCell::new(
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN, HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
                fan_in_uniform(&[4 * HIDDEN], HIDDEN, dt.clone()),
            ),
            final_conv: mk_conv([1, 128, 1], true, |c| c),
        }
    }

    /// Per-window convolutional front-end (STFT filterbank + four conv blocks),
    /// **batched** over the leading axis. Input `chunks: [B, CHUNK_LEN]`, output
    /// `[B, HIDDEN]` — the LSTM input feature for each window. This part of the
    /// forward pass is **not** recurrent, so all `B` windows run in one batched
    /// dispatch; the recurrent LSTM + head runs separately (on the host) over
    /// these features. See [`VadInference::probs`].
    pub fn forward_features(&self, chunks: &Tensor) -> Result<Tensor> {
        let x = chunks
            .pad_with()
            .padding(&[(0, 0), (0, STFT_PAD as isize)])
            .mode(PadMode::Reflect)
            .call()
            .context(TensorSnafu)?
            .try_unsqueeze(1)
            .context(TensorSnafu)?;

        let x = self.stft_conv.forward(&x).context(TensorSnafu)?;

        // Keep the full (symbolic) batch dim (`None`); split STFT real/imag
        // channels and the fixed 4 time frames.
        let real = x
            .try_shrink([None, Some((SInt::Const(0), SInt::Const(CUTOFF))), Some((SInt::Const(0), SInt::Const(4)))])
            .context(TensorSnafu)?;
        let imag = x
            .try_shrink([None, Some((SInt::Const(CUTOFF), SInt::Const(258))), Some((SInt::Const(0), SInt::Const(4)))])
            .context(TensorSnafu)?;
        let x = real
            .square()
            .context(TensorSnafu)?
            .try_add(&imag.square().context(TensorSnafu)?)
            .context(TensorSnafu)?
            .try_sqrt()
            .context(TensorSnafu)?;

        let x = self.conv1.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let x = self.conv2.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let x = self.conv3.forward(&x).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        self.conv4
            .forward(&x)
            .context(TensorSnafu)?
            .relu()
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)
    }

    pub fn forward_chunk(&self, chunk: &Tensor, state_h: &Tensor, state_c: &Tensor) -> Result<Tensor> {
        let x = self.forward_features(chunk)?;

        let (new_h, new_c) = self.lstm.step(&x, state_h, state_c).context(TensorSnafu)?;

        let prob = new_h.try_unsqueeze(-1).context(TensorSnafu)?.relu().context(TensorSnafu)?;
        let prob = self
            .final_conv
            .forward(&prob)
            .context(TensorSnafu)?
            .sigmoid()
            .context(TensorSnafu)?
            .try_squeeze(Some(-1))
            .context(TensorSnafu)?
            .mean_with()
            .axes(-1isize)
            .keepdim(true)
            .call()
            .context(TensorSnafu)?;

        Tensor::cat(&[&prob, &new_h, &new_c], 1).context(TensorSnafu)
    }
}

fn get(sd: &state::StateDict, key: &str) -> Result<Tensor> {
    sd.get(key)
        .cloned()
        .ok_or_else(|| Error::State { source: Box::new(state::Error::MissingKey { key: key.to_string() }) })
}

/// Max windows per batched conv-front-end dispatch. Larger = fewer dispatches
/// (less per-dispatch round-trip latency) but more VRAM + a longer compile.
const FEATURE_BATCH: usize = 4096;

jit_wrapper! {
    SileroVadFeatureJit(SileroVad) {
        chunks: Tensor,

        build(chunks) {
            // [FEATURE_BATCH, CHUNK_LEN] -> [FEATURE_BATCH, HIDDEN] conv features.
            // Fixed batch (not a runtime var): the conv front-end is row-
            // independent, so partial batches just fill fewer rows and ignore the
            // rest — and a symbolic leading dim trips the reflect-pad lowering.
            model.forward_features(chunks)
        }
    }
}

/// Host-resident LSTM cell + sigmoid head weights for the recurrent scan.
struct VadHead {
    w_ih: ndarray::Array2<f32>, // [4H, H]
    w_hh: ndarray::Array2<f32>, // [4H, H]
    b: ndarray::Array1<f32>,    // [4H] = bias_ih + bias_hh (only the sum is
    // ever used; PyTorch keeps them split)
    final_w: ndarray::Array1<f32>, // [H]
    final_b: f32,
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

impl VadHead {
    /// Recurrent scan over batched conv features `[n, H]` (row-major). Mirrors
    /// [`LSTMCell::step`] (PyTorch `[i,f,g,o]` order) + `sigmoid(final_conv ∘
    /// relu(h))`. The input projection `W_ih·feat` is non-recurrent, so it's
    /// computed batched up front; only `W_hh·h` stays in the per-step loop.
    fn scan(&self, features: &[f32], n: usize) -> Vec<f32> {
        let h = self.final_w.len();
        let feat = ndarray::ArrayView2::from_shape((n, h), features).expect("features shape");
        let gates_x = feat.dot(&self.w_ih.t()); // [n, 4H]

        let mut hs = ndarray::Array1::<f32>::zeros(h);
        let mut cs = vec![0.0f32; h];
        let mut gh = ndarray::Array1::<f32>::zeros(4 * h); // recurrent projection scratch, reused per step
        let mut probs = Vec::with_capacity(n);
        for t in 0..n {
            let gx = gates_x.row(t);
            // gh = W_hh · h, written in place to avoid a per-step heap allocation.
            ndarray::linalg::general_mat_vec_mul(1.0, &self.w_hh, &hs, 0.0, &mut gh);
            let mut p = self.final_b;
            for j in 0..h {
                let gate = |k: usize| gx[k] + gh[k] + self.b[k];
                let i = sigmoid(gate(j));
                let f = sigmoid(gate(h + j));
                let g = gate(2 * h + j).tanh();
                let o = sigmoid(gate(3 * h + j));
                cs[j] = f * cs[j] + i * g;
                hs[j] = o * cs[j].tanh();
                p += self.final_w[j] * hs[j].max(0.0);
            }
            probs.push(sigmoid(p));
        }
        probs
    }
}

pub struct VadInference {
    jit: SileroVadFeatureJit,
    head: VadHead,
}

impl VadInference {
    pub fn new(vad: SileroVad) -> crate::jit::Result<Self> {
        use crate::jit::{InputSpec, TensorSnafu};
        let h = HIDDEN;
        // Pull the recurrent weights to host before `vad` moves into the JIT.
        // Clone+realize first: checkpoint weights are already realized, but
        // random/lazy weights need materialization before `as_vec`.
        let to_vec = |t: &Tensor| -> crate::jit::Result<Vec<f32>> {
            let mut t = t.clone();
            t.realize().context(TensorSnafu)?;
            t.as_vec::<f32>().context(TensorSnafu)
        };
        let w_ih = ndarray::Array2::from_shape_vec((4 * h, h), to_vec(&vad.lstm.weight_ih)?).expect("w_ih shape");
        let w_hh = ndarray::Array2::from_shape_vec((4 * h, h), to_vec(&vad.lstm.weight_hh)?).expect("w_hh shape");
        let b_ih = ndarray::Array1::from_vec(to_vec(&vad.lstm.bias_ih)?);
        let b_hh = ndarray::Array1::from_vec(to_vec(&vad.lstm.bias_hh)?);
        let b = b_ih + b_hh; // only the sum is used in the gate; fold it once here
        let final_w = ndarray::Array1::from_vec(to_vec(&vad.final_conv.weight)?); // [1,H,1] flat = H
        let final_b = match &vad.final_conv.bias {
            Some(b) => to_vec(b)?[0],
            None => 0.0,
        };
        let head = VadHead { w_ih, w_hh, b, final_w, final_b };

        let mut jit = SileroVadFeatureJit::new(vad);
        jit.prepare(InputSpec::f32(&[FEATURE_BATCH, CHUNK_LEN]))?;
        Ok(Self { jit, head })
    }

    /// Run Silero V5 across the waveform and collect one speech probability per
    /// [`NUM_SAMPLES`]-sample window. Output length is
    /// `ceil(waveform.len() / NUM_SAMPLES)`. The conv front-end runs **batched**
    /// on the GPU (a handful of dispatches); the recurrent LSTM + sigmoid head
    /// scan runs on the host — eliminating the old one-tiny-dispatch-per-window
    /// path whose per-dispatch round-trip latency dominated.
    pub fn probs(&mut self, waveform: &[f32]) -> crate::jit::Result<Vec<f32>> {
        let total = waveform.len();
        if total == 0 {
            return Ok(Vec::new());
        }
        let pad_len = (NUM_SAMPLES - total % NUM_SAMPLES) % NUM_SAMPLES;
        let padded_len = CONTEXT_SIZE + total + pad_len;
        let mut padded = vec![0.0f32; padded_len];
        padded[CONTEXT_SIZE..CONTEXT_SIZE + total].copy_from_slice(waveform);

        let n_chunks = (total + pad_len) / NUM_SAMPLES;
        let h = HIDDEN;

        // Phase 1: batched conv front-end on the GPU -> features [n_chunks, H].
        let t_feat = std::time::Instant::now();
        let mut features = vec![0.0f32; n_chunks * h];
        let mut done = 0usize;
        while done < n_chunks {
            let b = (n_chunks - done).min(FEATURE_BATCH);
            {
                let buf = self.jit.chunks_mut()?;
                let mut view = buf.as_array_mut::<f32>().context(crate::jit::DeviceSnafu)?;
                let slice = view.as_slice_mut().expect("contiguous chunks");
                for i in 0..b {
                    let start = (done + i) * NUM_SAMPLES;
                    slice[i * CHUNK_LEN..(i + 1) * CHUNK_LEN].copy_from_slice(&padded[start..start + CHUNK_LEN]);
                }
            }
            self.jit.execute()?;
            let out = self.jit.output()?.as_array::<f32>().context(crate::jit::DeviceSnafu)?;
            let flat = out.as_slice().expect("contiguous features");
            features[done * h..(done + b) * h].copy_from_slice(&flat[..b * h]);
            done += b;
        }
        let feature_ms = t_feat.elapsed().as_secs_f64() * 1e3;

        // Phase 2: recurrent LSTM + sigmoid head on the host.
        let t_scan = std::time::Instant::now();
        let probs = self.head.scan(&features, n_chunks);
        let scan_ms = t_scan.elapsed().as_secs_f64() * 1e3;

        tracing::info!(
            target: "svod_model::silero_vad",
            n_chunks,
            feature_ms,
            scan_ms,
            "silero vad probs breakdown (batched conv + host LSTM scan)",
        );
        Ok(probs)
    }

    /// Convenience wrapper around [`Self::probs`] +
    /// [`svod_arch::vad::chunks_from_probs`] with default chunker knobs and
    /// the given `threshold`. Errors from the JIT or chunker are swallowed —
    /// callers that need fault-visibility should drive `probs()` and
    /// `chunks_from_probs` directly.
    ///
    /// Chunk ends are clamped to `waveform.len()` via `ChunkerOpts::
    /// max_total_samples`: the prob→sample mapping rounds the final window up
    /// past the audio (the trailing zero-pad), and a speech region can't extend
    /// beyond the waveform.
    pub fn segment(&mut self, waveform: &[f32], threshold: f32) -> Vec<(usize, usize)> {
        let Ok(probs) = self.probs(waveform) else { return Vec::new() };
        let opts = svod_arch::vad::ChunkerOpts {
            threshold,
            samples_per_prob: NUM_SAMPLES,
            max_total_samples: Some(waveform.len()),
            ..svod_arch::vad::ChunkerOpts::default()
        };
        svod_arch::vad::chunks_from_probs(&probs, &opts)
            .unwrap_or_default()
            .into_iter()
            .map(|c| (c.start_sample, c.end_sample))
            .collect()
    }
}
