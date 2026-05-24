//! Configuration for the DiariZen segmentation model.
//!
//! Mirrors `Model.__init__` from
//! `submodules/DiariZen/diarizen/models/eend/model_wavlm_conformer.py:25-76`
//! plus `[model.args]` from the published `config.toml`.

use crate::wavlm::WavLmConfig;

/// Per-block Conformer hyperparameters + powerset speaker bounds.
#[derive(Clone, Debug)]
pub struct DiariZenConfig {
    /// WavLM backbone configuration.
    pub wavlm: WavLmConfig,
    /// Output dim of `proj` (= input dim to the Conformer). Default 256.
    pub attention_in: usize,
    /// Conformer FFN hidden size. Default 1024.
    pub ffn_hidden: usize,
    /// Conformer attention heads. Default 4.
    pub num_head: usize,
    /// Number of Conformer blocks. Default 4.
    pub num_layer: usize,
    /// Depthwise conv kernel size. Default 31.
    pub kernel_size: usize,
    /// Per-block dropout rate. Default 0.1. (Inference treats this as a no-op.)
    pub dropout: f32,
    /// Whether the Conformer's MHSA uses relative positional bias. Default
    /// `false` for `s80-md-v2`.
    pub use_posi: bool,
    /// Maximum simultaneous speakers in a chunk. Default 4.
    pub max_speakers_per_chunk: usize,
    /// Maximum simultaneous speakers in a single frame. Default 4 (matches
    /// the published config — yields 16 powerset classes via
    /// `powerset_class_count`).
    pub max_speakers_per_frame: usize,
    /// Chunk length in seconds. Default 16.
    pub chunk_size_seconds: f32,
    /// Sample rate. Default 16 kHz.
    pub sample_rate: u32,
    /// Inter-window hop as a ratio of the window duration (pyannote
    /// `segmentation_step`). Default 0.1 → 90 % overlap.
    pub segmentation_step: f32,
    /// Max chunks per JIT batch for capture-graph-reuse inference. Default 32.
    pub inference_batch_size: usize,
}

impl DiariZenConfig {
    /// Default config for `BUT-FIT/diarizen-wavlm-large-s80-md-v2`. WavLM
    /// backbone is [`crate::wavlm::wavlm_large_s80_md`].
    pub fn diarizen_wavlm_large_s80_md_v2() -> Self {
        Self {
            wavlm: crate::wavlm::wavlm_large_s80_md(),
            attention_in: 256,
            ffn_hidden: 1024,
            num_head: 4,
            num_layer: 4,
            kernel_size: 31,
            dropout: 0.1,
            use_posi: false,
            max_speakers_per_chunk: 4,
            max_speakers_per_frame: 4,
            chunk_size_seconds: 16.0,
            sample_rate: 16_000,
            segmentation_step: 0.1,
            inference_batch_size: 32,
        }
    }

    /// Number of powerset output classes for the given speaker bounds.
    /// `sum_{k=0..=max_per_frame} C(max_per_chunk, k)`.
    ///
    /// For `(max_per_chunk=4, max_per_frame=4)` this is `2^4 = 16`.
    pub fn powerset_class_count(&self) -> usize {
        powerset_class_count(self.max_speakers_per_chunk, self.max_speakers_per_frame)
    }

    /// Number of WavLM intermediates produced by the backbone
    /// (`num_layers + 1`).
    pub fn wavlm_layer_num(&self) -> usize {
        self.wavlm.encoder_num_layers + 1
    }

    /// WavLM output embedding dim.
    pub fn wavlm_feat_dim(&self) -> usize {
        self.wavlm.encoder_embed_dim
    }

    /// Samples per analysis window = `floor(chunk_size_seconds * sample_rate)`
    /// (pyannote `Audio.get_num_samples`).
    pub fn window_samples(&self) -> usize {
        window_samples(self.chunk_size_seconds, self.sample_rate)
    }

    /// Hop between consecutive windows = `round(segmentation_step * window)`
    /// (pyannote `step = ratio * duration`, `step_size = round(step * sr)`).
    pub fn hop_samples(&self) -> usize {
        hop_samples(self.segmentation_step, self.window_samples())
    }

    /// Number of sliding windows (incl. a zero-padded trailing chunk) covering
    /// an `n`-sample waveform.
    pub fn num_chunks(&self, n: usize) -> usize {
        chunk_plan(n, self.window_samples(), self.hop_samples())
    }
}

/// `sum_{k=0..=max_per_frame} C(max_per_chunk, k)` — powerset class count.
pub fn powerset_class_count(max_per_chunk: usize, max_per_frame: usize) -> usize {
    let limit = max_per_frame.min(max_per_chunk);
    (0..=limit).map(|k| binomial(max_per_chunk, k)).sum()
}

/// Samples per analysis window = `floor(chunk_size_seconds * sample_rate)`.
pub fn window_samples(chunk_size_seconds: f32, sample_rate: u32) -> usize {
    (chunk_size_seconds as f64 * sample_rate as f64).floor() as usize
}

/// Hop between consecutive windows = `round(segmentation_step * window_samples)`.
pub fn hop_samples(segmentation_step: f32, window_samples: usize) -> usize {
    (segmentation_step as f64 * window_samples as f64).round() as usize
}

/// Number of sliding windows (incl. a zero-padded trailing chunk) covering an
/// `n`-sample waveform. Equivalent to `torch.unfold(1, window, hop)` plus
/// pyannote's last-incomplete-chunk zero-pad.
pub fn chunk_plan(n: usize, window: usize, hop: usize) -> usize {
    if n <= window { 1 } else { 1 + (n - window).div_ceil(hop) }
}

fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let k = k.min(n - k);
    let mut acc = 1usize;
    for i in 0..k {
        acc = acc * (n - i) / (i + 1);
    }
    acc
}

/// Enumerate the powerset table as a `Vec<Vec<usize>>` where each entry is a
/// sorted speaker-index subset. The ordering matches `pyannote.audio`'s
/// `Powerset.cardinal()`-then-`combinations` enumeration:
/// `[], [0], [1], …, [N-1], [0,1], [0,2], …`.
///
/// Returned slice length equals [`powerset_class_count`].
pub fn powerset_table(max_per_chunk: usize, max_per_frame: usize) -> Vec<Vec<usize>> {
    let limit = max_per_frame.min(max_per_chunk);
    let mut out = Vec::new();
    for k in 0..=limit {
        for combo in combinations(max_per_chunk, k) {
            out.push(combo);
        }
    }
    out
}

fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    let mut out = Vec::new();
    let mut buf = Vec::with_capacity(k);
    fn rec(start: usize, n: usize, k: usize, buf: &mut Vec<usize>, out: &mut Vec<Vec<usize>>) {
        if buf.len() == k {
            out.push(buf.clone());
            return;
        }
        let needed = k - buf.len();
        for i in start..=n.saturating_sub(needed) {
            buf.push(i);
            rec(i + 1, n, k, buf, out);
            buf.pop();
        }
    }
    rec(0, n, k, &mut buf, &mut out);
    out
}
