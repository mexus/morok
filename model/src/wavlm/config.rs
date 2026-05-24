//! WavLM model configuration with predefined builders that match the upstream
//! Python configs at `submodules/DiariZen/diarizen/models/module/wavlm_config.py`
//! verbatim (Base / Large / Base-S80-MD / Large-S80-MD).
//!
//! Per-layer pruning vectors (`use_attention`, `remaining_heads`,
//! `ff_interm_features`) are the heart of the `s80-md` variants — they encode
//! which transformer layers have attention at all, which heads survive in
//! each layer, and what FFN intermediate dim each layer carries.

/// Feature extractor normalization mode. Affects which conv blocks carry a
/// per-block normalization layer.
///
/// - `GroupNorm`: only block 0 has `GroupNorm(num_groups=out_ch, num_channels=out_ch)`.
/// - `LayerNorm`: every block has `LayerNorm` after the conv.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExtractorMode {
    GroupNorm,
    LayerNorm,
}

/// `(out_channels, kernel_size, stride)` triple for one feature-extractor
/// conv block.
pub type ConvLayerConfig = (usize, usize, usize);

/// WavLM model configuration.
///
/// Mirrors the kwargs of `wavlm_model(**configs)` in
/// `submodules/DiariZen/diarizen/models/module/wav2vec2/model.py:779-913`.
/// Per-layer pruning fields are concretized at construction (no runtime
/// gating) — the loader expects state-dict shapes to match these.
#[derive(Clone, Debug)]
pub struct WavLmConfig {
    // --- Feature extractor -------------------------------------------------
    pub extractor_mode: ExtractorMode,
    pub extractor_conv_layer_config: Vec<ConvLayerConfig>,
    pub extractor_conv_bias: bool,
    pub normalize_waveform: bool,

    // --- Encoder top-level -------------------------------------------------
    pub encoder_embed_dim: usize,
    pub encoder_pos_conv_kernel: usize,
    pub encoder_pos_conv_groups: usize,
    pub encoder_num_layers: usize,
    pub encoder_layer_norm_first: bool,
    pub encoder_head_dim: usize,

    // --- Relative position bias -------------------------------------------
    pub encoder_num_buckets: usize,
    pub encoder_max_distance: usize,

    // --- Per-layer pruning ------------------------------------------------
    /// Whether each layer's attention block is present at all (Vec of length
    /// `encoder_num_layers`). `False` ⇒ attention is skipped *and* its
    /// state-dict keys are absent.
    pub encoder_use_attention: Vec<bool>,
    /// Whether each layer's feed-forward block is present.
    pub encoder_use_feed_forward: Vec<bool>,
    /// Total head count per layer (before pruning); typically constant
    /// (16 for Large, 12 for Base). Used to size the un-pruned
    /// `rel_attn_embed` / `gru_rel_pos_const` tensors.
    pub encoder_total_num_heads: Vec<usize>,
    /// Surviving head indices per layer. Empty ⇒ that layer's attention is a
    /// pass-through.
    pub encoder_remaining_heads: Vec<Vec<usize>>,
    /// Intermediate FFN dim per layer.
    pub encoder_ff_interm_features: Vec<usize>,

    // --- Inference time ---------------------------------------------------
    /// Upper bound on the symbolic batch variable in the JIT wrapper.
    pub max_batch_size: usize,
}

impl WavLmConfig {
    /// Feature extractor output channel count (channels of the last block).
    pub fn extractor_out_dim(&self) -> usize {
        self.extractor_conv_layer_config.last().expect("at least one extractor block").0
    }

    /// Cumulative downsampling factor of the feature extractor (product of
    /// all block strides).
    pub fn extractor_stride(&self) -> usize {
        self.extractor_conv_layer_config.iter().map(|(_, _, s)| *s).product()
    }

    /// Input samples spanned by one output frame (the conv stack's receptive
    /// field). Inverse of the per-frame formula `in = (out-1)*stride + kernel`,
    /// folded from the last conv to the first. Used for the per-frame time grid.
    pub fn receptive_field_samples(&self) -> usize {
        self.extractor_conv_layer_config.iter().rev().fold(1usize, |out, conv| (out - 1) * conv.2 + conv.1)
    }
}

// ---------------------------------------------------------------------------
// Predefined configs — verbatim from
// submodules/DiariZen/diarizen/models/module/wavlm_config.py:38-239
// ---------------------------------------------------------------------------

const fn full_heads<const N: usize>(total: usize) -> [usize; N] {
    let mut out = [0usize; N];
    let mut i = 0;
    while i < N {
        out[i] = total;
        i += 1;
    }
    out
}

/// `WAVLM_BASE` — 12 layers, embed 768, no pruning.
pub fn wavlm_base() -> WavLmConfig {
    let n = 12;
    WavLmConfig {
        extractor_mode: ExtractorMode::GroupNorm,
        extractor_conv_layer_config: vec![
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias: false,
        normalize_waveform: false,

        encoder_embed_dim: 768,
        encoder_pos_conv_kernel: 128,
        encoder_pos_conv_groups: 16,
        encoder_num_layers: n,
        encoder_layer_norm_first: false,
        encoder_head_dim: 64,

        encoder_num_buckets: 320,
        encoder_max_distance: 800,

        encoder_use_attention: vec![true; n],
        encoder_use_feed_forward: vec![true; n],
        encoder_total_num_heads: full_heads::<12>(12).to_vec(),
        encoder_remaining_heads: (0..n).map(|_| (0..12).collect()).collect(),
        encoder_ff_interm_features: vec![3072; n],

        max_batch_size: 1,
    }
}

/// `WAVLM_LARGE` — 24 layers, embed 1024, no pruning.
pub fn wavlm_large() -> WavLmConfig {
    let n = 24;
    WavLmConfig {
        extractor_mode: ExtractorMode::LayerNorm,
        extractor_conv_layer_config: vec![
            (512, 10, 5),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 3, 2),
            (512, 2, 2),
            (512, 2, 2),
        ],
        extractor_conv_bias: false,
        normalize_waveform: true,

        encoder_embed_dim: 1024,
        encoder_pos_conv_kernel: 128,
        encoder_pos_conv_groups: 16,
        encoder_num_layers: n,
        encoder_layer_norm_first: true,
        encoder_head_dim: 64,

        encoder_num_buckets: 320,
        encoder_max_distance: 800,

        encoder_use_attention: vec![true; n],
        encoder_use_feed_forward: vec![true; n],
        encoder_total_num_heads: vec![16; n],
        encoder_remaining_heads: (0..n).map(|_| (0..16).collect()).collect(),
        encoder_ff_interm_features: vec![4096; n],

        max_batch_size: 1,
    }
}

/// `WAVLM_LARGE_S80_MD` — pruned Large with per-layer head subsets, per-layer
/// FFN dims, and attention skipped at layers 9 / 12 / 16 / 17.
///
/// Source: `wavlm_config.py:170-239`. The vectors below are copied verbatim;
/// changes here MUST be cross-checked against that file.
pub fn wavlm_large_s80_md() -> WavLmConfig {
    let n = 24;

    let use_attention = vec![
        true, true, true, true, true, true, true, true, true, /* 0..=8 */
        false, /* 9 */ true, true, /* 10, 11 */
        false, /* 12 */ true, true, true, /* 13, 14, 15 */
        false, false, /* 16, 17 */ true, true, /* 18, 19 */
        true, true, true, true, /* 20..=23 */
    ];
    let remaining_heads: Vec<Vec<usize>> = vec![
        vec![1, 2, 4, 5, 6],
        vec![9, 10, 14],
        vec![0, 1, 2, 4, 5, 7],
        vec![1, 4, 7, 12, 13, 14],
        vec![0, 2, 3, 4, 13],
        vec![1, 7, 13, 14, 15],
        vec![11, 13, 15],
        vec![2, 3, 4, 8, 15],
        vec![2, 5, 6, 15],
        vec![],
        vec![0, 1],
        vec![1, 3, 5, 12],
        vec![],
        vec![4, 7, 11],
        vec![6, 9],
        vec![11],
        vec![],
        vec![],
        vec![14],
        vec![5, 15],
        vec![0, 2, 8, 11, 13, 15],
        vec![0, 1, 3, 4, 5, 6, 7, 10, 13],
        vec![0, 1, 3, 6, 7, 9, 10, 11, 12, 14],
        vec![1, 2, 3, 4, 7, 13, 14, 15],
    ];
    let ff_interm_features = vec![
        1092, 925, 759, 646, 745, 615, 684, 958, 286, 294, 406, 377, 463, 542, 298, 236, 96, 104, 134, 211, 473, 1011,
        1770, 1316,
    ];

    debug_assert_eq!(use_attention.len(), n);
    debug_assert_eq!(remaining_heads.len(), n);
    debug_assert_eq!(ff_interm_features.len(), n);

    WavLmConfig {
        extractor_mode: ExtractorMode::LayerNorm,
        extractor_conv_layer_config: vec![
            (512, 10, 5),
            (153, 3, 2),
            (224, 3, 2),
            (255, 3, 2),
            (302, 3, 2),
            (368, 2, 2),
            (211, 2, 2),
        ],
        extractor_conv_bias: false,
        normalize_waveform: true,

        encoder_embed_dim: 1024,
        encoder_pos_conv_kernel: 128,
        encoder_pos_conv_groups: 16,
        encoder_num_layers: n,
        encoder_layer_norm_first: true,
        encoder_head_dim: 64,

        encoder_num_buckets: 320,
        encoder_max_distance: 800,

        encoder_use_attention: use_attention,
        encoder_use_feed_forward: vec![true; n],
        encoder_total_num_heads: vec![16; n],
        encoder_remaining_heads: remaining_heads,
        encoder_ff_interm_features: ff_interm_features,

        max_batch_size: 1,
    }
}
