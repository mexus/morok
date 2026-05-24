use crate::wavlm::{ExtractorMode, wavlm_base, wavlm_large, wavlm_large_s80_md};

/// `wavlm_base()` matches the WAVLM_BASE table in upstream Python.
#[test]
fn base_config_shape() {
    let c = wavlm_base();
    assert_eq!(c.extractor_mode, ExtractorMode::GroupNorm);
    assert_eq!(c.encoder_embed_dim, 768);
    assert_eq!(c.encoder_num_layers, 12);
    assert!(!c.encoder_layer_norm_first);
    assert!(!c.normalize_waveform);
    assert!(c.encoder_use_attention.iter().all(|&b| b));
    assert!(c.encoder_remaining_heads.iter().all(|h| h.len() == 12));
    assert!(c.encoder_ff_interm_features.iter().all(|&f| f == 3072));
    assert_eq!(c.extractor_conv_layer_config.len(), 7);
    assert_eq!(c.extractor_conv_layer_config[0], (512, 10, 5));
    assert_eq!(c.extractor_stride(), 5 * 2 * 2 * 2 * 2 * 2 * 2);
    assert_eq!(c.extractor_out_dim(), 512);
}

#[test]
fn large_config_shape() {
    let c = wavlm_large();
    assert_eq!(c.extractor_mode, ExtractorMode::LayerNorm);
    assert_eq!(c.encoder_embed_dim, 1024);
    assert_eq!(c.encoder_num_layers, 24);
    assert!(c.encoder_layer_norm_first);
    assert!(c.normalize_waveform);
    assert!(c.encoder_use_attention.iter().all(|&b| b));
    assert!(c.encoder_remaining_heads.iter().all(|h| h.len() == 16));
    assert!(c.encoder_ff_interm_features.iter().all(|&f| f == 4096));
}

/// `wavlm_large_s80_md()` carries the prune-mask layout from the published
/// WAVLM_LARGE_S80_MD config. The expected values come from
/// `submodules/DiariZen/diarizen/models/module/wavlm_config.py:170-239`.
#[test]
fn large_s80_md_config_shape() {
    let c = wavlm_large_s80_md();
    assert_eq!(c.extractor_mode, ExtractorMode::LayerNorm);
    assert_eq!(c.encoder_embed_dim, 1024);
    assert_eq!(c.encoder_num_layers, 24);
    assert!(c.encoder_layer_norm_first);

    // s80-md pruned feature extractor.
    let expected_conv = [(512, 10, 5), (153, 3, 2), (224, 3, 2), (255, 3, 2), (302, 3, 2), (368, 2, 2), (211, 2, 2)];
    assert_eq!(c.extractor_conv_layer_config, expected_conv);
    assert_eq!(c.extractor_out_dim(), 211);

    // Layers 9, 12, 16, 17 skip attention.
    let skip = [9, 12, 16, 17];
    for i in 0..c.encoder_num_layers {
        let expected = !skip.contains(&i);
        assert_eq!(c.encoder_use_attention[i], expected, "use_attention[{i}] mismatch");
    }

    // remaining_heads sizes per layer.
    let expected_heads = [5, 3, 6, 6, 5, 5, 3, 5, 4, 0, 2, 4, 0, 3, 2, 1, 0, 0, 1, 2, 6, 9, 10, 8];
    for (i, want) in expected_heads.iter().enumerate() {
        assert_eq!(c.encoder_remaining_heads[i].len(), *want, "remaining_heads[{i}] length");
    }
    // Spot-check a few exact head index lists.
    assert_eq!(c.encoder_remaining_heads[0], vec![1, 2, 4, 5, 6]);
    assert_eq!(c.encoder_remaining_heads[23], vec![1, 2, 3, 4, 7, 13, 14, 15]);
    assert!(c.encoder_remaining_heads[9].is_empty());

    // ff_interm_features per layer — full table.
    let expected_ff = [
        1092, 925, 759, 646, 745, 615, 684, 958, 286, 294, 406, 377, 463, 542, 298, 236, 96, 104, 134, 211, 473, 1011,
        1770, 1316,
    ];
    assert_eq!(c.encoder_ff_interm_features, expected_ff);
}
