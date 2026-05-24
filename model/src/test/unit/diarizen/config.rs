use crate::diarizen::{DiariZenConfig, powerset_class_count, powerset_table};

/// `(max_per_chunk=4, max_per_frame=4)` → full powerset = 16 classes.
#[test]
fn powerset_class_count_44() {
    assert_eq!(powerset_class_count(4, 4), 16);
}

/// `(max_per_chunk=4, max_per_frame=2)` would have been our earlier wrong
/// guess — confirm it produces a different count so the constant is
/// load-bearing.
#[test]
fn powerset_class_count_42_is_eleven() {
    // C(4,0)+C(4,1)+C(4,2) = 1 + 4 + 6 = 11
    assert_eq!(powerset_class_count(4, 2), 11);
}

/// The enumeration begins with the empty subset (silence), then singletons in
/// ascending order, then pairs in lex order, etc.
#[test]
fn powerset_table_layout() {
    let table = powerset_table(4, 4);
    assert_eq!(table.len(), 16);
    assert_eq!(table[0], Vec::<usize>::new());
    assert_eq!(table[1], vec![0]);
    assert_eq!(table[2], vec![1]);
    assert_eq!(table[3], vec![2]);
    assert_eq!(table[4], vec![3]);
    assert_eq!(table[5], vec![0, 1]);
    assert_eq!(table[6], vec![0, 2]);
    assert_eq!(table[15], vec![0, 1, 2, 3]);
}

#[test]
fn default_config_matches_published_args() {
    let cfg = DiariZenConfig::diarizen_wavlm_large_s80_md_v2();
    assert_eq!(cfg.attention_in, 256);
    assert_eq!(cfg.ffn_hidden, 1024);
    assert_eq!(cfg.num_head, 4);
    assert_eq!(cfg.num_layer, 4);
    assert_eq!(cfg.kernel_size, 31);
    assert!(!cfg.use_posi);
    assert_eq!(cfg.max_speakers_per_chunk, 4);
    assert_eq!(cfg.max_speakers_per_frame, 4);
    assert_eq!(cfg.powerset_class_count(), 16);
    assert_eq!(cfg.wavlm_layer_num(), 25);
    assert_eq!(cfg.wavlm_feat_dim(), 1024);
}
