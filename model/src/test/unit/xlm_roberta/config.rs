use crate::xlm_roberta::{XlmRobertaConfig, xlm_roberta_large};
use svod_dtype::DType;

#[test]
fn large_dims() {
    let cfg = xlm_roberta_large();
    assert_eq!(cfg.vocab_size, 250002);
    assert_eq!(cfg.hidden_size, 1024);
    assert_eq!(cfg.num_hidden_layers, 24);
    assert_eq!(cfg.num_attention_heads, 16);
    assert_eq!(cfg.head_dim(), 64);
    assert_eq!(cfg.intermediate_size, 4096);
    assert_eq!(cfg.max_position_embeddings, 8194);
    assert_eq!(cfg.type_vocab_size, 1);
    assert_eq!(cfg.pad_token_id, 1);
    assert_eq!(cfg.layer_norm_eps, 1e-5);
}

#[test]
fn parse_config_json() {
    let json = r#"{
        "vocab_size": 250002, "hidden_size": 1024, "num_hidden_layers": 24,
        "num_attention_heads": 16, "intermediate_size": 4096,
        "max_position_embeddings": 8194, "type_vocab_size": 1,
        "layer_norm_eps": 1e-05, "pad_token_id": 1
    }"#;
    let cfg = XlmRobertaConfig::from_json_str(json).unwrap();
    assert_eq!(cfg.vocab_size, 250002);
    assert_eq!(cfg.pad_token_id, 1);
}

#[test]
fn parse_norm_eps_alias() {
    let cfg = XlmRobertaConfig::from_json_str(r#"{"norm_eps": 1e-5}"#).unwrap();
    assert!((cfg.layer_norm_eps - 1e-5).abs() < 1e-12);
}

#[test]
fn merge_preserves_dtype() {
    let mut caller = XlmRobertaConfig { dtype: DType::Float32, max_batch_size: 4, ..xlm_roberta_large() };
    caller.merge_structural_from(&xlm_roberta_large());
    assert_eq!(caller.dtype, DType::Float32);
    assert_eq!(caller.max_batch_size, 4);
}
