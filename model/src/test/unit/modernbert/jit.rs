use svod_dtype::DType;

use crate::jit::InputSpec;
use crate::modernbert::{ModernBert, ModernBertConfig, ModernBertJit};

/// Tiny config (2 layers, hidden 32) so the CPU JIT graph compiles in seconds.
fn tiny_cfg() -> ModernBertConfig {
    ModernBertConfig {
        vocab_size: 64,
        hidden_size: 32,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        intermediate_size: 64,
        max_position_embeddings: 128,
        layer_norm_eps: 1e-5,
        global_rope_theta: 10_000.0,
        local_rope_theta: 10_000.0,
        local_attention: 16,
        global_attn_every_n_layers: 3,
        pad_token_id: 0,
        tie_word_embeddings: true,
        dtype: DType::Float32,
        max_batch_size: 1,
    }
}

/// The JIT wrapper must thread `attention_mask` through to the encoder. This
/// is the regression guard for the dropped-mask bug: an all-attend forward
/// (every mask entry 1) must differ from a masked forward (last half of the
/// sequence masked out). If the mask were hardcoded to `None` in the wrapper,
/// both runs would be byte-identical.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn jit_mask_changes_output() {
    let cfg = tiny_cfg();
    let mut jit = ModernBertJit::new(ModernBert::empty(cfg.clone()));
    let seq_len = 8usize;
    jit.prepare(InputSpec::i64(&[cfg.max_batch_size, seq_len]), InputSpec::i64(&[cfg.max_batch_size, seq_len]))
        .expect("prepare");

    let ids: Vec<i64> = (1..=seq_len as i64).collect();

    // Run A: all tokens attended (mask all ones).
    let out_all = run(&mut jit, &ids, &vec![1i64; seq_len]);
    // Run B: last half masked out (mask zeros there).
    let mut mask_half = vec![1i64; seq_len];
    for m in &mut mask_half[seq_len / 2..] {
        *m = 0;
    }
    let out_half = run(&mut jit, &ids, &mask_half);

    assert_ne!(out_all, out_half, "mask had no effect on JIT output — wrapper dropped it?");
}

fn run(jit: &mut ModernBertJit, ids: &[i64], mask: &[i64]) -> Vec<f32> {
    // Write input_ids.
    {
        let buf = jit.input_ids_mut().expect("input_ids buffer");
        let mut view = buf.as_array_mut::<i64>().expect("input_ids view");
        let flat = view.as_slice_mut().expect("contiguous");
        flat.copy_from_slice(ids);
    }
    // Write attention_mask (int64 1/0; the cast to bool happens in the graph).
    {
        let buf = jit.attention_mask_mut().expect("attention_mask buffer");
        let mut view = buf.as_array_mut::<i64>().expect("attention_mask view");
        let flat = view.as_slice_mut().expect("contiguous");
        flat.copy_from_slice(mask);
    }
    jit.execute().expect("execute");
    let out = jit.output().expect("output buffer");
    let view = out.as_array::<f32>().expect("output view");
    let flat = view.as_slice().expect("contiguous");
    flat.to_vec()
}

/// Smoke: the JIT plan for an input with a trailing-pad mask still runs the
/// full seq_len (the seq dim is baked, only attention is masked) and produces
/// `(B, L, D)` finite output.
#[test]
#[ignore = "heavy: 2-layer ModernBERT JIT graph compile through the CPU backend"]
fn jit_prepares_and_executes() {
    let cfg = tiny_cfg();
    let mut jit = ModernBertJit::new(ModernBert::empty(cfg.clone()));
    let seq_len = 6usize;
    jit.prepare(InputSpec::i64(&[cfg.max_batch_size, seq_len]), InputSpec::i64(&[cfg.max_batch_size, seq_len]))
        .expect("prepare");

    let ids: Vec<i64> = (1..=seq_len as i64).collect();
    let out = run(&mut jit, &ids, &vec![1i64; seq_len]);
    let d = cfg.hidden_size;
    assert_eq!(out.len(), cfg.max_batch_size * seq_len * d, "(B, L, D) element count");
    assert!(out.iter().all(|v| v.is_finite()), "non-finite JIT output");
}
