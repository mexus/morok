//! ModernBERT embeddings inference demo.
//!
//! Loads a HuggingFace ModernBERT checkpoint (weights + tokenizer) in one call
//! and runs a text input through an [`EmbeddingsPipeline`]: `HfTokenizer` →
//! `TruncatingChunker` → `ModernBertEmbedder`. This doubles as the runnable
//! end-to-end smoke test for the text pipeline — the analog of `gigaam_infer.rs`
//! for audio.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- "hello world"
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- --profile "hello world"
//!   cargo run -p svod-model --release --example modernbert_embed_infer -- --repo answerdotai/ModernBERT-base --max-batch 4 "text"
//!
//! Reads stdin when no positional text is given.

use std::io::{self, Read};
use std::time::Instant;

use clap::Parser;

use svod_arch::pipelines::text::{Embed, EmbeddingsPipeline, RunOptions, TruncatingChunker};
use svod_dtype::DType;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT embeddings demo", long_about = None)]
struct Args {
    /// Text to embed (reads stdin if omitted).
    text: Option<String>,

    /// HF Hub repo with the ModernBERT weights + tokenizer.
    #[arg(long, default_value = "answerdotai/ModernBERT-base")]
    repo: String,

    /// HF Hub revision.
    #[arg(long, default_value = "main")]
    revision: String,

    /// Compute dtype: `f32` (CPU) or `bf16` (GPU).
    #[arg(long, default_value = "f32")]
    dtype: String,

    /// Prepared max batch size (the JIT upper bound on the batch dimension).
    #[arg(long, default_value_t = 1)]
    max_batch: usize,

    /// Collect and print the per-stage profile.
    #[arg(long)]
    profile: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args = Args::parse();
    let text = match args.text {
        Some(t) => t,
        None => {
            let mut buf = String::new();
            io::stdin().read_to_string(&mut buf)?;
            buf
        }
    };

    let dtype = match args.dtype.as_str() {
        "f32" => DType::Float32,
        "bf16" => DType::BFloat16,
        other => return Err(format!("unknown dtype {other:?}; expected \"f32\" or \"bf16\"").into()),
    };

    let t_total = Instant::now();
    println!("Loading ModernBERT from {} ({})...", args.repo, args.revision);
    // One call fetches config.json + model.safetensors + tokenizer.json and
    // sizes the embedder JIT from the checkpoint's max_position_embeddings.
    let (tokenizer, embedder) = modernbert::from_hub_with_revision(&args.repo, &args.revision, args.max_batch, dtype)?;
    let (_, max_seq) = embedder.capacity();
    println!("Loaded: hidden_size={}, max_seq={}, max_batch={}", embedder.hidden_size(), max_seq, args.max_batch,);

    // The embedder is already JIT-prepared with max_seq, so compose directly
    // (rather than `assemble`, which would rebuild it) with a matching chunker.
    let mut pipeline = EmbeddingsPipeline::new(tokenizer, TruncatingChunker::new(max_seq), embedder);

    println!("\nEmbedding {} chars...", text.len());
    let t_embed = Instant::now();
    let result = pipeline.embed(&text, RunOptions { profile: args.profile })?;
    let dt_embed = t_embed.elapsed();

    for (i, chunk) in result.chunks.iter().enumerate() {
        let v = &chunk.values.values;
        println!(
            "  chunk {i} @ char {}: dim={} | L2={:.4} | first 5: {:?}",
            chunk.char_offset,
            v.len(),
            v.iter().map(|x| x * x).sum::<f32>().sqrt(),
            &v[..v.len().min(5)],
        );
    }
    if let Some(profile) = &result.profile {
        println!("\n--- Profile ---\n{profile}");
    }
    println!("\nTotal: {:.2}s; embed: {:.3}s", t_total.elapsed().as_secs_f32(), dt_embed.as_secs_f32());
    Ok(())
}
