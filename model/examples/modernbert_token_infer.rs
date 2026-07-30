//! ModernBERT token classification demo (NER / POS / chunking).
//!
//! Loads a HuggingFace ModernBERT token-classification checkpoint (weights +
//! tokenizer) in one call and runs a text input through a [`RecognizePipeline`]:
//! `HfTokenizer` → `TruncatingChunker` → `ModernBertTokenClassifier`. Decodes
//! per-token labels with [`labels_for_tokens`] and groups entity spans with
//! [`group_spans`] under a chosen [`Scheme`]. The token-classification analog of
//! `modernbert_classify_infer.rs`.
//!
//! Usage:
//!   cargo run -p svod-model --release --example modernbert_token_infer -- "Barack Obama was born in Hawaii"
//!   cargo run -p svod-model --release --example modernbert_token_infer -- --scheme bio --profile "Apple is based in Cupertino"
//!   cargo run -p svod-model --release --example modernbert_token_infer -- --repo sanketrai/modernbert-base-conll2003-english-ner "text"
//!
//! Reads stdin when no positional text is given.

use std::io::{self, Read};
use std::time::Instant;

use clap::Parser;

use svod_arch::pipelines::text::{
    Encoder, EncoderPipeline, Recognize, RunOptions, Scheme, TruncatingChunker, group_spans, labels_for_tokens,
};
use svod_dtype::DType;
use svod_model::modernbert;

#[derive(Parser, Debug)]
#[command(about = "ModernBERT token classification demo", long_about = None)]
struct Args {
    /// Text to tag (reads stdin if omitted).
    text: Option<String>,

    /// HF Hub repo with the ModernBERT token-classification weights + tokenizer.
    #[arg(long, default_value = "sanketrai/modernbert-base-conll2003-english-ner")]
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

    /// Label scheme for span grouping: `bio`, `bilou`, `iobes`, `flat`.
    #[arg(long, default_value = "bio")]
    scheme: String,

    /// Collect and print the per-stage profile.
    #[arg(long)]
    profile: bool,
}

/// Resolve the `id2label` map from a checkpoint's `config.json` into a dense
/// `Vec<String>` indexed by label id. The model itself deals in label indices;
/// only the demo needs the names for display + BIO span grouping.
fn load_id2label(repo: &str, revision: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    let api = hf_hub::api::sync::Api::new()?;
    let repo = api.repo(hf_hub::Repo::with_revision(repo.to_string(), hf_hub::RepoType::Model, revision.to_string()));
    let cfg_path = repo.get("config.json")?;
    let raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(cfg_path)?)?;
    let map = raw.get("id2label").ok_or("no id2label in config.json")?.as_object().ok_or("id2label not an object")?;
    let n = map.len();
    let mut out = vec![String::new(); n];
    for (k, v) in map {
        let id: usize = k.parse()?;
        if id < n {
            out[id] = v.as_str().unwrap_or("").to_string();
        }
    }
    Ok(out)
}

fn parse_scheme(s: &str) -> Result<Scheme, Box<dyn std::error::Error>> {
    Ok(match s.to_ascii_lowercase().as_str() {
        "bio" | "iob2" => Scheme::Bio,
        "bilou" => Scheme::Bilou,
        "iobes" => Scheme::Iobes,
        "flat" | "none" => Scheme::Flat,
        other => return Err(format!("unknown scheme {other:?}; expected bio|bilou|iobes|flat").into()),
    })
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
    let scheme = parse_scheme(&args.scheme)?;
    let dtype = match args.dtype.as_str() {
        "f32" => DType::Float32,
        "bf16" => DType::BFloat16,
        other => return Err(format!("unknown dtype {other:?}; expected \"f32\" or \"bf16\"").into()),
    };

    let t_total = Instant::now();
    println!("Loading ModernBERT token classifier from {} ({})...", args.repo, args.revision);
    let (tokenizer, recognizer) =
        modernbert::from_hub_token_classification_with_revision(&args.repo, &args.revision, args.max_batch, dtype)?;
    let (_, max_seq) = recognizer.capacity();
    let num_labels = recognizer.num_labels();
    println!("Loaded: max_seq={max_seq}, max_batch={}, num_labels={num_labels}", args.max_batch);

    let id2label = load_id2label(&args.repo, &args.revision)?;
    let label_of =
        |id: u32| id2label.get(id as usize).cloned().filter(|s| !s.is_empty()).unwrap_or_else(|| format!("LABEL_{id}"));

    println!("Chunker: truncating max_seq={max_seq}");
    let mut pipeline = EncoderPipeline::new(tokenizer, TruncatingChunker::new(max_seq), recognizer);

    println!("\nTagging {} chars (scheme = {})...", text.len(), args.scheme);
    let t = Instant::now();
    let result = pipeline.recognize(&text, RunOptions { profile: args.profile })?;
    let dt = t.elapsed();

    println!("  {} chunk(s)", result.chunks.len());
    for (i, chunk) in result.chunks.iter().enumerate() {
        let tokens = labels_for_tokens(chunk, &label_of);
        println!("  chunk {i} @ byte {}: {} content token(s)", chunk.byte_offset, tokens.len());
        for tok in &tokens {
            let surface = text.get(tok.start..tok.end).unwrap_or("<oob>");
            println!("    token {}: {:<8} [{},{}) {:?}", tok.token_index, tok.label, tok.start, tok.end, surface);
        }
        let spans = group_spans(&tokens, scheme);
        if !spans.is_empty() {
            println!("    → {} entit(y/ies):", spans.len());
            for e in spans {
                let surface = text.get(e.start..e.end).unwrap_or("<oob>");
                println!("       {:<8} [{},{}) {:?}", e.label, e.start, e.end, surface);
            }
        }
    }

    if let Some(prof) = &result.profile {
        println!("\n--- Profile ---\n{prof}");
    }
    println!("Recognize: {:.3}s", dt.as_secs_f32());
    println!("\nTotal: {:.2}s", t_total.elapsed().as_secs_f32());
    Ok(())
}
