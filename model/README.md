# morok-model

High-level inference for pretrained speech models on top of `morok-tensor`.
Each model is a pure-Rust port of an upstream checkpoint, fetched from
HuggingFace Hub at runtime and executed through JIT-compiled plans.

## Common infrastructure

| Module | Role |
|---|---|
| `jit` | `jit_wrapper!`-generated wrappers, `JitRecurrent<J>`, `InputSpec`, `JitError`. Build-once / run-many execution. See [JIT Graphs](../website/docs/architecture/jit-graphs.md). |
| `audio` | Log-mel spectrogram, `Splitter` trait for long-form chunking (default: `SileroVadSplitter`). |
| `state` | `HasStateDict` + `state_field!` macros for loading PyTorch / safetensors checkpoints into Rust weight structs. |
| `sentencepiece` | Minimal SentencePiece `.model` protobuf loader (vocab piece extraction). |

## Models

| Name | Module | Upstream | HuggingFace |
|---|---|---|---|
| GigaAM v3 (CTC + RN-T) | `gigaam` | [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) | [`vpermilp/GigaAM-v3`](https://huggingface.co/vpermilp/GigaAM-v3) |
| Silero VAD 16k | `silero_vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | [`vpermilp/silero-vad`](https://huggingface.co/vpermilp/silero-vad) |

## Examples

```bash
cargo run -p morok-model --release --example gigaam_infer -- audio.wav
cargo run -p morok-model --release --example gigaam_rnnt_infer -- audio.wav
cargo run -p morok-model --release --example test_vad -- audio.wav
```
