# morok-model

High-level inference for pretrained deep learning models on top of
`morok-tensor`. Each model is a pure-Rust port of an upstream checkpoint,
fetched from HuggingFace Hub at runtime and executed through JIT-compiled
plans.

## Common infrastructure

| Module | Role |
|---|---|
| `jit` | `jit_wrapper!`-generated wrappers, `JitRecurrent<J>`, `InputSpec`, `JitError`. Build-once / run-many execution. See [JIT Graphs](../website/docs/architecture/jit-graphs.md). |
| `audio` | Log-mel spectrogram, `Splitter` trait for long-form chunking (default: `SileroVadSplitter`). |
| `state` | `HasStateDict` + `state_field!` macros for loading PyTorch / safetensors checkpoints into Rust weight structs. |
| `sentencepiece` | Minimal SentencePiece `.model` protobuf loader (vocab piece extraction). |

## Models

| Name | Domain | Module | Upstream | HuggingFace |
|---|---|---|---|---|
| GigaAM v3 (CTC + RN-T) | Speech | `gigaam` | [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) | [`vpermilp/GigaAM-v3`](https://huggingface.co/vpermilp/GigaAM-v3) |
| Silero VAD 16k | Speech | `silero_vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | [`vpermilp/silero-vad`](https://huggingface.co/vpermilp/silero-vad) |
| ResNet (18 / 34 / 50 / 101 / 152) | Vision | `resnet` | [He et al. 2015](https://arxiv.org/abs/1512.03385) | [`timm/resnet*.a1_in1k`](https://huggingface.co/timm) |

## Examples

```bash
cargo run -p morok-model --release --example gigaam_infer -- audio.wav
cargo run -p morok-model --release --example gigaam_rnnt_infer -- audio.wav
cargo run -p morok-model --release --example test_vad -- audio.wav
cargo run -p morok-model --release --example resnet_classify -- --hub --image dog.bin --side 224
```
