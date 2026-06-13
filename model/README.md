# svod-model

High-level inference for pretrained deep learning models on top of
`svod-tensor`. Each model is a pure-Rust port of an upstream checkpoint,
fetched from HuggingFace Hub at runtime and executed through JIT-compiled
plans.

## Common infrastructure

| Module | Role |
|---|---|
| `jit` | `jit_wrapper!`-generated wrappers, `JitRecurrent<J>`, `InputSpec`, `JitError`. Build-once / run-many execution. See [JIT Graphs](../website/docs/architecture/jit-graphs.md). |
| `audio` | Log-mel spectrogram, `Splitter` trait for long-form chunking (`FireRedVadSplitter`, `SileroVadSplitter`, `FixedLengthSplitter`). |
| `state` | `HasStateDict` + `state_field!` macros for loading PyTorch / safetensors checkpoints into Rust weight structs. |
| `blocks` | Shared `Conv2dWeights`, `BatchNormWeights`, `BasicBlock`, `Bottleneck`, `ResidualStage` reused by every ResNet-shaped backbone. timm/torchvision key convention. |
| `wavlm` | WavLM speech-representation backbone (Conv1d feature extractor + gated rel-pos Transformer, per-layer pruning) consumed by `diarizen`. |
| `sentencepiece` | Minimal SentencePiece `.model` protobuf loader (vocab piece extraction). |

## Models

| Name | Domain | Module | Upstream | HuggingFace |
|---|---|---|---|---|
| GigaAM v3 (CTC + RN-T) | Speech | `gigaam` | [salute-developers/GigaAM](https://github.com/salute-developers/GigaAM) | [`vpermilp/GigaAM-v3`](https://huggingface.co/vpermilp/GigaAM-v3) |
| FireRedVAD (batch + streaming) | Voice activity | `firered_vad` | [FireRedTeam/FireRedVAD](https://github.com/FireRedTeam/FireRedVAD) | [`vpermilp/firered_vad`](https://huggingface.co/vpermilp/firered_vad) |
| Silero VAD 16k | Voice activity | `silero_vad` | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) | [`vpermilp/silero-vad`](https://huggingface.co/vpermilp/silero-vad) |
| DiariZen segmentation (WavLM + Conformer) | Speaker diarization | `diarizen` | [BUT-FIT/DiariZen](https://github.com/BUTSpeechFIT/DiariZen) | [`BUT-FIT/diarizen-wavlm-large-s80-md-v2`](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2) |
| ResNet (18 / 34 / 50 / 101 / 152) | Vision | `resnet` | [He et al. 2015](https://arxiv.org/abs/1512.03385) | [`timm/resnet*.a1_in1k`](https://huggingface.co/timm) |
| WeSpeaker ResNet34 | Speaker embedding | `wespeaker` | [wenet-e2e/wespeaker](https://github.com/wenet-e2e/wespeaker) | [`pyannote/wespeaker-voxceleb-resnet34-LM`](https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM) |

FireRedVAD ships two variants: the non-streaming model (`FireRedVadSplitter`
— the whole DFSMN runs as one batched device JIT, no host recurrence) and
the causal `Stream-VAD` checkpoint (`FireRedVadStreamer` — incremental
push/flush API with per-layer conv caches recycled on-device between
chunks).

## Examples

```bash
cargo run -p svod-model --release --example gigaam_infer -- audio.wav [--rnnt] [--timestamps]
cargo run -p svod-model --release --example vad_bench -- audio.wav            # Silero vs FireRedVAD timing
cargo run -p svod-model --release --example vad_stream -- audio.wav           # streaming VAD events (simulated mic)
cargo run -p svod-model --release --example resnet_classify -- --hub --image dog.bin --side 224
cargo run -p svod-model --release --example wespeaker_parity -- --hub --data reference.npz
```
