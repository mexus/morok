#!/usr/bin/env python3
"""Generate golden reference outputs for Whisper parity tests.

Downloads whisper-tiny from HuggingFace, runs the encoder + decoder on a
fixed mel input, and saves the inputs + outputs as ``golden.safetensors``.

Usage::

    pip install torch transformers safetensors numpy
    python scripts/convert_whisper.py --output data/whisper/golden.safetensors
    python scripts/convert_whisper.py --output data/whisper/golden.safetensors \
        --model openai/whisper-base

The generated file contains:
    - ``mel``         : float32 [1, n_mels, 3000]  — mel spectrogram input
    - ``mel_shape``   : int64 [3]                  — shape of mel
    - ``encoder_output``      : float32 [1, 1500, D]  — encoder features
    - ``encoder_output_shape``: int64 [3]
    - ``tokens``      : int64 [L]                  — decoder input token ids
    - ``logits``      : float32 [1, L, n_vocab]    — decoder output logits
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
import safetensors.torch as st


def main():
    parser = argparse.ArgumentParser(description="Generate Whisper golden reference")
    parser.add_argument("--model", default="openai/whisper-tiny", help="HF model name")
    parser.add_argument("--output", required=True, help="Output safetensors path")
    parser.add_argument("--n-mels", type=int, default=None, help="Override n_mels (auto from config)")
    args = parser.parse_args()

    # Load model via transformers
    from transformers import WhisperForConditionalGeneration, WhisperConfig

    print(f"Loading {args.model} ...")
    config = WhisperConfig.from_pretrained(args.model)
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.eval()

    n_mels = args.n_mels or getattr(config, "num_mel_bins", 80)
    d_model = getattr(config, "d_model", 384)
    n_vocab = getattr(config, "vocab_size", 51865)

    print(f"  dims: n_mels={n_mels}, d_model={d_model}, n_vocab={n_vocab}")

    # Create a deterministic mel input (zeros — deterministic across runs)
    # Using a fixed seed for reproducibility
    rng = np.random.RandomState(42)
    mel = rng.randn(1, n_mels, 3000).astype(np.float32) * 0.1  # small random values

    mel_tensor = torch.from_numpy(mel)

    # Encoder forward
    with torch.no_grad():
        # The HF model expects input_features of shape [batch, n_mels, seq_len]
        encoder_output = model.model.encoder(input_features=mel_tensor).last_hidden_state

    print(f"  encoder output: {encoder_output.shape}")  # [1, 1500, D]

    # Decoder forward — use a simple SOT sequence
    # SOT (50257) + no_timestamps (50362) for English-only tiny
    # For multilingual: SOT + language + task + no_timestamps
    is_multilingual = n_vocab >= 51865
    if is_multilingual:
        tokens = [50258, 50259, 50359, 50363]  # SOT, en, transcribe, notimestamps
    else:
        tokens = [50257, 50362]  # SOT, notimestamps

    token_tensor = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        # Full forward (encoder + decoder)
        outputs = model(input_features=mel_tensor, decoder_input_ids=token_tensor)
        logits = outputs.logits

    print(f"  tokens: {tokens}")
    print(f"  logits: {logits.shape}")  # [1, L, n_vocab]

    # Save as safetensors
    out_dir = Path(args.output).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    tensors = {
        "mel": mel_tensor,
        "mel_shape": torch.tensor(list(mel.shape), dtype=torch.int64),
        "encoder_output": encoder_output,
        "encoder_output_shape": torch.tensor(list(encoder_output.shape), dtype=torch.int64),
        "tokens": token_tensor.squeeze(0),
        "logits": logits,
    }

    # Convert to contiguous float32/int64 for safetensors
    tensors = {k: v.contiguous().to(torch.float32) if v.dtype in (torch.float16, torch.bfloat16) else v.contiguous()
               for k, v in tensors.items()}

    st.save_file(tensors, str(args.output))
    print(f"Saved golden reference to {args.output}")

    # Also download model.safetensors if not already present
    model_dir = out_dir / "model.safetensors"
    if not model_dir.exists():
        print(f"\nNOTE: Place model.safetensors from {args.model} in {out_dir}")


if __name__ == "__main__":
    main()
