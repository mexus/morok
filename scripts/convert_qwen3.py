# /// script
# dependencies = ["torch", "numpy", "safetensors", "transformers"]
# requires-python = ">=3.10"
# ///
"""Generate golden parity fixtures for Qwen3-Embedding-0.6B."""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file
from transformers import AutoModel, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Embedding-0.6B"
PROMPT = "What is Qwen3?"
MAX_LEN = 32


def main():
    local = Path(__file__).resolve().parent.parent / "submodules" / "Qwen3-Embedding-0.6B"
    model_path = str(local) if local.exists() else MODEL_ID

    print(f"Loading {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float32)
    model.eval()

    # Left-padding so last token = position L-1.
    tokenizer.padding_side = "left"
    enc = tokenizer(
        PROMPT,
        padding="max_length",
        max_length=MAX_LEN,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    print(f"input_ids shape: {tuple(input_ids.shape)}")
    print(f"attention_mask shape: {tuple(attention_mask.shape)}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = out.last_hidden_state  # (1, 32, 1024)

        # Last-token pooling (left-padded → last position is the real last token)
        pooled = last_hidden_state[:, -1, :]  # (1, 1024)
        embeddings = F.normalize(pooled, p=2, dim=-1)  # (1, 1024)

    print(f"last_hidden_state shape: {tuple(last_hidden_state.shape)}")
    print(f"embeddings shape: {tuple(embeddings.shape)}")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "qwen3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "golden.safetensors"

    save_file(
        {
            "input_ids": input_ids.numpy().astype(np.int64).flatten(),
            "attention_mask": attention_mask.numpy().astype(np.int64).flatten(),
            "input_ids_shape": np.array([1, MAX_LEN], dtype=np.int64),
            "last_hidden_state": last_hidden_state.numpy().astype(np.float32),
            "embeddings": embeddings.numpy().astype(np.float32),
        },
        str(out_path),
    )
    print(f"Saved golden to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
