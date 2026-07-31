# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "transformers", "huggingface_hub"]
# ///
"""Generate BGE-reranker-v2-m3 parity fixtures for the Rust `svod-model` port.

Downloads `BAAI/bge-reranker-v2-m3`, runs the cross-encoder on a fixed
query-passage pair, and dumps a `golden_reranker.safetensors` storing:

  - `input_ids`          (T,)  int64 — the exact token ids (query+passage pair)
  - `attention_mask`     (T,)  int64 — 1=real, 0=pad
  - `input_ids_shape`    (2,)  int64 — (batch, seq_len)
  - `logits`             (B, 1) f32 — raw cross-encoder logits
  - `normalized_scores`  (B, 1) f32 — sigmoid(logits)

Usage:
  uv run scripts/convert_bgem3_reranker.py
  uv run scripts/convert_bgem3_reranker.py --out path/to/golden_reranker.safetensors

Run the Rust parity test with the local fixture:
  SVOD_BGEM3=$PWD/data/bgem3 \
      cargo test -p svod-model --lib bgem3::parity::reranker -- --ignored
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForSequenceClassification, AutoTokenizer

HUB = "BAAI/bge-reranker-v2-m3"
DEFAULT_QUERY = "What is BGE-M3?"
DEFAULT_PASSAGE = "BGE-M3 is an embedding model supporting dense retrieval, lexical matching and multi-vector interaction."


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query", default=DEFAULT_QUERY)
    p.add_argument("--passage", default=DEFAULT_PASSAGE)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--max-length", type=int, default=64)
    args = p.parse_args()

    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "bgem3" / "golden_reranker.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(HUB)
    model = AutoModelForSequenceClassification.from_pretrained(HUB, torch_dtype=torch.float32)
    model.eval()

    # Tokenize as a query-passage pair (cross-encoder convention).
    enc = tok(
        args.query,
        args.passage,
        return_tensors="pt",
        padding="max_length",
        max_length=args.max_length,
        truncation=True,
    )
    input_ids = enc["input_ids"]  # (1, T)
    attn = enc["attention_mask"]  # (1, T)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attn).logits  # (1, 1)
        scores = torch.sigmoid(logits)

    ids_np = input_ids.squeeze(0).to(torch.int64).numpy()
    attn_np = attn.squeeze(0).to(torch.int64).numpy()
    logits_np = logits.squeeze(0).to(torch.float32).numpy()  # (1,)
    scores_np = scores.squeeze(0).to(torch.float32).numpy()
    shape_np = np.array(input_ids.shape, dtype=np.int64)

    save_file(
        {
            "input_ids": ids_np,
            "attention_mask": attn_np,
            "input_ids_shape": shape_np,
            "logits": logits_np,
            "normalized_scores": scores_np,
        },
        str(out),
    )
    print(f"wrote {out}")
    print(f"  input_ids {tuple(ids_np.shape)}  logits {tuple(logits_np.shape)}  ({HUB})")


if __name__ == "__main__":
    main()
