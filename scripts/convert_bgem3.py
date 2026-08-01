# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "transformers", "huggingface_hub"]
# ///
"""Generate BGE-M3 parity fixtures for the Rust `svod-model` port.

Downloads `BAAI/bge-m3`, runs the backbone and all three embedding heads
(dense, sparse, ColBERT) on a fixed input sequence, and dumps a
`golden.safetensors` storing:

  - `input_ids`          (T,)  int64 — the exact token ids fed to the model
  - `attention_mask`     (T,)  int64 — 1=real, 0=pad
  - `input_ids_shape`    (2,)  int64 — (batch, seq_len)
  - `last_hidden_state`  (B, T, D)    f32 — `model(...).last_hidden_state`
  - `dense_vecs`         (B, D)       f32 — CLS-pooled L2-normalized dense embedding
  - `sparse_vecs`        (B, V)       f32 — sparse lexical embedding (full vocab)
  - `colbert_vecs`       (B, L-1, Dc) f32 — ColBERT multi-vector embeddings

Usage:
  uv run scripts/convert_bgem3.py
  uv run scripts/convert_bgem3.py --out path/to/golden.safetensors

Run the Rust parity test with the local fixture:
  SVOD_BGEM3=$PWD/data/bgem3 \
      cargo test -p svod-model --lib bgem3::parity -- --ignored
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModel, AutoTokenizer

HUB = "BAAI/bge-m3"
DEFAULT_PROMPT = "What is BGE-M3?"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="text to tokenize for the golden forward")
    p.add_argument("--out", type=Path, default=None, help="output golden.safetensors path")
    p.add_argument("--max-length", type=int, default=32, help="tokenizer max_length / padding target")
    args = p.parse_args()

    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "bgem3" / "golden.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(HUB)
    model = AutoModel.from_pretrained(HUB, torch_dtype=torch.float32)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    input_ids = enc["input_ids"]  # (1, T)
    attn = enc["attention_mask"]  # (1, T)

    with torch.no_grad():
        out_t = model(input_ids=input_ids, attention_mask=attn).last_hidden_state  # (1, T, D)

    # Dense embedding: CLS pooling + L2 normalize.
    dense_vecs = out_t[:, 0]  # (1, D)
    dense_vecs = torch.nn.functional.normalize(dense_vecs, dim=-1)

    # ColBERT vectors: skip CLS, project + mask + normalize.
    # We need to load the colbert_linear weights from the .pt file.
    import os
    from huggingface_hub import hf_hub_download

    colbert_path = hf_hub_download(repo_id=HUB, filename="colbert_linear.pt")
    sparse_path = hf_hub_download(repo_id=HUB, filename="sparse_linear.pt")
    colbert_sd = torch.load(colbert_path, map_location="cpu", weights_only=True)
    sparse_sd = torch.load(sparse_path, map_location="cpu", weights_only=True)

    colbert_linear = torch.nn.Linear(model.config.hidden_size, model.config.hidden_size)
    colbert_linear.load_state_dict(colbert_sd)
    colbert_linear.eval()
    sparse_linear = torch.nn.Linear(model.config.hidden_size, 1)
    sparse_linear.load_state_dict(sparse_sd)
    sparse_linear.eval()

    with torch.no_grad():
        # ColBERT: skip CLS, linear, mask, normalize.
        colbert_vecs = colbert_linear(out_t[:, 1:])  # (1, T-1, Dc)
        colbert_vecs = colbert_vecs * attn[:, 1:].unsqueeze(-1).float()
        colbert_vecs = torch.nn.functional.normalize(colbert_vecs, dim=-1)

        # Sparse: relu(linear(hidden)) → scatter to vocab.
        token_weights = torch.relu(sparse_linear(out_t)).squeeze(-1)  # (1, T)
        sparse_vecs = torch.zeros(1, model.config.vocab_size, dtype=token_weights.dtype)
        sparse_vecs = sparse_vecs.scatter_reduce(
            dim=-1, index=input_ids, src=token_weights, reduce="amax"
        )
        unused_tokens = [
            tok.cls_token_id, tok.eos_token_id, tok.pad_token_id, tok.unk_token_id,
        ]
        sparse_vecs[:, unused_tokens] *= 0.0

    ids_np = input_ids.squeeze(0).to(torch.int64).numpy()
    attn_np = attn.squeeze(0).to(torch.int64).numpy()
    hidden_np = out_t.squeeze(0).to(torch.float32).numpy()
    dense_np = dense_vecs.squeeze(0).to(torch.float32).numpy()
    sparse_np = sparse_vecs.squeeze(0).to(torch.float32).numpy()
    colbert_np = colbert_vecs.squeeze(0).to(torch.float32).numpy()
    shape_np = np.array(input_ids.shape, dtype=np.int64)

    save_file(
        {
            "input_ids": ids_np,
            "attention_mask": attn_np,
            "input_ids_shape": shape_np,
            "last_hidden_state": hidden_np,
            "dense_vecs": dense_np,
            "sparse_vecs": sparse_np,
            "colbert_vecs": colbert_np,
        },
        str(out),
    )
    print(f"wrote {out}")
    print(
        f"  input_ids {tuple(ids_np.shape)}  hidden {tuple(hidden_np.shape)}  "
        f"dense {tuple(dense_np.shape)}  sparse {tuple(sparse_np.shape)}  "
        f"colbert {tuple(colbert_np.shape)}  ({HUB})"
    )


if __name__ == "__main__":
    main()
