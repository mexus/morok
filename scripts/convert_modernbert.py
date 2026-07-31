# /// script
# requires-python = ">=3.10"
# dependencies = ["torch", "numpy", "safetensors", "transformers", "huggingface_hub"]
# ///
"""Generate ModernBERT parity fixtures for the Rust `svod-model` port.

Downloads `answerdotai/ModernBERT-base`, runs the backbone (no MLM head) on a
fixed `input_ids` sequence, and dumps a `golden.safetensors` storing:

  - `input_ids`        (T,)  int64 — the exact token ids fed to the model
  - `input_ids_shape`  (2,)  int64 — (batch, seq_len)
  - `last_hidden_state` (B, T, D) f32 — `model(...).last_hidden_state`
  - `mlm_logits`       (B, T, V) f32 — `AutoModelForMaskedLM(...).logits`

The weights themselves are left as-is (`model.safetensors` is fetched by the
Rust test via the same HF repo, so no copy is needed here).

Usage:
  uv run scripts/convert_modernbert.py            # writes ../data/modernbert/golden.safetensors
  uv run scripts/convert_modernbert.py --large    # ModernBERT-large
  uv run scripts/convert_modernbert.py --out path/to/golden.safetensors

Run the Rust parity test with the local fixture:
  SVOD_MODERNBERT=$PWD/data/modernbert \
      cargo test -p svod-model --lib modernbert::parity -- --ignored
"""

import argparse
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

DEFAULT_PROMPT = "The quick brown fox jumps over the lazy dog."
HUB_BASE = "answerdotai/ModernBERT-base"
HUB_LARGE = "answerdotai/ModernBERT-large"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--large", action="store_true", help="use ModernBERT-large (default: base)")
    p.add_argument("--prompt", default=DEFAULT_PROMPT, help="text to tokenize for the golden forward")
    p.add_argument("--out", type=Path, default=None, help="output golden.safetensors path")
    p.add_argument("--max-length", type=int, default=32, help="tokenizer max_length / padding target")
    args = p.parse_args()

    hub = HUB_LARGE if args.large else HUB_BASE
    out = args.out or (Path(__file__).resolve().parent.parent / "data" / "modernbert" / "golden.safetensors")
    out.parent.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(hub)
    model = AutoModel.from_pretrained(hub, torch_dtype=torch.float32)
    model.eval()

    enc = tok(args.prompt, return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True)
    input_ids = enc["input_ids"]  # (1, T)
    attn = enc["attention_mask"]  # (1, T)

    with torch.no_grad():
        out_t = model(input_ids=input_ids, attention_mask=attn).last_hidden_state  # (1, T, D)

    # MLM head: `AutoModelForMaskedLM` shares the backbone weights and adds the
    # `head.dense` / `head.norm` / tied-decoder path producing (B, T, V) logits.
    mlm = AutoModelForMaskedLM.from_pretrained(hub, torch_dtype=torch.float32)
    mlm.eval()
    with torch.no_grad():
        logits_t = mlm(input_ids=input_ids, attention_mask=attn).logits  # (1, T, V)

    ids_np = input_ids.squeeze(0).to(torch.int64).numpy()
    attn_np = attn.squeeze(0).to(torch.int64).numpy()  # (T,) 1=real, 0=pad
    hidden_np = out_t.squeeze(0).to(torch.float32).numpy()  # (T, D)
    logits_np = logits_t.squeeze(0).to(torch.float32).numpy()  # (T, V)

    # The Rust test reads input_ids_shape to recover (B, T).
    shape_np = np.array(input_ids.shape, dtype=np.int64)

    save_file(
        {
            "input_ids": ids_np,
            "attention_mask": attn_np,
            "input_ids_shape": shape_np,
            "last_hidden_state": hidden_np,
            "mlm_logits": logits_np,
        },
        str(out),
    )
    print(f"wrote {out}")
    print(
        f"  input_ids {tuple(ids_np.shape)}  attention_mask {tuple(attn_np.shape)}  "
        f"hidden {tuple(hidden_np.shape)}  logits {tuple(logits_np.shape)}  ({hub})"
    )


if __name__ == "__main__":
    main()
