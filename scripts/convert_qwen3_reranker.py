# /// script
# dependencies = ["torch", "numpy", "safetensors", "transformers"]
# requires-python = ">=3.10"
# ///
"""Generate golden parity fixtures for Qwen3-Reranker-0.6B."""

import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.numpy import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-Reranker-0.6B"
QUERY = "What is Qwen3?"
PASSAGE = "Qwen3 is a large language model series developed by Qwen team."
MAX_LEN = 32


def last_logit_pool(logits, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return logits[:, -1, :]
    else:
        seq_lens = attention_mask.sum(dim=1) - 1
        return torch.stack([logits[i, seq_lens[i]] for i in range(logits.shape[0])])


def main():
    emb_local = Path(__file__).resolve().parent.parent / "submodules" / "Qwen3-Embedding-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(str(emb_local))

    reranker_local = Path(__file__).resolve().parent.parent / "submodules" / "Qwen3-Reranker-0.6B"
    model_path = str(reranker_local) if reranker_local.exists() else MODEL_ID

    print(f"Loading {model_path} ...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.float32)
    model.eval()

    tokenizer.padding_side = "left"
    prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    prompt = (
        f"{prefix}<Instruct>: {''}\n<Query>: {QUERY}\n<Document>: {PASSAGE}{suffix}"
    )
    enc = tokenizer(prompt, padding="max_length", max_length=MAX_LEN, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    print(f"input_ids shape: {tuple(input_ids.shape)}")

    yes_loc = tokenizer("Yes", add_special_tokens=False)["input_ids"][0]
    print(f"yes_loc: {yes_loc}")

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits
        pooled = last_logit_pool(logits, attention_mask)
        scores = pooled[:, yes_loc]
        normalized = torch.sigmoid(scores.float())

    print(f"scores shape: {tuple(scores.shape)}")
    print(f"score: {normalized.item():.6f}")

    out_dir = Path(__file__).resolve().parent.parent / "data" / "qwen3"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "golden_reranker.safetensors"

    save_file(
        {
            "input_ids": input_ids.numpy().astype(np.int64).flatten(),
            "attention_mask": attention_mask.numpy().astype(np.int64).flatten(),
            "input_ids_shape": np.array([1, MAX_LEN], dtype=np.int64),
            "scores": normalized.numpy().astype(np.float32),
        },
        str(out_path),
    )
    print(f"Saved golden to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
