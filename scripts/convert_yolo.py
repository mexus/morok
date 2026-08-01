#!/usr/bin/env python3
"""Generate golden test data for YOLO26n parity test.

Outputs golden.safetensors with:
  - images:        [1, 3, 640, 640] f32 — a deterministic test image
  - images_shape:  [4] i64 — shape of images
  - output:        [1, 84, 2100] f32 — PyTorch inference output

Usage:
  pip install ultralytics safetensors torch
  python scripts/convert_yolo.py  # writes to data/yolo/golden.safetensors
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

WEIGHTS_URL_REPO = "ultralytics/yolo26n"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "yolo"


def main() -> None:
    from ultralytics import YOLO

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = YOLO(f"{WEIGHTS_URL_REPO}.pt")
    model.eval()

    # Create deterministic input: gradient pattern normalized to [0, 1]
    img = np.zeros((1, 3, 640, 640), dtype=np.float32)
    for c in range(3):
        for h in range(640):
            for w in range(640):
                img[0, c, h, w] = (h + w + c * 213) / (640 + 640 + 3 * 213)

    images_t = torch.from_numpy(img)

    # Run inference to get the raw model output (pre-NMS)
    # Ultralytics YOLO.predict returns Results objects; we need the raw tensor.
    # Use model.model directly (the nn.Module):
    with torch.no_grad():
        # YOLO export-mode forward returns [B, 4+nc, A]
        output = model.model(images_t)

    if isinstance(output, (list, tuple)):
        output = output[0]

    # Ensure output is [B, C, A]
    if output.ndim == 3 and output.shape[0] != 1:
        output = output.permute(0, 2, 1)  # [B, A, C] -> [B, C, A]

    output_np = output.cpu().float().numpy()
    print(f"output shape: {output_np.shape}")

    # Save
    golden = {
        "images": images_t,
        "images_shape": torch.tensor(list(images_t.shape), dtype=torch.int64),
        "output": torch.from_numpy(output_np),
    }
    out_path = OUTPUT_DIR / "golden.safetensors"
    save_file(golden, str(out_path))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
