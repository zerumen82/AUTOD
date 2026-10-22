"""Acceptance criteria of the measured regime for the selected region.

The defect: the selected region answered a body request with a copy of the
photo. The fix has to satisfy BOTH measured conditions, and this script
measures them on the real photo:

  1. the change EXECUTES: the clothes of the selected subject are not the
     ones of the source photo;
  2. the generation stays REGISTERED with the source crop: a re-staged scene
     pasted into the measured support is a collage, not an edit.

Each engine regime of the measured recipe is measured on both conditions and
reported, so the chosen one rests on numbers and not on an opinion.

Nothing is downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"


def torso(source):
    """The torso of the selected subject, measured on the source photo."""
    arr = np.asarray(source.convert("RGB"))
    height, width = arr.shape[0], arr.shape[1]
    box = (int(width * 0.32), int(height * 0.40), int(width * 0.60), int(height * 0.60))
    return arr[box[1]:box[3], box[0]:box[2]].astype(np.float32)


def registration(source, generated, limit=48):
    src = np.asarray(source.convert("L"), dtype=np.float32)
    gen = np.asarray(generated.convert("L").resize(source.size, Image.Resampling.LANCZOS), dtype=np.float32)
    src = (src - src.mean()) / (src.std() + 1e-6)
    gen = (gen - gen.mean()) / (gen.std() + 1e-6)
    best = (0.0, 0, 0)
    for dy in range(-limit, limit + 1, 2):
        for dx in range(-limit, limit + 1, 2):
            ys = slice(max(0, dy), src.shape[0] + min(0, dy))
            yd = slice(max(0, -dy), src.shape[0] + min(0, -dy))
            xs = slice(max(0, dx), src.shape[1] + min(0, dx))
            xd = slice(max(0, -dx), src.shape[1] + min(0, -dx))
            a, b = src[ys, xs], gen[yd, xd]
            if a.size < 1000:
                continue
            score = float((a * b).mean())
            if score > best[0]:
                best = (score, dy, dx)
    return best


def measure(name, path, source):
    if not path.exists():
        print(f"[Acceptance] {name}: missing", flush=True)
        return
    generated = Image.open(path).convert("RGB")
    score, dy, dx = registration(source, generated)
    change = float(np.abs(torso(generated) - torso(source)).mean())
    executed = change >= 20.0
    registered = score >= 0.95 and abs(dy) <= 4 and abs(dx) <= 4
    print(f"[Acceptance] {name}: torso change={change:.1f} levels "
          f"(executed={executed}), registration score={score:.3f} shift=({dy},{dx}) "
          f"(registered={registered})", flush=True)


def main():
    source = Image.open(SOURCE).convert("RGB")
    for name in ("paired_anchored_raw", "paired_puro_raw", "lever_index_raw",
                 "scoped_index_raw", "prompt_scoped_raw", "prompt_plain_raw",
                 "strength_083_raw", "lever_tsz_raw", "lever_tsz_plain_raw"):
        measure(name, OUT / f"{name}.png", source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
