"""Measure the registration of every generated crop against the source.

The regional route pastes the generated crop back inside the measured
support. That is only coherent when the generation stays registered with the
source crop: a re-staged scene pasted into the support of the original photo
is a collage, not an edit.

Nothing is downloaded.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"


def best_shift(source, generated, limit=64):
    src = np.asarray(source.convert("L"), dtype=np.float32)
    gen = np.asarray(generated.convert("L"), dtype=np.float32)
    if gen.shape != src.shape:
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


def report(name, path, source):
    if not path.exists():
        print(f"[Registration] {name}: missing", flush=True)
        return
    generated = Image.open(path).convert("RGB")
    score, dy, dx = best_shift(source, generated)
    src = np.asarray(source.convert("L"), dtype=np.float32)
    gen = np.asarray(generated.resize(source.size, Image.Resampling.LANCZOS).convert("L"), dtype=np.float32)
    difference = float(np.abs(gen - src).mean())
    verdict = "REGISTERED" if score >= 0.95 else "RE-STAGED"
    print(f"[Registration] {name}: score={score:.3f} shift=({dy},{dx}) px "
          f"mean difference={difference:.1f} -> {verdict}", flush=True)


def main():
    source = Image.open(SOURCE).convert("RGB")
    for name in ("paired_anchored_raw", "paired_puro_raw", "lever_index_raw",
                 "scoped_uxo_raw", "scoped_index_raw", "prompt_scoped_raw",
                 "prompt_plain_raw", "strength_083_raw", "strength_global083_raw"):
        report(name, OUT / f"{name}.png", source)
    # The raw files ComfyUI wrote, in order, for this session's runs.
    for number in range(737, 752):
        path = COMFY_OUT / f"FastEdit_{number:05d}_.png"
        if path.exists():
            report(f"FastEdit_{number:05d}", path, source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
