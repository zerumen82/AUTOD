"""Measure the PRODUCTION path for a hair request on the real photo.

On the two-person test photo the production route abstains ("the focal core
is over 35% of the photo"), which protects the other person but leaves the
request unfulfilled. This measures the same request on the photo of the
case (400x600, one person): the hair of the measured head is the surface
that must change, the face must stay, and the exterior must be intact.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPTS = [
    ("bald", "la mujer debe ser calva"),
    ("long_hair", "la mujer debe tener el pelo largo"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = [r for r in analysis.regions if r.kind == "head"]
    head = max(heads, key=lambda r: r.coverage) if heads else None
    head_mask = (np.asarray(head.mask.convert("L")) > 127) if head else None
    if head_mask is None:
        print("[HairProd] no measured head", flush=True)
        return 1
    ys, xs = np.where(head_mask)
    split = int(ys.min() + (ys.max() - ys.min()) * 0.45)
    hair = head_mask.copy()
    hair[split:, :] = False
    face = head_mask & ~hair
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[HairProd] head {head.id}, hair px={int(hair.sum())}, "
          f"hair luminance before={luma_before:.1f}", flush=True)

    for name, prompt in PROMPTS:
        started = time.time()
        output, message, _mask = manager.generate_intelligent(
            image=photo, prompt=prompt, engine="kontext",
            use_semantic=True, use_rewriter=True, enhance_faces=False,
            auto_upscale=False, seed=424242,
        )
        print(f"[HairProd {name}] {time.time() - started:.1f}s: {message}", flush=True)
        if output is None:
            continue
        output = output.convert("RGB")
        if output.size != photo.size:
            output = output.resize(photo.size, Image.Resampling.LANCZOS)
        output.save(OUT / f"hairprod_{name}.png")
        out_arr = np.asarray(output).astype(np.int16)
        luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
        delta = np.abs(out_arr - source_arr).mean(axis=2)
        changed = delta > 8
        print(f"[HairProd {name}] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
              f"face delta={float(delta[face].mean()):.1f}, "
              f"changed px={int(changed.sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
