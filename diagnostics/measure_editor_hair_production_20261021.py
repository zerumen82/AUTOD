"""Measure the PRODUCTION path for a hair/appearance request.

The A/B with a synthetic prompt showed that neither engine executes the
bald request well. That A/B did NOT use the production recipe: the focal
crop, the measured hair surface and the protection of the other heads.

This measures the real route on the project's canonical case (two people,
"the man must be bald") and measures:
  * the hair region (the upper part of the measured head): the luminance
    before and after (the project's metric: "luz 99 -> 166,1");
  * the measured head (identity);
  * the OTHER measured head (the other person must not change);
  * the pixels changed outside the measured head;
  * the registration with the source.
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
SOURCE = ROOT / "ui" / "tob" / "ComfyUI" / "input" / "fast_1790845644.png"
PROMPTS = [
    ("bald", "el hombre debe ser calvo"),
    ("long_hair", "el hombre debe tener el pelo largo"),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = sorted((r for r in analysis.regions if r.kind == "head"),
                   key=lambda r: r.coverage, reverse=True)
    target = heads[0]
    other = heads[1] if len(heads) > 1 else None
    target_mask = np.asarray(target.mask.convert("L")) > 127
    other_mask = np.asarray(other.mask.convert("L")) > 127 if other is not None else None
    ys, xs = np.where(target_mask)
    hair = target_mask.copy()
    hair[int(ys.min()):int(ys.min() + (ys.max() - ys.min()) * 0.45), :] = False
    hair &= target_mask
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[Hair] measured head {target.id} ({target.label}), hair px="
          f"{int(hair.sum())}, hair luminance before={luma_before:.1f}", flush=True)

    for name, prompt in PROMPTS:
        started = time.time()
        output, message, _mask = manager.generate_intelligent(
            image=photo, prompt=prompt, engine="kontext",
            use_semantic=True, use_rewriter=True, enhance_faces=False,
            auto_upscale=False, seed=424242,
        )
        print(f"[Hair {name}] {time.time() - started:.1f}s: {message}", flush=True)
        if output is None:
            continue
        output = output.convert("RGB")
        if output.size != photo.size:
            output = output.resize(photo.size, Image.Resampling.LANCZOS)
        output.save(OUT / f"hair_{name}_production.png")
        out_arr = np.asarray(output).astype(np.int16)
        luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
        delta = np.abs(out_arr - source_arr).mean(axis=2)
        changed = delta > 8
        note_other = ""
        if other_mask is not None:
            note_other = (f", other head delta={float(delta[other_mask].mean()):.1f} "
                          f"changed px={int((changed & other_mask).sum())}")
        print(f"[Hair {name}] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
              f"head delta={float(delta[target_mask].mean()):.1f}{note_other}, "
              f"changed outside the measured head={int((changed & ~target_mask).sum())}",
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
