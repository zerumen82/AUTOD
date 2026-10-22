"""The GLOBAL route (prompt only, no zone marked) with the same instruction.

The user asks whether the job can be done WITHOUT analysing zones, only
with the prompt. The global route measures the subject itself and applies
its focal recipe (the crop window with context, the anchored edit and the
measured appearance text).

Measures the same way: the hair luminance of the measured heads, the pixels
changed and the head delta.
"""
from __future__ import annotations

import os
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
PROMPT = "EL HOMBRE DEBE SER CALVO"


def newest_source():
    rows = []
    for dirpath, _dirs, files in os.walk(ROOT / ".autodeep_temp"):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(dirpath, name)
                try:
                    rows.append((os.path.getmtime(path), path))
                except OSError:
                    pass
    rows.sort(reverse=True)
    return rows[0][1] if rows else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = newest_source()
    if source is None:
        print("[Global] no source photo", flush=True)
        return 1
    photo = Image.open(source).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    print(f"[Global] source {Path(source).name} {photo.size}", flush=True)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = [r for r in analysis.regions if r.kind == "head"]
    hair_zones = {}
    for region in heads:
        mask = np.asarray(region.mask.convert("L")) > 127
        ys, xs = np.where(mask)
        hair = mask.copy()
        hair[int(ys.min() + (ys.max() - ys.min()) * 0.45):, :] = False
        hair_zones[region.id] = hair
        luma = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
        print(f"[Global] head {region.id} hair px={int(hair.sum())} "
              f"luminance before={luma:.1f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[Global] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    if output.size != photo.size:
        output = output.resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "global_only_prompt.png")
    out_arr = np.asarray(output).astype(np.int16)
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    luma_out = np.asarray(output.convert("L"), dtype=np.float32)
    for region in heads:
        hair = hair_zones[region.id]
        before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
        after = float(luma_out[hair].mean())
        print(f"[Global] head {region.id}: hair luminance {before:.1f} -> {after:.1f}, "
              f"head delta={float(delta[np.asarray(region.mask.convert('L')) > 127].mean()):.1f}", flush=True)
    print(f"[Global] changed px={int(changed.sum())} of {changed.size}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
