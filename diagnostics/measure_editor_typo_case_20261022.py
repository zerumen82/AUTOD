"""The user's case with the typo: the anchored recipe changes nothing.

The user wrote «el hombre de la derecha debe ser cxalvo» (a word the
analyser does not cover). The route measured nothing, used the anchored
Kontext recipe and returned the photo untouched. Now the route measures the
written zone: if it did not move, it retries with the engine that follows
the text.

Measures: the change inside the written zone (the hair of the selected
head), the pixels changed outside it, and whether the retry happened.
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
PROMPT = "el hombre de la derecha debe ser cxalvo"


def newest_source():
    rows = []
    for dirpath, _dirnames, filenames in os.walk(ROOT / ".autodeep_temp"):
        for name in filenames:
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
        print("[Typo] no source photo found", flush=True)
        return 1
    photo = Image.open(source).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    print(f"[Typo] source {Path(source).name} {photo.size}", flush=True)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = [r for r in analysis.regions if r.kind == "head"]
    target = next((r for r in heads if r.id == "head-1"), None) or (
        max(heads, key=lambda r: r.coverage) if heads else None)
    if target is None:
        print("[Typo] no measured head", flush=True)
        return 1
    head_mask = np.asarray(target.mask.convert("L")) > 127
    ys, xs = np.where(head_mask)
    split = int(ys.min() + (ys.max() - ys.min()) * 0.45)
    hair = head_mask.copy()
    hair[split:, :] = False
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[Typo] selected {target.id}, hair px={int(hair.sum())}, "
          f"hair luminance before={luma_before:.1f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        selection_analysis=analysis, selected_region_ids=[target.id],
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[Typo] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    if output.size != photo.size:
        output = output.resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "typo_case_head.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    print(f"[Typo] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"changed inside the head={int((changed & head_mask).sum())} "
          f"outside={int((changed & ~head_mask).sum())}", flush=True)
    box = (int(xs.min()) - 40, int(ys.min()) - 40, int(xs.max()) + 40, int(ys.max()) + 40)
    output.crop(box).save(OUT / "typo_case_head_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
