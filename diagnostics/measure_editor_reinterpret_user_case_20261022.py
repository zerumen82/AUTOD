"""The user's case with the frame reinterpreted (the chosen trade-off).

The user's photo, the head-0 selection and the instruction of 01:40
(«EL HOMBRE DE LA IZQUIERDA DEBE SER CALVO»). The frame is reinterpreted by
the engine with vision (no hole, no anchor): the user decided (22/10) that
a re-done scene is better than a bad patch of a head.

Measures: the hair luminance of the selected head (execution), the pixels
changed outside it, and saves a crop for the visual check.
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
SOURCE = ROOT / ".autodeep_temp" / "710f7c4b0b76a303c1373a62b3cd9abed562677cef46d1b35fd5833a8be233e6" / "216313561_14-33-34_08-07-27.jpg"
PROMPT = "EL HOMBRE DE LA IZQUIERDA DEBE SER CALVO"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    head = next((r for r in analysis.regions if r.id == "head-0"), None)
    head_mask = np.asarray(head.mask.convert("L")) > 127
    ys, xs = np.where(head_mask)
    hair = head_mask.copy()
    hair[int(ys.min() + (ys.max() - ys.min()) * 0.45):, :] = False
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[Reint] head-0 bbox=({xs.min()},{ys.min()},{xs.max()},{ys.max()}), "
          f"hair luminance before={luma_before:.1f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        selection_analysis=analysis, selected_region_ids=["head-0"],
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[Reint] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    if output.size != photo.size:
        output = output.resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "reinterpret_user_case.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    print(f"[Reint] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"changed inside the head={int((changed & head_mask).sum())} "
          f"outside={int((changed & ~head_mask).sum())}", flush=True)
    box = (0, 0, photo.width, min(photo.height, 260))
    output.crop(box).save(OUT / "reinterpret_user_case_after.png")
    photo.crop(box).save(OUT / "reinterpret_user_case_before.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
