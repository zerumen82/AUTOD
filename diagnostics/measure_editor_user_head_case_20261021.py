"""The user's real case: the head selected + «debe ser calvo».

Reproduces the run of the user (log 22:14) on the same photo with the same
selection and instruction, and measures whether the change EXECUTES (the hair
of the selected head), the identity (the face of that head) and the exterior.

Only metrics are printed; the head crop is saved for the visual check.
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
SOURCE = ROOT / ".autodeep_temp" / "fa7ba290db529a64c05f968349bd13ccbc26cfe18b692adece8b6f9c1a39d31f" / "142969032.jpg"
PROMPT = "debe ser calvo"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = [r for r in analysis.regions if r.kind == "head"]
    print(f"[User] photo {photo.size}, heads: "
          f"{[(r.id, round(r.coverage, 4)) for r in heads]}", flush=True)
    target = next((r for r in heads if r.id == "head-1"), None) or (
        max(heads, key=lambda r: r.coverage) if heads else None)
    if target is None:
        print("[User] no measured head", flush=True)
        return 1
    head_mask = np.asarray(target.mask.convert("L")) > 127
    ys, xs = np.where(head_mask)
    split = int(ys.min() + (ys.max() - ys.min()) * 0.45)
    hair = head_mask.copy()
    hair[split:, :] = False
    face = head_mask & ~hair
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[User] selected {target.id}, hair px={int(hair.sum())}, "
          f"face px={int(face.sum())}, hair luminance before={luma_before:.1f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        selection_analysis=analysis, selected_region_ids=[target.id],
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[User] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    if output.size != photo.size:
        output = output.resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "user_case_head_bald.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    support = head_mask
    print(f"[User] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}, "
          f"changed inside the head={int((changed & support).sum())} "
          f"outside={int((changed & ~support).sum())}", flush=True)
    # The head crop for the visual check.
    box = (int(xs.min()) - 40, int(ys.min()) - 40, int(xs.max()) + 40, int(ys.max()) + 40)
    Image.open(SOURCE).convert("RGB").crop(box).save(OUT / "user_case_head_before.png")
    output.crop(box).save(OUT / "user_case_head_after.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
