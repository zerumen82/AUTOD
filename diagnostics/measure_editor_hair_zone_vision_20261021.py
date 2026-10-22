"""Measure a hair/appearance change with the engine that follows the text.

The production path on Kontext preserves the photo but does not execute the
request (measured: the hair luminance barely moves). The engine with vision
executes it but, with the whole head as the hole, it redraws the person and
touches the other one.

This measures the third combination: the hole is only the HAIR of the
measured head (its upper part, the scalp zone), so the model repaints the
hair while the face and the rest of the head stay, and the other person is
outside the hole.

Metrics: the hair luminance (execution), the lower part of the head
(identity), the other measured head (the other person) and the pixels
changed outside the hole.
"""
from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.selected_region_edit import VISION_BODY_MODEL  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = ("Instruction: The woman is bald. Apply the requested change as "
          "completely, strongly and obviously as possible. The requested change "
          "must be clearly visible in the result; do not preserve the previous "
          "state of what was asked to change. Leave every other person in the "
          "photo exactly as they are.")
SEED = 424242


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    heads = sorted((r for r in analysis.regions if r.kind == "head"),
                   key=lambda r: r.coverage, reverse=True)
    target, other = heads[0], (heads[1] if len(heads) > 1 else None)
    head_mask = np.asarray(target.mask.convert("L")) > 127
    other_mask = np.asarray(other.mask.convert("L")) > 127 if other is not None else None
    ys, xs = np.where(head_mask)
    split = int(ys.min() + (ys.max() - ys.min()) * 0.45)
    hair = head_mask.copy()
    hair[split:, :] = False
    face = head_mask & ~hair
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[HairZone] hair px={int(hair.sum())}, face px={int(face.sum())}, "
          f"hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[HairZone] model not available: {note}", flush=True)
        return 1
    started = time.time()
    result, note = client.generate(
        image=photo, prompt=PROMPT, negative_prompt="",
        guidance_scale=1.0, seed=SEED, num_inference_steps=12, denoise=1.0,
        prompt_authority=True, no_kontext=True, preserve_frame=True,
        mask_image=Image.fromarray(hair.astype(np.uint8) * 255, "L"),
        keep_source_pixels=True, progress_callback=None,
    )
    print(f"[HairZone] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "hairzone_vision_raw.png")
    patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray((hair.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
    output = Image.composite(patch, photo, alpha)
    output.save(OUT / "hairzone_vision.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    note_other = ""
    if other_mask is not None:
        note_other = (f", other head delta={float(delta[other_mask].mean()):.1f} "
                      f"changed px={int((changed & other_mask).sum())}")
    print(f"[HairZone] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}{note_other}, "
          f"changed outside the hair hole={int((changed & ~hair).sum())}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
