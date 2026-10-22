"""The hair change with the engine that follows the text, on the MEASURED hair.

The white cap measured in the previous run is the known artifact of this
project ("el modelo quitó el pelo pero conservó la diadema... la cabeza salió
como un gorrito blanco"), and the project has the measured text that removes
it (`_bald_recipe_tail`).

This combines the two measured pieces:
  * the hole is the MEASURED hair surface (the segmenter), not a geometric
    slice of the head;
  * the prompt carries the measured bald recipe (no headband, no hat, the
    natural skin tone).

The face and everything outside the measured hair must not move.
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
    head = max((r for r in analysis.regions if r.kind == "head"), key=lambda r: r.coverage)
    head_mask = np.asarray(head.mask.convert("L")) > 127
    ys, xs = np.where(head_mask)
    face = head_mask.copy()
    face[int(ys.min() + (ys.max() - ys.min()) * 0.45):, :] = False

    # The MEASURED hair surface (the segmenter), restricted to the head.
    masker = manager._get_clipseg_masker()
    hair_mask = masker.generate_mask(photo, "hair")
    if hair_mask is None:
        print("[HairMeasured] the segmenter did not find the hair surface", flush=True)
        return 1
    hair = (np.asarray(hair_mask.convert("L")) >= 96) & head_mask
    if not hair.any():
        print("[HairMeasured] the measured hair does not overlap the head", flush=True)
        return 1
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[HairMeasured] measured hair px={int(hair.sum())}, "
          f"face px={int(face.sum())}, hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[HairMeasured] model not available: {note}", flush=True)
        return 1
    prompt = (
        "Instruction: The woman is bald. Apply the requested change as "
        "completely, strongly and obviously as possible."
        + manager._bald_recipe_tail("the woman is bald")
        + " The requested change must be clearly visible in the result; do not "
        "preserve the previous state of what was asked to change. Leave every "
        "other person in the photo exactly as they are."
    )
    started = time.time()
    result, note = client.generate(
        image=photo, prompt=prompt, negative_prompt="",
        guidance_scale=1.0, seed=SEED, num_inference_steps=12, denoise=1.0,
        prompt_authority=True, no_kontext=True, preserve_frame=True,
        mask_image=Image.fromarray(hair.astype(np.uint8) * 255, "L"),
        keep_source_pixels=True, progress_callback=None,
    )
    print(f"[HairMeasured] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "hairmeasured_vision_raw.png")
    patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray((hair.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
    output = Image.composite(patch, photo, alpha)
    output.save(OUT / "hairmeasured_vision.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    # The residual hair, measured with the segmenter on the result.
    try:
        after_hair = masker.generate_mask(output, "hair")
        residual = ""
        if after_hair is not None:
            residual = (f", residual hair in the head="
                        f"{float((np.asarray(after_hair.convert('L')) >= 96)[head_mask].mean()):.4f}")
    except Exception:
        residual = ""
    print(f"[HairMeasured] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}, "
          f"changed outside the measured hair={int((changed & ~hair).sum())}{residual}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
