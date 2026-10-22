"""The hair change: the model repaints the head, the MEASURED face is restored.

Measured so far on this photo:
  * production (Kontext focal): the hair goes (139.4 -> 196.9) but the face
    changes (60.7 levels): it is another person.
  * vision engine with a small hole: the face is perfect (0.1) but the model
    paints the shape of the hole (a white cap).

This combines both: the hole is the whole MEASURED head (so the model paints
a coherent bald head, not a cap) and the MEASURED face (the segmenter) is
restored from the original photo in the composition, the same rule the
project already uses to protect other people's faces.

Measures: the hair luminance (execution), the face (identity), the pixels
changed outside the measured head and the residual hair.
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
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())

    # The face, measured by the segmenter inside the measured head.
    masker = manager._get_clipseg_masker()
    face_surface = masker.generate_mask(photo, "face")
    face = np.zeros_like(head_mask)
    if face_surface is not None:
        face = (np.asarray(face_surface.convert("L")) >= 96) & head_mask
    if not face.any():
        # No face surface: the lower half of the measured head.
        face = head_mask.copy()
        face[int(y0 + (y1 - y0) * 0.45):, x0:x1] = True
    hair = head_mask & ~face
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[HairRestore] head box=({x0},{y0},{x1},{y1}), hair px={int(hair.sum())}, "
          f"face px={int(face.sum())}, hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[HairRestore] model not available: {note}", flush=True)
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
        mask_image=Image.fromarray(head_mask.astype(np.uint8) * 255, "L"),
        keep_source_pixels=True, progress_callback=None,
    )
    print(f"[HairRestore] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "hairrestore_vision_raw.png")
    patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray((head_mask.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
    output = Image.composite(patch, photo, alpha)
    # The measured face comes from the original photo.
    output = Image.composite(photo, output, Image.fromarray(face.astype(np.uint8) * 255, "L"))
    output.save(OUT / "hairrestore_vision.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    residual = ""
    try:
        after_hair = masker.generate_mask(output, "hair")
        if after_hair is not None:
            residual = (f", residual hair in the head="
                        f"{float((np.asarray(after_hair.convert('L')) >= 96)[head_mask].mean()):.4f}")
    except Exception:
        pass
    print(f"[HairRestore] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}, "
          f"changed outside the measured head={int((changed & ~head_mask).sum())}{residual}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
