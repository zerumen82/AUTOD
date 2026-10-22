"""The user's case: the head selected + «el hombre de la izquierda ...».

The vision engine with the WHOLE HEAD as the hole replaces the head with an
invented one (the user's report at 01:40: «una edición pésima»: the measured
head became a bald stranger). This measures the other recipe: the hole is
the MEASURED HAIR of the selected head (the segmenter), so the face is
outside the hole and the model only repaints the hair.

Metrics: the hair luminance (execution), the face (identity), the pixels
changed outside the hair and a head crop for the visual check.
"""
from __future__ import annotations

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
SOURCE = ROOT / ".autodeep_temp" / "710f7c4b0b76a303c1373a62b3cd9abed562677cef46d1b35fd5833a8be233e6" / "216313561_14-33-34_08-07-27.jpg"
PROMPT = "the man on the left must be bald"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    head = next((r for r in analysis.regions if r.id == "head-0"), None) or max(
        (r for r in analysis.regions if r.kind == "head"), key=lambda r: r.coverage)
    head_mask = np.asarray(head.mask.convert("L")) > 127
    ys, xs = np.where(head_mask)
    print(f"[Hair] selected {head.id} bbox={head_mask.getbbox() if hasattr(head_mask,'getbbox') else (xs.min(), ys.min(), xs.max(), ys.max())}",
          flush=True)

    # The MEASURED hair of that head (the segmenter), and the face = the rest.
    masker = manager._get_clipseg_masker()
    surface = masker.generate_mask(photo, "hair")
    hair = (np.asarray(surface.convert("L")) >= 96) & head_mask if surface is not None else None
    if hair is None or not hair.any():
        # No surface: the upper half of the measured head.
        hair = head_mask.copy()
        hair[int(ys.min() + (ys.max() - ys.min()) * 0.45):, :] = False
    face = head_mask & ~hair
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[Hair] measured hair px={int(hair.sum())}, face px={int(face.sum())}, "
          f"hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[Hair] model not available: {note}", flush=True)
        return 1
    prompt = (
        "Instruction: The man on the left must be bald. Apply the requested "
        "change as completely, strongly and obviously as possible."
        + manager._bald_recipe_tail("the man on the left must be bald")
        + " The requested change must be clearly visible in the result; do not "
        "preserve the previous state of what was asked to change. Leave every "
        "other person in the photo exactly as they are."
    )
    started = time.time()
    result, note = client.generate(
        image=photo, prompt=prompt, negative_prompt="",
        guidance_scale=1.0, seed=424242, num_inference_steps=12, denoise=1.0,
        prompt_authority=True, no_kontext=True, preserve_frame=True,
        mask_image=Image.fromarray(hair.astype(np.uint8) * 255, "L"),
        keep_source_pixels=True, progress_callback=None,
    )
    print(f"[Hair] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray((hair.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
    output = Image.composite(patch, photo, alpha)
    output.save(OUT / "hairzone_user_case.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    print(f"[Hair] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}, "
          f"changed inside the hair={int((changed & hair).sum())} "
          f"outside={int((changed & ~hair).sum())}", flush=True)
    box = (max(0, int(xs.min()) - 60), max(0, int(ys.min()) - 60),
           min(photo.width, int(xs.max()) + 60), min(photo.height, int(ys.max()) + 60))
    output.crop(box).save(OUT / "hairzone_user_case_after.png")
    photo.crop(box).save(OUT / "hairzone_user_case_before.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
