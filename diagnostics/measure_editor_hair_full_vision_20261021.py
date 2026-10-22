"""The hair change with the whole MEASURED hair as the hole.

The white cap measured before has a cause: the model paints the shape of
the hole. With a small hole (the top of the head) it painted a cap that
matched it. So the hole must be the whole hair of the measured head: the
measured head minus the measured face.

The face is measured (the face detector), so it is outside the hole: the
model repaints the hair and the face must not move.

Measures: the hair luminance (execution), the face (identity), the pixels
changed outside the hole and the residual hair measured with the segmenter.
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


def measured_face_box(photo, head_mask):
    """The measured face box inside the head (the face detector)."""
    try:
        import cv2
        from roop.face_util import get_face_analyser
        analyser = get_face_analyser()
        if analyser is None:
            return None
        import roop.face_util as face_util
        faces = face_util.get_face_analyser().get(np.asarray(photo)[:, :, ::-1]) \
            if hasattr(face_util, "get_face_analyser") else None
    except Exception:
        return None
    return faces


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

    # The face, measured by the project's own face detector.
    face_box = None
    try:
        from roop.face_util import get_face_analyser
        analyser = get_face_analyser()
        if analyser is not None:
            import cv2
            det = analyser.get(np.asarray(photo)[:, :, ::-1])
            for face in det:
                fx, fy, fw, fh = face["bbox"]
                if x0 - 20 <= fx <= x1 and y0 - 20 <= fy <= y1:
                    face_box = (int(fx), int(fy), int(fx + fw), int(fy + fh))
                    break
    except Exception as error:
        print(f"[HairFull] face detection failed: {error}", flush=True)
    face = np.zeros_like(head_mask)
    if face_box is not None:
        face[face_box[1]:face_box[3], face_box[0]:face_box[2]] = True
        face &= head_mask
    else:
        # No detector: the face is the lower half of the measured head.
        face[int(y0 + (y1 - y0) * 0.45):, x0:x1] = True
        face &= head_mask
    hair = head_mask & ~face
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[HairFull] head {head.id} box=({x0},{y0},{x1},{y1}), face box={face_box}, "
          f"hair px={int(hair.sum())}, face px={int(face.sum())}, "
          f"hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[HairFull] model not available: {note}", flush=True)
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
    print(f"[HairFull] {time.time() - started:.1f}s: {note}", flush=True)
    if result is None or result.image is None:
        return 1
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "hairfull_vision_raw.png")
    patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
    alpha = Image.fromarray((hair.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
    output = Image.composite(patch, photo, alpha)
    output.save(OUT / "hairfull_vision.png")
    out_arr = np.asarray(output).astype(np.int16)
    luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
    delta = np.abs(out_arr - source_arr).mean(axis=2)
    changed = delta > 8
    residual = ""
    try:
        masker = manager._get_clipseg_masker()
        after_hair = masker.generate_mask(output, "hair")
        if after_hair is not None:
            residual = (f", residual hair in the head="
                        f"{float((np.asarray(after_hair.convert('L')) >= 96)[head_mask].mean()):.4f}")
    except Exception:
        pass
    print(f"[HairFull] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
          f"face delta={float(delta[face].mean()):.1f}, "
          f"changed outside the measured hair={int((changed & ~hair).sum())}{residual}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
