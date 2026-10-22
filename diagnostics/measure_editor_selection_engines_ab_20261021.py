"""A/B of the two engines for the same measured edit: hair/appearance.

The Editor's engine is decided by the measurement per category. The body
change is already measured (the vision engine executes it: skin 0.5299 vs
0.0676 with Kontext). The face/appearance category still rests on September
measurements made with a different code base, so it is measured again here
with the SAME hole (the measured head) and the SAME text in both engines.

Metrics:
  * the hair region: how much of the measured head's upper part changed
    (the project measures the hair luminance: "luz 99 -> 166,1");
  * the head itself (identity): the mean absolute difference;
  * the OTHER measured head (the other person must not change);
  * the frame: the registration with the source and the pixels changed
    outside the measured head.
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / "ui" / "tob" / "ComfyUI" / "input" / "fast_1790845644.png"
PROMPT = ("Instruction: The man is bald. Apply the requested change as "
          "completely, strongly and obviously as possible. The requested change "
          "must be clearly visible in the result; do not preserve the previous "
          "state of what was asked to change. Leave every other person in the "
          "photo exactly as they are.")
SEED = 424242
TURBO = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def registration(source, generated, limit=48):
    src = np.asarray(source.convert("L"), dtype=np.float32)
    gen = np.asarray(generated.convert("L").resize(source.size, Image.Resampling.LANCZOS), dtype=np.float32)
    src = (src - src.mean()) / (src.std() + 1e-6)
    gen = (gen - gen.mean()) / (gen.std() + 1e-6)
    height, width = src.shape
    best = 0.0
    for dy in range(-limit, limit + 1, 2):
        for dx in range(-limit, limit + 1, 2):
            y0, y1 = max(0, dy), min(height, height + dy)
            x0, x1 = max(0, dx), min(width, width + dx)
            if y1 - y0 < 60 or x1 - x0 < 60:
                continue
            a = src[y0:y1, x0:x1]
            b = gen[y0 - dy:y1 - dy, x0 - dx:x1 - dx]
            best = max(best, float((a * b).mean()))
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    photo = Image.open(SOURCE).convert("RGB")
    source_arr = np.asarray(photo).astype(np.int16)
    analysis = es.analyze_elements(photo)
    heads = sorted((r for r in analysis.regions if r.kind == "head"),
                   key=lambda r: r.coverage, reverse=True)
    target = heads[0]
    other = heads[1] if len(heads) > 1 else None
    target_mask = np.asarray(target.mask.convert("L")) > 127
    other_mask = np.asarray(other.mask.convert("L")) > 127 if other is not None else None
    # The hair is the upper part of the measured head.
    ys, xs = np.where(target_mask)
    hair = target_mask.copy()
    hair[int(ys.min()):int(ys.min() + (ys.max() - ys.min()) * 0.45), :] = False
    hair &= target_mask
    luma_before = float(np.asarray(photo.convert("L"), dtype=np.float32)[hair].mean())
    print(f"[AB] target head {target.id} coverage={target.coverage:.4f}, "
          f"hair px={int(hair.sum())}, hair luminance before={luma_before:.1f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    from roop.img_editor.selected_region_edit import VISION_BODY_MODEL
    client = get_flux_edit_comfy_client()
    manager = ImgEditorManager()

    runs = [
        ("vision_turbo", VISION_BODY_MODEL, True, True),
        ("vision_turbo_ref", VISION_BODY_MODEL, True, False),
        ("kontext", None, False, False),
    ]
    for name, version, vision, unanchored in runs:
        if version is None:
            from roop.img_editor.gen_prompt_config import get_kontext_model_config
            version = get_kontext_model_config()["model_file"]
        ok, note = client.load(flux_version=version, allow_vision_engine=vision)
        if not ok:
            print(f"[AB {name}] model not available: {note}", flush=True)
            continue
        started = time.time()
        result, note = client.generate(
            image=photo, prompt=PROMPT, negative_prompt="",
            guidance_scale=1.0 if vision else 3.5, seed=SEED,
            num_inference_steps=12, denoise=1.0,
            prompt_authority=vision, no_kontext=vision and unanchored,
            preserve_frame=vision,
            mask_image=Image.fromarray(target_mask.astype(np.uint8) * 255, "L"),
            keep_source_pixels=True, progress_callback=None,
        )
        print(f"[AB {name}] {time.time() - started:.1f}s: {note}", flush=True)
        if result is None or result.image is None:
            continue
        raw = newest_fastedit(started - 1)
        if raw is not None:
            shutil.copy(raw, OUT / f"ab_{name}_raw.png")
        patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
        alpha = Image.fromarray(
            (target_mask.astype(np.uint8) * 255), "L").filter(ImageFilter.GaussianBlur(6))
        output = Image.composite(patch, photo, alpha)
        output.save(OUT / f"ab_{name}.png")
        out_arr = np.asarray(output).astype(np.int16)
        luma_after = float(np.asarray(output.convert("L"), dtype=np.float32)[hair].mean())
        delta = np.abs(out_arr - source_arr).mean(axis=2)
        changed = delta > 8
        note_other = ""
        if other_mask is not None:
            note_other = (f", other head delta={float(delta[other_mask].mean()):.1f} "
                          f"changed px={int((changed & other_mask).sum())}")
        print(f"[AB {name}] hair luminance {luma_before:.1f} -> {luma_after:.1f}, "
              f"head delta={float(delta[target_mask].mean()):.1f}{note_other}, "
              f"changed outside the measured head={int((changed & ~target_mask).sum())}, "
              f"registration={registration(photo, output):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
