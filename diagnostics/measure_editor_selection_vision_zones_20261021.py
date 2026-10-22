"""Measure the zone variants of the measured body change, vision engine.

Recipe A (shipped): the hole is the measured GARMENT zone. The garment goes
away but the model leaves a bikini; the identity and the frame survive
(measured: skin 0.4088, head delta 1.7, registration 0.865).

Recipe B: no hole. The instruction is fully executed (nude, no shoes) but the
person and the frame are replaced (head delta 96.7, registration 0.643).

Recipe C: the hole is the measured SUBJECT silhouette; the head is restored
from the original photo in the composition.

Recipe D: the hole is the measured subject silhouette WITHOUT the head: the
sampler keeps the face and the model repaints the whole body.

Each recipe is composed the way the selection route composes it: only the
measured support is written and the exterior stays pixel-exact.
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
PROMPT = "la mujer esta desnuda, sin zapatos"
SEED = 424242


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def skin_share(image, zone):
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    maximum = arr.max(axis=2)
    minimum = arr.min(axis=2)
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    skin = (maximum > 90) & (maximum < 235) & (saturation < 0.42)
    return float((skin & zone).sum()) / max(1, int(zone.sum()))


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
    source_arr = np.asarray(photo)
    manager = ImgEditorManager()
    analysis = es.analyze_elements(photo)
    region = max((item for item in analysis.regions if item.kind == "object"),
                 key=lambda item: item.coverage)
    support = np.asarray(es.union_mask(analysis, [region.id])) > 0
    head = None
    for item in analysis.regions:
        if item.kind == "head" and isinstance(item.mask, Image.Image):
            candidate = np.asarray(item.mask.convert("L")) > 127
            if candidate.shape == support.shape:
                head = candidate
    segmenter = manager._get_clothing_segmenter()
    cloth, _ = segmenter.segment_clothing(photo, threshold=0.45)
    garment = (np.asarray(cloth.convert("L")) > 127) & support
    if head is not None:
        garment = garment & ~head
    before = skin_share(photo, garment)
    print(f"[Zone] garment zone={int(garment.sum())} px, subject={int(support.sum())} px, "
          f"original skin share={before:.4f}", flush=True)

    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
    client = get_flux_edit_comfy_client()
    ok, note = client.load(flux_version=VISION_BODY_MODEL, allow_vision_engine=True)
    if not ok:
        print(f"[Zone] model not available: {note}", flush=True)
        return 1
    from roop.img_editor.prompt_translator import translate_prompt
    instruction = translate_prompt(PROMPT.strip())
    regime = manager.measured_body_regime(PROMPT.strip(), instruction, engine="kontext")
    body_prompt = manager._compose_generation_prompt(
        instruction, engine="longcat", magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True, body_direction=regime["direction"],
        body_intensity=regime["intensity"])
    negative = regime["negative_prompt"]
    if negative:
        body_prompt += f" Avoid: {negative}."

    effective = Image.fromarray(support.astype(np.uint8) * 255, "L")
    alpha = np.asarray(effective.filter(ImageFilter.GaussianBlur(6))).copy()
    alpha[~support] = 0
    alpha = Image.fromarray(alpha, "L")

    body_without_head = support & ~head if head is not None else support
    recipes = {
        "C_silueta": {"zone": support, "restore_head": True},
        "D_cuerpo_sin_cabeza": {"zone": body_without_head, "restore_head": False},
    }
    for name, extra in recipes.items():
        started = time.time()
        result, note = client.generate(
            image=photo, prompt=body_prompt, negative_prompt=negative,
            guidance_scale=1.0, seed=SEED, num_inference_steps=12, denoise=1.0,
            prompt_authority=True, no_kontext=True, preserve_frame=True,
            mask_image=Image.fromarray(extra["zone"].astype(np.uint8) * 255, "L"),
            keep_source_pixels=True, progress_callback=None,
        )
        print(f"[Zone {name}] {time.time() - started:.1f}s: {note}", flush=True)
        if result is None or result.image is None:
            continue
        raw = newest_fastedit(started - 1)
        if raw is not None:
            shutil.copy(raw, OUT / f"zone_{name}_raw.png")
        patch = result.image.convert("RGB").resize(photo.size, Image.Resampling.LANCZOS)
        output = Image.composite(patch, photo, alpha)
        if extra["restore_head"] and head is not None:
            output = Image.composite(photo, output, Image.fromarray(head.astype(np.uint8) * 255, "L"))
        output.save(OUT / f"zone_{name}.png")
        after = skin_share(output, garment)
        changed = np.any(source_arr != np.asarray(output), axis=2)
        face_note = ""
        if head is not None and head.any():
            delta = np.abs(source_arr.astype(np.int16) - np.asarray(output).astype(np.int16))
            face_note = f", head mean delta={float(delta[head].mean()):.1f}"
        print(f"[Zone {name}] skin share={after:.4f} (original {before:.4f}), "
              f"changed inside={int((changed & support).sum())} "
              f"outside={int((changed & ~support).sum())}{face_note}, "
              f"registration={registration(photo, output):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
