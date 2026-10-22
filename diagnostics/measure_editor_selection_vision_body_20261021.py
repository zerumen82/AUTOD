"""Measure the measured body change on the vision engine, on the real photo.

The selection route measured a body change (remove the clothing) with the
vision engine: the photo is visible to the model, the text is the only
authority and the measured garment zone is the surface that is repainted.

For each run it measures, objectively:
  * the skin share inside the measured garment zone, against the original
    (the model either removes the garment or replaces it with another one);
  * the pixels changed outside the measured selection (must stay 0);
  * the change inside the measured head of the selected subject (identity);
  * the registration of the result with the source photo.

Nothing is downloaded.
"""
from __future__ import annotations

import shutil
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
COMFY_OUT = ROOT / "ui" / "tob" / "ComfyUI" / "output"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = "la mujer esta desnuda, sin zapatos"


def newest_fastedit(after):
    candidates = [path for path in COMFY_OUT.glob("FastEdit_*.png") if path.stat().st_mtime > after]
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def skin_share(image, zone):
    """Share of skin-like pixels inside the zone.

    Skin is bright and not saturated like the garments; the rule is a
    luminance range with a loose saturation bound. It is a rough measure,
    used only to compare the runs against each other.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float32)
    maximum = arr.max(axis=2)
    minimum = arr.min(axis=2)
    value = maximum
    saturation = np.where(maximum > 0, (maximum - minimum) / np.maximum(maximum, 1e-6), 0.0)
    skin = (value > 90) & (value < 235) & (saturation < 0.42)
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
    objects = [region for region in analysis.regions if region.kind == "object"]
    region = max(objects, key=lambda item: item.coverage)
    support = np.asarray(es.union_mask(analysis, [region.id])) > 0
    head = None
    for item in analysis.regions:
        if item.kind == "head" and isinstance(item.mask, Image.Image):
            candidate = np.asarray(item.mask.convert("L")) > 127
            if candidate.shape == support.shape:
                head = candidate
    segmenter = manager._get_clothing_segmenter()
    cloth, _ = segmenter.segment_clothing(photo, threshold=0.45)
    zone = (np.asarray(cloth.convert("L")) > 127) & support
    if head is not None:
        zone = zone & ~head
    before = skin_share(photo, zone)
    print(f"[Vision] measured garment zone={int(zone.sum())} px, "
          f"original skin share={before:.4f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        selection_analysis=analysis, selected_region_ids=[region.id],
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[Vision] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    output.save(OUT / "vision_body_seed424242.png")
    raw = newest_fastedit(started - 1)
    if raw is not None:
        shutil.copy(raw, OUT / "vision_body_seed424242_raw.png")
    after = skin_share(output, zone)
    changed = np.any(source_arr != np.asarray(output), axis=2)
    face_note = ""
    if head is not None and head.any():
        delta = np.abs(source_arr.astype(np.int16) - np.asarray(output).astype(np.int16))
        face_delta = float(delta[head].mean())
        face_note = f", head mean delta={face_delta:.1f}"
    print(f"[Vision] skin share={after:.4f} (original {before:.4f}), "
          f"changed inside={int((changed & support).sum())} "
          f"outside={int((changed & ~support).sum())}"
          f"{face_note}, registration={registration(photo, output):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
