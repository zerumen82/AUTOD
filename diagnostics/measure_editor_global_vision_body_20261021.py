"""Verify the GLOBAL route (no selection) with the measured body change.

The global route is the one that used to answer a body change with an anime
character (measured: 100 % of the pixels changed, the garment still on). Now
the measured body change runs on the engine with vision there too.

Measures the skin share inside the measured garment zone, the registration
with the source and the change in the measured head.
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
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"
PROMPT = "la mujer esta desnuda, sin zapatos"


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
    source_arr = np.asarray(photo).astype(np.int16)
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
    print(f"[Global] measured garment zone={int(garment.sum())} px, "
          f"original skin share={before:.4f}", flush=True)

    started = time.time()
    output, message, _mask = manager.generate_intelligent(
        image=photo, prompt=PROMPT, engine="kontext",
        use_semantic=True, use_rewriter=True, enhance_faces=False,
        auto_upscale=False, seed=424242,
    )
    print(f"[Global] {time.time() - started:.1f}s: {message}", flush=True)
    if output is None:
        return 1
    output = output.convert("RGB")
    if output.size != photo.size:
        output = output.resize(photo.size, Image.Resampling.LANCZOS)
    output.save(OUT / "global_vision_body_seed424242.png")
    after = skin_share(output, garment)
    delta = np.abs(source_arr - np.asarray(output).astype(np.int16)).mean(axis=2)
    changed = delta > 8
    face_note = ""
    if head is not None and head.any():
        face_note = f", head mean delta={float(delta[head].mean()):.1f}"
    print(f"[Global] skin share={after:.4f} (original {before:.4f}), "
          f"changed px={int(changed.sum())} of {changed.size}"
          f"{face_note}, registration={registration(photo, output):.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
