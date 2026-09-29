"""Inspect person-mask residuals around a generated office background."""
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.clipseg_masker import get_clipseg_masker


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "ui/tob/ComfyUI/input/fast_1790203567.png"
GENERATED = ROOT / "ui/tob/ComfyUI/output/FastEdit_00595_.png"
OUT = ROOT / "diagnostics"


def foreground(masker, image):
    arr = np.zeros((image.height, image.width), dtype=np.uint8)
    for query in ("person", "human", "man", "woman"):
        part = masker.generate_mask(image, query)
        if part is None:
            continue
        arr = np.maximum(arr, np.asarray(part.convert("L"), dtype=np.uint8))
    return (arr >= 96).astype(np.uint8)


def main():
    source = Image.open(SOURCE).convert("RGB")
    generated = Image.open(GENERATED).convert("RGB")
    if generated.size != source.size:
        generated = generated.resize(source.size, Image.Resampling.LANCZOS)
    masker = get_clipseg_masker()
    src_mask = foreground(masker, source)
    gen_mask = foreground(masker, generated)
    protected = cv2.dilate(src_mask, np.ones((7, 7), np.uint8))
    residual = gen_mask & (1 - protected)
    print(
        f"[PROBE] source={src_mask.mean():.1%} generated={gen_mask.mean():.1%} "
        f"residual={residual.mean():.1%}", flush=True,
    )
    Image.fromarray(residual * 255).save(OUT / "scene_ghost_residual.png")
    for width in (21, 41, 61):
        guard = cv2.dilate(src_mask, np.ones((width, width), np.uint8))
        matte = cv2.GaussianBlur((1 - guard).astype(np.float32), (0, 0), 3)
        final = (
            np.asarray(generated).astype(np.float32) * matte[:, :, None]
            + np.asarray(source).astype(np.float32) * (1 - matte[:, :, None])
        ).clip(0, 255).astype(np.uint8)
        Image.fromarray(final).save(OUT / f"scene_guard_{width}.png")
    for width in (11, 21, 31):
        core = cv2.erode(src_mask, np.ones((width, width), np.uint8))
        core = cv2.GaussianBlur(core.astype(np.float32), (0, 0), 4)
        final = (
            np.asarray(source).astype(np.float32) * core[:, :, None]
            + np.asarray(generated).astype(np.float32) * (1 - core[:, :, None])
        ).clip(0, 255).astype(np.uint8)
        Image.fromarray(final).save(OUT / f"scene_core_{width}.png")
    for radius in (5, 11):
        expanded = cv2.dilate(residual, np.ones((radius, radius), np.uint8)) * 255
        repaired = cv2.inpaint(np.asarray(generated)[:, :, ::-1].copy(), expanded, 7, cv2.INPAINT_TELEA)
        repaired = cv2.cvtColor(repaired, cv2.COLOR_BGR2RGB)
        matte = cv2.GaussianBlur((1 - protected).astype(np.float32), (0, 0), 3)
        final = (
            repaired.astype(np.float32) * matte[:, :, None]
            + np.asarray(source).astype(np.float32) * (1 - matte[:, :, None])
        ).clip(0, 255).astype(np.uint8)
        Image.fromarray(final).save(OUT / f"scene_ghost_cleanup_{radius}.png")
    masker.unload()


if __name__ == "__main__":
    main()
