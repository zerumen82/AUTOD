"""Compare support dilation on the last recorded source and LongCat output."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path("diagnostics/out/editor_bald_last_ui_replay_20261126")


def main() -> None:
    source = Image.open(ROOT / "source.png").convert("RGB")
    raw = Image.open("ui/tob/ComfyUI/output/FastEdit_01436_.png").convert("RGB")
    raw = raw.resize(source.size, Image.Resampling.LANCZOS)
    measured = np.asarray(Image.open(ROOT / "paste_mask.png").convert("L")) >= 96
    for radius in (0, 4, 8, 12, 16, 24):
        if radius:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1),
            )
            support = cv2.dilate(measured.astype(np.uint8), kernel) > 0
        else:
            support = measured
        alpha = Image.fromarray(support.astype(np.uint8) * 255, "L")
        alpha = alpha.filter(ImageFilter.GaussianBlur(radius=3))
        result = Image.composite(raw, source, alpha)
        result.save(ROOT / f"paste_dilate_{radius}.png")
        result.crop((50, 75, 270, 280)).save(ROOT / f"head_dilate_{radius}.png")
        print(radius, int(support.sum()))
    hard = Image.fromarray(measured.astype(np.uint8) * 255, "L")
    softened = np.asarray(hard.filter(ImageFilter.GaussianBlur(radius=1)))
    clipped = Image.fromarray(np.where(measured, softened, 0).astype(np.uint8), "L")
    result = Image.composite(raw, source, clipped)
    result.crop((50, 75, 270, 280)).save(ROOT / "head_original_clip.png")
    manager = ImgEditorManager()
    head_mask = Image.open(ROOT / "paste_mask.png").convert("L")
    for name, path in (
        ("first", "ui/tob/ComfyUI/output/FastEdit_01436_.png"),
        ("second", "ui/tob/ComfyUI/output/FastEdit_01437_.png"),
    ):
        generated = Image.open(path).convert("RGB")
        integrated = manager._integrate_local_generation(
            source, generated, head_mask,
            (0, 0, source.width, source.height),
            keep_measured_mask=True, expand_head_support=True,
        )
        integrated.save(ROOT / f"production_{name}.png")
        integrated.crop((50, 75, 270, 280)).save(ROOT / f"production_{name}_head.png")


if __name__ == "__main__":
    main()
