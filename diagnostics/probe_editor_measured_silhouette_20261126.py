"""Inspect the class-neutral silhouette refiner on a real animal photo."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from skimage import data

from roop.img_editor.element_selection import load_settings
from roop.img_editor.img_editor_manager import ImgEditorManager
from roop.img_editor.clipseg_masker import get_clipseg_masker


def main():
    output = Path("diagnostics/out/measured_silhouette_cat_20261126")
    output.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(data.chelsea()).convert("RGB")
    segment = get_clipseg_masker().generate_mask(image, "cat")
    assert segment is not None, "real cat was not measured"
    seed = np.asarray(segment.convert("L")) >= 96
    broad = cv2.dilate(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (101, 101)),
    ) > 0
    settings = load_settings()["edit"]["measured_silhouette"]
    refined = ImgEditorManager._refine_measured_silhouette(
        image, broad, seed, settings,
    )
    assert refined is not None, "real animal region was not refined"
    assert int((refined & seed).sum()) == int(seed.sum())
    print(f"cat seed={int(seed.sum())} broad={int(broad.sum())} refined={int(refined.sum())}")
    rgb = np.asarray(image).copy()
    for name, mask in (("seed", seed), ("broad", broad), ("refined", refined)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"{name}.png")
        overlay = rgb.copy()
        overlay[mask] = (
            0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"{name}_overlay.jpg", quality=90)


if __name__ == "__main__":
    main()
