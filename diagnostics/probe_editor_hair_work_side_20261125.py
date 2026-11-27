"""Compare the measured head-and-hair region at two work resolutions.

Read-only with respect to the source photo. This does not call a generator.
"""

import argparse
from copy import deepcopy
from pathlib import Path
import traceback
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser
from roop.img_editor.clipseg_masker import get_clipseg_masker
from roop.img_editor.element_selection import load_settings
from roop.img_editor.img_editor_manager import ImgEditorManager


def _mask_at_side(manager, image, head, faces, masker, side):
    settings = deepcopy(load_settings())
    settings["edit"]["measured_silhouette"]["work_side"] = side
    with patch("roop.img_editor.element_selection.load_settings", return_value=settings):
        return manager._head_with_its_hair(image, head, faces, masker)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo", type=Path)
    parser.add_argument("face_x", type=float)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    image = Image.open(args.photo).convert("RGB")
    faces = detect_faces_robust(
        cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR),
        get_face_analyser(), 0.20,
    )
    face = min(faces, key=lambda item: abs(float(np.mean(item.bbox[0::2])) - args.face_x))
    manager = ImgEditorManager.__new__(ImgEditorManager)
    head = manager._head_mask_from_faces(image, [face])
    masker = get_clipseg_masker()
    args.output.mkdir(parents=True, exist_ok=True)
    original_refine = manager._refine_head_silhouette

    def record_refine(work_image, candidate, hair, *rest):
        side = max(work_image.size)
        Image.fromarray(hair.astype(np.uint8) * 255).save(
            args.output / f"selected_hair_{side}.png",
        )
        try:
            return original_refine(work_image, candidate, hair, *rest)
        except Exception:
            traceback.print_exc()
            raise

    manager._refine_head_silhouette = record_refine
    print(f"face={tuple(float(v) for v in face.bbox)}")
    raw_hair = masker.generate_mask(image, "hair")
    if raw_hair is not None:
        raw_hair.save(args.output / "raw_hair.png")
    for side in (768, 1536):
        region = _mask_at_side(manager, image, head, faces, masker, side)
        pixels = np.asarray(region.convert("L")) >= 96
        print(f"side={side} region_pixels={int(pixels.sum())}")
        region.save(args.output / f"head_hair_{side}.png")
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        box = (
            max(0, x1 - (x2 - x1)), max(0, y1 - (y2 - y1)),
            min(image.width, x2 + (x2 - x1)), min(image.height, y2 + (y2 - y1)),
        )
        crop = image.crop(box).convert("RGBA")
        overlay = Image.new("RGBA", crop.size, (255, 0, 0, 90))
        mask_crop = region.crop(box).convert("L")
        Image.composite(Image.alpha_composite(crop, overlay), crop, mask_crop).save(
            args.output / f"head_hair_overlay_{side}.png",
        )


if __name__ == "__main__":
    main()
