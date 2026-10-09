"""Capture the same bald edit with and without hair-surface completion."""

from unittest.mock import patch

import numpy as np
from PIL import Image

from diagnostics.probe_editor_bald_compliance import capture, case_dir
from roop.img_editor.img_editor_manager import ImgEditorManager


PHOTO = "diagnostics/out/bath800_seedA/source_real.png"


def main() -> None:
    original = ImgEditorManager._complete_hair_surface

    def record(image, seed, subject, faces):
        target = case_dir("bald_mask_current")
        target.mkdir(parents=True, exist_ok=True)
        Image.fromarray(subject.astype(np.uint8) * 255, "L").save(target / "subject.png")
        Image.fromarray(seed.astype(np.uint8) * 255, "L").save(target / "seed.png")
        return original(image, seed, subject, faces)

    with patch.object(ImgEditorManager, "_complete_hair_surface", staticmethod(record)):
        capture("bald_mask_current", PHOTO, "el hombre debe ser calvo", "The man must be bald")

    with patch.object(
        ImgEditorManager, "_complete_hair_surface",
        staticmethod(lambda image, seed, subject, faces: np.zeros(seed.shape, dtype=bool)),
    ):
        capture("bald_mask_base", PHOTO, "el hombre debe ser calvo", "The man must be bald")

    current = np.asarray(Image.open(case_dir("bald_mask_current") / "mask.png").convert("L")) >= 96
    base = np.asarray(Image.open(case_dir("bald_mask_base") / "mask.png").convert("L")) >= 96
    added = current & ~base
    assert added.any()
    print("base", int(base.sum()), "current", int(current.sum()), "added", int(added.sum()))
    Image.fromarray(added.astype(np.uint8) * 255, "L").save(
        case_dir("bald_mask_current") / "added_only.png"
    )


if __name__ == "__main__":
    main()
