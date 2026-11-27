"""Measure real animal and object masks through the editor's region routes."""

from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image
from skimage import data

from roop.img_editor.img_editor_manager import ImgEditorManager


def _overlay(image: Image.Image, mask: Image.Image, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB")).copy()
    selected = np.asarray(mask.convert("L")) > 128
    rgb[selected] = (
        0.55 * rgb[selected] + 0.45 * np.array([255, 0, 0])
    ).astype(np.uint8)
    Image.fromarray(rgb).save(path, quality=90)


def main() -> None:
    output = Path("diagnostics/out/generic_surface_route_20261126")
    output.mkdir(parents=True, exist_ok=True)
    manager = ImgEditorManager()

    cat = Image.fromarray(data.chelsea()).convert("RGB")
    instruction = "The cat should be blue"
    base = manager._measured_subject_passes(cat, instruction, instruction, [])
    refined = manager._measured_subject_passes(
        cat, instruction, instruction, [], refine_existing_surface=True,
    )
    assert len(base) == len(refined) == 1, "animal subject was not grounded"
    plain = np.asarray(base[0]["mask"].convert("L")) > 128
    tightened = np.asarray(refined[0]["mask"].convert("L")) > 128
    assert np.any(plain != tightened), "animal appearance route skipped refinement"
    _overlay(cat, base[0]["mask"], output / "cat_before.jpg")
    base[0]["mask"].save(output / "cat_base_subject.png")
    _overlay(cat, refined[0]["mask"], output / "cat_after.jpg")
    refined[0]["mask"].save(output / "cat_refined_subject.png")
    print(f"cat route: {int(plain.sum())} -> {int(tightened.sum())} px")
    region = manager._measured_region_passes(
        cat, instruction, instruction, surface_recolor=True,
    )
    assert len(region) == 1, "animal recolor region was not grounded"
    recolor_mask = np.asarray(region[0]["mask"].convert("L")) >= 96
    assert recolor_mask[150, 220], "measured cat was excluded"
    assert not recolor_mask[150, 440], "background was swallowed by color growth"
    continuation = manager._complete_request_surface(
        cat, region[0]["mask"], Image.new("L", cat.size, 255), [],
        complete_hair=False,
    )
    if continuation is not None:
        continued = np.asarray(continuation.convert("L")) >= 96
        assert not continued[150, 440], "recolor continuation swallowed background"
    region[0]["mask"].save(output / "cat_recolor_region.png")
    _overlay(cat, region[0]["mask"], output / "cat_recolor_region.jpg")
    with patch.object(
        ImgEditorManager, "_refine_measured_silhouette",
        side_effect=cv2.error("GrabCut unavailable"),
    ):
        fallback = manager._grow_recolor_surface(cat, base[0]["mask"])
    assert fallback is None, "GrabCut failure must retain the measured region"

    coffee = Image.fromarray(data.coffee()).convert("RGB")
    instruction = "Change the cup"
    cups = manager._measured_region_passes(coffee, instruction, instruction)
    assert len(cups) == 1, "object region was not grounded"
    _overlay(coffee, cups[0]["mask"], output / "cup_route.jpg")
    print(f"cup route: {int((np.asarray(cups[0]['mask']) > 128).sum())} px")


if __name__ == "__main__":
    main()
