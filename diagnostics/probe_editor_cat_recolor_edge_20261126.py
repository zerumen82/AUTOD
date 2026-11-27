"""Inspect GrabCut refinement of a measured recolor region on a real cat."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.element_selection import load_settings
from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    output = Path("diagnostics/out/generic_surface_route_20261126")
    sample = Path("diagnostics/out/appearance_reference_20261023/1795723663305177100")
    image = Image.open(sample / "source.png").convert("RGB")
    raw = Image.open(sample / "raw_0.png").convert("RGB").resize(
        image.size, Image.Resampling.LANCZOS,
    )
    candidate = np.asarray(Image.open(output / "cat_recolor_region.png").convert("L")) >= 96
    settings = load_settings()["edit"]["measured_silhouette"]
    rgb = np.asarray(image)
    for radius in (3, 6, 10):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1),
        )
        seed = cv2.erode(candidate.astype(np.uint8), kernel) > 0
        refined = ImgEditorManager._refine_measured_silhouette(
            image, candidate, seed, settings,
        )
        if refined is None:
            continue
        print(radius, int(candidate.sum()), int(refined.sum()))
        mask = Image.fromarray(refined.astype(np.uint8) * 255, "L")
        mask.save(output / f"cat_edge_mask_{radius}.png")
        overlay = rgb.copy()
        overlay[refined] = (
            0.55 * overlay[refined] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"cat_edge_overlay_{radius}.png")
        Image.composite(raw, image, mask).save(output / f"cat_edge_result_{radius}.png")
    for reach in (2, 4, 8, 12):
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (reach * 2 + 1, reach * 2 + 1),
        )
        proposed = cv2.dilate(candidate.astype(np.uint8), kernel) > 0
        core = cv2.erode(candidate.astype(np.uint8), kernel) > 0
        refined = ImgEditorManager._refine_measured_silhouette(
            image, proposed, core, settings,
        )
        if refined is None:
            continue
        print("reach", reach, int(refined.sum()))
        overlay = rgb.copy()
        overlay[refined] = (
            0.55 * overlay[refined] + 0.45 * np.array([255, 0, 0])
        ).astype(np.uint8)
        Image.fromarray(overlay).save(output / f"cat_edge_overlay_expand_{reach}.png")
        Image.composite(
            raw, image, Image.fromarray(refined.astype(np.uint8) * 255, "L"),
        ).save(output / f"cat_edge_result_expand_{reach}.png")
    base = np.asarray(Image.open(output / "cat_base_subject.png").convert("L")) >= 96
    original_lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    edited_lab = cv2.cvtColor(np.asarray(raw), cv2.COLOR_RGB2LAB).astype(np.float32)
    for name, region in (("base", base), ("refined", candidate)):
        core = cv2.erode(region.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
        delta = np.median(
            edited_lab[core, 1:] - original_lab[core, 1:], axis=0,
        )
        shifted = original_lab.copy()
        shifted[..., 1:] = np.clip(shifted[..., 1:] + delta, 0, 255)
        colored = cv2.cvtColor(shifted.astype(np.uint8), cv2.COLOR_LAB2RGB)
        alpha = cv2.GaussianBlur(region.astype(np.float32), (0, 0), 2.0)
        fused = colored.astype(np.float32) * alpha[..., None] + rgb.astype(np.float32) * (1 - alpha[..., None])
        Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
            output / f"cat_source_recolor_{name}.png",
        )
        print("source recolor", name, delta)
        direct = original_lab.copy()
        direct[..., 1:] = np.median(edited_lab[core, 1:], axis=0)
        direct_rgb = cv2.cvtColor(direct.astype(np.uint8), cv2.COLOR_LAB2RGB)
        fused = direct_rgb.astype(np.float32) * alpha[..., None] + rgb.astype(np.float32) * (1 - alpha[..., None])
        Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
            output / f"cat_source_recolor_direct_{name}.png",
        )
        spatial = original_lab.copy()
        spatial[..., 1:] = edited_lab[..., 1:]
        spatial_rgb = cv2.cvtColor(spatial.astype(np.uint8), cv2.COLOR_LAB2RGB)
        fused = spatial_rgb.astype(np.float32) * alpha[..., None] + rgb.astype(np.float32) * (1 - alpha[..., None])
        Image.fromarray(np.clip(fused, 0, 255).astype(np.uint8)).save(
            output / f"cat_source_recolor_spatial_{name}.png",
        )


if __name__ == "__main__":
    main()
