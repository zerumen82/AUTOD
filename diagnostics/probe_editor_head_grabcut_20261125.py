"""Inspect whether measured face and hair can bound a head edit region."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.clipseg_masker import get_clipseg_masker


ROOT = Path("diagnostics/out")


def inspect_case(name, source, region_path, face_x):
    out = ROOT / "head_grabcut_20261125" / name
    out.mkdir(parents=True, exist_ok=True)
    photo = Image.open(source).convert("RGB")
    photo.thumbnail((768, 768))
    rgb = np.asarray(photo)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    region = np.asarray(Image.open(region_path).convert("L").resize(photo.size)) >= 96
    masker = get_clipseg_masker()
    hair = masker.generate_mask(photo, "hair")
    face = masker.generate_mask(photo, "face")
    hair = np.asarray(hair.convert("L")) >= 96
    face = np.asarray(face.convert("L")) >= 96
    selected_x = face_x * photo.width / Image.open(source).width
    face_n, face_labels, face_stats, face_centers = cv2.connectedComponentsWithStats(
        face.astype(np.uint8), 8,
    )
    chosen = min(
        range(1, face_n),
        key=lambda i: abs(face_centers[i, 0] - selected_x),
    )
    face = face_labels == chosen
    hair_n, hair_labels, hair_stats, hair_centers = cv2.connectedComponentsWithStats(
        hair.astype(np.uint8), 8,
    )
    hair = np.zeros_like(hair)
    for i in range(1, hair_n):
        if abs(hair_centers[i, 0] - selected_x) < max(30, face_stats[chosen, cv2.CC_STAT_WIDTH]):
            hair |= hair_labels == i
    seed = face | hair
    dilate_radius = max(5, int(max(photo.size) * 0.02))
    wide = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (dilate_radius * 2 + 1, dilate_radius * 2 + 1),
    )) > 0
    gc = np.full(region.shape, cv2.GC_PR_BGD, np.uint8)
    gc[wide] = cv2.GC_PR_FGD
    gc[~wide] = cv2.GC_BGD
    sure = cv2.erode(seed.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    gc[sure] = cv2.GC_FGD
    cv2.grabCut(bgr, gc, None, np.zeros((1, 65), np.float64),
                np.zeros((1, 65), np.float64), 5, cv2.GC_INIT_WITH_MASK)
    found = np.isin(gc, (cv2.GC_FGD, cv2.GC_PR_FGD))
    count, labels, _, _ = cv2.connectedComponentsWithStats(found.astype(np.uint8), 8)
    keep = np.zeros_like(found)
    for i in range(1, count):
        part = labels == i
        if (part & seed).any():
            keep |= part
    print(name, "region", region.sum(), "seed", seed.sum(), "grabcut", keep.sum())
    for key, mask in (("region", region), ("seed", seed), ("grabcut", keep)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / f"{key}.png")
        overlay = rgb.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
        Image.fromarray(overlay).save(out / f"{key}_overlay.jpg", quality=90)


if __name__ == "__main__":
    inspect_case(
        "headfix",
        ROOT / "review_headfix_real_escalated_20261125/source_real.png",
        ROOT / "hair_resolution_headfix_20261125/head_hair_768.png", 4700,
    )
    inspect_case(
        "last_edit",
        Path(".autodeep_temp/c9d0164f13255d3455fed3276963a3f62cbc304ac3d0a48d94ad8e72ef96ce4d/954898847-1.jpg"),
        ROOT / "hair_resolution_last_edit_20261125/head_hair_768.png", 4690,
    )
