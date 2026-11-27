"""Inspect difference-guided paste support on the saved neutral sample."""

from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "diagnostics/out/appearance_reference_20261023/1795764767693508900"
MASKS = ROOT / "diagnostics/out/editor_grace_replay_20261127"
OUT = ROOT / "diagnostics/out/editor_grace_support_candidates_20261127"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = np.asarray(Image.open(SAMPLE / "source.png").convert("RGB"))
    raw = np.asarray(Image.open(SAMPLE / "raw_0.png").convert("RGB").resize(
        (source.shape[1], source.shape[0]), Image.Resampling.LANCZOS))
    core = np.asarray(Image.open(MASKS / "head_refined.png").convert("L")) >= 96
    ellipse = np.asarray(Image.open(MASKS / "head.png").convert("L")) >= 96
    subject = np.asarray(Image.open(MASKS / "subject.png").convert("L")) >= 96
    diff = np.max(np.abs(raw.astype(np.int16) - source.astype(np.int16)), axis=2).astype(np.uint8)
    span = max(np.ptp(np.where(core)[0]) + 1, np.ptp(np.where(core)[1]) + 1)
    radius = round(span * .22)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)
    near = cv2.dilate(core.astype(np.uint8), kernel) > 0
    search = near & ellipse
    threshold, _ = cv2.threshold(diff[search].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    print(f"span={span}, radius={radius}, Otsu={threshold}, search={search.sum()}, core={core.sum()}")
    Image.fromarray(diff).save(OUT / "diff.png")
    for cutoff in (int(threshold), 40, 60, 80):
        changed = (diff >= cutoff) & search
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
        count, labels, stats, _ = cv2.connectedComponentsWithStats(changed.astype(np.uint8), 8)
        support = core.copy()
        close_core = cv2.dilate(core.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
        for index in range(1, count):
            piece = labels == index
            if stats[index, cv2.CC_STAT_AREA] >= 20 and (piece & close_core).any():
                support |= piece
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 3.0)[:, :, None]
        output = np.clip(source * (1 - alpha) + raw * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(support.astype(np.uint8) * 255).save(OUT / f"mask_{cutoff}.png")
        Image.fromarray(output).save(OUT / f"composite_{cutoff}.png")
        print(f"cutoff={cutoff}, support={support.sum()}, added={(support & ~core).sum()}")
    core_y = np.where(core)[0]
    top, bottom = int(core_y.min()), int(core_y.max())
    for fraction in (.25, .4, .5):
        ceiling = int(top + (bottom - top) * fraction)
        support = core | (ellipse & (np.indices(core.shape)[0] <= ceiling))
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 5.0)[:, :, None]
        output = np.clip(source * (1 - alpha) + raw * alpha, 0, 255).astype(np.uint8)
        label = round(fraction * 100)
        Image.fromarray(support.astype(np.uint8) * 255).save(OUT / f"upper_mask_{label}.png")
        Image.fromarray(output).save(OUT / f"upper_composite_{label}.png")
        print(f"upper={label}%, ceiling={ceiling}, support={support.sum()}")
        support = core | (ellipse & subject & (np.indices(core.shape)[0] <= ceiling))
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 5.0)[:, :, None]
        output = np.clip(source * (1 - alpha) + raw * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(support.astype(np.uint8) * 255).save(OUT / f"subject_upper_mask_{label}.png")
        Image.fromarray(output).save(OUT / f"subject_upper_composite_{label}.png")
        print(f"subject upper={label}%, support={support.sum()}")
    ceiling = int(top + (bottom - top) * .4)
    for radius in (10, 20, 30):
        wider = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * radius + 1,) * 2)) > 0
        support = core | (ellipse & wider & (np.indices(core.shape)[0] <= ceiling))
        alpha = cv2.GaussianBlur(support.astype(np.float32), (0, 0), 5.0)[:, :, None]
        output = np.clip(source * (1 - alpha) + raw * alpha, 0, 255).astype(np.uint8)
        Image.fromarray(output).save(OUT / f"dilated_subject_{radius}.png")
        print(f"dilated subject {radius}px, support={support.sum()}")


if __name__ == "__main__":
    main()
