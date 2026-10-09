"""Inspect weak semantic support excluding measured foreground, offline only."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("probability")
    parser.add_argument("seed")
    parser.add_argument("exclusion")
    parser.add_argument("--thresholds", type=float, nargs="+", required=True)
    args = parser.parse_args()
    photo = Image.open(args.photo).convert("RGB")
    heat = np.asarray(Image.open(args.probability).convert("L"), np.float32) / 255
    seed = np.asarray(Image.open(args.seed).convert("L")) >= 96
    exclusion = np.asarray(Image.open(args.exclusion).convert("L")) >= 128
    assert heat.shape == seed.shape == exclusion.shape == (photo.height, photo.width)
    output = Path("diagnostics/out") / f"connected_probability_{time.time_ns()}"
    output.mkdir()
    panels = [np.asarray(photo)]
    meta = {"photo": args.photo, "probability": args.probability,
            "seed": args.seed, "exclusion": args.exclusion, "thresholds": {}}
    for threshold in args.thresholds:
        assert 0 < threshold < 1
        support = (heat >= threshold) & ~exclusion
        _, labels = cv2.connectedComponents(support.astype(np.uint8))
        touching = np.unique(labels[seed & ~exclusion])
        touching = touching[touching != 0]
        selected = np.isin(labels, touching)
        name = str(threshold).replace(".", "_")
        Image.fromarray((selected * 255).astype(np.uint8)).save(output / f"mask_{name}.png")
        overlay = np.asarray(photo).copy()
        overlay[selected] = (overlay[selected] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
        panels.append(overlay)
        meta["thresholds"][str(threshold)] = {"coverage": float(selected.mean()),
                                             "seed_retained": float(selected[seed].mean())}
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "compare.png")
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(output, meta, flush=True)


if __name__ == "__main__":
    main()
