"""Inspect actual CLIPSeg probability coverage; no production mask change."""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("query")
    args = parser.parse_args()
    image = Image.open(args.photo).convert("RGB")
    manager = ImgEditorManager()
    masker = manager._get_clipseg_masker()
    seed = masker.generate_mask(image, args.query)
    heat = masker.probability(image, args.query)
    assert seed is not None and heat is not None
    quantized = (heat.clip(0, 1) * 255).astype(np.uint8)
    cutoff, _ = cv2.threshold(quantized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, labels = cv2.connectedComponents((quantized > cutoff).astype(np.uint8))
    seed_pixels = np.asarray(seed) >= 96
    touching = np.unique(labels[seed_pixels])
    touching = touching[touching != 0]
    proposed = np.isin(labels, touching)
    output = Path(__file__).resolve().parents[1] / "diagnostics/out" / f"region_probability_{time.time_ns()}"
    output.mkdir()
    seed.save(output / "baseline_mask.png")
    Image.fromarray((proposed * 255).astype(np.uint8)).save(output / "connected_otsu_mask.png")
    Image.fromarray(quantized).save(output / "probability.png")
    photo = np.asarray(image)
    panels = []
    for selected in (seed_pixels, proposed):
        overlay = photo.copy()
        overlay[selected] = (overlay[selected].astype(np.float32) * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
        panels.append(overlay)
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "coverage_compare.png")
    meta = {"photo": args.photo, "query": args.query, "cutoff": float(cutoff) / 255,
            "baseline_coverage": float(seed_pixels.mean()), "proposed_coverage": float(proposed.mean()),
            "max_probability": float(heat.max())}
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(output, meta, flush=True)


if __name__ == "__main__":
    main()
