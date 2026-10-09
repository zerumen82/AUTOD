"""Offline source-color refinement of an actual measured object mask."""
import argparse
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image

from diagnostics.probe_editor_delta_support_replay_real import foreground


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("call_directory")
    parser.add_argument("sample")
    parser.add_argument("measured_mask")
    parser.add_argument("exclusion")
    parser.add_argument("--band-pixels", type=int, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    args = parser.parse_args()
    assert args.band_pixels > 0 and args.iterations > 0
    source = Image.open(args.source).convert("RGB")
    call = Path(args.call_directory)
    window = Image.open(call / "window.png").convert("RGB")
    sample = Image.open(args.sample).convert("RGB").resize(window.size, Image.Resampling.LANCZOS)
    _, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(np.asarray(source), np.asarray(window), cv2.TM_SQDIFF))
    box = (x, y, x + window.width, y + window.height)
    assert np.array_equal(np.asarray(source.crop(box)), np.asarray(window))
    measured = Image.open(args.measured_mask).convert("L")
    assert measured.size == source.size
    seed = np.asarray(measured.crop(box)) >= 128
    protect = (np.asarray(Image.open(args.exclusion).convert("L").crop(box)) >= 128) | foreground(sample)
    seed &= ~protect
    assert seed.any()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.band_pixels * 2 + 1,) * 2)
    near = cv2.dilate(seed.astype(np.uint8), kernel) > 0
    classes = np.full(seed.shape, cv2.GC_BGD, np.uint8)
    classes[near] = cv2.GC_PR_BGD
    classes[seed] = cv2.GC_FGD
    classes[protect] = cv2.GC_BGD
    background_model = np.zeros((1, 65), np.float64)
    foreground_model = np.zeros((1, 65), np.float64)
    cv2.grabCut(cv2.cvtColor(np.asarray(window), cv2.COLOR_RGB2BGR), classes, None,
                background_model, foreground_model, args.iterations, cv2.GC_INIT_WITH_MASK)
    selected = ((classes == cv2.GC_FGD) | (classes == cv2.GC_PR_FGD)) & ~protect
    # Hard replacement is intentional here: assess whether source-color edges
    # can remove the old object without blending its silhouette back in.
    result = source.copy()
    blended = np.where(selected[:, :, None], np.asarray(sample), np.asarray(window))
    assert np.array_equal(blended[protect], np.asarray(window)[protect])
    result.paste(Image.fromarray(blended), (x, y))
    output = Path("diagnostics/out") / f"grabcut_replay_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    output.mkdir()
    result.save(output / "result.png")
    Image.fromarray(selected.astype(np.uint8) * 255).save(output / "mask.png")
    overlay = np.asarray(window).copy()
    overlay[selected] = (overlay[selected] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
    Image.fromarray(overlay).save(output / "coverage.png")
    strip = Image.new("RGB", (source.width * 2, source.height))
    strip.paste(source)
    strip.paste(result, (source.width, 0))
    strip.save(output / "compare.png")
    metadata = {"source": args.source, "sample": args.sample, "measured_mask": args.measured_mask,
                "band_pixels": args.band_pixels, "iterations": args.iterations, "box": box,
                "seed_coverage": float(seed.mean()), "selected_coverage": float(selected.mean()),
                "protected_pixel_error": 0, "production_modified": False}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output, metadata, flush=True)


if __name__ == "__main__":
    main()
