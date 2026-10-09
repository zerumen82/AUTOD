"""Offline support measurement on an actual full-noise sample, no production edit."""
import argparse
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from roop.img_editor.img_editor_manager import _otsu_threshold


def foreground(photo):
    session = ort.InferenceSession("models/Frame/isnet-general-use.onnx", providers=["CPUExecutionProvider"])
    entry = session.get_inputs()[0]
    _, _, height, width = entry.shape
    rgb = cv2.resize(np.asarray(photo), (width, height)).astype(np.float32) / 255
    prediction = session.run([session.get_outputs()[0].name], {entry.name: (rgb - .5).transpose(2, 0, 1)[None]})[0].squeeze()
    assert np.isfinite(prediction).all()
    prediction = (prediction - prediction.min()) / max(float(prediction.max() - prediction.min()), np.finfo(np.float32).eps)
    return cv2.resize(prediction, photo.size) >= .5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("call_directory")
    parser.add_argument("sample")
    parser.add_argument("exclusion")
    parser.add_argument("--bridge-pixels", type=int, required=True)
    parser.add_argument("--edge-pixels", type=int, required=True)
    parser.add_argument("--blur-pixels", type=float, required=True)
    args = parser.parse_args()
    assert args.bridge_pixels >= 0 and args.edge_pixels >= 0 and args.blur_pixels > 0
    source = Image.open(args.source).convert("RGB")
    call = Path(args.call_directory)
    window = Image.open(call / "window.png").convert("RGB")
    seed = np.asarray(Image.open(call / "mask.png").convert("L")) >= 96
    rendered = Image.open(args.sample).convert("RGB").resize(window.size, Image.Resampling.LANCZOS)
    _, _, (x, y), _ = cv2.minMaxLoc(cv2.matchTemplate(np.asarray(source), np.asarray(window), cv2.TM_SQDIFF))
    box = (x, y, x + window.width, y + window.height)
    assert np.array_equal(np.asarray(source.crop(box)), np.asarray(window)), "Actual source crop required"
    source_exclusion = np.asarray(Image.open(args.exclusion).convert("L").crop(box)) >= 128
    assert source_exclusion.shape == seed.shape
    generated_exclusion = foreground(rendered)
    protect = source_exclusion | generated_exclusion
    delta = np.abs(np.asarray(window, np.int16) - np.asarray(rendered, np.int16)).max(axis=2).astype(np.uint8)
    cutoff = _otsu_threshold(delta[~protect])
    support = (delta > cutoff) & ~protect
    if args.bridge_pixels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.bridge_pixels * 2 + 1,) * 2)
        support = (cv2.morphologyEx(support.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0) & ~protect
    _, labels = cv2.connectedComponents(support.astype(np.uint8))
    touching = np.unique(labels[seed & ~protect])
    touching = touching[touching != 0]
    selected = np.isin(labels, touching)
    alpha = selected.astype(np.uint8)
    if args.edge_pixels:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.edge_pixels * 2 + 1,) * 2)
        alpha = cv2.dilate(alpha, kernel)
    alpha = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), args.blur_pixels)
    alpha[protect] = 0
    before = np.asarray(window, np.float32)
    after = np.asarray(rendered, np.float32)
    blended = np.round(after * alpha[:, :, None] + before * (1 - alpha[:, :, None])).clip(0, 255).astype(np.uint8)
    assert np.array_equal(blended[protect], before.astype(np.uint8)[protect])
    result = source.copy()
    result.paste(Image.fromarray(blended), (x, y))
    output = Path("diagnostics/out") / f"delta_support_replay_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    output.mkdir()
    result.save(output / "result.png")
    Image.fromarray(selected.astype(np.uint8) * 255).save(output / "connected_support.png")
    Image.fromarray(protect.astype(np.uint8) * 255).save(output / "protected.png")
    Image.fromarray((alpha * 255).astype(np.uint8)).save(output / "alpha.png")
    strip = Image.new("RGB", (source.width * 2, source.height))
    strip.paste(source)
    strip.paste(result, (source.width, 0))
    strip.save(output / "compare.png")
    metadata = {"source": args.source, "sample": args.sample, "box": box, "cutoff": float(cutoff),
                "connected_coverage": float(selected.mean()), "alpha_coverage": float((alpha > .01).mean()),
                "protected_pixel_error": 0, "bridge_pixels": args.bridge_pixels,
                "edge_pixels": args.edge_pixels, "blur_pixels": args.blur_pixels,
                "production_modified": False}
    (output / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(output, metadata, flush=True)


if __name__ == "__main__":
    main()
