"""Measure installed ISNet foreground against a real semantic seed, CPU only.

No production mask/routing changes. Foreground may include unrelated people;
saved overlays reveal whether connected foreground is usable for this object.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("photo")
    parser.add_argument("seed")
    parser.add_argument("--threshold", type=float, default=.5)
    parser.add_argument("--model", default="models/Frame/isnet-general-use.onnx")
    args = parser.parse_args()
    photo = Image.open(args.photo).convert("RGB")
    seed = np.asarray(Image.open(args.seed).convert("L")) >= 96
    assert seed.shape == (photo.height, photo.width), "Seed must match actual photo"
    model = Path(args.model)
    assert model.is_file(), "Only already installed models are allowed"
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    _, _, height, width = input_meta.shape
    assert isinstance(height, int) and isinstance(width, int), input_meta.shape
    rgb = cv2.resize(np.asarray(photo), (width, height)).astype(np.float32) / 255
    tensor = (rgb - .5).transpose(2, 0, 1)[None]
    started = time.perf_counter()
    raw = np.asarray(session.run([session.get_outputs()[0].name], {input_meta.name: tensor})[0]).squeeze()
    assert np.isfinite(raw).all() and raw.ndim == 2
    probability = (raw - raw.min()) / max(float(raw.max() - raw.min()), np.finfo(np.float32).eps)
    probability = cv2.resize(probability, photo.size)
    _, labels = cv2.connectedComponents((probability >= args.threshold).astype(np.uint8))
    touching = np.unique(labels[seed])
    touching = touching[touching != 0]
    connected = np.isin(labels, touching)
    output = Path("diagnostics/out") / f"foreground_region_{time.time_ns()}"
    output.mkdir()
    Image.fromarray((probability * 255).astype(np.uint8)).save(output / "foreground_probability.png")
    Image.fromarray((connected * 255).astype(np.uint8)).save(output / "connected_foreground.png")
    panels = []
    for selected in (seed, connected):
        overlay = np.asarray(photo).copy()
        overlay[selected] = (overlay[selected].astype(np.float32)*.55 + np.array([255, 0, 0])*.45).astype(np.uint8)
        panels.append(overlay)
    Image.fromarray(np.concatenate(panels, axis=1)).save(output / "coverage_compare.png")
    meta = {"photo": args.photo, "seed": args.seed, "model": args.model,
            "threshold": args.threshold, "providers": session.get_providers(),
            "seconds": time.perf_counter()-started, "seed_coverage": float(seed.mean()),
            "connected_coverage": float(connected.mean()), "components": touching.tolist()}
    (output / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(output, meta, flush=True)


if __name__ == "__main__":
    main()
