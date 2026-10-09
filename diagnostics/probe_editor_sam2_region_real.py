"""Isolated SAM2 comparison with measured seed points and local files only.

All alternative masks are saved for visual inspection. Model confidence is
recorded without declaring completeness or automatically changing production.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
from PIL import Image
import torch
from transformers import Sam2Model, Sam2Processor


def interior_point(mask):
    assert mask.any(), "No measured point available"
    # Padding also measures distance from the photo boundary for border seeds.
    distance = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    y, x = np.unravel_index(distance.argmax(), distance.shape)
    return [int(x), int(y)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_directory")
    parser.add_argument("photo")
    parser.add_argument("seed")
    parser.add_argument("--negative-mask")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model_directory = Path(args.model_directory)
    weights = model_directory / "model.safetensors"
    assert weights.is_file(), "Model download requires explicit approval; this probe never downloads"
    photo = Image.open(args.photo).convert("RGB")
    seed = np.asarray(Image.open(args.seed).convert("L")) >= 96
    assert seed.shape == (photo.height, photo.width) and seed.any()
    negative = np.zeros(seed.shape, bool)
    if args.negative_mask:
        negative = np.asarray(Image.open(args.negative_mask).convert("L")) >= 128
        assert negative.shape == seed.shape
    points = [interior_point(seed & ~negative)]
    labels = [1]
    count, components = cv2.connectedComponents((negative & ~seed).astype(np.uint8))
    for component in range(1, count):
        # Exclusion points always come from the actual measured mask.
        points.append(interior_point(components == component))
        labels.append(0)
    processor = Sam2Processor.from_pretrained(model_directory, local_files_only=True)
    model, loading_info = Sam2Model.from_pretrained(
        model_directory, local_files_only=True, output_loading_info=True,
    )
    loading_info = {name: sorted(value) if isinstance(value, set) else value
                    for name, value in loading_info.items()}
    assert not loading_info.get("missing_keys"), loading_info
    assert not loading_info.get("mismatched_keys"), loading_info
    assert not loading_info.get("error_msgs"), loading_info
    model = model.to(args.device).eval()
    inputs = processor(images=photo, input_points=[[points]], input_labels=[[labels]], return_tensors="pt")
    original_sizes = inputs["original_sizes"]
    model_inputs = {key: value.to(args.device) for key, value in inputs.items() if key != "original_sizes"}
    started = time.perf_counter()
    with torch.inference_mode():
        outputs = model(**model_inputs, multimask_output=True)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), original_sizes)[0][0].numpy()
    elapsed = time.perf_counter() - started
    assert masks.ndim == 3 and masks.shape[1:] == seed.shape
    target = Path("diagnostics/out") / f"sam2_region_{time.time_ns()}_{uuid.uuid4().hex[:8]}"
    target.mkdir()
    panels = [np.asarray(photo)]
    measurements = []
    for index, selected in enumerate(masks):
        selected = selected.astype(bool)
        Image.fromarray((selected * 255).astype(np.uint8)).save(target / f"mask_{index}.png")
        overlay = np.asarray(photo).copy()
        overlay[selected] = (overlay[selected] * .55 + np.array([255, 0, 0]) * .45).astype(np.uint8)
        panels.append(overlay)
        measurements.append({"index": index, "model_iou_score": float(outputs.iou_scores[0, 0, index]),
                             "coverage": float(selected.mean()), "seed_recall": float(selected[seed].mean()),
                             "exclusion_overlap": int((selected & negative).sum())})
    digest = hashlib.sha256()
    with weights.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    metadata = {"model_directory": str(model_directory), "sha256": digest.hexdigest(),
                "photo": args.photo, "seed": args.seed, "negative_mask": args.negative_mask,
                "device": args.device, "points": points, "labels": labels,
                "seconds": elapsed, "masks": measurements, "loading_info": loading_info}
    (target / "meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    Image.fromarray(np.concatenate(panels, axis=1)).save(target / "compare.png")
    print(target, metadata, flush=True)


if __name__ == "__main__":
    main()
