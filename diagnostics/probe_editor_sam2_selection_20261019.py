"""Local SAM2 image proposals, without annotations or generator conditioning.

Compare actual CLIPSeg-seeded prompts and an image-wide uniform point grid.
Save alternatives: predicted IoU/stability never certify object completeness.
No GrabCut, downloads, production mutations or category-specific rules.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from transformers import Sam2Model, Sam2Processor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from roop.img_editor.element_selection import image_fingerprint  # noqa: E402


def spread_points(mask, count):
    depth = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    yy, xx = np.indices(mask.shape)
    nearest = np.full(mask.shape, np.inf)
    points = []
    for _ in range(count):
        score = depth if not points else np.sqrt(nearest) * np.sqrt(depth)
        score = np.where(mask, score, 0)
        if not score.max() > 0:
            break
        y, x = np.unravel_index(score.argmax(), score.shape)
        points.append([int(x), int(y)])
        nearest = np.minimum(nearest, (xx - x) ** 2 + (yy - y) ** 2)
    return points


def overlay(photo, mask):
    rgb = np.asarray(photo).copy()
    rgb[mask] = (rgb[mask] * .55 + np.array([255, 30, 0]) * .45).astype(np.uint8)
    return Image.fromarray(rgb)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photo", type=Path)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "diagnostics/editor_sam2_20261019.json")
    parser.add_argument("--output", type=Path, default=ROOT / "diagnostics/out/editor_sam2_selection_20261019")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    photo = Image.open(args.photo).convert("RGB")
    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    assert analysis["fingerprint"] == image_fingerprint(photo)
    assert tuple(analysis["size"]) == photo.size
    model_path = ROOT / cfg["model_path"]
    assert (model_path / "model.safetensors").is_file(), "This probe never downloads models"
    target = args.output / str(time.time_ns())
    target.mkdir(parents=True, exist_ok=False)
    photo.save(target / "source.png")
    torch.set_num_threads(cfg["threads"])
    started = time.monotonic()
    processor = Sam2Processor.from_pretrained(model_path, local_files_only=True)
    model, info = Sam2Model.from_pretrained(model_path, local_files_only=True, output_loading_info=True)
    info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
    assert not any(info.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")), info
    # The installed checkpoint also carries video memory parameters, unused here.
    model.eval()
    base = processor(images=photo, return_tensors="pt")
    with torch.inference_mode():
        embeddings = model.get_image_embeddings(base["pixel_values"])
    report = {"source": str(args.photo), "analysis": str(args.analysis), "settings": cfg,
              "fingerprint": analysis["fingerprint"], "size": photo.size,
              "model_sha256": hashlib.sha256((model_path / "model.safetensors").read_bytes()).hexdigest(),
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "loading_info": info, "embedding_seconds": time.monotonic() - started,
              "scope": "Measured proposals only; no full-object success inferred", "candidates": []}

    def save_report():
        report["elapsed_seconds"] = time.monotonic() - started
        (target / "results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def decode(points, boxes=None):
        kwargs = {"original_sizes": base["original_sizes"], "input_points": [points],
                  "input_labels": [[[1] * len(item) for item in points]], "return_tensors": "pt"}
        if boxes is not None:
            kwargs["input_boxes"] = [boxes]
        inputs = processor(**kwargs)
        with torch.inference_mode():
            result = model(image_embeddings=embeddings, **{key: value for key, value in inputs.items()
                           if key != "original_sizes"}, multimask_output=True)
            logits = processor.post_process_masks(result.pred_masks, inputs["original_sizes"], binarize=False)[0]
        masks = logits.numpy() > 0
        upper = (logits > cfg["stability_offset"]).sum((-1, -2))
        lower = (logits > -cfg["stability_offset"]).sum((-1, -2)).clamp_min(1)
        return masks, result.iou_scores[0].numpy(), (upper / lower).numpy()

    save_report()

    def save(mask, score, stability, method, points, region=None, box=None):
        name = f"candidate_{len(report['candidates']):04d}"
        Image.fromarray(mask.astype(np.uint8) * 255).save(target / f"{name}.png")
        record = {"mask": f"{name}.png", "method": method, "points": points, "box": box,
                  "model_iou": float(score), "stability": float(stability),
                  "coverage": float(mask.mean()), "region_id": None if region is None else region["id"],
                  "label": None if region is None else region["label"]}
        if region is not None:
            seed = np.asarray(Image.open(args.analysis.parent / f"{region['id']}.png").convert("L")) > 0
            record["seed_recall"] = float(mask[seed].mean())
        report["candidates"].append(record)
        return record

    for region in analysis["regions"]:
        if region["kind"] != "object":
            continue
        seed = np.asarray(Image.open(args.analysis.parent / f"{region['id']}.png").convert("L")) > 0
        assert seed.shape == (photo.height, photo.width) and seed.any()
        points = spread_points(seed, cfg["distributed_points"])
        ys, xs = np.where(seed)
        box = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
        px = int(np.ceil((box[2] - box[0]) * cfg["context_fraction"]))
        py = int(np.ceil((box[3] - box[1]) * cfg["context_fraction"]))
        context = [max(0, box[0] - px), max(0, box[1] - py),
                   min(photo.width, box[2] + px), min(photo.height, box[3] + py)]
        plans = [("single", points[:1], None), ("distributed", points, None),
                 ("seed_box", points[:1], box), ("context_box", points[:1], context)]
        panels = [photo, overlay(photo, seed)]
        for method, prompt, bounds in plans:
            masks, scores, stabilities = decode([prompt], None if bounds is None else [bounds])
            for mask, score, stability in zip(masks[0], scores[0], stabilities[0]):
                save(mask, score, stability, method, prompt, region, bounds)
                panels.append(overlay(photo, mask))
        # All alternatives, not a visually cherry-picked mask.
        thumb = (min(photo.width, 400), round(photo.height * min(photo.width, 400) / photo.width))
        sheet = Image.new("RGB", (thumb[0] * 4, (thumb[1] + 24) * 4), "white")
        draw = ImageDraw.Draw(sheet)
        records = report["candidates"][-12:]
        names = ["source", "CLIPSeg seed"] + [f"{r['mask']} {r['method']} IoU{r['model_iou']:.2f}" for r in records]
        for index, (panel, title) in enumerate(zip(panels, names)):
            x, y = index % 4 * thumb[0], index // 4 * (thumb[1] + 24)
            sheet.paste(panel.resize(thumb), (x, y))
            draw.text((x, y + thumb[1]), title, fill="black")
        sheet.save(target / f"{region['id']}_alternatives.png")
        save_report()
        print(f"{region['id']} {region['label']}: 12 actual candidates", flush=True)

    # Uniform image proposals can reveal elements absent from the text seed.
    side = cfg["grid_side"]
    grid = [[[float(x), float(y)]] for y in (np.arange(side) + .5) * photo.height / side
            for x in (np.arange(side) + .5) * photo.width / side]
    kept = []
    for begin in range(0, len(grid), cfg["grid_batch"]):
        prompts = grid[begin:begin + cfg["grid_batch"]]
        masks, scores, stabilities = decode(prompts)
        for prompt, group, ious, stability_scores in zip(prompts, masks, scores, stabilities):
            for mask, score, stability in zip(group, ious, stability_scores):
                coverage = float(mask.mean())
                if (score < cfg["grid_min_iou"] or stability < cfg["grid_min_stability"] or
                        not cfg["min_coverage"] <= coverage <= cfg["max_coverage"]):
                    continue
                if any((mask & prior).sum() / max(1, (mask | prior).sum()) >= cfg["duplicate_iou"] for prior in kept):
                    continue
                kept.append(mask)
                save(mask, score, stability, "grid", prompt)
        save_report()
        print(f"Grid {min(begin + cfg['grid_batch'], len(grid))}/{len(grid)}; unique masks {len(kept)}", flush=True)
    report["completed"] = True
    save_report()
    print(f"EVIDENCE {target}", flush=True)


if __name__ == "__main__":
    main()
