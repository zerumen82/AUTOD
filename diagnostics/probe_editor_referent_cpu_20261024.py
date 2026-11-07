"""Measure local CPU CLIPSeg/face association on an existing image; no render.

Explicit local backbone/weights and CPUExecutionProvider; no downloads or GPU
queries. Source/masks/face scores are retained to separate segmentation from
association errors. CLI queries are passed unchanged, never rewritten.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor import element_selection as es  # noqa: E402
from roop.img_editor.clipseg_masker import CLIPSegMasker  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--det-threshold", type=float)
    parser.add_argument("--subject-mode", action="store_true")
    parser.add_argument("--expect-face-index", type=int)
    args = parser.parse_args()
    settings = es.load_settings()
    backbone = Path(settings["segmenter"]["backbone_path"]).expanduser()
    weights = ROOT / settings["segmenter"]["weights_path"]
    detector_file = ROOT / settings["detector"]["model_path"]
    assert all(path.is_file() for path in (backbone, weights, detector_file)), "local models required"
    torch.set_num_threads(int(settings["inventory"]["threads"]))
    image = Image.open(args.source).convert("RGB")
    out = ROOT / "diagnostics/out/editor_referent_cpu_20261024" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.source, out / f"source{args.source.suffix}")
    started = time.monotonic()
    from insightface.model_zoo import get_model

    detector = get_model(str(detector_file), providers=["CPUExecutionProvider"])
    det_size = int(settings["detector"]["input_size"])
    threshold = settings["detector"]["threshold"] if args.det_threshold is None else args.det_threshold
    detector.prepare(ctx_id=-1, input_size=(det_size, det_size), det_thresh=threshold)
    boxes, landmarks = detector.detect(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
    assert boxes is not None and len(boxes), "no measured faces"
    faces = [SimpleNamespace(bbox=row[:4], det_score=float(row[4]),
                             kps=landmarks[index] if landmarks is not None else None)
             for index, row in enumerate(boxes)]
    manager = ImgEditorManager()
    eligible = manager._credible_faces(faces)
    rows = []
    with (patch.object(torch.cuda, "is_available", return_value=False),
          patch.object(torch.cuda, "init", side_effect=AssertionError("GPU forbidden"))):
        masker = CLIPSegMasker(backbone_path=str(backbone))
        assert masker.device == "cpu"
        for index, query in enumerate(args.query):
            conditioned_query = query
            if args.subject_mode:
                original_probability = masker.probability
                prediction = {}

                def observe(photo, text):
                    prediction["query"] = text
                    prediction["heat"] = original_probability(photo, text)
                    return prediction["heat"]

                masker.probability = observe
                try:
                    mask = masker.generate_subject_mask(
                        image, query, settings["segmenter"]["subject_prompt_template"])
                finally:
                    masker.probability = original_probability
                heat = prediction.get("heat")
                conditioned_query = prediction.get("query")
                assert mask is not None, query
            else:
                heat = masker.probability(image, query)
            assert heat is not None, query
            binary = (heat > float(settings["segmenter"]["probability_threshold"])).astype(np.uint8) * 255
            if not args.subject_mode:
                mask = Image.fromarray(cv2.GaussianBlur(binary, (5, 5), 0))
            mask.save(out / f"mask_{index}.png")
            head = manager._head_mask_from_region_plan(
                image, faces, mask, continue_region=False, eligible_faces=eligible)
            if head is not None:
                head.save(out / f"chosen_head_{index}.png")
            overlay = image.copy()
            overlay.thumbnail((1024, 1024))
            draw = ImageDraw.Draw(overlay)
            scale = overlay.width / image.width
            scored = []
            region = np.asarray(mask)
            for ident, face in enumerate(faces):
                x1, y1, x2, y2 = [int(round(float(value))) for value in face.bbox]
                fw, fh = max(1, x2 - x1), max(1, y2 - y1)
                ax1, ay1 = max(0, x1 - int(fw * 0.25)), max(0, y1 - int(fh * 0.65))
                ax2, ay2 = min(image.width, x2 + int(fw * 0.25)), min(image.height, y2 + int(fh * 0.15))
                face_region = region[ay1:ay2, ax1:ax2]
                face_heat = heat[max(0,y1):min(image.height,y2), max(0,x1):min(image.width,x2)]
                score = float((face_region > 128).mean()) if face_region.size else 0.0
                scored.append({"id": ident, "bbox": list(map(float, face.bbox)),
                               "det_score": face.det_score, "expanded_coverage": score,
                               "face_probability": float(face_heat.mean()) if face_heat.size else None})
                box = tuple(round(value * scale) for value in (x1, y1, x2, y2))
                draw.rectangle(box, outline="red", width=3)
                draw.text((box[0], box[1]), f"{ident}: {score:.3f}", fill="red")
            overlay.save(out / f"faces_{index}.png")
            Image.fromarray((heat * 255).astype(np.uint8)).resize(overlay.size).save(out / f"heat_{index}.png")
            chosen = []
            if head is not None:
                head_pixels = np.asarray(head)
                for ident, face in enumerate(faces):
                    cx, cy = [int(round(value)) for value in
                              ((face.bbox[0] + face.bbox[2]) / 2, (face.bbox[1] + face.bbox[3]) / 2)]
                    if 0 <= cx < image.width and 0 <= cy < image.height and head_pixels[cy, cx] >= 128:
                        chosen.append(ident)
            rows.append({"query": query, "conditioned_query": conditioned_query,
                         "coverage": float((binary > 0).mean()), "chosen_face_indices": chosen,
                         "faces": scored, "has_chosen_head": head is not None})
            print(json.dumps(rows[-1], indent=2), flush=True)
            if args.expect_face_index is not None:
                assert chosen == [args.expect_face_index], chosen
        masker.unload()
    metrics = {"scope": "Actual CPU detection+segmentation only; no model image generation",
               "source": str(args.source.resolve()), "size": image.size,
               "det_threshold": threshold, "subject_mode": args.subject_mode,
               "elapsed_s": time.monotonic() - started, "queries": rows}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"EVIDENCE {out}", flush=True)


if __name__ == "__main__":
    main()
