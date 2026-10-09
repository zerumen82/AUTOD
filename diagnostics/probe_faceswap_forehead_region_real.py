"""CPU semantic diagnosis of a raw swap artifact, using installed CLIPSeg.

Queries and an optional measured inspection rectangle are explicit CLI inputs.
This probe never changes routing, production masks, configuration, or images
used by the application. Semantic probabilities are diagnostic, not truth.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw

from roop.img_editor.clipseg_masker import CLIPSegMasker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("aligned_input")
    parser.add_argument("raw_output")
    parser.add_argument("--query", action="append", required=True)
    parser.add_argument("--inspect", type=int, nargs=4)
    parser.add_argument("--candidate-metadata")
    parser.add_argument("--candidate-full-render")
    parser.add_argument("--candidate-blur", type=int, default=5)
    parser.add_argument("--candidate-kind", choices=("semantic-hair", "landmark-area"), default="semantic-hair")
    parser.add_argument("--candidate-compose", choices=("gaussian", "poisson"), default="gaussian")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    backbone = Path.home() / ".cache/clip/ViT-B-16.pt"
    assert backbone.is_file(), "Installed CLIP backbone required; no download"
    digest = hashlib.sha256()
    with backbone.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f", "Invalid installed backbone; downloading is forbidden"
    assert (root / "models/CLIP/rd64-uni-refined.pth").is_file()
    masker = CLIPSegMasker()
    assert masker.device == "cpu"
    ok, message = masker.initialize()
    assert ok and "(con pesos)" in message, message
    output = root / "diagnostics/out" / f"faceswap_forehead_region_{time.time_ns()}"
    output.mkdir()
    records = []
    maps = {}
    images = {}
    try:
        for label, path in (("input", args.aligned_input), ("raw", args.raw_output)):
            photo = Image.open(path).convert("RGB")
            array = np.asarray(photo)
            images[label] = array
            inspection = np.zeros(array.shape[:2], bool)
            if args.inspect:
                x0, y0, x1, y1 = args.inspect
                assert 0 <= x0 < x1 <= photo.width and 0 <= y0 < y1 <= photo.height
                inspection[y0:y1, x0:x1] = True
            panels = []
            for index, query in enumerate(args.query):
                heat = masker.probability(photo, query)
                assert heat is not None and np.isfinite(heat).all(), query
                maps[label, query] = heat
                np.save(output / f"{label}_{index}_probability.npy", heat)
                quantized = (heat.clip(0, 1) * 255).astype(np.uint8)
                color = cv2.applyColorMap(quantized, cv2.COLORMAP_TURBO)[:, :, ::-1]
                overlay = (array.astype(float) * .55 + color.astype(float) * .45).astype(np.uint8)
                if args.inspect:
                    cv2.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (255, 255, 255), 1)
                panel = Image.new("RGB", (photo.width, photo.height + 30), "white")
                panel.paste(Image.fromarray(overlay), (0, 30))
                ImageDraw.Draw(panel).text((5, 5), f"{label}: {query}", fill="black")
                panels.append(np.asarray(panel))
                item = {"label": label, "query": query, "max_probability": float(heat.max()),
                        "coverage_gt_04": float((heat > .4).mean())}
                if inspection.any():
                    values = heat[inspection]
                    item.update(inspection_mean=float(values.mean()), inspection_max=float(values.max()),
                                inspection_coverage_gt_04=float((values > .4).mean()))
                records.append(item)
                print(item, flush=True)
            Image.fromarray(np.concatenate(panels, axis=1)).save(output / f"{label}_semantic_compare.png")
        candidate = None
        if args.candidate_metadata or args.candidate_full_render:
            assert args.candidate_metadata and args.candidate_full_render
            assert args.candidate_blur > 0 and args.candidate_blur % 2 == 1
            # This is an explicit experimental hypothesis, never production:
            # new measured hair on previously measured forehead, with both
            # semantic queries agreeing on the input/output ownership change.
            metadata = json.loads(Path(args.candidate_metadata).read_text(encoding="utf-8"))
            original = cv2.imread(metadata["target"])
            rendered = cv2.imread(args.candidate_full_render)
            assert original is not None and rendered is not None and original.shape == rendered.shape
            from insightface.utils import face_align
            measured, matrix = face_align.norm_crop2(original, np.asarray(metadata["kps"], np.float32), 256)
            assert np.array_equal(measured[:, :, ::-1], images["input"]), "Geometry does not match saved raw input"
            selected = ((maps["raw", "hair"] > .4) & (maps["input", "hair"] <= .4)
                        & (maps["input", "forehead"] > maps["input", "hair"])
                        & (maps["raw", "hair"] > maps["raw", "forehead"]))
            landmarks = None
            if args.candidate_kind == "landmark-area":
                from insightface.app import FaceAnalysis
                for name in ("det_10g.onnx", "1k3d68.onnx"):
                    assert (root / "models/models/buffalo_l" / name).is_file(), "Installed landmark models required"
                analyser = FaceAnalysis(name="buffalo_l", root=str(root / "models"),
                    allowed_modules=["detection", "landmark_3d_68"], providers=["CPUExecutionProvider"])
                analyser.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=.1)
                detections = analyser.get(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
                assert detections, "No real facial geometry measured"
                center = np.asarray(metadata["kps"], np.float32).mean(axis=0)
                face = min(detections, key=lambda item: np.linalg.norm((item.bbox[:2] + item.bbox[2:]) / 2 - center))
                points = np.asarray(face.landmark_3d_68, np.float32)[:, :2]
                aligned_points = cv2.transform(points[None], matrix)[0]
                support = np.zeros(measured.shape[:2], np.uint8)
                cv2.fillConvexPoly(support, cv2.convexHull(aligned_points.astype(np.int32)), 255)
                Image.fromarray(support).save(output / "measured_landmark_support.png")
                selected = ((support == 0) & (maps["input", "forehead"] > .4)
                            & (maps["input", "forehead"] > maps["input", "hair"]))
                landmarks = points.tolist()
            assert selected.any(), "No measured preservation region"
            Image.fromarray(selected.astype(np.uint8) * 255).save(output / "candidate_mask.png")
            alpha = cv2.GaussianBlur(selected.astype(np.float32),
                                     (args.candidate_blur, args.candidate_blur), 0)
            raw_corrected = (images["input"] * alpha[:, :, None]
                             + images["raw"] * (1 - alpha[:, :, None])).clip(0, 255).astype(np.uint8)
            Image.fromarray(raw_corrected).save(output / "candidate_raw.png")
            restored = cv2.warpAffine(alpha, cv2.invertAffineTransform(matrix),
                                     (original.shape[1], original.shape[0]), flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            corrected = (original.astype(np.float32) * restored[:, :, None]
                         + rendered.astype(np.float32) * (1 - restored[:, :, None])).clip(0, 255).astype(np.uint8)
            if args.candidate_compose == "poisson":
                mask = (restored > .5).astype(np.uint8) * 255
                x, y, width, height = cv2.boundingRect(mask)
                assert width > 0 and height > 0
                corrected = cv2.seamlessClone(original, rendered, mask,
                                              (x + width // 2, y + height // 2), cv2.NORMAL_CLONE)
            assert cv2.imwrite(str(output / "candidate_full.png"), corrected)
            candidate = {"metadata": args.candidate_metadata, "render": args.candidate_full_render,
                         "kind": args.candidate_kind, "composition": args.candidate_compose, "landmarks": landmarks,
                         "selected_aligned_pixels": int(selected.sum()), "blur": args.candidate_blur,
                         "changed_full_pixels_gt_10": int((cv2.absdiff(corrected, rendered).max(axis=2) > 10).sum()),
                         "limit": "Original forehead restored using measured support; diagnostic composition, not a model correction"}
        (output / "meta.json").write_text(json.dumps({"input": args.aligned_input,
            "raw": args.raw_output, "device": masker.device, "inspection": args.inspect,
            "queries": args.query, "weights_message": message, "records": records,
            "candidate": candidate}, indent=2), encoding="utf-8")
    finally:
        masker.unload()
    print(output, flush=True)


if __name__ == "__main__":
    main()
