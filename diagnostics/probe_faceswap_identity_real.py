"""CPU identity measurements of real renders, with fixed and detected geometry.

Cosine similarity is diagnostic evidence, not a visual quality verdict.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from insightface.app.common import Face

import roop.face_util
from roop.face_util import extract_face_images


def norm(vector):
    vector = np.asarray(vector, np.float32)
    return vector / max(float(np.linalg.norm(vector)), np.finfo(np.float32).eps)


def overlap(left, right):
    left, right = np.asarray(left), np.asarray(right)
    extent = np.maximum(0, np.minimum(left[2:], right[2:]) - np.maximum(left[:2], right[:2]))
    shared = float(extent.prod())
    areas = [float(np.maximum(0, box[2:] - box[:2]).prod()) for box in (left, right)]
    return shared / max(areas[0] + areas[1] - shared, np.finfo(np.float32).eps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("target")
    parser.add_argument("renders", nargs="+")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    analyser = FaceAnalysis(name="buffalo_l", root=str(root / "models"),
                            allowed_modules=["detection", "recognition"],
                            providers=["CPUExecutionProvider"])
    analyser.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=0.1)
    roop.face_util.FACE_ANALYSER = analyser
    source_faces = extract_face_images(args.source, (False, 0), is_source_face=True)
    target_faces = extract_face_images(args.target, (False, 0))
    assert source_faces and target_faces, "Source or target face was not measured"
    source = source_faces[0][0]
    reference = target_faces[0][0]
    source_embedding = norm(source.embedding)
    recognizer = analyser.models["recognition"]
    measurements = []
    for path in [args.target, *args.renders]:
        photo = cv2.imread(path)
        assert photo is not None, path
        fixed = Face(bbox=np.asarray(reference.bbox, np.float32).copy(),
                     kps=np.asarray(reference.kps, np.float32).copy())
        recognizer.get(photo, fixed)
        detections = analyser.get(photo)
        matched = max(detections, key=lambda face: overlap(face.bbox, reference.bbox)) if detections else None
        values = {"path": path, "fixed_geometry_cosine": float(source_embedding @ norm(fixed.embedding)),
                  "detected_geometry_cosine": float(source_embedding @ norm(matched.embedding)) if matched is not None else None,
                  "detected_bbox_iou": overlap(matched.bbox, reference.bbox) if matched is not None else None}
        measurements.append(values)
        print(values, flush=True)
    output = root / "diagnostics/out" / f"faceswap_identity_{time.time_ns()}.json"
    output.write_text(json.dumps({"source": args.source, "target": args.target,
                                 "source_bbox": np.asarray(source.bbox).tolist(),
                                 "target_bbox": np.asarray(reference.bbox).tolist(),
                                 "measurements": measurements}, indent=2))
    print(output)


if __name__ == "__main__":
    main()
