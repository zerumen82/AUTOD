"""H-FS-004 CPU geometry contracts; no model inference or visual quality claim."""
import ast
import json
from pathlib import Path

import cv2
import numpy as np
from insightface.utils import face_align

from roop.faceswap_config import get_faceswap_config
from roop.processors.FaceSwap import FaceWarpEngine
from roop.ProcessMgr import ProcessMgr

ROOT = Path(__file__).resolve().parents[1]
results = []


def check(name, condition, detail=""):
    results.append(bool(condition))
    print(("OK   " if condition else "FAIL ") + name + (f" | {detail}" if detail else ""))


def inverse_side(matrix, output_size):
    return output_size / np.sqrt(np.linalg.det(matrix[:, :2]))

def rejects(name, **overrides):
    parameters = dict(image_size=(1000, 1000), kps=healthy, bbox=healthy_bbox)
    parameters.update(overrides)
    try:
        FaceWarpEngine.estimate_constrained_norm(**parameters)
    except (ValueError, TypeError, KeyError):
        check(name, True)
    else:
        check(name, False, "invalid input was accepted")

config = get_faceswap_config()
template = face_align.arcface_dst
bbox_bad = np.array([184, 184, 816, 816], dtype=np.float64)
# Exact reviewer counterexample: all points are a similarity, eye_rel=0.02787794.
collapsed_points = (template - template.mean(axis=0)) * 0.5 + 500
healthy = (template - template.mean(axis=0)) * 3 + 500
healthy_bbox = np.array([350, 350, 650, 650], dtype=np.float64)

for output_size in (112, 128, 224, 256, 384):
    official = cv2.transform(template[:, None], face_align.estimate_norm(template, output_size))[:, 0]
    check(f"InsightFace template convention {output_size}",
          np.allclose(FaceWarpEngine.arcface_template(output_size), official, atol=2e-5))
    matrix, method = FaceWarpEngine.estimate_constrained_norm(
        (1000, 1000), healthy, output_size, bbox=healthy_bbox,
    )
    check(f"healthy M unchanged {output_size}",
          method == "kps_ok" and np.array_equal(matrix, face_align.estimate_norm(healthy, output_size)))

for output_size in (128, 256):
    matrix, method = FaceWarpEngine.estimate_constrained_norm(
        (1000, 1000), collapsed_points, output_size, bbox=bbox_bad,
    )
    check(f"reviewer collapsed similarity uses bbox {output_size}", method == "bbox_anchored", method)
    expected_side = np.sqrt(np.prod(bbox_bad[2:] - bbox_bad[:2])) * config["align_constrain_crop_to_bbox_ratio"]
    lower_side = expected_side * (1 - config["align_constrain_scale_dev"])
    check(f"projected crop side limited by bbox {output_size}",
          np.isclose(inverse_side(matrix, output_size), lower_side), f"side={inverse_side(matrix, output_size):.4f}")
    mapped_center = cv2.transform(np.array([[[500.0, 500.0]]]), matrix)[0, 0]
    check(f"bbox center retains official template convention {output_size}",
          np.allclose(mapped_center, FaceWarpEngine.arcface_template(output_size).mean(axis=0), atol=2e-5))
    wider_matrix, _ = FaceWarpEngine.estimate_constrained_norm(
        (1000, 1000), collapsed_points, output_size, bbox=bbox_bad, max_scale_dev=0.1,
    )
    check(f"scale parameter changes constrained M {output_size}",
          np.isclose(inverse_side(wider_matrix, output_size), expected_side * 0.9))

    oversized = (template - template.mean(axis=0)) * 20 + 500
    rotation = np.array([[0.8, -0.6], [0.6, 0.8]], dtype=np.float32)
    oversized = (oversized - 500) @ rotation.T + 500
    oversized_original = face_align.estimate_norm(oversized, output_size)
    oversized_matrix, oversized_method = FaceWarpEngine.estimate_constrained_norm(
        (2000, 2000), oversized, output_size, bbox=bbox_bad,
    )
    check(f"oversized rotated warp constrained to bbox {output_size}",
          oversized_method == "bbox_anchored" and np.isclose(
              inverse_side(oversized_matrix, output_size), expected_side * (1 + config["align_constrain_scale_dev"])))
    check(f"constrained warp retains measured roll {output_size}",
          np.allclose(oversized_matrix[:, :2] / np.linalg.norm(oversized_matrix[0, :2]),
                      oversized_original[:, :2] / np.linalg.norm(oversized_original[0, :2])))

# Read logged frame2/frame3 points without importing a probe that runs tests.
tree = ast.parse((ROOT / "diagnostics/probe_faceswap_align_and_mouth_gate.py").read_text(encoding="utf-8"))
samples = [(node.targets[0].id, ast.literal_eval(node.value)) for node in tree.body
           if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
           and node.targets[0].id in {"frame2", "frame3"}]
samples.append(("edge profile", dict(kps=[[100, 100], [110, 101], [104, 130], [100, 150], [108, 151]],
                                    bbox=[80, 80, 200, 200])))
reference_file = ROOT / "diagnostics/out/faceswap_landmarks_1791196415282182700/meta.json"
for sample in json.loads(reference_file.read_text(encoding="utf-8")):
    if sample["label"] in {"reference", "detector_bgr", "detector_rgb"}:
        samples.append((sample["label"], dict(kps=sample["kps"], bbox=[655, 366, 1287, 842])))
for label, sample in samples:
    points = np.asarray(sample["kps"], dtype=np.float32)
    for output_size in (128, 256):
        matrix, method = FaceWarpEngine.estimate_constrained_norm(
            (1000, 1600), points, output_size, bbox=sample["bbox"],
        )
        dimensions = np.asarray(sample["bbox"])[2:] - np.asarray(sample["bbox"])[:2]
        ratio = inverse_side(matrix, output_size) / np.sqrt(np.prod(dimensions))
        check(f"logged geometry preserved {label} {output_size}",
              method == "kps_ok" and np.array_equal(matrix, face_align.estimate_norm(points, output_size)),
              f"crop/bbox_area_sqrt={ratio:.6f}")

offset_bbox = healthy_bbox + [40, 40, 40, 40]
translation_matrix, translation_method = FaceWarpEngine.estimate_constrained_norm(
    (1000, 1000), healthy, 128, bbox=offset_bbox, max_translate_px=0,
)
check("explicit zero translation limit is honored", translation_method == "bbox_anchored")
check("translation fallback preserves scale and roll",
      np.allclose(translation_matrix[:, :2], face_align.estimate_norm(healthy, 128)[:, :2]))

# A large declared frame must not cause a large dummy allocation.
large_matrix, _ = FaceWarpEngine.estimate_constrained_norm((10**8, 10**8), healthy, bbox=healthy_bbox)
check("estimation independent of full frame allocation", np.isfinite(large_matrix).all())
rejects("missing bbox rejected", bbox=None)
rejects("reversed bbox rejected", bbox=[500, 500, 400, 400])
rejects("nonfinite bbox rejected", bbox=[0, 0, np.nan, 100])
rejects("nonfinite kps rejected", kps=np.full((5, 2), np.inf))
rejects("singular kps rejected", kps=np.zeros((5, 2)))
rejects("incorrect kps shape rejected", kps=np.zeros((10,)))
rejects("invalid image dimensions rejected", image_size=(0, 1000))
rejects("nonfinite tolerance rejected", max_scale_dev=np.nan)
rejects("invalid tolerance range rejected", max_scale_dev=1)
rejects("invalid expected crop ratio rejected", crop_to_bbox_ratio=0)
rejects("negative translation limit rejected", max_translate_px=-1)
rejects("unsupported output convention rejected", image_size_out=123)

# Existing measured-shape contract; this does not exercise every mask fallback.
collapsed_mask = np.zeros((200, 200), dtype=np.float32)
collapsed_mask[80:130, 70:140] = 0.12
shaped = ProcessMgr._build_xseg_shaped_safety_mask(
    (200, 200), collapsed_mask, (60, 60, 150, 150), tight=False, floor=0.55,
)
check("measured shape fallback exists", shaped is not None)
if shaped is not None:
    check("measured shape outside corner remains empty", shaped[5, 5] < 0.05)
    check("measured shape preserves nucleus", shaped[100, 100] > 0.10)
    check("measured shape retains asymmetry", np.abs(shaped - shaped[:, ::-1]).mean() > 1e-4)

print(f"\nResult: {sum(results)} OK, {len(results) - sum(results)} FAIL")
raise SystemExit(0 if all(results) else 1)
