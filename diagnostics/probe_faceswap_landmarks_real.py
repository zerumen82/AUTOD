"""Compare measured detector/68-point alignment on the difficult real profile.

The 68-to-5 conversion follows FaceFusion's face_helper.convert_to_face_landmark_5.
Installed InsightFace models only; production landmarks remain unchanged.
"""
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import roop.face_util
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
TARGET = Path(r"D:\PROJECTS\uee\AUTO-DEEP\input\posti\1994859933.jpg")
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\IMG-20240111-WA0000.jpg")


def main():
    frame = cv2.imread(str(TARGET))
    assert frame is not None
    analyser = FaceAnalysis(name="buffalo_l", root=str(ROOT / "models"),
        allowed_modules=["detection", "recognition", "landmark_3d_68", "landmark_2d_106"],
        providers=["CPUExecutionProvider"])
    analyser.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=0.1)
    roop.face_util.FACE_ANALYSER = analyser
    reference = extract_face_images(str(TARGET), (False, 0))[0][0]
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]
    candidates = [("reference", copy.deepcopy(reference))]
    for colour, photo in [("bgr", frame), ("rgb", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))]:
        measured = analyser.get(photo)
        assert measured
        detected = min(measured, key=lambda face: np.linalg.norm(np.asarray(face.bbox) - np.asarray(reference.bbox)))
        candidate = copy.deepcopy(reference)
        candidate.kps = detected.kps
        candidates.append((f"detector_{colour}", candidate))
        points = np.asarray(detected.landmark_3d_68, np.float32)[:, :2]
        candidate = copy.deepcopy(reference)
        candidate.kps = np.array([points[36:42].mean(axis=0), points[42:48].mean(axis=0),
            points[30], points[48], points[54]], np.float32)
        candidates.append((f"landmark68_{colour}", candidate))
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    roop.globals.enhancer_blend_factor = 0.70
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [reference], SimpleNamespace(blend_ratio=1.0,
        current_video_path=None), face_swap_mode="selected_faces")
    manager._is_single_photo = True
    output = ROOT / "diagnostics" / "out" / f"faceswap_landmarks_{time.time_ns()}"
    output.mkdir(parents=True)
    boxes = np.asarray(reference.bbox, int)
    margin = int(max(boxes[2]-boxes[0], boxes[3]-boxes[1]) * .08)
    x1, y1 = np.maximum(boxes[:2] - margin, 0)
    x2, y2 = np.minimum(boxes[2:] + margin, [frame.shape[1], frame.shape[0]])
    sheet = []
    metadata = []
    for label, target in [("original", None)] + candidates:
        ProcessMgr._swap_call_count = 0
        result = frame if target is None else manager._process_face_swap_v21(source,
            target, frame.copy(), frame, photo_path=str(TARGET))
        assert result is not None
        assert cv2.imwrite(str(output / f"{label}.png"), result)
        crop = result[y1:y2, x1:x2]
        crop = cv2.resize(crop, (400, int(round(crop.shape[0]*400/crop.shape[1]))))
        cv2.putText(crop, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, .6, (0,255,255), 2)
        sheet.append(crop)
        if target is not None:
            metadata.append({"label":label, "kps":np.asarray(target.kps).tolist()})
    cv2.imwrite(str(output / "comparison.png"), np.concatenate(sheet, axis=1))
    (output / "meta.json").write_text(json.dumps(metadata, indent=2))
    print("[PROBE] output:", output, flush=True)


if __name__ == "__main__":
    main()
