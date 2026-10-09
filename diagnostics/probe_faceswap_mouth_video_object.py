"""Inspect the object touching the mouth in one real profile video frame."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import roop.globals
import roop.ProcessMgr as process_module
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")
VIDEO = Path(r"D:\PROJECTS\uee\tlk\RDT_20260916_115926.mp4")
PREFIX = ROOT / "diagnostics" / "mouth_video_object_936"


def main():
    source = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)[0][0]
    analyser = FaceAnalysis(
        name="buffalo_l", root=str(ROOT / "models"),
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CPUExecutionProvider"],
    )
    analyser.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    capture = cv2.VideoCapture(str(VIDEO))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 936)
    ok, frame = capture.read()
    capture.release()
    assert ok
    target = max(
        (f for f in analyser.get(frame)
         if min(f.bbox[2] - f.bbox[0], f.bbox[3] - f.bbox[1]) > 140),
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    manager = ProcessMgr()
    manager.initialize(
        [FaceSet([source])], [],
        SimpleNamespace(blend_ratio=1.0, current_video_path=str(VIDEO)),
        face_swap_mode="selected_faces",
    )
    manager._current_video_key = f"video_{VIDEO.name}"
    original_occ = process_module.detect_foreground_occlusion
    original_mouth = manager._mouth_mask_in_aligned_face

    def record_occ(swapped, reference, **kwargs):
        mask = original_occ(swapped, reference, **kwargs)
        cv2.imwrite(str(PREFIX) + "_target.png", reference)
        cv2.imwrite(str(PREFIX) + "_occ.png", np.uint8(np.clip(mask, 0, 1) * 255))
        return mask

    def record_mouth(*args, **kwargs):
        mask = original_mouth(*args, **kwargs)
        cv2.imwrite(str(PREFIX) + "_mouth.png", np.uint8(np.clip(mask, 0, 1) * 255))
        from roop.processors.FaceSwap import detect_mouth_open
        _, mouth_106, _ = detect_mouth_open(
            target, target.landmark_2d_106, None, face_bbox=target.bbox,
        )
        if mouth_106 is not None:
            aligned_106 = original_mouth(
                args[0], mouth_106, args[2], args[3], args[4], (0, 0), None,
            )
            cv2.imwrite(str(PREFIX) + "_mouth_106.png", np.uint8(np.clip(aligned_106, 0, 1) * 255))
        return mask

    process_module.detect_foreground_occlusion = record_occ
    manager._mouth_mask_in_aligned_face = record_mouth
    try:
        result = manager._process_face_swap_v21(source, target, frame.copy(), frame)
    finally:
        process_module.detect_foreground_occlusion = original_occ
    assert result is not None
    cv2.imwrite(str(PREFIX) + "_result.png", result)
    print(f"[PROBE] saved {PREFIX}_*.png", flush=True)


if __name__ == "__main__":
    main()
