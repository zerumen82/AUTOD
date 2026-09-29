"""Exercise real open-mouth frames through the video FaceSwap path."""
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")
VIDEO = Path(r"D:\PROJECTS\uee\tlk\RDT_20260916_115926.mp4")


def main():
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    source_faces = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)
    assert source_faces
    source = source_faces[0][0]
    analyser = FaceAnalysis(
        name="buffalo_l", root=str(ROOT / "models"),
        allowed_modules=["detection", "landmark_2d_106"],
        providers=["CPUExecutionProvider"],
    )
    analyser.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    manager = ProcessMgr()
    manager.initialize(
        [FaceSet([source])], [],
        SimpleNamespace(blend_ratio=1.0, current_video_path=str(VIDEO)),
        face_swap_mode="selected_faces",
    )
    manager._current_video_key = f"video_{VIDEO.name}"
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    active, changes = 0, []
    try:
        for number in range(936, 948):
            capture.set(cv2.CAP_PROP_POS_FRAMES, number)
            ok, frame = capture.read()
            assert ok
            faces = [
                face for face in analyser.get(frame)
                if min(face.bbox[2]-face.bbox[0], face.bbox[3]-face.bbox[1]) > 140
            ]
            assert faces, f"No main face in frame {number}"
            target = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
            mouth = np.asarray(target.landmark_2d_106)[52:72]
            ratio = float(np.ptp(mouth[:, 1]) / max(1, target.bbox[2]-target.bbox[0]))
            active += ratio > 0.22
            result = manager._process_face_swap_v21(
                source, target, frame.copy(), frame, enable_temporal_smoothing=True,
            )
            assert result is not None and result.shape == frame.shape
            x1, y1, x2, y2 = np.clip(target.bbox.astype(int), [0, 0, 0, 0], [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            change = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
            changes.append(change)
            print(f"[PROBE] frame={number} mouth_ratio={ratio:.2f} face_change={change:.2f}", flush=True)
        assert active >= 1, "No open mouth found"
        assert min(changes) > 1.0, f"One or more swaps invisible: {changes}"
        print(f"[PROBE] completed=12 open_mouth_frames={active} mean_face_change={np.mean(changes):.2f}", flush=True)
    finally:
        capture.release()


if __name__ == "__main__":
    main()
