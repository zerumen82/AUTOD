"""Real frontal swap: measure changed pixels outside the face paste window."""
from pathlib import Path

import cv2
import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")
TARGET = ROOT / "testdata" / "test_face.webp"
OUTPUT = ROOT / "diagnostics" / "faceswap_frontal_halo_probe.png"


def main():
    frame = cv2.imread(str(TARGET))
    source_faces = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)
    target_faces = extract_face_images(str(TARGET), (False, 0))
    assert frame is not None and source_faces and target_faces
    source, target = source_faces[0][0], target_faces[0][0]
    manager = ProcessMgr()
    assert not manager._is_profile_face(target.kps, target.det_score)
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    manager.initialize([FaceSet([source])], [target], None, face_swap_mode="selected_faces")
    manager._is_single_photo = True
    result = manager._process_face_swap_v21(source, target, frame.copy(), frame)
    assert result is not None and result.shape == frame.shape
    window = manager._profile_head_window(frame.shape, target.bbox, is_profile=False)
    changed = cv2.absdiff(result, frame).max(axis=2)
    outside = window <= 0.001
    outside_changed = int(np.count_nonzero(changed[outside]))
    assert outside_changed == 0, f"{outside_changed} changed pixels outside face window"
    x1, y1, x2, y2 = np.asarray(target.bbox, dtype=int)
    face_change = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
    assert face_change > 3.0, face_change
    assert cv2.imwrite(str(OUTPUT), result)
    print(f"frontal face_change={face_change:.2f} outside_changed={outside_changed} output={OUTPUT}")


if __name__ == "__main__":
    main()
