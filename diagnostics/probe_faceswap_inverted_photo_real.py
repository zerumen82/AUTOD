"""Exercise the rotated photo face through the real swap and mask pipeline."""
from pathlib import Path
from types import SimpleNamespace
import os

import cv2
import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.face_util_rotation import get_all_faces_smart
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
PHOTO = ROOT / '.autodeep_temp' / 'f3c8fe4c764dc3db23f738f5d5c0dc39626c662517f259f1dbd7157a1102bc46' / '449508945.jpg'
SOURCE = Path(r'D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG')
OUTPUT = ROOT / 'diagnostics' / 'faceswap_inverted_photo_probe.png'


def main():
    frame = cv2.imread(str(PHOTO))
    assert frame is not None
    faces = get_all_faces_smart(frame, for_target=True, include_180=True)
    target = max((f for f in faces if f.bbox[1] > 300 and f.det_score > 0.5), key=lambda f: f.det_score)
    source_faces = extract_face_images(str(SOURCE), is_source_face=True)
    assert source_faces
    source = source_faces[0][0]
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = 'GFPGAN'
    roop.globals.use_enhancer = True
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], SimpleNamespace(blend_ratio=1.0, current_video_path=None), face_swap_mode='selected_faces')
    manager._is_single_photo = True
    result = manager._process_face_swap_v21(source, target, frame.copy(), frame)
    x1, y1, x2, y2 = np.clip(np.asarray(target.bbox, dtype=int), [0, 0, 0, 0], [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
    change = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
    cv2.imwrite(str(OUTPUT), result)
    print(f'[PROBE] inverted_face={target.bbox} face_change={change:.2f} output={OUTPUT}', flush=True)
    assert change > 1.0

    old_refs = getattr(roop.globals, 'selected_face_references', None)
    old_targets = getattr(roop.globals, 'TARGET_FACES', None)
    old_random = getattr(roop.globals, 'source_random', False)
    try:
        roop.globals.selected_face_references = {
            f'selected_face_ref_{PHOTO.name}': [{
                'face_obj': target, 'embedding': target.embedding,
                'bbox': target.bbox,
                'source_path': os.path.normcase(os.path.abspath(str(PHOTO))),
            }],
        }
        roop.globals.TARGET_FACES = [target]
        roop.globals.source_random = False
        routed = manager.process_frame(frame.copy(), file_path=str(PHOTO))
        routed_change = float(cv2.absdiff(routed, frame)[y1:y2, x1:x2].mean())
        print(f'[PROBE] full_photo_route_face_change={routed_change:.2f}', flush=True)
        assert routed_change > 1.0
    finally:
        roop.globals.selected_face_references = old_refs
        roop.globals.TARGET_FACES = old_targets
        roop.globals.source_random = old_random


if __name__ == '__main__':
    main()
