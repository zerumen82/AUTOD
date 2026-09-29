"""Check the selected-face tracker on consecutive real profile frames."""
from pathlib import Path
from types import SimpleNamespace
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import roop.globals
from roop.ProcessMgr import ProcessMgr


ROOT = Path(__file__).resolve().parents[1]
VIDEO = Path(r"D:\PROJECTS\uee\tlk\RDT_20260917_234436.mp4")


def main():
    analyser = FaceAnalysis(
        name="buffalo_l", root=str(ROOT / "models"),
        allowed_modules=["detection", "recognition", "landmark_2d_106"],
        providers=["CPUExecutionProvider"],
    )
    analyser.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    capture = cv2.VideoCapture(str(VIDEO))
    assert capture.isOpened()
    manager = ProcessMgr()
    manager.options = SimpleNamespace(current_video_path=str(VIDEO))
    manager._last_swap_bbox = {}
    assert manager._bbox_iou(np.array([0, 0, 10, 10]), np.array([0, 0, 10, 10])) == 1.0
    old_refs = getattr(roop.globals, 'selected_face_references', None)
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, 90)
        ok, frame = capture.read()
        assert ok
        faces = analyser.get(frame)
        target = max(
            (face for face in faces if (face.bbox[0] + face.bbox[2]) * .5 > 700
             and (face.bbox[1] + face.bbox[3]) * .5 < 320),
            key=lambda face: face.det_score,
        )
        name = VIDEO.name
        roop.globals.selected_face_references = {
            f'selected_face_ref_{name}': [{
                'face_obj': target, 'embedding': target.embedding,
                'bbox': target.bbox,
                'source_path': os.path.normcase(os.path.abspath(str(VIDEO))),
            }],
        }
        tracked = []
        for number in range(90, 102):
            if number != 90:
                ok, frame = capture.read()
                assert ok
                faces = analyser.get(frame)
            if not hasattr(manager, f'_reference_face_selected_{name}'):
                manager._setup_selected_faces_frame_for_video(str(VIDEO), faces)
            selected = manager._find_target_face_for_selected_mode(str(VIDEO), faces, frame)
            is_target = selected is not None and manager._bbox_iou(selected.bbox, target.bbox) > .35
            tracked.append(bool(is_target))
            print(f'[TRACK_PROBE] frame={number} faces={len(faces)} target={bool(is_target)}', flush=True)
        assert sum(tracked) >= 10, f'Profile target tracked only {sum(tracked)}/12 frames'
        # Same embedding at a distant position must not jump the lock in one
        # frame, even when identity matching alone would look excellent.
        distant = SimpleNamespace(**{
            key: getattr(selected, key, None)
            for key in ('bbox', 'kps', 'embedding', 'normed_embedding',
                        'det_score', 'landmark_2d_106', 'gender', 'age')
        })
        distant.bbox = np.asarray(selected.bbox, dtype=np.float32) + [-250, 0, -250, 0]
        distant.kps = np.asarray(selected.kps, dtype=np.float32) + [-250, 0]
        rejected = manager._find_target_face_for_selected_mode(str(VIDEO), [distant], frame)
        assert rejected is None, 'tracker accepted a 250 px one-frame face jump'
        print(f'[TRACK_PROBE] selected profile tracked {sum(tracked)}/12 frames', flush=True)
    finally:
        capture.release()
        roop.globals.selected_face_references = old_refs


if __name__ == '__main__':
    main()
