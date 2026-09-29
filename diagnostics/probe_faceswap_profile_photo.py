"""Run one real profile photo through the FaceSwap composite, no UI/server."""
from pathlib import Path
from types import SimpleNamespace
import os

import cv2
import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.types import FaceSet
from roop.face_util import extract_face_images


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")
TARGET = ROOT / "testdata/test_person.jpg"
OUTPUT = ROOT / "diagnostics/faceswap_profile_photo_probe.png"


def main():
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = "GFPGAN"
    roop.globals.use_enhancer = True
    source_faces = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)
    target_faces = extract_face_images(str(TARGET), (False, 0))
    print(f"[PROBE] faces source={len(source_faces)} target={len(target_faces)}", flush=True)
    assert source_faces and target_faces
    source, target = source_faces[0][0], target_faces[0][0]
    frame = cv2.imread(str(TARGET))
    assert frame is not None
    manager = ProcessMgr()
    manager.initialize([FaceSet([source])], [target], None, face_swap_mode="selected_faces")
    manager._is_single_photo = True
    result = manager._process_face_swap_v21(source, target, frame.copy(), frame)
    assert result is not None and result.shape == frame.shape
    print(f"[PROBE] profile={manager._is_profile_face(target.kps, target.det_score)}", flush=True)
    x1, y1, x2, y2 = [int(v) for v in target.bbox]
    local_change = cv2.absdiff(result, frame)[y1:y2, x1:x2].mean()
    print(
        f"[PROBE] mean_abs_change={cv2.absdiff(result, frame).mean():.2f} "
        f"face_change={local_change:.2f}", flush=True,
    )
    assert local_change > 3.0, "swap did not change the face"
    assert cv2.imwrite(str(OUTPUT), result)
    print(f"[PROBE] output={OUTPUT}", flush=True)

    # Exercise the full still-photo route, including automatic source and
    # destination selection, rather than only the low-level swap operation.
    old_refs = getattr(roop.globals, 'selected_face_references', None)
    old_targets = getattr(roop.globals, 'TARGET_FACES', None)
    old_random = getattr(roop.globals, 'source_random', False)
    try:
        roop.globals.selected_face_references = {
            f'selected_face_ref_{TARGET.name}': [{
                'face_obj': target, 'embedding': target.embedding,
                'bbox': target.bbox,
                'source_path': os.path.normcase(os.path.abspath(str(TARGET))),
            }],
        }
        roop.globals.TARGET_FACES = [target]
        roop.globals.source_random = False
        manager.options = SimpleNamespace(
            current_video_path=None, face_swap_mode='selected_faces', blend_ratio=1.0,
        )
        routed = manager.process_frame(frame.copy(), file_path=str(TARGET))
        routed_change = cv2.absdiff(routed, frame)[y1:y2, x1:x2].mean()
        assert routed_change > 3.0, f'photo route unchanged: {routed_change:.2f}'
        print(f'[PROBE] complete photo route face_change={routed_change:.2f}', flush=True)
    finally:
        roop.globals.selected_face_references = old_refs
        roop.globals.TARGET_FACES = old_targets
        roop.globals.source_random = old_random


if __name__ == "__main__":
    main()
