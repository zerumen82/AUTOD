"""Render the old and quality-first source picks on a real last-batch photo."""
import copy
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import roop.globals
from roop.ProcessMgr import ProcessMgr
from roop.face_util import extract_face_images
from roop.face_util_rotation import get_all_faces_smart
from roop.types import FaceSet


ROOT = Path(__file__).resolve().parents[1]
TEMP = ROOT / '.autodeep_temp'
PHOTO = TEMP / 'e80f75b083cb24fe430d742ea9bdcd2db5e44f00cddf1463456b945f771177b5' / '713373234.jpg'
OLD = TEMP / '4ddaf50181cd993baf88d525bd5b09e0b20a5b77a52b2d452947802787313d53' / '20220609_171308_1.jpg'
NEW = TEMP / 'd8d0eef342a032208f4609b340ed17908288aea69c28a97ba9ace85dc0aa3c57' / '00039.jpg'


def main():
    frame = cv2.imread(str(PHOTO))
    assert frame is not None
    targets = get_all_faces_smart(frame, for_target=True, include_180=True)
    target = max(
        (f for f in targets if (f.bbox[0] + f.bbox[2]) * 0.5 > frame.shape[1] * 0.55),
        key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
    )
    old = extract_face_images(str(OLD), is_source_face=True)[0][0]
    new = extract_face_images(str(NEW), is_source_face=True)[0][0]
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = 'GFPGAN'
    roop.globals.use_enhancer = True
    x1, y1, x2, y2 = np.clip(np.asarray(target.bbox, dtype=int), [0, 0, 0, 0],
                              [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
    for label, source in [('old', old), ('new', new)]:
        manager = ProcessMgr()
        manager.initialize([FaceSet([source])], [target],
                           SimpleNamespace(blend_ratio=1.0, current_video_path=None),
                           face_swap_mode='selected_faces')
        manager._is_single_photo = True
        result = manager._process_face_swap_v21(source, copy.deepcopy(target), frame.copy(), frame)
        output = ROOT / 'diagnostics' / f'faceswap_last_auto_{label}.jpg'
        cv2.imwrite(str(output), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
        change = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
        print(f'[PROBE] {label} face_change={change:.2f} output={output}', flush=True)


if __name__ == '__main__':
    main()
