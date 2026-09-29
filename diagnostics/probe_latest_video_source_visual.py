"""Check that a sharp photo of another identity cannot replace the video source."""
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
VIDEO = ROOT / '.autodeep_temp/3adf818f7cc4034f93928efeb84b56f9d73b609754d20f9ef6f64d40ab72f701/1754378114229.mp4'
OLD_SOURCE = ROOT / '.autodeep_temp/c027db4ce7925562a238178b140a1aec4bd268e55e92d123e9fbefb8fefe9173/00689.jpg'
OTHER_SOURCE = ROOT / '.autodeep_temp/689d28d07991d122fd5c3593dc36aa5f73455d1cd3b7c3afa8a1a753527d85d0/mar.JPG'


def main():
    roop.globals.enhancer_blend_factor = 0.70
    roop.globals.selected_enhancer = 'GFPGAN'
    roop.globals.use_enhancer = True
    old = extract_face_images(str(OLD_SOURCE), (False, 0), is_source_face=True)[0][0]
    other = extract_face_images(str(OTHER_SOURCE), (False, 0), is_source_face=True)[0][0]
    analyser = FaceAnalysis(name='buffalo_l', root=str(ROOT / 'models'),
                            allowed_modules=['detection', 'recognition', 'landmark_2d_106'],
                            providers=['CPUExecutionProvider'])
    analyser.prepare(ctx_id=-1, det_size=(640, 640), det_thresh=0.1)
    cap = cv2.VideoCapture(str(VIDEO))
    ok, frame = cap.read()
    cap.release()
    assert ok
    faces = analyser.get(frame)
    target = min(faces, key=lambda f: np.linalg.norm(np.asarray(f.bbox) - [442, 204, 528, 330]))
    manager = ProcessMgr()
    manager.initialize([FaceSet([old]), FaceSet([other])], [],
                       SimpleNamespace(blend_ratio=1.0, current_video_path=str(VIDEO)),
                       face_swap_mode='selected_faces')
    chosen = manager._select_source_face_by_similarity(target, [old, other], is_profile=True)
    same_identity = manager._cosine_similarity(old.embedding, other.embedding)
    print('[PROBE] old_quality', manager._score_source_face_quality(old),
          'other_quality', manager._score_source_face_quality(other),
          'source_cosine', same_identity,
          'selected', 'old' if chosen is old else 'other', flush=True)
    assert same_identity < 0.60 and chosen is old
    for label, source in [('old', old), ('other_identity', other)]:
        result = manager._process_face_swap_v21(source, target, frame.copy(), frame,
                                                enable_temporal_smoothing=False)
        x1, y1, x2, y2 = np.asarray(target.bbox, dtype=int)
        change = float(cv2.absdiff(result, frame)[y1:y2, x1:x2].mean())
        out = ROOT / 'diagnostics' / f'latest_video_source_{label}_f1.png'
        cv2.imwrite(str(out), result)
        print('[PROBE]', label, 'face_change', round(change, 2), 'output', out, flush=True)


if __name__ == '__main__':
    main()
