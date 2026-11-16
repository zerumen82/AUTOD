import sys
from pathlib import Path
from PIL import Image
import numpy as np
ROOT = Path(r"D:\PROJECTS\AUTOAUTO")
sys.path.insert(0, str(ROOT))
from roop.img_editor.top_tier_finish import _get_face_landmarks, _camera_change_score
before = Image.open(ROOT/"diagnostics/out/editor_camera_neutral_20261110/before.png").convert("RGB")
after = Image.open(ROOT/"diagnostics/out/editor_categories_20261114/camera.png").convert("RGB")
print("sizes", before.size, after.size)
if after.size != before.size:
    after = after.resize(before.size, Image.Resampling.LANCZOS)
lb = _get_face_landmarks(before); la = _get_face_landmarks(after)
print("landmarks:", None if lb is None else lb.shape, None if la is None else la.shape)
if lb is not None and la is not None:
    d = float(np.linalg.norm(lb-la, axis=1).mean())
    print("mean diff", d, "old score", float(np.clip(d/0.1,0,1)))
print("old _camera_change_score:", _camera_change_score(before, after))
