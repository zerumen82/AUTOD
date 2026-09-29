"""Short integration probe for the FaceSwap enhancer preview."""
from pathlib import Path

import numpy as np

import roop.globals
from roop.face_util import extract_face_images
from roop.types import FaceSet
from ui.tabs.faceswap.logic import get_enhancer_preview


ROOT = Path(__file__).resolve().parents[1]
source_path = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")
target_path = ROOT / "testdata" / "test_person.jpg"

source_faces = extract_face_images(str(source_path), (False, 0), is_source_face=True)
assert source_faces, "No source face"
roop.globals.INPUT_FACESETS = [FaceSet([source_faces[0][0]])]
preview, status = get_enhancer_preview(str(target_path), 0, 0, "GFPGAN", 0.70)
assert status == "Preview generado", status
assert isinstance(preview, np.ndarray) and preview.size > 0
print(f"[PROBE] preview={status} shape={preview.shape}", flush=True)
