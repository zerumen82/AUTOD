"""Short integration probe for the FaceSwap enhancer preview."""
from pathlib import Path
from contextlib import redirect_stdout
import io
import sys
import time

import cv2
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
enhancer = sys.argv[1] if len(sys.argv) > 1 else "GFPGAN"
roop.globals.selected_enhancer = "GFPGAN" if enhancer != "GFPGAN" else "CodeFormer"
roop.globals.use_enhancer = True
log = io.StringIO()
with redirect_stdout(log):
    preview, status = get_enhancer_preview(str(target_path), 0, 0, enhancer, 0.70)
print(log.getvalue(), end="", flush=True)
assert status == "Preview generado", status
assert isinstance(preview, np.ndarray) and preview.size > 0
assert f"{enhancer} Enhancer initialized" in log.getvalue(), "Preview initialized the previous enhancer"
output = ROOT / "diagnostics" / "out" / f"faceswap_preview_{enhancer}_{time.time_ns()}.png"
output.parent.mkdir(parents=True, exist_ok=True)
assert cv2.imwrite(str(output), cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
print(f"[PROBE] preview={status} shape={preview.shape}", flush=True)
print(f"[PROBE] selected={enhancer} output={output}", flush=True)
