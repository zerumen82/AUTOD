from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager


ROOT = Path(__file__).resolve().parents[1]
inputs = [
    ROOT / "ui/tob/ComfyUI/input" / f"fast_{timestamp}.png"
    for timestamp in (1790204211, 1790204520, 1790204739)
]
masker = CLIPSegMasker()
manager = ImgEditorManager.__new__(ImgEditorManager)
analyser = get_face_analyser()
for path in inputs:
    image = Image.open(path).convert("RGB")
    faces = detect_faces_robust(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), analyser, 0.20) or []
    print(path.name, image.size, "faces", [tuple(round(float(x)) for x in f.bbox) for f in faces], flush=True)
    mask = masker.generate_mask(image, "man")
    head = manager._head_mask_from_region_plan(image, faces, mask) if mask is not None else None
    focal = manager._crop_focal_context(image, head) if head is not None else None
    print("mask", mask is not None, "head", head is not None, "crop", focal[2] if focal else None, flush=True)
masker.unload()
