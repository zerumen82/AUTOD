from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from roop.face_util import detect_faces_robust, get_face_analyser
from roop.img_editor.clipseg_masker import CLIPSegMasker
from roop.img_editor.img_editor_manager import ImgEditorManager

root = Path(__file__).resolve().parents[1]
img = Image.open(root / "ui/tob/ComfyUI/input/fast_1790204739.png").convert("RGB")
faces = detect_faces_robust(cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR), get_face_analyser(), 0.20)
masker = CLIPSegMasker()
man = masker.generate_mask(img, "man")
hair = masker.generate_mask(img, "hair")
head = ImgEditorManager._head_mask_from_region_plan(ImgEditorManager.__new__(ImgEditorManager), img, faces, man)
for name, mask in (("man", man), ("hair", hair), ("head", head)):
    if mask is not None:
        mask.save(root / "diagnostics" / f"bald_mask_{name}.png")
arr_head = np.asarray(head.convert("L"))
arr_hair = np.asarray(hair.convert("L"))
arr_man = np.asarray(man.convert("L"))
near = cv2.dilate((arr_head > 80).astype(np.uint8), np.ones((91, 91), np.uint8)) > 0
union = np.maximum(arr_head, np.where((arr_hair > 80) & (arr_man > 80) & near, 255, 0).astype(np.uint8))
Image.fromarray(union).save(root / "diagnostics/bald_mask_union.png")
print("hair", float((arr_hair > 80).mean()), "added", float(((union > 80) & (arr_head <= 80)).mean()), flush=True)
masker.unload()
