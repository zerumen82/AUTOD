"""Measure which detected head still has hair in the last failed input."""
import cv2
import numpy as np
from PIL import Image

from roop.face_util import get_face_analyser_cpu, detect_faces_robust
from roop.img_editor.img_editor_manager import ImgEditorManager


image = Image.open("img_editor/edit_1790254710.png").convert("RGB")
bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
faces = list(detect_faces_robust(bgr, get_face_analyser_cpu(), 0.20) or [])
manager = ImgEditorManager()
masker = manager._get_clipseg_masker()
man = masker.generate_mask(image, "man")
hair = masker.generate_mask(image, "hair")
for label, mask in (("man", man), ("hair", hair)):
    raw = np.asarray(mask.convert("L")) >= 96
    count, labels, stats, centers = cv2.connectedComponentsWithStats(raw.astype(np.uint8), 8)
    print(label, sorted([(int(s[4]), [int(v) for v in s[:4]]) for s in stats[1:] if s[4] >= 15], reverse=True)[:10], flush=True)
    mask.save(f"diagnostics/last_bald_{label}_mask.png")
subject = np.asarray(man.convert("L")) >= 80
hraw = np.asarray(hair.convert("L")) >= 96
count, labels, stats, centers = cv2.connectedComponentsWithStats(hraw.astype(np.uint8), 8)
for idx in range(1, count):
    area = int(stats[idx, cv2.CC_STAT_AREA])
    if area < 15:
        continue
    comp = labels == idx
    print("hair_component", idx, area, "subject_overlap", float((comp & subject).sum())/area, flush=True)
alternate = manager._unique_subject_hair_mask(image, man, hair)
assert alternate is not None
assert manager._unique_subject_hair_mask(image, Image.new("L", image.size, 255), hair) is None
alternate.save("diagnostics/last_bald_selected_hair.png")
editable = np.asarray(alternate.convert("L")) >= 128
print("editable_outside_subject", int((editable & ~subject).sum()), int(editable.sum()), flush=True)
assert np.asarray(alternate.convert("L"))[170, 85] >= 160
assert np.asarray(alternate.convert("L"))[108, 145] == 0
assert np.asarray(alternate.convert("L"))[80, 310] == 0
print("alternate_bbox", alternate.getbbox(), flush=True)
for i, face in enumerate(faces):
    head = manager._head_mask_from_faces(image, [face])
    print(f"face={i} bbox={np.asarray(face.bbox).round(1).tolist()} gender={getattr(face, 'gender', None)} "
          f"age={getattr(face, 'age', None)} hair={manager._bald_hair_coverage(image, image, head, masker)}", flush=True)
    x1, y1, x2, y2 = [int(round(float(v))) for v in face.bbox]
    fw, fh = max(1, x2-x1), max(1, y2-y1)
    px, py = int(fw*.25), int(fh*.65)
    patch = np.asarray(man.convert("L"))[max(0,y1-py):min(image.height,y2+int(fh*.15)),
                                       max(0,x1-px):min(image.width,x2+px)]
    print(f"  man_score={float((patch>128).mean()) if patch.size else 0:.3f}", flush=True)
masker.unload()
