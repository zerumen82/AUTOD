"""Check whether subject-bound compositing removes the visible head halo."""
import cv2
import numpy as np
from PIL import Image, ImageFilter


source = Image.open("img_editor/edit_1790254710.png").convert("RGB")
raw = Image.open("ui/tob/ComfyUI/output/FastEdit_00614_.png").convert("RGB")
raw = raw.resize(source.size, Image.Resampling.LANCZOS)
hair = np.asarray(Image.open("diagnostics/last_bald_selected_hair.png").convert("L")) >= 128
subject = np.asarray(Image.open("diagnostics/last_bald_man_mask.png").convert("L")) >= 80
subject = cv2.dilate(subject.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))) > 0
editable = (hair & subject).astype(np.uint8) * 255
mask = Image.fromarray(editable, "L").filter(ImageFilter.GaussianBlur(6))
result = Image.composite(raw, source, mask)
result.save("diagnostics/last_bald_subject_bound_composite.png")
print("editable", int((editable > 0).sum()), "hair_mask", int(hair.sum()), flush=True)
