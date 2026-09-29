from pathlib import Path

import cv2
import numpy as np
from PIL import Image


root = Path(__file__).resolve().parents[1]
source = np.asarray(Image.open(root / "ui/tob/ComfyUI/input/fast_1790204739.png").convert("RGB"))
hair = np.asarray(Image.open(root / "diagnostics/bald_mask_hair.png").convert("L")) > 80
man = np.asarray(Image.open(root / "diagnostics/bald_mask_man.png").convert("L")) > 80
region = hair & man
skin = np.median(cv2.cvtColor(source[660:710, 700:790], cv2.COLOR_RGB2LAB).reshape(-1, 3), axis=0)
box = (459, 334, 998, 875)
for number in range(598, 605):
    path = root / "ui/tob/ComfyUI/output" / f"FastEdit_{number:05d}_.png"
    if not path.exists():
        continue
    crop = Image.open(path).convert("RGB").resize((box[2] - box[0], box[3] - box[1]), Image.Resampling.LANCZOS)
    full = source.copy()
    full[box[1]:box[3], box[0]:box[2]] = np.asarray(crop)
    lab = cv2.cvtColor(full, cv2.COLOR_RGB2LAB)
    scalp = np.median(lab[region], axis=0)
    detail = cv2.Laplacian(cv2.cvtColor(full, cv2.COLOR_RGB2GRAY), cv2.CV_64F)[region].var()
    print(path.name, "skin_dist", round(float(np.linalg.norm(scalp - skin)), 1),
          "laplacian_var", round(float(detail), 1), flush=True)
