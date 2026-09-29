"""Recover the original full frame from a saved crop and unchanged exterior."""
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    ("profile", 1790280808, 1790280681, 213, 387),
    ("first", 1790275633, 1790275495, 333, 363),
]

for name, output_stamp, crop_stamp, w, h in CASES:
    final = np.asarray(Image.open(ROOT / "img_editor" / f"edit_{output_stamp}.png").convert("RGB"))
    crop = np.asarray(Image.open(ROOT / "ui/tob/ComfyUI/input" / f"fast_{crop_stamp}.png").convert("RGB"))
    small = cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(cv2.cvtColor(small, cv2.COLOR_RGB2GRAY), None)
    kp2, des2 = sift.detectAndCompute(cv2.cvtColor(final, cv2.COLOR_RGB2GRAY), None)
    matches = cv2.BFMatcher().knnMatch(des1, des2, k=2)
    good = [m for m, n in matches if m.distance < .72 * n.distance]
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
    affine, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3)
    if affine is None:
        raise RuntimeError(f"No crop alignment for {name}")
    x, y = [int(round(v)) for v in affine[:, 2]]
    if abs(x) <= 2:
        x = max(0, x)
    if abs(y) <= 2:
        y = max(0, y)
    print(f"{name}: sift_matches={len(good)} inliers={int(inliers.sum())} affine={affine.tolist()}")
    if not (0 <= x <= final.shape[1]-w and 0 <= y <= final.shape[0]-h):
        raise RuntimeError(f"Crop box out of frame for {name}")
    recovered = final.copy()
    recovered[y:y+h, x:x+w] = small
    print(f"{name}: box={(x, y, x+w, y+h)}")
    target = ("profile_reconstructed_source.png" if name == "profile"
              else f"{name}_reconstructed_source.png")
    Image.fromarray(recovered).save(ROOT / "diagnostics" / target)
