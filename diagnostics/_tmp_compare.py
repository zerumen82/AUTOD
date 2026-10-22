"""Compare raw FLUX Kontext output against the reference crop it received."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image

from roop.img_editor import element_selection as es
from roop.img_editor.reference_canvas import prepare_reference_canvas

SRC = "img_editor/edit_1792479695.png"
FINAL = "img_editor/edit_1792479788.png"
RAW = "ui/tob/ComfyUI/output/FastEdit_00732_.png"
BOX = (289, 92, 585, 349)

src = Image.open(SRC).convert("RGB")
final = Image.open(FINAL).convert("RGB")

# Only the face detector is needed: the VLM/LLM would only add minutes.
settings = es.load_settings()
work = src.copy()
work.thumbnail((settings["max_analysis_side"], settings["max_analysis_side"]))
faces = es._detect_faces(work, settings)
scale = np.array([src.width / work.width, src.height / work.height])
harr = {}
for index, (box, kps) in enumerate(faces):
    full_box = np.asarray(box) * np.tile(scale, 2)
    full_kps = np.asarray(kps) * scale if kps is not None else None
    mask = es._head_mask_for_box(src, full_box, full_kps)
    if mask is None:
        continue
    harr[f"head-{index}"] = np.asarray(mask.convert("L")) >= 128
print("heads detected:", {k: int(v.sum()) for k, v in harr.items()})

# Use the real mapping: content is centred inside the aligned canvas.
reference = src.crop(BOX)
encoded, mapping = prepare_reference_canvas(reference, max_side=768, multiple=64)
print("reference size:", reference.size, "content:", mapping.content_size,
      "canvas:", mapping.canvas_size, "content_box:", mapping.content_box)
canvas = Image.open(RAW).convert("RGB")
print("raw canvas size:", canvas.size)
assert canvas.size == mapping.canvas_size, "raw canvas does not match the mapping"
restored = mapping.restore(canvas)

ref_arr = np.asarray(reference).astype(np.int16)
raw_arr = np.asarray(restored).astype(np.int16)
diff = np.abs(ref_arr - raw_arr).max(axis=2) > 12
ys, xs = np.where(diff)
print("raw FLUX changed px:", int(diff.sum()), "of", diff.size)
if len(xs):
    print("diff bbox in crop coords:", (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    print("diff bbox in image coords:", (BOX[0] + int(xs.min()), BOX[1] + int(ys.min()),
                                         BOX[0] + int(xs.max()) + 1, BOX[1] + int(ys.max()) + 1))

# Which head does the raw FLUX change fall on? (image coordinates)
diff_full = np.zeros((src.height, src.width), dtype=bool)
diff_full[BOX[1]:BOX[3], BOX[0]:BOX[2]] = diff
for rid, arr in harr.items():
    inside = int((diff_full & arr).sum())
    print(f"  raw FLUX change inside {rid}: {inside} (head px {int(arr.sum())})")

# What does the FINAL composed image change?
fin_arr = np.asarray(final).astype(np.int16)
src_arr = np.asarray(src).astype(np.int16)
changed = np.abs(src_arr - fin_arr).max(axis=2) > 10
print("final changed px:", int(changed.sum()))
ys, xs = np.where(changed)
if len(xs):
    print("final changed bbox:", (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
for rid, arr in harr.items():
    inside = int((changed & arr).sum())
    print(f"  final change inside {rid}: {inside} (head px {int(arr.sum())})")

# Hair proxy: darkness/variance of the upper part of each head.
def hair_stats(arr):
    ys_, xs_ = np.where(arr)
    if len(xs_) == 0:
        return None
    top = ys_.min() + max(1, int((ys_.max() - ys_.min()) * 0.45))
    band = arr.copy()
    band[top:, :] = False
    sub = src_arr[band]
    return round(float(sub.mean()), 1), round(float(sub.std()), 1)

for rid, arr in harr.items():
    print(f"  hair band {rid}: src mean/std = {hair_stats(arr)}")

def hair_stats_img(img, arr):
    ys_, xs_ = np.where(arr)
    if len(xs_) == 0:
        return None
    top = ys_.min() + max(1, int((ys_.max() - ys_.min()) * 0.45))
    band = arr.copy()
    band[top:, :] = False
    sub = np.asarray(img).astype(np.int16)[band]
    return round(float(sub.mean()), 1), round(float(sub.std()), 1)

raw_full = src.copy()
raw_full.paste(restored, BOX)
print("--- hair band after raw FLUX (image coords) ---")
for rid, arr in harr.items():
    print(f"  {rid}: src {hair_stats(arr)} | rawFLUX {hair_stats_img(raw_full, arr)}")

# Save side by sides
Image.open(RAW).save("diagnostics/_tmp_raw_canvas.png")
restored.save("diagnostics/_tmp_raw_restored.png")
reference.save("diagnostics/_tmp_reference_crop.png")
vis = np.zeros((src.height, src.width, 3), dtype=np.uint8)
vis[..., 0] = (diff_full * 255).astype(np.uint8)
Image.fromarray(vis).save("diagnostics/_tmp_raw_diff_mask.png")
print("saved diagnostics/_tmp_raw_{canvas,restored}.png, _tmp_reference_crop.png, _tmp_raw_diff_mask.png")
