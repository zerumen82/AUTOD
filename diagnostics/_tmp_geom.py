"""Temporary: head geometry + candidate tight crops for the two-men photo."""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())

from roop.img_editor import element_selection as es

SRC = "img_editor/edit_1792479695.png"

src = Image.open(SRC).convert("RGB")
settings = es.load_settings()
work = src.copy()
work.thumbnail((settings["max_analysis_side"], settings["max_analysis_side"]))
faces = es._detect_faces(work, settings)
scale = np.array([src.width / work.width, src.height / work.height])

masks = {}
for index, (box, kps) in enumerate(faces):
    full_box = np.asarray(box) * np.tile(scale, 2)
    mask = es._head_mask_for_box(src, full_box)
    if mask is None:
        continue
    arr = np.asarray(mask.convert("L")) >= 128
    ys, xs = np.where(arr)
    name = f"head-{index}"
    masks[name] = (arr, (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    print(f"{name}: bbox={[int(v) for v in full_box]} mask_bbox={masks[name][1]} px={int(arr.sum())}")

edit = settings["edit"]
context = float(edit["head_crop_context_fraction"])
print("head_crop_context_fraction =", context)

for name, (arr, (x1, y1, x2, y2)) in masks.items():
    pad_x = int(np.ceil((x2 - x1) * context))
    pad_y = int(np.ceil((y2 - y1) * context))
    cx1, cx2 = max(0, x1 - pad_x), min(src.width, x2 + pad_x)
    cy1, cy2 = max(0, y1 - pad_y), min(src.height, y2 + pad_y)
    others = []
    for other, (oarr, _) in masks.items():
        if other == name:
            continue
        overlap = int((oarr & arr).sum())
        inside = int(oarr[cy1:cy2, cx1:cx2].sum())
        others.append((other, overlap, inside))
    print(f"{name}: tight crop=({cx1},{cy1},{cx2},{cy2}) size={cx2-cx1}x{cy2-cy1} others_in_crop={others}")

# Visual: masks over the crop of head-2, plus the overlap area.
vis = src.convert("RGB").copy()
colors = {"head-0": (255, 0, 0), "head-1": (0, 255, 0), "head-2": (0, 128, 255)}
from PIL import ImageDraw
draw = ImageDraw.Draw(vis)
for name, (arr, bbox) in masks.items():
    overlay = Image.new("RGB", src.size, colors[name])
    vis = Image.composite(overlay, vis, Image.fromarray(arr.astype(np.uint8) * 255, "L"))
    draw.rectangle(bbox, outline=colors[name], width=3)
    draw.text((bbox[0] + 4, bbox[1] + 4), name, fill=(255, 255, 255))

cx1, cy1, cx2, cy2 = 331, 115, 554, 320
crop = vis.crop((cx1, cy1, cx2, cy2)).resize(((cx2 - cx1) * 2, (cy2 - cy1) * 2), Image.Resampling.NEAREST)
crop.save("diagnostics/_tmp_heads_visual.png")

h0 = masks["head-0"][0]
h2 = masks["head-2"][0]
overlap = h0 & h2
print("overlap head-0/head-2 px:", int(overlap.sum()))
ys, xs = np.where(overlap)
if len(xs):
    print("overlap bbox:", (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
print("head-2 px outside head-0:", int((h2 & ~h0).sum()))
