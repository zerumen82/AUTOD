import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from PIL import Image
from roop.img_editor import element_selection as es

img = Image.open("img_editor/edit_1792479695.png").convert("RGB")
an = es.analyze_elements(img)
print("SIZE", an.size)
for r in an.regions:
    print("REG", r.id, "|", r.label, "| kind", r.kind, "| bbox", r.bbox, "| cov", round(r.coverage, 4))

print("--- crop recompute (selected = head of blond man) ---")

import itertools
import math

settings = es.load_settings()["edit"]
context = float(settings["crop_context_fraction"])
minimum = int(settings["minimum_reference_side"])
maximum = int(settings["maximum_reference_side"])

heads = [r for r in an.regions if r.kind == "head"]
objs = [r for r in an.regions if r.kind == "object"]


def union_bbox(ids):
    masks = [r.mask for r in an.regions if r.id in set(ids)]
    arr = np.zeros((an.size[1], an.size[0]), dtype=bool)
    for m in masks:
        arr |= np.asarray(m.convert("L")) >= 128
    ys, xs = np.where(arr)
    return (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1), arr


def compute_crop(ids):
    (x1, y1, x2, y2), _ = union_bbox(ids)
    ctx = context
    if any(r.kind == "object" and r.id in ids for r in an.regions):
        ctx = max(ctx, float(settings["object_context_fraction"]))
    pad_x, pad_y = math.ceil((x2 - x1) * ctx), math.ceil((y2 - y1) * ctx)
    x1, x2 = max(0, x1 - pad_x), min(img.width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(img.height, y2 + pad_y)
    for axis in (0, 1):
        lo, hi, limit = (x1, x2, img.width) if axis == 0 else (y1, y2, img.height)
        extra = max(0, minimum - (hi - lo))
        lo = max(0, lo - (extra + 1) // 2)
        hi = min(limit, max(hi, lo + minimum))
        lo = max(0, min(lo, hi - minimum))
        if axis == 0:
            x1, x2 = lo, hi
        else:
            y1, y2 = lo, hi
    tw = max(minimum, math.ceil((y2 - y1) * minimum / maximum))
    th = max(minimum, math.ceil((x2 - x1) * minimum / maximum))
    for axis, target in ((0, tw), (1, th)):
        lo, hi, limit = (x1, x2, img.width) if axis == 0 else (y1, y2, img.height)
        extra = max(0, target - (hi - lo))
        lo = max(0, lo - (extra + 1) // 2)
        hi = min(limit, max(hi, lo + target))
        lo = max(0, min(lo, hi - target))
        if axis == 0:
            x1, x2 = lo, hi
        else:
            y1, y2 = lo, hi
    return (x1, y1, x2, y2)


TARGET = (289, 92, 585, 349)
print("TARGET crop:", TARGET)
all_ids = [r.id for r in an.regions]


def support_for(ids):
    """Exactly the write support used by selected_region_edit."""
    chosen = set(ids)
    arr = np.zeros((an.size[1], an.size[0]), dtype=bool)
    for r in an.regions:
        if r.id in chosen:
            arr |= np.asarray(r.mask.convert("L")) >= 128
    prot = es.protection_mask(an, list(ids))
    if prot is not None:
        arr &= np.asarray(prot) == 0
    return arr


def crop_from_support(arr, ids):
    ys, xs = np.where(arr)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1
    ctx = context
    if any(r.kind == "object" and r.id in ids for r in an.regions):
        ctx = max(ctx, float(settings["object_context_fraction"]))
    pad_x, pad_y = math.ceil((x2 - x1) * ctx), math.ceil((y2 - y1) * ctx)
    x1, x2 = max(0, x1 - pad_x), min(img.width, x2 + pad_x)
    y1, y2 = max(0, y1 - pad_y), min(img.height, y2 + pad_y)
    for axis in (0, 1):
        lo, hi, limit = (x1, x2, img.width) if axis == 0 else (y1, y2, img.height)
        extra = max(0, minimum - (hi - lo))
        lo = max(0, lo - (extra + 1) // 2)
        hi = min(limit, max(hi, lo + minimum))
        lo = max(0, min(lo, hi - minimum))
        if axis == 0:
            x1, x2 = lo, hi
        else:
            y1, y2 = lo, hi
    tw = max(minimum, math.ceil((y2 - y1) * minimum / maximum))
    th = max(minimum, math.ceil((x2 - x1) * minimum / maximum))
    for axis, target in ((0, tw), (1, th)):
        lo, hi, limit = (x1, x2, img.width) if axis == 0 else (y1, y2, img.height)
        extra = max(0, target - (hi - lo))
        lo = max(0, lo - (extra + 1) // 2)
        hi = min(limit, max(hi, lo + target))
        lo = max(0, min(lo, hi - target))
        if axis == 0:
            x1, x2 = lo, hi
        else:
            y1, y2 = lo, hi
    return (x1, y1, x2, y2)


for n in (1, 2):
    for combo in itertools.combinations(all_ids, n):
        try:
            arr = support_for(combo)
            if not arr.any():
                continue
            box = crop_from_support(arr, combo)
        except Exception:
            continue
        if box == TARGET:
            print("  *** MATCH:", combo)
        elif box and abs(box[0] - TARGET[0]) <= 12 and abs(box[1] - TARGET[1]) <= 12:
            print("  near:", combo, box)
print("--- single regions with protection ---")
for r in an.regions:
    arr = support_for([r.id])
    if not arr.any():
        print(f"  {r.id:10s} support EMPTY (fully protected)")
        continue
    ys, xs = np.where(arr)
    bb = (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1)
    print(f"  {r.id:10s} supportbbox={bb} crop={crop_from_support(arr, [r.id])}")

print("--- does the protection revert FLUX on head-0? ---")
SEL = ["head-2"]
prot = es.protection_mask(an, SEL)
print("protection mask present:", prot is not None)
if prot is not None:
    parr = np.asarray(prot) >= 128
    print("protected px:", int(parr.sum()))
    # head-0 region (background dark-haired man)
    h0 = [r for r in an.regions if r.id == "head-0"][0]
    h0arr = np.asarray(h0.mask.convert("L")) >= 128
    print("head-0 px:", int(h0arr.sum()), "| head-0 inside protected:",
          int((h0arr & parr).sum()), "=", round(float((h0arr & parr).sum()) / max(1, h0arr.sum()), 3))
    # Where did FLUX change pixels (final vs source)?
    src = np.asarray(Image.open("img_editor/edit_1792479695.png").convert("RGB")).astype(np.int16)
    fin = np.asarray(Image.open("img_editor/edit_1792479788.png").convert("RGB")).astype(np.int16)
    changed = np.abs(src - fin).max(axis=2) > 10
    print("changed px in final:", int(changed.sum()))
    print("changed px INSIDE protected:", int((changed & parr).sum()))
    print("changed px inside head-0 (protected):", int((changed & h0arr).sum()))
    print("changed px inside head-2 (selected):", int((changed & (np.asarray(h0.mask.convert('L')) >= 128)).sum()))
    ys, xs = np.where(changed)
    print("changed bbox:", (int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    print("=> FLUX bald head is at:", "head-0" if (changed & h0arr).sum() > 0 else "head-2")
