"""Before/after crops of a saved result, matched with ITS source.

The user keeps uploading photos, so "the newest file" is not necessarily the
source of a saved result: the source is the one with the same geometry.

`.autodeep_temp` has ~70k hashed directories, and the analysis of a 27 MP
photo is slow, so the element analysis runs on a reduced copy and the boxes
are scaled back up. Crops only, no inference.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
ANALYSIS_LONG_SIDE = 1280


def newest_source():
    rows = []
    for dirpath, _dirs, files in os.walk(ROOT / ".autodeep_temp"):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(dirpath, name)
                try:
                    rows.append((os.path.getmtime(path), path))
                except OSError:
                    pass
    rows.sort(reverse=True)
    return rows[0][1] if rows else None


def find_source(result: Image.Image):
    """The source of a saved result: the photo with the same geometry.

    `.autodeep_temp` holds every photo the user ever uploaded (~70k hashed
    directories), so the source has to be searched by geometry instead of by
    "newest": the user uploads new photos continuously.
    """
    candidates = []
    for dirpath, _dirs, files in os.walk(ROOT / ".autodeep_temp"):
        for name in files:
            if name.lower().endswith((".jpg", ".jpeg")):
                path = os.path.join(dirpath, name)
                try:
                    candidates.append((os.path.getmtime(path), path))
                except OSError:
                    pass
    candidates.sort(reverse=True)
    for _mtime, path in candidates:
        try:
            with Image.open(path) as candidate:
                if candidate.size == result.size:
                    return Image.open(path).convert("RGB")
        except Exception:
            continue
    return None


def main():
    result_path = sys.argv[1] if len(sys.argv) > 1 else str(
        OUT / "global_only_prompt.png")
    tag = sys.argv[2] if len(sys.argv) > 2 else "global"
    result = Image.open(result_path).convert("RGB")
    source = find_source(result)
    if source is None:
        print("[Compare] no source with the same geometry", flush=True)
        return 1
    print(f"[Compare] result {result.size} <- source {source.size}", flush=True)
    scale = ANALYSIS_LONG_SIDE / max(source.size)
    small = source.resize(
        (max(1, int(source.width * scale)), max(1, int(source.height * scale))),
        Image.Resampling.LANCZOS)
    analysis = es.analyze_elements(small)
    factor = 1.0 / scale
    for region in analysis.regions:
        if region.kind != "head":
            continue
        mask = np.asarray(region.mask.convert("L")) > 127
        ys, xs = np.where(mask)
        x0, y0, x1, y1 = (int(xs.min() * factor), int(ys.min() * factor),
                          int(xs.max() * factor), int(ys.max() * factor))
        pad = int((x1 - x0) * 0.35)
        box = (max(0, x0 - pad), max(0, y0 - pad),
               min(source.width, x1 + pad), min(source.height, y1 + pad))
        width = 380
        height = max(1, int(width * (box[3] - box[1]) / (box[2] - box[0])))
        source.crop(box).resize((width, height)).save(
            OUT / f"{tag}_before_{region.id}.png")
        result.crop(box).resize((width, height)).save(
            OUT / f"{tag}_after_{region.id}.png")
        print(f"[Compare] head {region.id} bbox={(x0, y0, x1, y1)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
