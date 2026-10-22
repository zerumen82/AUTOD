"""Measure the inclusion rule on the real photo.

The blind and the protection mask now share one rule: a head whose measured
mask lies mostly inside the selected object support belongs to the selected
subject. This measures that rule on the real photo, so the composition of the
real result can be explained.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import element_selection as es  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"


def main():
    photo = Image.open(SOURCE).convert("RGB")
    analysis = es.analyze_elements(photo)
    inclusion = float(es.load_settings()["edit"]["head_inclusion_coverage"])
    objects = [region for region in analysis.regions if region.kind == "object"]
    region = max(objects, key=lambda item: item.coverage)
    support = es._selected_object_support(analysis, [region.id])
    support_arr = np.asarray(support) > 0 if support is not None else None
    print(f"[Inclusion] selected object: {region.id} ({region.label}) "
          f"coverage={region.coverage:.4f}", flush=True)
    for head in analysis.regions:
        if head.kind != "head":
            continue
        head_arr = np.asarray(head.mask.convert("L")) >= 128
        area = int(head_arr.sum())
        inside = int((head_arr & support_arr).sum()) if support_arr is not None else 0
        ratio = inside / max(1, area)
        belongs = es._head_inside_selection(head.mask, support, inclusion)
        print(f"[Inclusion] head {head.id}: area={area} inside support={inside} "
              f"ratio={ratio:.4f} (threshold {inclusion}) -> belongs to selection={belongs}", flush=True)
        print(f"[Inclusion] head {head.id} would be blinded: {not belongs}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
