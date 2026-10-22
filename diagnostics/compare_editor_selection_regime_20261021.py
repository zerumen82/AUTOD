"""Comparison sheet: the failed result and the measured-regime results.

Builds one sheet with the photo that failed on 21/03 and the results of the
measured regime, so the change can be inspected at a glance.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "diagnostics" / "out" / "selection_regime_real_20261021"
SOURCE = ROOT / ".autodeep_temp" / "42ebfcf365a6c3b5c1dbfdbf39c36c32f25938bfc7ac489c44cd9366e2b27b3a" / "19978049_13-35-33.jpg"


def main():
    panels = [
        ("ORIGINAL (21/03: la ropa intacta)", Image.open(SOURCE).convert("RGB")),
    ]
    for name, path in (
        ("medido, semilla 424242", OUT / "regional_seed424242.png"),
        ("medido, semilla 7", OUT / "regional_seed7.png"),
    ):
        if path.exists():
            panels.append((name, Image.open(path).convert("RGB")))
    size = (360, 540)
    sheet = Image.new("RGB", (size[0] * len(panels), size[1] + 30), (24, 24, 24))
    draw = ImageDraw.Draw(sheet)
    for index, (title, image) in enumerate(panels):
        sheet.paste(image.resize(size, Image.Resampling.LANCZOS), (index * size[0], 30))
        draw.text((index * size[0] + 6, 9), title, fill=(255, 255, 255))
    target = OUT / "comparison_original_vs_measured.png"
    sheet.save(target)
    print(f"[Sheet] {target}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
