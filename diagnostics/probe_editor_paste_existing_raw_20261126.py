"""Compare paste limits using a completed neutral LongCat edit; no render."""

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path("diagnostics/out/appearance_reference_20261023/1795710447155110700")
MASKS = Path("diagnostics/out/editor_person_mask_20261126")
OUT = Path("diagnostics/out/editor_paste_existing_raw_20261126")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = Image.open(ROOT / "source.png").convert("RGB")
    reference = Image.open(ROOT / "reference_0.png").convert("RGB")
    generated = Image.open(ROOT / "raw_0.png").convert("RGB")
    raw_source = np.asarray(source)
    raw_ref = np.asarray(reference)
    width, height = reference.size
    matches = []
    for y in range(source.height - height + 1):
        for x in range(source.width - width + 1):
            region = raw_source[y:y + height, x:x + width]
            error = float(np.abs(region.astype(np.int16) - raw_ref.astype(np.int16)).mean())
            matches.append((error, x, y))
    error, x, y = min(matches)
    assert error < 1, f"reference crop not aligned with source: {error}"
    patch = generated.resize((width, height), Image.Resampling.LANCZOS)
    canvas = source.copy()
    canvas.paste(patch, (x, y))
    for name, path in (
        ("head", MASKS / "paste_head.png"),
        ("person", MASKS / "paste_limit.png"),
    ):
        mask = Image.open(path).convert("L")
        assert mask.size == source.size
        result = Image.composite(canvas, source, mask)
        result.save(OUT / f"whole_{name}.png")
    print(f"crop=({x}, {y}, {x + width}, {y + height}), error={error:.3f}")


if __name__ == "__main__":
    main()
