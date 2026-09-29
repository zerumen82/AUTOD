"""Measure whether pose descriptions ground the subjects in the user's photo."""
from pathlib import Path

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import get_img_editor_manager


root = Path(__file__).resolve().parents[1]
img = Image.open(root / "ui/tob/ComfyUI/input/fast_1790338257.png").convert("RGB")
img = img.resize((800, 533), Image.Resampling.LANCZOS)
manager = get_img_editor_manager()
masker = manager._get_clipseg_masker()
for query in (
    "woman on her knees", "woman", "stockings",
    "man on his knees", "man", "man on his knees must be bald",
):
    mask = masker.generate_mask(img, query)
    cov = float((np.asarray(mask.convert("L")) > 128).mean()) if mask else 0.0
    print(f"{query!r}: coverage={cov:.3%}")
    if mask:
        mask.save(root / "diagnostics" / ("descriptor_" + query.replace(" ", "_") + ".png"))
