"""Inspect the real inpaint result at the original crop size."""
from pathlib import Path
from PIL import Image, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790280681.png").convert("RGB")
edited = Image.open(ROOT / "diagnostics/profile_longcat_inpaint_raw.png").convert("RGB")
mask = Image.open(ROOT / "diagnostics/profile_hair_mask_00619.png").convert("L")
edited = edited.resize(source.size, Image.Resampling.LANCZOS)
mask = mask.filter(ImageFilter.GaussianBlur(radius=14))
composed = Image.composite(edited, source, mask)
composed.save(ROOT / "diagnostics/profile_longcat_inpaint_soft.png")
composed.resize((213, 387), Image.Resampling.LANCZOS).save(
    ROOT / "diagnostics/profile_longcat_inpaint_native.png")
