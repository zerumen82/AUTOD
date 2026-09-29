"""Check that an ambiguous focal request is surfaced by the UI event."""
from pathlib import Path

import numpy as np
from PIL import Image

from ui.tabs.img_editor.events import on_generate


ROOT = Path(__file__).resolve().parents[1]
source = Image.open(ROOT / "ui/tob/ComfyUI/input/fast_1790331799.png").convert("RGB")
source = source.resize((800, 533), Image.Resampling.LANCZOS)
output, status, _ = on_generate(
    source, "EL HOMBRE DEBE SER CALVO", "longcat", 0,
    False, 1, False, "hd",
)
assert output is not None, status
assert np.array_equal(np.asarray(source), np.asarray(output)), status
assert "no se pudo localizar" in status.lower(), status
assert "Guardado en" not in status, status
print(f"UI status={status}")
print("UI output identical to input; no global generation")
