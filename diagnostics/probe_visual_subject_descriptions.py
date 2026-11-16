"""Diagnostica _visual_subject_descriptions para ver si reconoce 'astronaut' y 'suit'."""
from pathlib import Path
from PIL import Image
import numpy as np

from roop.img_editor.img_editor_manager import get_img_editor_manager

root = Path(__file__).resolve().parents[1]
img = Image.open(root / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png").convert("RGB")
print("tamaño:", img.size)

manager = get_img_editor_manager()
descs = manager._visual_subject_descriptions("Change the astronaut suit to blue")
print("_visual_subject_descriptions:", descs)