"""Diagnostica por qué _measured_subject_passes devuelve [] en la foto del astronauta con prompts de clothing/body (matriz P1 FAIL)."""
from pathlib import Path
from PIL import Image
import numpy as np

from roop.img_editor.img_editor_manager import get_img_editor_manager

root = Path(__file__).resolve().parents[1]
img = Image.open(root / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png").convert("RGB")
print("tamaño:", img.size)

manager = get_img_editor_manager()
masker = manager._get_clipseg_masker()
print("masker:", masker)

for prompt in (
    "Change the astronaut suit to blue",
    "Make the astronaut taller",
):
    print("\n==== PROMPT:", prompt)
    queries = manager._subject_queries(prompt, prompt)
    print("queries:", queries)
    descs = manager._visual_subject_descriptions(prompt)
    print("_visual_subject_descriptions:", descs)
    for q in queries:
        try:
            cand = masker.generate_subject_mask(img, q)
        except Exception as e:
            print("generate_subject_mask error", q, type(e).__name__, e)
            continue
        cov = float((np.asarray(cand.convert("L")) > 96).mean()) if cand is not None else None
        print(f"  query={q!r}: mask={cand is not None} coverage={cov:.1%}" if cov is not None else f"  query={q!r}: mask=None")

    passes = manager._measured_subject_passes(img, prompt, prompt, faces=None)
    print("passes:", passes)