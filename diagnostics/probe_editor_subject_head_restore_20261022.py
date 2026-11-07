"""v5.1242 — el marco corporal se entrega entero, sin devolver la cabeza.

v5.1241 pegaba la cabeza medida de la foto original sobre el marco de
LongCat. Medido el 23/10: FastEdit_00836_.png editaba el cuerpo y
edit_1792706837.png solo difería en un óvalo del pecho (x 302-475,
y 396-702 de 777x1600): la cara original encima del torso. El usuario
pidió volver a la entrega que editaba.

Falla (rc!=0) cuando la ruta vuelve a exponer el pegado de cabeza o cuando
un marco sin otras personas no es, píxel a píxel, la imagen generada.

Uso:  PYTHONPATH=D:\\PROJECTS\\AUTOAUTO python diagnostics/probe_editor_subject_head_restore_20261022.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roop.img_editor import selected_region_edit as sre  # noqa: E402


def main() -> int:
    failures = []
    source = Path(sre.__file__).read_text(encoding="utf-8")

    for name in ("_subject_head_restore_mask", "_aligned_head_mask"):
        if hasattr(sre, name) or f"def {name}" in source:
            failures.append(f"vuelve el pegado de cabeza ({name})")
    if "devueltos de la foto original" in source:
        failures.append("el marco vuelve a anunciar la cabeza original")

    # Sin otras personas el compuesto es la imagen generada. Un óvalo de la
    # foto original no puede reaparecer en el torso.
    size = (80, 120)
    generated = Image.new("RGB", size, (20, 180, 40))
    original = Image.new("RGB", size, (200, 30, 30))
    original.paste((10, 10, 220), (30, 40, 55, 80))
    effective = Image.new("L", size, 255)
    result = np.asarray(Image.composite(generated, original, effective))
    if not np.array_equal(result, np.asarray(generated)):
        failures.append("el marco sin otras personas no es la imagen generada")
    chest = result[40:80, 30:55]
    if np.array_equal(chest, np.asarray(original)[40:80, 30:55]):
        failures.append("el torso vuelve a ser la foto original")

    for line in failures:
        print(f"[FAIL] {line}")
    if failures:
        print(f"[RESULTADO] {len(failures)} fallo(s) — la entrega que editaba NO está")
        return 1
    print("[RESULTADO] OK — el marco corporal se entrega entero, sin cabeza original")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
