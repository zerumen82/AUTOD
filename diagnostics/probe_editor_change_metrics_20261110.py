"""Probe: medición objetiva del cambio y recorte que ve el verificador (v5.1327).

Sin modelos ni GPU: valida `measure_change` (0 px cuando la foto no cambió,
diferencia dentro/fuera de la máscara) y `_detail_crop` (recorta la zona
cambiada, devuelve la imagen completa si no hay cambios).

Uso: venv/Scripts/python.exe diagnostics/probe_editor_change_metrics_20261110.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from roop.img_editor.instruction_verifier import (  # noqa: E402
    InstructionVerifier, measure_change,
)

W, H = 640, 480


def _photo(with_box: bool, box=(260, 180, 380, 300)) -> Image.Image:
    img = Image.new("RGB", (W, H), (210, 210, 210))
    draw = ImageDraw.Draw(img)
    draw.rectangle((40, 40, 200, 200), fill=(90, 90, 90))
    if with_box:
        draw.rectangle(box, fill=(200, 40, 40))
    return img


def main() -> int:
    fails = 0

    # 1) Sin cambios: 0 px y el verificador recibe la foto completa.
    same = measure_change(_photo(False), _photo(False))
    ok = same.get("available") and same["changed_px"] == 0
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] foto idéntica -> changed_px={same.get('changed_px')}")

    crop_full = InstructionVerifier._detail_crop(_photo(False), _photo(False))
    ok = crop_full.size == (W, H)
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] sin cambios, recorte completo -> {crop_full.size}")

    # 2) Cambio localizado: se mide y el recorte se centra en la zona.
    before, after = _photo(False), _photo(True)
    changed = measure_change(before, after)
    ok = changed.get("available") and changed["changed_px"] > 0
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] cambio localizado -> {changed.get('changed_px')} px "
          f"({changed.get('changed_ratio', 0):.2%})")

    crop = InstructionVerifier._detail_crop(after, before)
    ok = crop.size[0] < W or crop.size[1] < H
    ok = ok and crop.size[0] >= 384 and crop.size[1] <= H
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] recorte de la zona cambiada -> {crop.size}")

    # 3) Máscara: el cambio debe quedar dentro y nada fuera.
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).rectangle((240, 160, 400, 320), fill=255)
    inside = measure_change(before, after, mask)
    ok = inside.get("inside_px", 0) > 0 and inside.get("outside_ratio", 1) == 0
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] dentro={inside.get('inside_px')} px, "
          f"fuera={inside.get('outside_ratio', 1):.2%}")

    # 4) Cambio fuera de la máscara: se detecta como fuga.
    leak_mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(leak_mask).rectangle((0, 0, 100, 100), fill=255)
    leak = measure_change(before, after, leak_mask)
    ok = leak.get("outside_px", 0) == leak.get("changed_px", -1)
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] fuga fuera de máscara -> outside_px={leak.get('outside_px')}"
          f", changed={leak.get('changed_px')}")

    # 5) Tamaños distintos: no inventa una medida.
    diff_size = measure_change(Image.new("RGB", (10, 10)), Image.new("RGB", (11, 10)))
    ok = diff_size.get("available") is False
    fails += 0 if ok else 1
    print(f"  [{'PASS' if ok else 'FAIL'}] tamaños distintos -> {diff_size.get('reason')}")

    print(f"\nTOTAL FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
