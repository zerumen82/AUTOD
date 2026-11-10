"""Calibra la comprobación de destrucción corporal (v5.1325).

Casos:
  1. Emparejamiento real roto: `edit_1794265715.png` → `edit_1794266098.png`
     («la mujer debe ir descalza», piernas tapadas). Debe rechazarse.
  2. Cambio legítimo de tono sobre la misma banda: no debe rechazarse.
  3. Relleno plano sobre piel: debe rechazarse por «plana».
  4. Escritura fuera de la zona: debe rechazarse por «fuera».
  5. Sin cambios: no debe rechazarse.

Sin GPU, sin ComfyUI y sin modelos: solo PIL, numpy y OpenCV.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roop.img_editor.selected_region_edit import (  # noqa: E402
    _body_destruction_metrics,
    _body_destruction_reason,
)

BROKEN_FROM = ROOT / "img_editor" / "edit_1794265715.png"
BROKEN_TO = ROOT / "img_editor" / "edit_1794266098.png"
FALLBACK = (ROOT / "diagnostics" / "out" / "appearance_reference_20261023"
            / "1793812998803975900" / "result.png")

fails = 0


def check(name, condition, detail=""):
    global fails
    mark = "OK  " if condition else "FAIL"
    print(f"{mark} {name}" + (f" — {detail}" if detail else ""), flush=True)
    if not condition:
        fails += 1


def band_mask(size, top=0.65):
    width, height = size
    mask = np.zeros((height, width), dtype=bool)
    mask[int(round(top * height)):, :] = True
    return mask


def main():
    if BROKEN_FROM.is_file() and BROKEN_TO.is_file():
        base = Image.open(BROKEN_FROM).convert("RGB")
        broken = Image.open(BROKEN_TO).convert("RGB")
        zone = band_mask(base.size)
        metrics = _body_destruction_metrics(base, broken, zone)
        reason = _body_destruction_reason(base, broken, zone)
        print(f"[real] metrics={metrics}", flush=True)
        print(f"[real] reason={reason!r}", flush=True)
        check("caso real roto se rechaza", bool(reason), reason)
    else:
        print("SKIP caso real roto: faltan los archivos del log", flush=True)

    if FALLBACK.is_file():
        photo = Image.open(FALLBACK).convert("RGB")
    elif BROKEN_FROM.is_file():
        photo = Image.open(BROKEN_FROM).convert("RGB")
    else:
        print("FAIL sin foto de trabajo", flush=True)
        return 1
    work = photo.copy()
    zone = band_mask(photo.size)
    arr = np.asarray(photo, dtype=np.float32)
    width, height = photo.size

    print(f"[foto] {work.size}", flush=True)

    # 2. Cambio de tono legítimo: la textura se conserva.
    legit = arr.copy()
    legit[zone] = np.clip(legit[zone] + 14.0, 0, 255)
    legit_img = Image.fromarray(legit.astype(np.uint8), "RGB")
    metrics = _body_destruction_metrics(work, legit_img, zone)
    reason = _body_destruction_reason(work, legit_img, zone)
    print(f"[tono] metrics={metrics} reason={reason!r}", flush=True)
    check("cambio de tono legítimo no se rechaza", reason == "", reason)

    # 3. Relleno plano sobre la piel (el fallo de las piernas tapadas).
    flat = arr.copy()
    fill = np.median(arr[zone], axis=0)
    flat[zone] = fill
    flat_img = Image.fromarray(flat.astype(np.uint8), "RGB")
    metrics = _body_destruction_metrics(work, flat_img, zone)
    reason = _body_destruction_reason(work, flat_img, zone)
    print(f"[plano] metrics={metrics} reason={reason!r}", flush=True)
    check("relleno plano se rechaza por piel plana",
          "plana" in reason, reason)

    # 3b. Bloques planos como en la captura del caso real (3 parches).
    blocks = arr.copy()
    band_top = int(round(0.65 * height))
    band_h = max(1, height - band_top)
    for index in range(3):
        y0 = band_top + int(index * band_h / 3)
        y1 = band_top + int((index + 1) * band_h / 3)
        x0 = int(width * (0.10 + 0.28 * index))
        x1 = min(width, x0 + int(width * 0.24))
        blocks[y0:y1, x0:x1] = fill
    blocks_img = Image.fromarray(blocks.astype(np.uint8), "RGB")
    metrics = _body_destruction_metrics(work, blocks_img, zone)
    reason = _body_destruction_reason(work, blocks_img, zone)
    print(f"[bloques] metrics={metrics} reason={reason!r}", flush=True)
    check("bloques planos se rechazan por piel plana",
          "plana" in reason, reason)

    # 4. Escritura fuera de la zona medida.
    outside = arr.copy()
    x0, y0 = int(width * 0.1), int(height * 0.1)
    outside[y0:y0 + 40, x0:x0 + 40] = 255.0
    outside_img = Image.fromarray(outside.astype(np.uint8), "RGB")
    metrics = _body_destruction_metrics(work, outside_img, zone)
    reason = _body_destruction_reason(work, outside_img, zone)
    print(f"[fuera] metrics={metrics} reason={reason!r}", flush=True)
    check("escritura fuera de la zona se rechaza",
          "fuera de la zona" in reason, reason)

    # 5. Sin cambios.
    reason = _body_destruction_reason(work, work.copy(), zone)
    check("sin cambios no se rechaza", reason == "", reason)

    print(f"TOTAL FAIL={fails}", flush=True)
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
