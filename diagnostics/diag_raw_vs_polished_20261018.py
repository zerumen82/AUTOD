#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico: que es lo que sale NEGRO, el sampler o el post-acabado.

El cliente de Generate descarga la imagen de ComfyUI y luego le aplica
polish_result_image. La muestra cruda sigue en ui/tob/ComfyUI/output, asi que se
puede medir sin volver a renderizar: si la cruda es normal y la devuelta por el
cliente es negra, el culpable es el post-acabado.
"""
import glob
import os
import sys
import time

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "ui", "tob", "ComfyUI", "output")
RESULT = os.path.join(ROOT, "diagnostics", "out", "gen_uncensored_20261018")

MAX_AGE_H = 8


def stats(path):
    image = Image.open(path).convert("RGB")
    pixels = np.asarray(image, dtype=np.float32)
    return image.size, float(pixels.mean()), float(pixels.std()), float(np.percentile(pixels, 50))


def main():
    now = time.time()
    print("== Muestras crudas de ComfyUI (antes del post-acabado) ==")
    files = [p for p in glob.glob(os.path.join(OUT, "**", "*.png"), recursive=True)
             if now - os.path.getmtime(p) < MAX_AGE_H * 3600]
    for path in sorted(files, key=os.path.getmtime):
        size, mean, std, p50 = stats(path)
        marca = "NEGRA " if std < 20 else "ok    "
        momento = time.strftime("%H:%M:%S", time.localtime(os.path.getmtime(path)))
        print(f"{momento} {marca} {os.path.basename(path)[:44]:46s} {size} "
              f"mean={mean:6.1f} std={std:6.1f} mediana={p50:5.0f}")

    print()
    print("== Imagen final devuelta por el cliente (tras el post-acabado) ==")
    finals = sorted(glob.glob(os.path.join(RESULT, "*.png")))
    for path in finals:
        size, mean, std, p50 = stats(path)
        marca = "NEGRA " if std < 20 else "ok    "
        print(f"{marca} {os.path.basename(path)[:44]:46s} {size} "
              f"mean={mean:6.1f} std={std:6.1f} mediana={p50:5.0f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
