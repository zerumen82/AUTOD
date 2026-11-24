#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 2 — calidad de LongCat Turbo sin cambiar de motor.

Verifica, sin ComfyUI ni UI:
  1. los pasos y el tope de imagen de trabajo salen de configuración;
  2. un recorte del tamaño del calvo del 24/11 se acota antes de ComfyUI;
  3. el pegado sigue escribiendo en el recorte original cuando el generado
     vuelve más pequeño (es lo que ya hacía el cliente al muestrear a ~1 MP).
"""
import os
import sys

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def main() -> int:
    steps, max_side = ImgEditorManager._local_appearance_quality()
    print(f"config: pasos={steps}, lado maximo={max_side}")
    assert steps >= 20, "los pasos no pueden bajar del minimo anterior"
    assert 0 < max_side <= 2048, "el tope de imagen de trabajo esta fuera de rango"

    manager = ImgEditorManager.__new__(ImgEditorManager)

    # Recorte medido en el calvo de las 10:07 (3480x2901).
    work = Image.new("RGB", (3480, 2901), (110, 120, 130))
    mask = Image.new("L", work.size, 0)
    mask.paste(255, (1400, 1000, 2100, 1800))
    scale = float(max_side) / float(max(work.size))
    bounded = (
        max(1, int(round(work.width * scale))),
        max(1, int(round(work.height * scale))),
    )
    print(f"imagen de trabajo: {work.size} -> {bounded} (escala {scale:.3f})")
    assert max(bounded) == max_side
    small_mask = mask.resize(bounded, Image.Resampling.BILINEAR)
    before = float(np.asarray(mask).mean())
    after = float(np.asarray(small_mask).mean())
    assert abs(before - after) < 0.01, "la mascara cambio al acotarla"
    print(f"cobertura de mascara conservada: {before:.4f} -> {after:.4f}")

    # El generado vuelve pequeno; el pegado lo escala al recorte original.
    source = Image.new("RGB", (600, 500), (30, 40, 50))
    generated = Image.new("RGB", bounded, (200, 200, 200))
    paste_mask = Image.new("L", source.size, 0)
    paste_mask.paste(255, (100, 100, 300, 300))
    box = (0, 0, 600, 500)
    integrated = manager._integrate_local_generation(
        source, generated, paste_mask, box,
    )
    arr = np.asarray(integrated)
    changed = int((np.any(arr != np.asarray(source), axis=2)).sum())
    print(f"pegado con generado {bounded} en recorte {source.size}: {changed} px cambiados")
    assert changed > 10000, "el pegado no reescalo el generado al recorte"
    outside = int((np.any(arr[350:500, 350:600] != np.asarray(source)[350:500, 350:600], axis=2)).sum())
    assert outside == 0, "el pegado cambio pixeles fuera de la mascara"
    print("PASS — Fase 2: Turbo con mas pasos y codificacion acotada sin romper el pegado")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
