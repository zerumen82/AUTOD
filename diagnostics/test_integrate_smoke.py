# -*- coding: utf-8 -*-
"""Smoke end-to-end de _integrate_local_generation con los arreglos nuevos.

Usa una instancia sin __init__ (los metodos implicados son puro numpy/cv2/PIL)
y una foto real como fuente:

  1. Construye un "sujeto" (elipse) con un cambio fuerte en el interior
     (simula el repintado del modelo) y lo entrega escalado x3.51, como la
     generacion real de 1 MP.
  2. Llama al metodo real del manager.
  3. Verifica: corre sin excepcion, imprime nitidez multi + feather escalado,
     pega solo dentro del sujeto y recupera nitidez en la zona pegada.

Sale PASS/FAIL con numeros.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import cv2
from PIL import Image, ImageDraw

from roop.img_editor.img_editor_manager import ImgEditorManager

SRC_EDIT = ROOT / "img_editor" / "edit_1791586497.png"
BOX = (0, 0, 1700, 1700)
SCALE = 3.51
CENTER = (850, 850)
INNER = (320, 400)   # radio de la zona con cambio real
SUBJ = (430, 520)    # radio del sujeto (mascara)


def lapvar(a):
    g = a if a.dtype == np.uint8 else np.clip(a, 0, 255).astype(np.uint8)
    if g.ndim == 3:
        g = g.mean(axis=2).astype(np.uint8)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def main():
    photo = Image.open(SRC_EDIT).convert("RGB").crop(BOX)
    src = np.asarray(photo, dtype=np.float32)

    # Cambio fuente en el interior del sujeto: version tintada del propio
    # contenido a resolucion de generacion (texto blanda + color nuevo), que
    # es lo que el modelo entrega de verdad (no un blur inventado).
    yy, xx = np.mgrid[0:src.shape[0], 0:src.shape[1]]
    inner = (((xx - CENTER[0]) / INNER[0]) ** 2 +
             ((yy - CENTER[1]) / INNER[1]) ** 2) <= 1.0
    soft = cv2.resize(
        cv2.resize(src, (int(src.shape[1] / SCALE), int(src.shape[0] / SCALE)),
                   interpolation=cv2.INTER_AREA),
        (src.shape[1], src.shape[0]),
        interpolation=cv2.INTER_LANCZOS4,
    )
    changed = src.copy()
    changed[inner] = np.clip(soft[inner] * 0.70 + 90.0, 0, 255)

    gen_full = Image.fromarray(changed.astype(np.uint8))
    gen_small = gen_full.resize(
        (int(src.shape[1] / SCALE), int(src.shape[0] / SCALE)),
        Image.Resampling.LANCZOS,
    )

    # Mascara del sujeto a tamano de fuente
    mask = Image.new("L", photo.size, 0)
    ImageDraw.Draw(mask).ellipse(
        (CENTER[0] - SUBJ[0], CENTER[1] - SUBJ[1],
         CENTER[0] + SUBJ[0], CENTER[1] + SUBJ[1]),
        fill=255,
    )

    mgr = object.__new__(ImgEditorManager)
    out = mgr._integrate_local_generation(photo, gen_small, mask, BOX)

    ok = True
    checks = []

    def check(name, cond, detail):
        nonlocal ok
        ok = ok and bool(cond)
        checks.append(f"  [{'PASS' if cond else 'FAIL'}] {name}: {detail}")

    check("tamano", out.size == photo.size, f"{out.size}")

    o = np.asarray(out, dtype=np.float32)
    # Nitidez de la zona pegada (interior, sin margen de feather)
    inner_shrunk = (((xx - CENTER[0]) / (INNER[0] - 60)) ** 2 +
                    ((yy - CENTER[1]) / (INNER[1] - 60)) ** 2) <= 1.0
    r_in, s_in = lapvar(o[inner_shrunk]), lapvar(src[inner_shrunk])
    check("nitidez pegada >= 25% fuente", r_in >= 0.25 * max(s_in, 1.0),
          f"salida={r_in:.0f} fuente={s_in:.0f} ratio={r_in / max(1.0, s_in):.0%}")

    # Fuera del sujeto + margen de feather: debe ser identico a la fuente
    subj_grown = (((xx - CENTER[0]) / (SUBJ[0] + 120)) ** 2 +
                  ((yy - CENTER[1]) / (SUBJ[1] + 120)) ** 2) <= 1.0
    outside = np.abs(o - src)[~subj_grown]
    check("fuera del sujeto intacto", outside.max() <= 1.0,
          f"max diff={outside.max():.1f}")

    # El cambio si se aplico dentro
    inside_diff = np.abs(o - src)[inner_shrunk].mean()
    check("cambio aplicado", inside_diff > 5.0,
          f"MAE dentro={inside_diff:.1f}")

    # Transicion de feather: banda parcial (diff entre 10% y 90% del meseta)
    # en el recorrido horizontal por el centro, cerca del borde del cambio.
    row = int(CENTER[1])
    prof = np.abs(o - src)[row].mean(axis=1)
    edge = CENTER[0] + INNER[0]
    seg = prof[edge - 120:edge + 120]
    plateau = max(float(np.percentile(seg, 95)), 1.0)
    partial = int(np.sum((seg >= plateau * 0.1) & (seg <= plateau * 0.9)))
    check("rampa de feather >= 15px", partial >= 15,
          f"banda parcial ~{partial}px (feather pedido 34px; codigo viejo=6px)")

    print()
    for line in checks:
        print(line)
    print("RESULTADO:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
