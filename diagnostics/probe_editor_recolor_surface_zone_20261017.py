"""Recolor de superficie: la zona sigue la superficie entera, no el fragmento.

Evidencia roja (log real `ui_start_20261017_220140.log`, edit 22:39 «el hombre
es calvo, las paredes son de color gris» sobre foto hotel 6395x4263):

- CLIPSeg midió «the wall» en 5,8 % de la foto («the walls» 2,7 %, «la pared»
  0,06 %, «las paredes» 0,00 %) — un FRAGMENTO de la pared, no la pared.
- La ruta de recolor v5.1209 SÍ se activó (consultas singulares presentes),
  pero la zona devuelta es ese fragmento: `_continue_measured_region` solo
  rellena huecos DENTRO de la propia máscara CLIPSeg (+19766 px).
- El hueco del sampler y el pegado = zona → la pared se pintó gris SOLO en
  un parche difuminado junto a la lámpara; el resto de la pared quedó crema
  (resultado `edit_1792269760.png`). Es el problema recurrente de paredes:
  no el enrutado (v5.1209 lo arregló) sino la COBERTURA de la zona.

Contrato: una cláusula de recolor sin parte distinta («las paredes son de
color gris» pide la superficie COMPLETA) crece por el color de la propia
superficie (envelope p90 + la separación plana 18 que este camino ya usa)
y se detiene donde el color cambia. Las personas y los objetos medidos
siguen fuera (mismos guardas del camino de persona). El crecimiento no
atraviesa hacia otra superficie de distinto color.
"""
import os
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

SIZE = (320, 240)
WALL = (228, 218, 203)
WOOD = (150, 100, 62)
SHEET = (246, 246, 250)
SKIN = (205, 165, 140)
RED_WALL = (170, 55, 50)

FAILED = []


def check(name, ok, detail=""):
    print(("OK  " if ok else "FAIL") + " | " + name + (f" — {detail}" if detail else ""))
    if not ok:
        FAILED.append(name)


class _StubMasker:
    """CLIPSeg simulado: solo los sustantivos de superficie miden."""

    def __init__(self, fragment):
        self.fragment = fragment

    def generate_mask(self, image, query, **kwargs):
        key = (query or "").strip().lower()
        if key in ("the wall", "the walls", "la pared", "las paredes"):
            return self.fragment.copy()
        return None


def _blocks_scene():
    """Pared crema con degradado + cabecero + sábanas + persona."""
    height, width = SIZE[1], SIZE[0]
    photo = np.zeros((height, width, 3), np.float32)
    grad = np.linspace(-8, 8, width)[None, :] + np.linspace(-6, 6, height)[:, None]
    for channel, base in enumerate(WALL):
        photo[..., channel] = base + grad
    photo[120:200, 40:280] = np.array(WOOD, np.float32)
    photo[200:240, :] = np.array(SHEET, np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    person = ((yy - 150) / 52.0) ** 2 + ((xx - 280) / 32.0) ** 2 <= 1.0
    photo[person] = np.array(SKIN, np.float32)
    fragment = np.zeros((height, width), np.uint8)
    fragment[20:70, 20:90] = 255
    ground = {
        "wall": ~person & ~(
            np.zeros((height, width), bool)
            | _box(120, 200, 40, 280)
            | _box(200, 240, 0, width)
        ),
        "wood": _box(120, 200, 40, 280),
        "sheet": _box(200, 240, 0, width),
        "skin": person,
    }
    return Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8)), fragment, ground


def _box(y0, y1, x0, x1):
    mask = np.zeros((SIZE[1], SIZE[0]), bool)
    mask[y0:y1, x0:x1] = True
    return mask


def _run(manager, image, fragment, clause_raw, clause_en):
    stub = _StubMasker(Image.fromarray(fragment, "L"))
    with patch.object(ImgEditorManager, "_get_clipseg_masker", return_value=stub):
        passes = manager._measured_region_passes(
            image, clause_raw, clause_en, surface_recolor=True, faces=(),
        )
    if not passes or passes[0].get("mask") is None:
        return None
    return np.asarray(passes[0]["mask"].convert("L")) >= 96


def main():
    manager = ImgEditorManager()

    print("[1] «las paredes son de color gris»: la zona es la superficie entera")
    image, fragment, ground = _blocks_scene()
    zone = _run(manager, image, fragment, "las paredes son de color gris", "the walls are gray")
    check("la cláusula devolvió una zona", zone is not None)
    if zone is not None:
        for name in ("wood", "sheet", "skin"):
            total = int(ground[name].sum())
            inside = int((zone & ground[name]).sum())
            check(f"{name} fuera de la zona", inside / max(1, total) <= 0.05,
                  f"{inside}/{total} = {inside / max(1, total):.1%}")
        wall_total = int(ground["wall"].sum())
        wall_inside = int((zone & ground["wall"]).sum())
        check(
            "la pared entera entra en la zona",
            wall_inside / wall_total >= 0.60,
            f"{wall_inside}/{wall_total} = {wall_inside / wall_total:.1%}",
        )

    print("\n[2] sin sobrecrecimiento: otra superficie de otro color no se cruza")
    height, width = SIZE[1], SIZE[0]
    photo = np.zeros((height, width, 3), np.float32)
    for channel, base in enumerate(WALL):
        photo[..., channel] = base
    photo[0:120, 0:200] = np.array(RED_WALL, np.float32)
    fragment2 = np.zeros((height, width), np.uint8)
    fragment2[20:70, 20:90] = 255
    red_zone = _box(0, 120, 0, 200)
    cream_zone = ~red_zone
    zone2 = _run(manager, Image.fromarray(np.clip(photo, 0, 255).astype(np.uint8)),
                 fragment2, "pinta la pared de rojo", "paint the wall red")
    check("la cláusula devolvió una zona", zone2 is not None)
    if zone2 is not None:
        cream_total = int(cream_zone.sum())
        cream_inside = int((zone2 & cream_zone).sum())
        check("la superficie vecina queda fuera",
              cream_inside / cream_total <= 0.05,
              f"{cream_inside}/{cream_total} = {cream_inside / cream_total:.1%}")
        red_total = int(red_zone.sum())
        red_inside = int((zone2 & red_zone).sum())
        check("la pared roja medida se cubre entera",
              red_inside / red_total >= 0.60,
              f"{red_inside}/{red_total} = {red_inside / red_total:.1%}")

    print(f"\n{'=' * 60}\n{'PASS' if not FAILED else 'FAIL'} — {len(FAILED)} fallos")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
