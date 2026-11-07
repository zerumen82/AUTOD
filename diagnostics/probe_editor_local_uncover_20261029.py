# -*- coding: utf-8 -*-
"""Probe 2026-10-29 — «sin zapatos» debe quitar los zapatos, no desnudar.

Caso real (log 17:53, edit_1793292876.png): la instrucción «LA MUJER DEBE IR
SIN ZAPATOS» acabó en el prompt «The woman is fully nude, bare skin, no
clothing at all», con la zona «persona medida menos la cabeza» y los zapatos
intactos (la máscara de generación terminaba por encima de los tobillos).

Este probe mide las tres capas, sin render:

1. cómputo de alcance (full vs local) y sufijo de estado;
2. frase objetivo del «quitar» local;
3. máscara real del objetivo con CLIPSeg sobre la foto del caso
   (edit_1793292876.png conserva los zapatos originales: están fuera de la
   zona que escribió el render anterior).

Límite conocido: el analizador mide la exposición local de los pies
(«barefoot/feet only»), no la de cada prenda. Un «without pants» aislado en
INGLÉS sigue midiendo full=0,00/local=0,00 y conserva el régimen corporal
completo anterior; la frase española sí mide local. No se añaden listas de
prendas (AGENTS.md).

Uso:
    python diagnostics/probe_editor_local_uncover_20261029.py
    python diagnostics/probe_editor_local_uncover_20261029.py --no-segment
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor import selected_region_edit as sre  # noqa: E402

CASE_PHOTO = PROJECT_ROOT / "img_editor" / "edit_1793292876.png"
# Zapatos del caso, medidos con CLIPSeg «white sneakers» (0.6) sobre la foto:
# caja (4957, 2956)-(5487, 3410). El probe comprueba que la zona nueva cae ahí.
SHOES_BBOX = (4957, 2956, 5487, 3410)

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'} — {name}" + (f" | {detail}" if detail else ""))


def _overlap_ratio(mask, box):
    x1, y1, x2, y2 = box
    window = mask[y1:y2, x1:x2]
    total = int(mask.sum())
    if total <= 0:
        return 0.0
    return float(window.sum()) / float(total)


def part_scope(manager):
    print("\n[1] Alcance medido y sufijo de estado")
    shoes = "The woman must go without shoes"
    naked = "The woman must go naked"
    check("sin zapatos: no es exposición completa",
          manager.measured_full_body_exposure(shoes) is False)
    check("desnuda: sí es exposición completa",
          manager.measured_full_body_exposure(naked) is True)

    regime = manager.measured_body_regime(shoes, shoes, engine="kontext")
    prompt = manager._compose_generation_prompt(
        shoes, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True, body_direction=regime["direction"],
        body_intensity=regime["intensity"],
    )
    check("prompt sin zapatos: sin «fully nude»",
          "fully nude" not in prompt.lower(), prompt[:180])
    check("prompt sin zapatos: conserva la instrucción",
          "without shoes" in prompt.lower())
    check("prompt sin zapatos: dirección uncover",
          regime["direction"] == "uncover", regime["direction"])

    nude_prompt = manager._compose_generation_prompt(
        naked, engine="kontext", magnitude=0.75, body_transform=True,
        body_direction="uncover", body_intensity=0.70)
    check("prompt desnuda: conserva el estado final",
          "fully nude" in nude_prompt.lower())

    print("\n[2] Frase objetivo del «quitar» local")
    check("objetivo EN = shoes",
          manager._requested_removal_target(naked, shoes) == "shoes",
          manager._requested_removal_target(naked, shoes))
    check("objetivo ES traducido = shoes",
          manager._requested_removal_target(
              "LA MUJER DEBE IR SIN ZAPATOS",
              "The woman must go without shoes.") == "shoes")
    check("objetivo no catalogado EN = anklet",
          manager._extract_removal_target(
              "The person must go without an anklet") == "anklet")
    check("objetivo no catalogado ES = jarrón",
          manager._extract_removal_target(
              "Quita el jarrón de la mesa") == "jarrón")
    check("objetivo no catalogado con ubicación = plant",
          manager._extract_removal_target(
              "Remove the plant next to the window") == "plant")
    check("desnuda no produce objetivo (tiene su ruta)",
          manager._requested_removal_target(naked, naked) == "")

    feet_prompt = manager._compose_local_target_prompt(
        shoes, "shoes", lower_body=True)
    other_prompt = manager._compose_local_target_prompt(
        "The woman must go without a hat", "hat", lower_body=False)
    check("prompt local de pies: estado medido y sin sujeto entero",
          "the bare skin of the legs and feet" in feet_prompt.lower()
          and "the measured area shows bare skin" in feet_prompt.lower()
          and "the woman is fully nude" not in feet_prompt.lower(),
          feet_prompt[:120])
    check("prompt local genérico: el objeto desaparece",
          "remove the hat completely; nothing of it may remain"
          in other_prompt.lower()
          # «what» contiene «hat»: se cuenta la palabra, no la subcadena.
          and len(re.findall(r"\bhat\b", other_prompt.lower())) == 1,
          other_prompt[:120])


def part_zone(manager):
    print("\n[3] Máscara real del objetivo sobre la foto del caso")
    if not CASE_PHOTO.is_file():
        check("foto del caso disponible", False, str(CASE_PHOTO))
        return
    image = Image.open(CASE_PHOTO).convert("RGB")
    print(f"Foto del caso: {image.size[0]}x{image.size[1]}")

    support, query = sre._described_subject_mask(
        manager, image, ["The woman", "the woman"])
    if support is None:
        check("persona medida", False, "CLIPSeg no devolvió persona")
        return
    ys, xs = np.nonzero(support)
    print(f"Persona medida ({query}): {int(support.sum())} px, caja "
          f"({int(xs.min())}, {int(ys.min())})-({int(xs.max())}, {int(ys.max())})")
    # La silueta de 352 px cubre el borde superior del calzado pero no su
    # suela: por eso la zona de producción terminaba en el tobillo.
    check("la persona medida no cubre el calzado completo",
          int(ys.max()) < SHOES_BBOX[3] - 40,
          f"y_max={int(ys.max())} vs zapatos y2={SHOES_BBOX[3]}")

    zone = sre._measured_target_zone(manager, image, support, "shoes")
    if zone is None:
        check("objetivo «shoes» medido en la foto", False)
        return
    pixels = int(zone.sum())
    person = int(support.sum())
    zys, zxs = np.nonzero(zone)
    print(f"Zona objetivo: {pixels} px, caja "
          f"({int(zxs.min())}, {int(zys.min())})-({int(zxs.max())}, {int(zys.max())})")
    check("zona objetivo no vacía y minoritaria",
          400 <= pixels <= 0.5 * person,
          f"{pixels} px = {100.0 * pixels / person:.1f} % de la persona")
    ratio = _overlap_ratio(zone, SHOES_BBOX)
    check("≥80 % de la zona cae en la caja de los zapatos",
          ratio >= 0.80, f"solape={ratio:.2%}")
    check("la zona recorrida alcanza la suela (y inferior del calzado)",
          int(zys.max()) >= SHOES_BBOX[3] - 40,
          f"y_max={int(zys.max())} vs zapatos y2={SHOES_BBOX[3]}")

    # Recorte que usará _kontext_zone_pass (contexto 0.35, lado máximo 768).
    from roop.img_editor.element_selection import load_settings
    settings = load_settings()["edit"]
    context = float(settings.get("crop_context_fraction", 0.35))
    pad_x = int(np.ceil((zxs.max() + 1 - zxs.min()) * context))
    pad_y = int(np.ceil((zys.max() + 1 - zys.min()) * context))
    box = (max(0, int(zxs.min()) - pad_x), max(0, int(zys.min()) - pad_y),
           min(image.width, int(zxs.max()) + 1 + pad_x),
           min(image.height, int(zys.max()) + 1 + pad_y))
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    maximum = int(settings.get("maximum_reference_side", 768))
    side = maximum / max(1, max(crop_w, crop_h))
    print(f"Recorte: {crop_w}x{crop_h} → escala {side:.2f} "
          f"({int(round(crop_w * side))}x{int(round(crop_h * side))})")
    check("el recorte no pierde resolución (escala ≥ 0.6)", side >= 0.60,
          f"escala={side:.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-segment", action="store_true",
                        help="solo cómputo NLP, sin cargar CLIPSeg")
    args = parser.parse_args()

    manager = ImgEditorManager()
    part_scope(manager)
    if not args.no_segment:
        part_zone(manager)

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} PASS")
    for name in failed:
        print(f"FAIL: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
