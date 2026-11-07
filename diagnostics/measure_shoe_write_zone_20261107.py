# -*- coding: utf-8 -*-
"""¿Qué píxeles de la zona objetivo NO se escriben y por qué?

Mide sobre la foto del caso (2026-11-07) la zona de «shoes», la máscara de
piel (`_skin_pixels`) y el diff real entre la entrada y cada variante, para
ver si la guarda de piel está excluyendo el propio objeto que hay que quitar.

Uso:
    python diagnostics/measure_shoe_write_zone_20261107.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from roop.img_editor import selected_region_edit as sre  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager, _skin_pixels  # noqa: E402

SOURCE = ROOT / "img_editor" / "edit_1794006695.png"
OUT = ROOT / "diagnostics" / "out" / "shoe_removal_variants_20261107"


def main():
    photo = Image.open(SOURCE).convert("RGB")
    manager = ImgEditorManager()
    person, query = sre._described_subject_mask(manager, photo, ["The woman"])
    target = manager._requested_removal_target(
        "LA MUJER DEBE IR SIN ZAPATOS", "The woman must go without shoes")
    zone = sre._measured_target_zone(manager, photo, person, target)
    if person is None or zone is None:
        print("FAIL: sin persona/zona")
        return 1
    rgb = np.asarray(photo.convert("RGB"))
    skin = _skin_pixels(rgb)
    garment = zone & ~skin
    print(f"zona «{target}»: {int(zone.sum())} px")
    print(f"  piel dentro de la zona: {int((zone & skin).sum())} px "
          f"({100.0 * (zone & skin).sum() / zone.sum():.1f} %)")
    print(f"  write_zone (zona & ~piel): {int(garment.sum())} px "
          f"({100.0 * garment.sum() / zone.sum():.1f} %) "
          f"→ {'SE ESCRIBE TODO' if garment.sum() < 400 else 'se escribe solo lo no-piel'}")

    arr_in = np.asarray(photo.convert("RGB"), dtype=np.int16)
    for name in ("prod", "cover", "endstate", "cover_endstate"):
        path = OUT / f"result_{name}.png"
        if not path.exists():
            continue
        arr_out = np.asarray(Image.open(path).convert("RGB"), dtype=np.int16)
        diff = np.abs(arr_out - arr_in).max(axis=2) > 16
        in_zone = int((diff & zone).sum())
        in_skin_zone = int((diff & zone & skin).sum())
        in_garment_zone = int((diff & garment).sum())
        print(f"{name}: cambiado dentro de la zona {in_zone} px "
              f"({100.0 * in_zone / zone.sum():.1f} %) · "
              f"de los que eran piel {in_skin_zone}/{int((zone & skin).sum())} · "
              f"de los no-piel {in_garment_zone}/{int(garment.sum())}")

    overlay = rgb.copy()
    overlay[zone & skin] = [255, 0, 0]
    overlay[garment] = [0, 255, 0]
    crop = Image.fromarray(overlay).crop((140, 400, 350, 590))
    crop.resize((crop.width * 3, crop.height * 3), Image.NEAREST).save(
        OUT / "zone_skin_overlay.png")
    print(f"overlay → {OUT / 'zone_skin_overlay.png'} (rojo=piel excluida, verde=se escribe)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
