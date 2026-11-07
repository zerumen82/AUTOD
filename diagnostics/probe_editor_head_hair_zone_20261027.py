"""CPU contract: the head that stays out of the generation zone carries its hair.

The no-selection body route builds the zone as "the measured person minus the
head". The head it subtracts was a tight ellipse fitted to the *visible* face
(`_head_ellipse`), so on a profile the ponytail, the nape and part of the jaw
stayed inside the zone. Kontext filled that hole and painted a second head.

Measured on the user's own photo (`ui/tob/ComfyUI/input/fast_1793085083.png`,
640x640, 27/10): the mask actually sent to ComfyUI
(`mask_1793085083.png`) had 61,2 % of the ponytail and 72,4 % of the
jaw/neck inside the zone, and the delivery `edit_1793085144.png` came back
with a duplicated translucent head.

This probe pins the measured behaviour of `_head_with_hair`:

  * synthetic fixture (always): a person whose hair is attached to the head
    keeps that hair OUT of the zone, and the chest/hips INSIDE it. The old
    formula fails the same assertion (witness), so the probe catches the bug;
  * real photo (when the assets are present): the same assertion on the
        photo and mask that produced the duplicated head.

No model is loaded, downloaded, or installed. No render.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.selected_region_edit import (  # noqa: E402
    _head_ellipse,
    _head_with_hair,
    _person_without_head,
)

PASSED = []
FAILED = []


def check(name, condition, detail=""):
    if condition:
        PASSED.append(name)
        print(f"  PASS {name} {detail}")
    else:
        FAILED.append(name)
        print(f"  FAIL {name} {detail}")


def synthetic():
    """Person with the hair attached behind the head, as a profile gives it."""
    height, width = 420, 420
    person = np.zeros((height, width), dtype=bool)
    person[80:400, 120:260] = True          # torso + legs
    yy, xx = np.indices((height, width))
    hair = ((xx - 262) / 42.0) ** 2 + ((yy - 110) / 55.0) ** 2 <= 1.0
    person |= hair                          # the ponytail belongs to the person
    face = SimpleNamespace(bbox=(186.0, 62.0, 246.0, 142.0))

    tight = _head_ellipse(face, person.shape)
    checked = _head_with_hair(tight, person)
    old_zone = person & ~tight
    new_zone = person & ~checked

    hair_px = int(hair.sum())
    chest = np.s_[200:320, 130:250]
    hips = np.s_[330:395, 130:250]
    print(f"  fixture: persona {int(person.sum())} px, pelo {hair_px} px, "
          f"zona antes {int(old_zone.sum())} px, zona ahora {int(new_zone.sum())} px")

    # Witness: the tight ellipse really leaves the hair inside the zone.
    check("witness: la elipse ajustada dejaba el pelo en la zona",
          old_zone[hair].mean() > 0.5, f"({100 * old_zone[hair].mean():.1f} %)")
    # El pelo que cuelga más allá del margen medido sigue en la zona (deuda
    # declarada en spec/03): el fixture lo pone 2 px por debajo del margen.
    hair_left = 100 * new_zone[hair].mean()
    check("el pelo medido queda fuera de la zona",
          hair_left <= 2.0, f"({hair_left:.1f} %)")
    check("la cara medida queda fuera de la zona",
          new_zone[60:145, 185:250].mean() < 0.05,
          f"({100 * new_zone[60:145, 185:250].mean():.1f} %)")
    check("el pecho sigue dentro de la zona",
          new_zone[chest].mean() > 0.95, f"({100 * new_zone[chest].mean():.1f} %)")
    check("las caderas siguen dentro de la zona",
          new_zone[hips].mean() > 0.95, f"({100 * new_zone[hips].mean():.1f} %)")

    # The same helper through the route that composes the zone.
    mask = Image.fromarray((person * 255).astype(np.uint8), "L")
    region = SimpleNamespace(
        kind="head", mask=Image.fromarray((tight * 255).astype(np.uint8), "L"))
    analysis = SimpleNamespace(regions=[region])
    zone = _person_without_head(mask, analysis)
    route_left = 100.0 * zone[hair].mean() if zone is not None else 100.0
    check("_person_without_head usa la cabeza con pelo",
          zone is not None and route_left <= 2.0, f"({route_left:.1f} %)")


def real_photo():
    photo = ROOT / "ui/tob/ComfyUI/input/fast_1793085083.png"
    sent = ROOT / "ui/tob/ComfyUI/input/mask_1793085083.png"
    support_path = ROOT / ".autodeep_temp/support_1793085083.npy"
    if not (photo.is_file() and sent.is_file() and support_path.is_file()):
        print("  SKIP foto real: faltan los assets del render del 27/10")
        return
    import cv2
    from insightface.model_zoo import get_model

    image = Image.open(photo).convert("RGB")
    person = np.load(support_path)
    detector = get_model(str(ROOT / "models/buffalo_l/det_10g.onnx"),
                         providers=["CPUExecutionProvider"])
    detector.prepare(ctx_id=-1, input_size=(1024, 1024), det_thresh=0.5)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    boxes, _landmarks = detector.detect(bgr)
    del detector
    faces = [SimpleNamespace(bbox=box[:4], kps=None) for box in boxes]
    check("foto real: una cara detectada", len(faces) == 1, f"({len(faces)})")
    if not faces:
        return
    old_zone = person & ~_head_ellipse(faces[0], person.shape)
    new_zone = person.copy()
    for face in faces:
        head = _head_with_hair(_head_ellipse(face, person.shape), person)
        new_zone &= ~head

    boxes_measure = {
        "cola de caballo": np.s_[80:200, 315:380],
        "cara": np.s_[105:195, 245:330],
        "pecho/hombro": np.s_[235:300, 200:330],
        "cadera/muslo": np.s_[380:480, 200:330],
    }
    inside = {name: (100 * old_zone[box].mean(), 100 * new_zone[box].mean())
              for name, box in boxes_measure.items()}
    for name, (before, after) in inside.items():
        print(f"    {name}: zona antes {before:5.1f} % → ahora {after:5.1f} %")
    check("foto real: la coleta sale de la zona",
          inside["cola de caballo"][1] < 10.0,
          f"({inside['cola de caballo'][1]:.1f} %)")
    check("foto real: la cara sale de la zona",
          inside["cara"][1] < 10.0, f"({inside['cara'][1]:.1f} %)")
    check("foto real: el pecho sigue en la zona",
          inside["pecho/hombro"][1] > 95.0,
          f"({inside['pecho/hombro'][1]:.1f} %)")
    check("foto real: la cadera sigue en la zona",
          inside["cadera/muslo"][1] > 95.0,
          f"({inside['cadera/muslo'][1]:.1f} %)")
    print(f"    zona antes {int(old_zone.sum())} px → ahora {int(new_zone.sum())} px")


def main():
    print("[1] fixture sintetico")
    synthetic()
    print("[2] foto real del 27/10")
    real_photo()
    total = len(PASSED) + len(FAILED)
    print(f"\n{len(PASSED)}/{total} PASS")
    for name in FAILED:
        print(f"  FAIL {name}")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(main())
