"""Probe de la verificación medida (P6, P7, P8, P9).

Comprueba con casos reales:
- La gramática del objetivo añadido (`_extract_addition_target`).
- La firma de encuadre: un cambio de cámara REAL se marca como medido y una
  edición de ropa NO (calibrado con las salidas reales del 2026-11-14).
- La zona medida: el cambio dentro/fuera de la zona.
- La evidencia nueva de un concepto: el objeto añadido deja rastro medible y
  un caso sin objeto no.

Uso: venv\\Scripts\\python.exe diagnostics\\probe_measured_verification_20261115.py
"""

import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402
from roop.img_editor.measured_verification import (  # noqa: E402
    framing_change,
    new_concept_evidence,
    zone_change,
)

SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"
OUT = ROOT / "diagnostics/out/editor_categories_20261114"

checks = 0
failures = 0


def check(label: str, condition: bool, extra: str = "") -> None:
    global checks, failures
    checks += 1
    if condition:
        print(f"OK   {label} {extra}")
    else:
        failures += 1
        print(f"FAIL {label} {extra}")


def main() -> None:
    before = Image.open(SOURCE).convert("RGB")

    # --- Gramática del objetivo añadido -------------------------------
    cases = (
        ("Add a small flag beside the astronaut", "small flag"),
        ("Add another astronaut beside the first one", "astronaut"),
        ("add a red car to the left of the house", "red car"),
        ("Añade una bandera pequeña al lado del astronauta", "bandera pequeña"),
        ("pon un perro cerca de la mesa", "perro"),
        ("Change the background to a forest", ""),
        ("Remove the black helmet from the photo", ""),
    )
    for prompt, expected in cases:
        got = ImgEditorManager._extract_addition_target(prompt)
        check(f"objetivo añadido de «{prompt[:38]}»", got == expected,
              f"→ {got!r} (esperado {expected!r})")

    # --- Firma de encuadre (P9) ----------------------------------------
    camera = OUT / "camera.png"
    clothing = OUT / "clothing.png"
    if camera.is_file():
        report = framing_change(before, Image.open(camera).convert("RGB"))
        check("cambio de cámara: es medible", report.get("medible") is True,
              f"score={report.get('score')}")
        check("cambio de cámara: se detecta el cambio", report.get("changed") is True,
              f"score={report.get('score')} indicadores={report.get('indicators')}")
    if clothing.is_file():
        report = framing_change(before, Image.open(clothing).convert("RGB"))
        check("cambio de ropa: no se marca cambio de encuadre",
              report.get("changed") is False,
              f"score={report.get('score')} indicadores={report.get('indicators')}")

    # --- Zona medida (P7) ---------------------------------------------
    zone = Image.new("L", before.size, 0)
    zone.paste(255, (100, 100, 300, 300))
    changed_zone = before.copy()
    changed_zone.paste((255, 0, 0), (120, 120, 200, 200))
    report = zone_change(before, changed_zone, zone)
    check("zona medida: disponible", report.get("available") is True)
    check("zona medida: el cambio dentro de la zona cuenta",
          report.get("inside_ratio", 0) > 0.1, f"dentro={report.get('inside_ratio')}")
    check("zona medida: nada cambia fuera", report.get("outside_ratio", 1) < 0.01,
          f"fuera={report.get('outside_ratio')}")
    check("zona medida: veredicto OK", report.get("ok") is True)
    elsewhere = before.copy()
    elsewhere.paste((0, 255, 0), (0, 0, 80, 80))
    report = zone_change(before, elsewhere, zone)
    check("zona medida: cambio fuera se marca como no conforme",
          report.get("ok") is False, f"fuera={report.get('outside_ratio')}")

    # --- Evidencia nueva de un concepto (P8) --------------------------
    from roop.img_editor.clipseg_masker import get_clipseg_masker

    masker = get_clipseg_masker()
    ok, message = masker.initialize()
    if not ok:
        check("CLIPSeg disponible", False, message)
    else:
        masker.device = "cpu"
        masker.model.to("cpu")
        add = OUT / "add.png"
        remove = OUT / "remove.png"
        if add.is_file():
            report = new_concept_evidence(
                masker, before, Image.open(add).convert("RGB"), "flag")
            check("evidencia nueva del objeto añadido",
                  report.get("available") and report.get("fresh_area", 0) >= 0.001,
                  f"fresh_area={report.get('fresh_area')}")
        if remove.is_file():
            report = new_concept_evidence(
                masker, before, Image.open(remove).convert("RGB"), "helmet")
            check("sin objeto pedido no hay evidencia nueva",
                  report.get("fresh_area", 1) < 0.001,
                  f"fresh_area={report.get('fresh_area')}")
        masker.unload()

    print(f"\n{checks - failures}/{checks} comprobaciones OK")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
