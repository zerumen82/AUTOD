# -*- coding: utf-8 -*-
"""Contraprueba temporal del TESTER (sd-tester) — 2026-10-30.

Verificación independiente del cambio en
`roop/img_editor/img_editor_manager.py::_multi_clause_uncover_is_full_body`
(una cláusula-descripción, scope == cláusula entera con dirección `change`,
ya no cuenta para el mínimo de cláusulas).

Solo NLP/cómputo con el manager real: sin CLIPSeg, sin ComfyUI, sin render.
No modifica producción.

Uso:
    python diagnostics/tmp_tester_20261030.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

RESULTS = []

COMPOUND_EN = "the woman with red shoes, she must go without shoes"
COMPOUND_ES = "la mujer de zapatos rojos, debe ir sin zapatos"


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'} — {name}"
          + (f" | {detail}" if detail else ""))


def main():
    manager = ImgEditorManager()

    v1 = manager.measured_full_body_exposure(COMPOUND_EN)
    check("measured_full_body_exposure(compuesto EN sin zapatos) es False",
          v1 is False, f"valor={v1}")

    v2 = manager.measured_full_body_exposure("The woman must go naked")
    check("measured_full_body_exposure(desnuda) es True",
          v2 is True, f"valor={v2}")

    v3 = manager.measured_full_body_exposure(
        "The woman must go without shoes, without pants")
    check("measured_full_body_exposure(sin zapatos + sin pantalones) es True",
          v3 is True, f"valor={v3}")

    # _body_state_clause para el caso compuesto de zapatos (EN y ES crudo):
    # el sufijo NO debe contener «fully nude».
    for text, label in ((COMPOUND_EN, "EN"), (COMPOUND_ES, "ES")):
        regime = manager.measured_body_regime(
            COMPOUND_ES if label == "ES" else COMPOUND_EN,
            COMPOUND_EN, engine="kontext")
        suffix = manager._body_state_clause(
            text, regime["direction"], regime["intensity"])
        print(f"  [{label}] direction={regime['direction']} "
              f"intensity={regime['intensity']:.3f} suffix={suffix!r}")
        check(f"_body_state_clause del compuesto de zapatos [{label}] "
              f"NO contiene «fully nude»",
              "fully nude" not in (suffix or "").lower(), repr(suffix))

    v5 = manager.is_true_full_body_exposure(COMPOUND_ES)
    check("is_true_full_body_exposure(compuesto ES sin zapatos) es False",
          v5 is False, f"valor={v5}")

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"CONTRAPRUEBA TESTER: {len(RESULTS) - len(failed)}"
          f"/{len(RESULTS)} PASS")
    for name in failed:
        print(f"FAIL: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
