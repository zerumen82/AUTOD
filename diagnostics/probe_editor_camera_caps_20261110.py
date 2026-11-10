"""Probe: instrucciones de cámara en mayúsculas o sin tildes (v5.1326).

Comprueba que «mejora el ANGULO» (y variantes) se miden como petición de
cámara en vez de caer a la ruta de región y devolver la foto sin cambios.
Incluye controles negativos para no ampliar la ruta a cualquier frase.

Uso: venv/Scripts/python.exe diagnostics/probe_editor_camera_caps_20261110.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roop.img_editor.nlp.semantic_analyzer import is_camera_view_request  # noqa: E402
from roop.img_editor.prompt_translator import translate_prompt  # noqa: E402

CASES = [
    # (entrada, se espera cámara)
    ("mejora el ANGULO", True),
    ("CAMBIA el ANGULO", True),
    ("mejora el ángulo", True),
    ("cambia el angulo", True),
    ("cambia el ángulo, plano desde arriba", True),
    ("improves the angle", True),
    ("change the camera angle", True),
    # Controles: no deben activar la ruta de cámara
    ("mejora la calidad de la foto", False),
    ("haz un plano del salon", False),
    ("cambia el color del coche", False),
    ("el hombre debe ser calvo", False),
    ("la mujer debe ir descalza", False),
]


def main() -> int:
    fails = 0
    for raw, expected in CASES:
        translated = translate_prompt(raw)
        view_translated = is_camera_view_request(translated)
        view_raw = is_camera_view_request(raw)
        ok = view_translated == expected and view_raw == expected
        # La palabra suelta en mayúsculas no debe sobrevivir al traductor.
        caps_residue = "ANGULO" in translated
        if caps_residue:
            ok = False
        status = "PASS" if ok else "FAIL"
        if not ok:
            fails += 1
        print(
            f"  [{status}] {raw!r} -> {translated!r} | camara(raw)={view_raw} "
            f"camara(trad)={view_translated} esperado={expected}"
            + (" | RESIDUO ANGULO" if caps_residue else "")
        )
    print(f"\nTOTAL FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
