"""Probe: la traducción no repite puntuación y conserva la elipsis (v5.1327).

Mide el defecto real del log 2026-11-10 («Man must be bald....») sin cargar
modelos: ejercita `clean_text`, que es donde se normaliza la salida traducida.

Uso: venv/Scripts/python.exe diagnostics/probe_editor_prompt_punctuation_20261110.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roop.img_editor.prompt_translator import clean_text  # noqa: E402

CASES = [
    # (entrada traducida, esperado)
    ("Man must be bald....", "Man must be bald."),
    ("Man must be bald..", "Man must be bald."),
    ("The woman must go barefoot", "The woman must go barefoot"),
    ("wait... then she smiles", "wait... then she smiles"),
    ("Man must be bald .", "Man must be bald."),
    ("Hello!!", "Hello!"),
    ("Is she ok??", "Is she ok?"),
]


def main() -> int:
    fails = 0
    for raw, expected in CASES:
        got = clean_text(raw)
        ok = got == expected
        if not ok:
            fails += 1
        print(f"  [{'PASS' if ok else 'FAIL'}] {raw!r} -> {got!r} (esperado {expected!r})")
    # La instrucción completa no debe perder contenido al normalizar.
    sample = (
        "Edit this exact photo: The woman must go barefoot, keep the same people "
        "present with a strong likeness to them, and preserve the overall composition."
    )
    if clean_text(sample) != sample:
        fails += 1
        print(f"  [FAIL] La normalización alteró una instrucción sin puntuación repetida: {clean_text(sample)!r}")
    else:
        print("  [PASS] Instrucción normal intacta")
    print(f"\nTOTAL FAIL={fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
