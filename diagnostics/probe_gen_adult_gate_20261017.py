"""Gate adulto medido + coma pegada antes de Marian (regresión sesión 00:22-00:42).

Evidencia roja (log real `ui_start_20261016_231655.log`, pestaña Generate):
- un prompt largo con señal adulta medida quedó en score 0,121 vs umbral 0,13
  → «NSFW ON pero el prompt no pide contenido adulto» con el prefijo
  anti-censor caído; medido: «orgy» y «explicit» valían 0,000 (sin cobertura
  de concepto en adult_gate).
- «…FOLLANDOLA,EXPLICITO…» sin espacio tras la coma produjo un token pegado
  que el SentencePiece de Marian pasó sin traducir («explicito» en la salida
  EN). Solo cambia espacios: no toca vocabulario ni intención.
"""
import os
import sys
from contextlib import nullcontext
from types import SimpleNamespace

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.img_editor import prompt_translator  # noqa: E402
from roop.img_editor.gen_semantic import (  # noqa: E402
    prompt_requests_adult_content,
    score_adult_gate,
)

FAILED = []


def check(name, ok):
    print(("OK  " if ok else "FAIL") + " | " + name)
    if not ok:
        FAILED.append(name)


def main():
    print("[1] adult_gate: la señal larga (orgy/explicit) cruza el umbral")
    cases = [
        # Evidencia roja real (log 00:22-00:42): esta frase midió 0,121 vs
        # umbral 0,13 → gate caído con «orgy» y «explicit» a 0,000.
        ("mature in an orgy surrounded by people fucking her, explicit, ultra realistic", True),
        ("explicit orgy scene with several people, photorealistic", True),
        ("an orgy", True),
        ("castle with unicorns, drawing style", False),
        ("photorealistic portrait of a woman", False),
        ("family having barbecue in the garden", False),
        ("woman in a bikini by the pool", False),
        ("a detailed pencil sketch of an old fisherman", False),
        ("hot sexy woman in a leather jacket, studio light", False),
    ]
    for text, expected in cases:
        score = score_adult_gate(text)
        got = prompt_requests_adult_content(text)
        print(f"    score={score:.3f} gate={'ON' if got else 'OFF'} "
              f"(esperado {'ON' if expected else 'OFF'}) | {text}")
        check(f"adult_gate {'ON' if expected else 'OFF'}: {text}", got == expected)

    print("\n[2] coma pegada: la fuente que llega a Marian lleva espacio")
    captured = []

    class _FakeTokenizer:
        def __call__(self, source, **kwargs):
            captured.append(source)
            return {"input_ids": [[0]]}

        def decode(self, ids, **kwargs):
            # Condiciones suficientes para no tocar el guard de pérdida.
            return "translated, text, with, enough, clauses"

    class _FakeModel:
        def generate(self, **kwargs):
            return [[0]]

    fake_torch = SimpleNamespace(inference_mode=lambda: nullcontext())
    original = prompt_translator._offline_translator
    prompt_translator._offline_translator = (
        _FakeTokenizer(), _FakeModel(), fake_torch,
    )
    try:
        out = prompt_translator._offline_translate(
            "una mesa grande,roja,en la cocina,EXPLICITO"
        )
    finally:
        prompt_translator._offline_translator = original
    check("traducción local disponible con stub", out is not None)
    check(
        "sin tokens pegados en la fuente de Marian",
        bool(captured)
        and captured[0] == "una mesa grande, roja, en la cocina, EXPLICITO",
    )
    if captured:
        print(f"    fuente a Marian: {captured[0]!r}")

    print("\n[3] sin dobles espacios ni comas alteradas")
    captured.clear()
    prompt_translator._offline_translator = (
        _FakeTokenizer(), _FakeModel(), fake_torch,
    )
    try:
        prompt_translator._offline_translate("un gato blanco, sentado, en un sofa azul")
    finally:
        prompt_translator._offline_translator = original
    check(
        "entrada ya normalizada pasa idéntica",
        bool(captured) and captured[0] == "un gato blanco, sentado, en un sofa azul",
    )

    print(f"\n{'=' * 60}\n{'PASS' if not FAILED else 'FAIL'} — "
          f"{len(FAILED)} fallos")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
