"""Gate adulto: la traducción real de producción debe medir intención adulta.

Evidencia roja (log real `ui_start_20261017_154703.log`, pestaña Generate,
10 corridas 16:00-19:30): el MISMO prompt ES del usuario se traduce con
variantes de Marian y el gate solo disparaba en algunas de ellas —

- «practice SEX with … EXPLICITE PENETRATION» → score 0.1430 (dispara, umbral 0.13)
- «SEXUAL RELATIONS WITH … EXPLICITE PENETRACION» → score 0.0715 (NO dispara)
- «practice dogging …» sin palabras adultas explícitas → score ≈ 0.02 (NO dispara)

Fórmula medida del analizador ligero (_score_texts):
score = coverage(matched/palabras_del_prompt)·0.75 + density(matched/palabras_del_concepto)·0.35,
y con prompts largos (~23 tokens) hacen falta 2+ coincidencias exactas en un
solo concepto para cruzar 0.13. «SEXUAL RELATIONS» solo cae en «sexual»
(1 match en el concepto largo) y «relations» no está en ningún concepto →
media pasada → sin prefijo adulto/anti-censor en corridas adultas reales.

Fix (config, umbral intacto): una línea nueva en adult_gate.concepts con las
salidas del traductor medidas en producción («sexual relations» por
«relaciones sexuales», «penetration» por «penetracion», «cuckold» y «dogging»
pasan en la traducción). Ninguna palabra suelta cruza el umbral en prompts
largos: hacen falta 2+ de la línea para disparar (misma característica del
gate actual).

Grupo 1: traducciones EN reales del log → gate ON.
Grupo 2: regresión — positivos y negativos del contrato previo intactos.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from roop.img_editor.gen_semantic import (  # noqa: E402
    prompt_requests_adult_content,
    score_adult_gate,
)

FAILED = []

# Variantes reales de Marian sobre el mismo prompt ES (log 15:47, corridas
# 16:04-19:30). Las tres son contenido adulto y el prefijo adulto/anti-censor
# debía activarse en las tres.
PRODUCTION_EN = [
    ("wide plan, cuckold pov, a 45 year old woman, thin, brown hair, white skin, "
     "SEXUAL RELATIONS WITH Several children on the beach, EXPLICITE PENETRATION, "
     "realistic photo, amateur photo, hyper realistic with details"),
    ("wide plan, cuckold pov, a 45 year old woman, thin, brown hair, white skin, "
     "practice SEX with several children on the beach, EXPLICIT PENETRATION, "
     "realistic photo, amateur photo, hyper realistic with details"),
    ("wide plan, cuckold pov, a 45 year old woman, skinny, dark hair, white skin, "
     "practice dogging with several children on the beach, realistic photo, "
     "amateur photo, hyper realistic with details"),
]

REGRESSION_TRUE = [
    "mature in an orgy surrounded by people fucking her, explicit, ultra realistic",
    "explicit orgy scene with several people, photorealistic",
    "an orgy",
]

REGRESSION_FALSE = [
    "castle with unicorns, drawing style",
    "photorealistic portrait of a woman",
    "family having barbecue in the garden",
    "woman in a bikini by the pool",
    "a detailed pencil sketch of an old fisherman",
    "hot sexy woman in a leather jacket, studio light",
]


def check(name, ok):
    print(("OK  " if ok else "FAIL") + " | " + name)
    if not ok:
        FAILED.append(name)


def main():
    print("[1] traducciones EN reales de producción → gate ON")
    for text in PRODUCTION_EN:
        score = score_adult_gate(text)
        got = prompt_requests_adult_content(text)
        print(f"    score={score:.4f} gate={'ON' if got else 'OFF'} | {text[:80]}...")
        check(f"producción gate ON: {text[:52]}...", got)

    print("\n[2] regresión del contrato previo del gate")
    for text in REGRESSION_TRUE:
        got = prompt_requests_adult_content(text)
        print(f"    gate={'ON' if got else 'OFF'} | {text}")
        check(f"gate ON: {text}", got)
    for text in REGRESSION_FALSE:
        got = prompt_requests_adult_content(text)
        print(f"    gate={'ON' if got else 'OFF'} | {text}")
        check(f"gate OFF: {text}", not got)

    print(f"\n{'=' * 60}\n{'PASS' if not FAILED else 'FAIL'} — {len(FAILED)} fallos")
    if FAILED:
        for name in FAILED:
            print(f"  FAIL: {name}")
        sys.exit(1)


if __name__ == "__main__":
    main()
