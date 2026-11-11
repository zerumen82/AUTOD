"""Probe del saneado de posesivos en _proposed_region_passes (v5.1334).

Reproduce la proyección de queries de soporte sobre los tokens de la
instrucción, antes y después del fix, con la instrucción del log
2026-11-11 («el hombre debe ir con un top de mujer»).
"""
import re

POSSESSIVE = re.compile(r"['\u2019\u02bc\u00b4`]s\b")


def strip_possessive(text: str) -> str:
    return POSSESSIVE.sub("", text or "")


def project(instruction: str, proposals, fixed: bool) -> list:
    base = instruction.casefold()
    if fixed:
        base = strip_possessive(base)
    tokens = set(re.findall(r"\w+", base))
    grounded = []
    seen = set()
    for query in proposals:
        qtext = query.casefold()
        if fixed:
            qtext = strip_possessive(qtext)
        words = re.findall(r"\w+", qtext)
        projected = " ".join(
            word
            for word in dict.fromkeys(w for w in words if w in tokens)
            if len(word) > 1
        ) if fixed else " ".join(dict.fromkeys(w for w in words if w in tokens))
        if projected and projected not in seen:
            grounded.append(projected)
            seen.add(projected)
    return grounded


INSTRUCTION = "The man must go with a woman's top"
PROPOSALS = ["man's neck", "woman's shoulder"]

before = project(INSTRUCTION, PROPOSALS, fixed=False)
after = project(INSTRUCTION, PROPOSALS, fixed=True)
print("ANTES :", before)
print("DESPUES:", after)
assert before == ["man s", "woman s"], before
assert after == ["man", "woman"], after
print("PROBE OK: queries saneadas y ancladas a la instrucción")
