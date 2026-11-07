# -*- coding: utf-8 -*-
"""Medición 2026-10-30 — ¿qué distingue una cláusula-descripción de una
cláusula-petición para la heurística multicláusula?

Solo lectura, sin render.

Uso:
    python diagnostics/measure_clause_facts_20261030.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

CASES = [
    "the woman with red shoes, she must go without shoes",
    "la mujer de zapatos rojos, debe ir sin zapatos",
    "debe ir descalza y sin pantalones",
    "The woman must go without shoes, without pants",
    "the woman is barefoot, without trousers",
    "the woman with red shoes, she must go naked",
    "the sofa must be grey, the woman naked",
    "the woman must go without a hat, she must smile",
    "cambia la camisa a rojo, y quítale los zapatos",
]


def main():
    manager = ImgEditorManager()
    scorer = manager._get_semantic_analyzer(full_ai=False)
    for text in CASES:
        print("\n" + "=" * 100)
        print(f"TEXTO: {text}")
        clauses = manager._split_compound_instruction(text)
        if len(clauses) < 2:
            import re
            clauses = [c.strip() for c in re.split(r"\s*[,;]\s*", text) if c.strip()]
        for clause in clauses:
            scope = manager._requested_change_scope(clause) or ""
            facts = manager._prompt_authority_facts(clause, clause)
            visible = {k: v for k, v in facts.items() if v}
            try:
                direction = scorer.get_body_direction(scope or clause)
                intensity = scorer.get_body_transform_intensity(scope or clause)
            except Exception as error:
                direction, intensity = f"ERR {error}", float("nan")
            subj = manager._visual_subject_description(clause)
            unfmeasured_ok = manager._has_unmeasured_clause(text)
            print(f"  cláusula : {clause!r}")
            print(f"    scope  : {scope!r}")
            print(f"    facts  : {visible}")
            print(f"    dir/int: {direction} / {intensity:.3f}")
            print(f"    subject: {subj!r}")
        print(f"  _has_unmeasured_clause(texto) = {unfmeasured_ok}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
