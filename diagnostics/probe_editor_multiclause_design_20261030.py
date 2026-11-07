# -*- coding: utf-8 -*-
"""Validación de diseño 2026-10-30 — heurística multicláusula.

Compara la condición ACTUAL de `_multi_clause_uncover_is_full_body`
(`relevant_count >= min and uncover_count >= 1`) frente a la propuesta en
producción (v5.1256: una cláusula con scope igual a la cláusula entera y
dirección medida «change» es descripción pura del sujeto y no cuenta),
sobre los casos ya medidos en memory.md y el caso nuevo.

Nota: `proposed_multi` DUPLICA la lógica de producción a propósito, para
poder comparar sin tocarla; el cruce `actual (manager) == propuesto` de cada
fila hace visible cualquier drift como FAIL.

Solo lectura: no modifica producción.

Uso:
    python diagnostics/probe_editor_multiclause_design_20261030.py
"""

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

CASES = [
    # (texto, esperado por diseño/medición previa, nota)
    ("the woman with red shoes, she must go without shoes", False,
     "CASO NUEVO (log 12:19): descripción + un quitar local"),
    ("la mujer de zapatos rojos, debe ir sin zapatos", False,
     "CASO NUEVO en español"),
    ("The woman must go without shoes", False, "único (probe 29/10)"),
    ("The woman must go naked", True, "desnudez medida por vocabulario"),
    ("the woman with red shoes, she must go naked", True,
     "desnudez en cláusula 2 (vocabulario)"),
    ("The woman must go without shoes, without pants", True,
     "diseño: dos quitar fuertes = desvestir"),
    ("the woman must go without stockings, barefoot, dressed only in a thong", False,
     "la prenda conservada impide inferir desnudez total"),
    ("la mujer debe ir sin medias, descalza, vestida solo con un tanga", False,
     "la prenda conservada en español impide inferir desnudez total"),
    ("debe ir descalza y sin pantalones", True,
     "log 12:06/12:13 del 29/10: conjunción sin coma"),
    ("the woman is barefoot, without trousers", True, "diseño multicláusula"),
    ("the sofa must be grey, the woman naked", True, "v5.1216 sujeto equivocado"),
    ("cambia la camisa a rojo, y quítale los zapatos", False,
     "cambio + quitar local: no debe desnudar"),
    ("the woman must go without a hat, she must smile", False,
     "quitar local + gesto: no debe desnudar"),
    ("sin zapatos y sin pantalones", True,
     "dos imperativos desnudos en español sin modal (reviewer 2026-10-30)"),
    ("sin zapatos", False, "imperativo español suelto: sin promoción"),
]

RESULTS = []


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'} — {name}"
          + (f" | {detail}" if detail else ""))


def proposed_multi(manager, text):
    """Condición propuesta: la cláusula-descripción pura del sujeto no cuenta
    para el mínimo de cláusulas.

    Se midió (2026-10-30, diagnostics/measure_clause_facts_20261030.py) que
    «the woman with red shoes» llega con scope == cláusula entera (sin
    predicado pedido) y dirección medida «change»: es una descripción, no un
    segundo «quitar». El guard solo excluye dirección «change»: «sin
    pantalones» (scope == cláusula pero dirección «uncover») sigue contando,
    y «debe ir descalza» además separa el predicado (scope «ir descalza»).
    """
    try:
        policy = manager._body_scope_policy()
        scope_cfg = manager._load_instruction_scope()
        if not scope_cfg.get("multi_clause_uncover_enabled", False):
            return False
        min_clauses = int(scope_cfg.get("multi_clause_uncover_min_clauses", 2))
        if min_clauses < 2:
            return False
    except Exception:
        return False
    clauses = manager._split_compound_instruction(text or "")
    if len(clauses) < 2:
        clauses = [c.strip() for c in re.split(r"\s*[,;]\s*", text or "") if c.strip()]
    if len(clauses) < min_clauses:
        return False
    try:
        scorer = manager._get_semantic_analyzer(full_ai=False)
    except Exception:
        return False
    relevant_count = 0
    uncover_count = 0
    for clause in clauses:
        try:
            facts = manager._prompt_authority_facts(clause, clause)
        except Exception:
            continue
        if (
            facts.get("pose") or facts.get("scenario")
            or facts.get("local_region") or facts.get("local_appearance")
            or facts.get("structural_remove")
        ):
            continue
        scope = manager._requested_change_scope(clause) or clause
        try:
            scores = scorer.score_concepts(scope)
            cover_score = float(scores.get("attribute_cover", 0.0))
            direction = str(scorer.get_body_direction(scope)).lower()
            intensity = float(scorer.get_body_transform_intensity(scope))
        except Exception:
            return False
        if not np.isfinite(cover_score) or cover_score > 0.03:
            return False
        if scope.strip() == clause.strip() and direction == "change":
            continue          # descripción pura del sujeto, sin predicado
        relevant_count += 1
        if (
            direction == "uncover"
            and np.isfinite(intensity)
            and intensity >= policy["body_state_min_intensity"]
        ):
            uncover_count += 1
    return relevant_count >= min_clauses and uncover_count >= 1


def main():
    manager = ImgEditorManager()
    print(f"{'caso':<58} {'dir/int por cláusula':<44} "
          f"{'actual':<7} {'propuesto':<9} {'esperado'}")
    print("-" * 140)
    for text, expected, note in CASES:
        clauses = manager._split_compound_instruction(text)
        if len(clauses) < 2:
            clauses = [c.strip() for c in
                       re.split(r"\s*[,;]\s*", text) if c.strip()]
        scorer = manager._get_semantic_analyzer(full_ai=False)
        detail = []
        for clause in clauses:
            scope = manager._requested_change_scope(clause) or clause
            try:
                direction = str(scorer.get_body_direction(scope)).lower()
                intensity = float(scorer.get_body_transform_intensity(scope))
                detail.append(f"{clause[:26]!r}:{direction}/{intensity:.2f}")
            except Exception as error:
                detail.append(f"{clause[:26]!r}:ERR {error}")
        proposed = proposed_multi(manager, text)
        # El resultado GLOBAL medido incluye _full_body_exposure_clauses.
        actual_global = manager.measured_full_body_exposure(text)
        proposed_global = bool(manager._full_body_exposure_clauses(text)) or proposed
        mark = "OK " if proposed_global == expected else "!! "
        print(f"{mark}{text[:56]:<56} {' | '.join(detail)[:42]:<44} "
              f"{str(actual_global):<7} {str(proposed_global):<9} {expected}")
        check(f"{note} → {expected}",
              proposed_global == expected,
              f"propuesto={proposed_global}, actual={actual_global}")
    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} PASS")
    for name in failed:
        print(f"FAIL: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
