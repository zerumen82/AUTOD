# -*- coding: utf-8 -*-
"""Probe 2026-10-30 — «la mujer de zapatos rojos, debe ir sin zapatos».

Caso real (log 12:19, 2026-10-30): la instrucción compuesta (coma) acabó en
el prompt «She is fully nude, bare skin, no clothing at all.» con zona
«persona medida menos la cabeza» (2 476 769 px), en lugar de la zona local
del objetivo «shoes» que el arreglo v5.1255 activa para un «quitar» local.

Hipótesis: `_multi_clause_uncover_is_full_body` (config
multi_clause_uncover_enabled=true, min_clauses=2) promueve a exposición
completa una petición de un solo objeto porque la primera cláusula
(«the woman with red shoes») cuenta como cláusula relevante.

Mide solo NLP/cómputo, sin CLIPSeg, sin ComfyUI, sin render.

Uso:
    python diagnostics/probe_editor_compound_uncover_20261030.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

RESULTS = []

RAW_ES = "la mujer de zapatos rojos, debe ir sin zapatos"
EN = "the woman with red shoes, she must go without shoes"


def check(name, condition, detail=""):
    RESULTS.append((name, bool(condition), detail))
    print(f"{'PASS' if condition else 'FAIL'} — {name}"
          + (f" | {detail}" if detail else ""))


def main():
    manager = ImgEditorManager()

    print("\n[1] Separación en cláusulas")
    clauses = manager._split_compound_instruction(EN)
    print(f"  _split_compound_instruction → {clauses}")
    iclauses = manager._instruction_clauses(EN)
    print(f"  _instruction_clauses → {iclauses}")

    print("\n[2] Alcance medido")
    full_clauses = manager._full_body_exposure_clauses(EN)
    print(f"  _full_body_exposure_clauses → {full_clauses}")
    multi = manager._multi_clause_uncover_is_full_body(EN)
    print(f"  _multi_clause_uncover_is_full_body → {multi}")
    measured = manager.measured_full_body_exposure(EN)
    print(f"  measured_full_body_exposure → {measured}")
    true_full = manager.is_true_full_body_exposure(EN)
    print(f"  is_true_full_body_exposure → {true_full}")
    check("compuesto sin zapatos NO es exposición completa",
          measured is False, f"measured={measured}")
    check("la promoción viene de la heurística multicláusula",
          (multi is False) or (full_clauses == []),
          f"multi={multi}, full_clauses={full_clauses}")

    print("\n[3] Sufijo de estado y régimen")
    for text, label in ((EN, "EN"), (RAW_ES, "ES crudo")):
        regime = manager.measured_body_regime(RAW_ES if label == "ES crudo" else EN,
                                               EN, engine="kontext")
        suffix = manager._body_state_clause(
            EN if label == "EN" else RAW_ES,
            regime["direction"], regime["intensity"])
        print(f"  [{label}] direction={regime['direction']} "
              f"intensity={regime['intensity']:.3f} suffix={suffix!r}")
        check(f"{label}: sin sufijo «fully nude»",
              "fully nude" not in suffix.lower(), suffix)

    regime = manager.measured_body_regime(RAW_ES, EN, engine="kontext")
    prompt = manager._compose_generation_prompt(
        EN, engine="kontext",
        magnitude=max(0.75, float(regime["magnitude"])),
        body_transform=True, body_direction=regime["direction"],
        body_intensity=regime["intensity"],
    )
    print(f"\n  prompt compuesto → {prompt[:260]}")
    check("prompt compuesto: sin «fully nude»",
          "fully nude" not in prompt.lower())
    check("prompt compuesto: conserva la instrucción",
          "without shoes" in prompt.lower())

    print("\n[4] Objetivo local del quitar")
    target = manager._requested_removal_target(RAW_ES, EN)
    print(f"  _requested_removal_target → {target!r}")
    check("objetivo = shoes", target == "shoes", target)

    print("\n[5] Regresión: casos ya medidos")
    single = "The woman must go without shoes"
    check("caso único sin zapatos: no es exposición completa",
          manager.measured_full_body_exposure(single) is False)
    naked = "The woman must go naked"
    check("desnuda sí es exposición completa",
          manager.measured_full_body_exposure(naked) is True)
    compound_naked = "the woman with red shoes, she must go naked"
    check("compuesto con desnudez sí es exposición completa",
          manager.measured_full_body_exposure(compound_naked) is True)
    two_local = "the woman must go without shoes, without pants"
    check("dos quitar fuertes sigue siendo exposición completa (multicláusula)",
          manager.measured_full_body_exposure(two_local) is True,
          f"multi={manager._multi_clause_uncover_is_full_body(two_local)}")

    print("\n[6] Simulación de la ruta de zona (píxeles del log 12:19)")
    # selected_region_edit.py: full_body = measured_full_body_exposure;
    # si no, decide la foto con measured_uncover_is_whole_person.
    person_px, garment_px = 3131064, 496771
    for clause in clauses:
        facts = manager._prompt_authority_facts(clause, clause)
        print(f"  cláusula {clause!r} → facts="
              f"{ {k: v for k, v in facts.items() if v} }")
    has_unmeasured = manager._has_unmeasured_clause(EN)
    whole = manager.measured_uncover_is_whole_person(
        EN, person_px, garment_px)
    print(f"  _has_unmeasured_clause → {has_unmeasured}")
    print(f"  measured_uncover_is_whole_person(ropa={garment_px}, "
          f"persona={person_px}) → {whole}")
    full_body = manager.measured_full_body_exposure(EN) or whole
    check("la ruta NO promueve «sin zapatos» a cuerpo entero",
          full_body is False, f"full_body={full_body}")
    target_available = not full_body
    check("la ruta alcanza la zona local del objetivo («shoes»)",
          target_available, f"target_zone disponible={target_available}")

    failed = [name for name, ok, _ in RESULTS if not ok]
    print("\n" + "=" * 60)
    print(f"{len(RESULTS) - len(failed)}/{len(RESULTS)} PASS")
    for name in failed:
        print(f"FAIL: {name}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
