# -*- coding: utf-8 -*-
"""Probe: el Modo Calidad (híbrido) no debe forzar edición GLOBAL de una
instrucción que no pide calidad.

Contexto (log 2026-11-11, «el hombre debe ser calvo» con Modo Calidad):
`has_quality_request` se contaminaba con `or quality_enhance` (análisis
9742-9755) y ese valor contaminado entraba en `force_global` (9994-9997):
cualquier instrucción con Modo Calidad activo se derivaba a edición global
(identidad 0,92→0,81) en vez de su ruta focal medida. El fix usa la petición
de calidad MEDIDA en el texto (`_measured_quality_request`) solo para el
enrutado; las cláusulas de prompt del híbrido siguen usando el valor amplio.

Verificación CPU: AST sobre el manager + ejecución real de `_light_score`
con los anchors extraídos del propio fuente (sin stubs ni listas paralelas).
"""

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
MANAGER = os.path.join(ROOT, "roop", "img_editor", "img_editor_manager.py")

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")


def main():
    src = open(MANAGER, encoding="utf-8").read()
    tree = ast.parse(src)

    # 1) La medida se guarda justo tras calcular has_quality_request.
    assign_after = False
    lines = src.splitlines()
    for idx, line in enumerate(lines):
        if "_light_score(prompt_lower, quality_anchors) > 0.03" in line and "has_quality_request =" in line:
            window = "\n".join(lines[idx:idx + 10])
            assign_after = "_measured_quality_request = has_quality_request" in window
            break
    check("medida_guardada_junto_al_calculo", assign_after)

    # 2) force_global usa la medida, no el valor contaminado.
    force_global_ok = False
    force_src = ""
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "force_global" in targets:
                force_src = ast.unparse(node)
                force_global_ok = (
                    "_measured_quality_request" in force_src
                    and "has_quality_request" not in force_src.split("high_structural")[0]
                )
    check("force_global_usa_medida", force_global_ok, force_src[:120])

    # 2b) Ninguna otra condición de enrutado (skip_focal_cloth / print
    # compound) usa el valor contaminado para decidir la RUTA focal→global.
    compound_print_measured = "_measured_quality_request and mag_suggested" in src
    check("print_compound_usa_medida", compound_print_measured)

    # 3) Las cláusulas de prompt del híbrido conservan el valor amplio
    #    (`has_quality_request or quality_enhance`), que es deliberado.
    check("analisis_conserva_valor_amplio", '"has_quality_request": has_quality_request or quality_enhance' in src)

    # 4) Funcional: ejecutar la decisión real con los anchors del fuente.
    from roop.img_editor.img_editor_manager import ImgEditorManager

    anchors = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if "quality_anchors" in targets:
                anchors = ast.literal_eval(node.value)
                break
    check("anchors_extraidos_del_fuente", isinstance(anchors, list) and len(anchors) >= 4, f"{len(anchors or [])} anchors")

    class _Stub:
        _STOPWORDS = ImgEditorManager._STOPWORDS

    light = lambda text: ImgEditorManager._light_score(_Stub(), text, anchors)

    # La instrucción del log NO pide calidad → no fuerza global.
    bald = light("man must be bald..")
    check("instruccin_sin_calidad_no_fuerza_global", bald <= 0.03, f"score={bald:.3f}")
    # Una petición de calidad REAL en el texto sí sigue forzando global.
    quality = light("improve quality sharpness detail resolution clarity enhance")
    check("peticin_de_calidad_s_fuerza_global", quality > 0.03, f"score={quality:.3f}")
    # Y en español, como la escribe el usuario.
    quality_es = light("mejorar calidad nitidez detalle")
    check("peticin_de_calidad_es_s_fuerza_global", quality_es > 0.03, f"score={quality_es:.3f}")

    fails = [n for n, ok, _ in results if not ok]
    print(f"\nRESULTADO: {'PASS' if not fails else 'FAIL'} ({len(fails)} fallos)")
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
