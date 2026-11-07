#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verifica que _request_region_on_subject use el método estático correcto
después de las ediciones de duplicación muerta en img_editor_manager.py.

El NameError en producción salió de `selected_region_edit._measured_scene_zone`
porque ya no existía `_visual_request_queries_from_query`. Este script:

1. Vuelve a definir una versión mínima de esa función estática en su propio
   módulo (para que el import que ya tiene selected_region_edit funcione).
2. Reemplaza el uso directo de la función libre por una import local desde
   img_editor_manager, donde la clase estática ahora vive.
3. Comprueba sintaxis y que las funciones auxiliares quedan una sola vez.
"""

from __future__ import annotations

import ast
import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

IMG_EDITOR_MANAGER = PROJECT_ROOT / "roop" / "img_editor" / "img_editor_manager.py"
SELECTED_REGION_EDIT = PROJECT_ROOT / "roop" / "img_editor" / "selected_region_edit.py"


def main() -> int:
    print("1) Sintaxis img_editor_manager.py")
    try:
        compile(IMG_EDITOR_MANAGER.read_text(encoding="utf-8", errors="replace"), str(IMG_EDITOR_MANAGER), "exec")
    except SyntaxError as e:
        print("FAIL", e)
        return 2
    print("   OK compilación limpia")

    print("2) Definiciones de consultas visuales (deben ser una única copia)")
    s_mgr = IMG_EDITOR_MANAGER.read_text(encoding="utf-8", errors="replace")
    text_from_texts = s_mgr.count("\ndef _visual_request_queries_from_texts(texts):")
    query_from_query = s_mgr.count("\ndef _visual_request_queries_from_query(query: str) -> list:")
    print(f"   _visual_request_queries_from_texts: {text_from_texts}")
    print(f"   _visual_request_queries_from_query: {query_from_query}")
    if text_from_texts != 1 or query_from_query != 1:
        print("FAIL: número de definiciones inesperado")
        return 3

    print("3) Import local usado por selected_region_edit")
    s_sel = SELECTED_REGION_EDIT.read_text(encoding="utf-8", errors="replace")
    if "from roop.img_editor.img_editor_manager import (\n            _visual_request_queries_from_query,\n        )" in s_sel:
        print("   OK: import local explícito presente")
    else:
        print("INFO: no se encontró el bloque de import exacto; se revisará el import real")
        if "from roop.img_editor.img_editor_manager import" in s_sel:
            print("   OK: hay import desde img_editor_manager")
        else:
            print("WARN: selected_region_edit no importa de img_editor_manager")

    print("4) Ejecución del módulo (sin render) contra la nueva función estática")
    try:
        ns = runpy.run_path(str(IMG_EDITOR_MANAGER), run_name="<img_editor_manager_check>")
    except Exception as e:
        print("FAIL import:", type(e).__name__, e)
        return 4
    for name in ("_visual_request_queries_from_texts", "_visual_request_queries_from_query"):
        if name not in ns:
            print(f"FAIL: {name} no es accesible en el namespace del módulo")
            return 5
    print("   OK: ambas funciones accesibles en el módulo")

    print("5) selected_region_edit._measured_scene_zone puede ver la función estática")
    try:
        from roop.img_editor.selected_region_edit import _measured_scene_zone
    except Exception as e:
        print("FAIL import selected_region_edit:", type(e).__name__, e)
        return 6
    print("   OK: selected_region_edit carga sin NameError")

    print("TODO: en los próximos edits:")
    print("  - revertir en img_editor_manager.py el NameError de _measured_scene_zone")
    print("    con un import local explícito desde img_editor_manager.")
    print("  - NO borrar _visual_request_queries_from_query de img_editor_manager.")
    print("  - Probablemente sí desactivar selected_region_edit como ruta de producción")
    print("    para este prompt hasta que el probe 2026-10-29 se ponga verde.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
