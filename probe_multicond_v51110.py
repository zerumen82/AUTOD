# -*- coding: utf-8 -*-
"""Probe v5.1110: prompts compuestos en TODAS las rutas (no solo uncover).
1. _discard_rewrite_text: no-EN → SIEMPRE descarta (change, cover, kontext, cualquier engine).
2. _count_conditions / _translation_lost_conditions: detecta pérdida de mitades.
"""
import sys, ast, io

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
ok = 0
total = 0

def check(name, cond):
    global ok, total
    total += 1
    if cond:
        ok += 1
        print(f"  OK  {name}")
    else:
        print(f"  FAIL {name}")

print("=== 1. AST: gate sin branching por dirección/engine ===")
src = open("roop/img_editor/img_editor_manager.py", encoding="utf-8").read()
tree = ast.parse(src)
fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_discard_rewrite_text":
        fn = node
        break
check("funcion existe", fn is not None)
body_src = ast.get_source_segment(src, fn)
# El cuerpo ejecutable no debe ramificar por direccion ni engine (los
# comentarios explican la historia, por eso se parsea el AST real:
# solo debe haber el return de _is_english).
has_branch = any(isinstance(n, (ast.If, ast.IfExp)) for n in ast.walk(fn))
returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
check("cero ifs ejecutables (solo return _is_english)", not has_branch and len(returns) == 1
      and isinstance(returns[0].value, ast.UnaryOp))

print("=== 2. Funcional: gate general ===")
from roop.img_editor.img_editor_manager import ImgEditorManager
d = ImgEditorManager._discard_rewrite_text
# ES "desnuda y descalza" en TODAS las combinaciones → descarta
check("ES uncover+longcat", d("debe ir desnuda y descalza", "uncover", "longcat") is True)
check("ES change+kontext", d("ponle falda azul y sombrero", "change", "kontext") is True)
check("ES cover+kontext", d("abrígoala de nieve y botas", "cover", "kontext") is True)
check("ES change+klein", d("vestido rojo y pelo corto", "change", "klein") is True)
# EN → conserva (polish seguro)
check("EN conserva (uncover)", d("she must be naked and barefoot", "uncover", "longcat") is False)
check("EN conserva (change)", d("red dress and short hair", "change", "kontext") is False)
check("vacio → conserva", d("", "change", "longcat") is False)

print("=== 3. Contador de condiciones ===")
from roop.img_editor.prompt_translator import _count_conditions, _translation_lost_conditions
check("desnuda y descalza = 2", _count_conditions("debe ir desnuda y descalza") == 2)
check("3 condiciones = 3", _count_conditions("desnuda, descalza y sentada") == 3)
check("simple = 1", _count_conditions("hazla desnuda") == 1)
check("EN simple = 1", _count_conditions("she must be naked") == 1)

print("=== 4. Deteccion de perdida de mitad ===")
# API se comio la mitad de pies (caso log 12:12)
check("perdida detectada (2->1)", _translation_lost_conditions(
    "debe ir desnuda y descalza", "she must go completely naked") is True)
check("completo NO dispara (2->2)", _translation_lost_conditions(
    "debe ir desnuda y descalza", "she must go naked and barefoot") is False)
check("simple nunca dispara", _translation_lost_conditions(
    "hazla desnuda", "make her naked") is False)
check("vacio seguro", _translation_lost_conditions("", "") is False)

print(f"\n{ok}/{total} OK")
sys.exit(0 if ok == total else 1)
