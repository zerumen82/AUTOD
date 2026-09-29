#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe v5.1131b — NameError `i` en _process_face_swap_v21 (0 swaps en batch).

RAÍZ: v5.1130 introdujo keys por persona (`f"{video_key}_p{i}"`) para el EMA
bucal/histéresis/hold/paste DENTRO de _process_face_swap_v21, pero `i` SOLO
existe en el loop del caller (`for i, target_face in enumerate(...)`) →
NameError en CADA frame → el except del caller lo traga → frame SIN swap.
Log 2026-09-15: 'name \\'i\\' is not defined' l.4109 en todos los frames.

Checks (AST sobre roop/ProcessMgr.py):
  T1  firma de _process_face_swap_v21 incluye person_idx
  T2  NINGÚN Name `i` libre dentro de _process_face_swap_v21
  T3  call site pasa person_idx=i
  T4  cero literales `p{i}` residuales en el módulo
  T5  TODAS las keys por persona usan person_idx (AST de JoinedStr)
  T6  py_compile del módulo OK
"""
import ast
import os
import py_compile
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
PATH = os.path.join(BASE, "roop", "ProcessMgr.py")
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"{'OK ' if ok else 'FAIL'} {name}")


with open(PATH, encoding="utf-8") as f:
    src = f.read()
tree = ast.parse(src)

fn = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "_process_face_swap_v21":
        fn = node
        break

# T1 — firma
args = [a.arg for a in fn.args.args] if fn else []
check("T1 firma tiene person_idx", fn is not None and "person_idx" in args)

# T2 — sin `i` libre dentro de la función
names = set()
if fn:
    for node in ast.walk(fn):
        if isinstance(node, ast.Name):
            names.add(node.id)
check("T2 sin Name `i` dentro de v21", "i" not in names)

# T3 — call site pasa person_idx
call_ok = False
call_count = 0
for node in ast.walk(tree):
    if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "_process_face_swap_v21":
        call_count += 1
        if any(k.arg == "person_idx" for k in node.keywords):
            call_ok = True
check("T3 call site pasa person_idx", call_ok and call_count >= 1)

# T4 — sin `p{i}` literales residuales
check("T4 cero p{i} literales", "_p\\{i\\}" not in src.replace("\\", ""))

# T5 — ninguna key por persona formatea `i` (el bug exacto de v5.1130).
# Las f-strings de PREFIJO (startswith(f"{video_key}_p")) en el reset del
# scene cut son legítimas: iteran TODAS las personas por prefijo.
person_keys_found = 0
bad_i_keys = 0
for node in ast.walk(tree):
    if isinstance(node, ast.JoinedStr):
        consts = [
            p.value for p in node.values
            if isinstance(p, ast.Constant) and isinstance(p.value, str)
        ]
        if not any(c.endswith("_p") for c in consts):
            continue
        person_keys_found += 1
        if any(
            isinstance(p, ast.FormattedValue)
            and isinstance(p.value, ast.Name)
            and p.value.id == "i"
            for p in node.values
        ):
            bad_i_keys += 1
check(
    f"T5 keys por persona sin `i` ({bad_i_keys} bad de {person_keys_found})",
    person_keys_found > 0 and bad_i_keys == 0,
)

# T6 — compila
try:
    py_compile.compile(PATH, doraise=True)
    check("T6 py_compile OK", True)
except Exception as e:
    check(f"T6 py_compile OK ({e})", False)

n_ok = sum(1 for _, ok in RESULTS if ok)
print(f"\n{n_ok}/{len(RESULTS)} OK")
sys.exit(0 if n_ok == len(RESULTS) else 1)
