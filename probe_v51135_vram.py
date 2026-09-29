#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe v5.1135 — liberación de VRAM del proceso principal en el pipeline TOP.

Checks (AST sobre los módulos editados):
  T1  top_tier_finish._generative_polish_pass libera CodeFormer (_preserve_faces._cf = None) ANTES de client.load
  T2  top_tier_finish._generative_polish_pass llama release_gpu_caches ANTES de client.load
  T3  img_editor_manager: el bloque de calidad llama _release_local_gpu_before_comfy antes de run_top_tier_quality
  T4  flux_edit_comfy_client: guard de VRAM usa umbral 2200 (no 1500)
  T5  flux_edit_comfy_client: no queda ningún '< 1500' activo
"""
import ast
import os

BASE = os.path.dirname(os.path.abspath(__file__))
RESULTS = []


def check(name, ok):
    RESULTS.append((name, bool(ok)))
    print(f"{'OK ' if ok else 'FAIL'} {name}")


def _ast_of(path):
    with open(os.path.join(BASE, path), encoding="utf-8") as f:
        return ast.parse(f.read())


def _find_func(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    return None


def _line_of(tree, fn_name, needle):
    """1-based line of the first statement containing needle inside function fn_name."""
    fn = _find_func(tree, fn_name)
    if fn is None:
        return None, None
    for node in ast.walk(fn):
        if isinstance(node, ast.Call):
            src = ast.dump(node.func)
            if needle in src:
                return node.lineno, fn.lineno
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if needle in ast.dump(t):
                    return node.lineno, fn.lineno
    return None, fn.lineno if fn else None


# T1/T2 — top_tier_finish
tt = _ast_of("roop/img_editor/top_tier_finish.py")
fn = _find_func(tt, "_generative_polish_pass")
src_lines = []
with open(os.path.join(BASE, "roop/img_editor/top_tier_finish.py"), encoding="utf-8") as f:
    src_lines = f.readlines()

# locate load call and release statements by line numbers
load_line = rel_line = cf_line = finisher_line = None
in_fn = False
for i, line in enumerate(src_lines, 1):
    if line.startswith("def _generative_polish_pass"):
        in_fn = True
    elif in_fn and line.startswith("def ") and not line.startswith("def _generative_polish_pass"):
        break
    if in_fn:
        if "client.load(version)" in line and load_line is None:
            load_line = i
        if '_preserve_faces._cf = None' in line and cf_line is None:
            cf_line = i
        if "release_gpu_caches()" in line and finisher_line is None:
            finisher_line = i

check("T1 CodeFormer liberado dentro de _generative_polish_pass", cf_line is not None)
check("T2 release_gpu_caches dentro de _generative_polish_pass", finisher_line is not None)
check("T1b releases ANTES de client.load", (cf_line and load_line and cf_line < load_line) and (finisher_line and finisher_line < load_line))

# T3 — manager quality block calls release before run_top_tier_quality
mg = _ast_of("roop/img_editor/img_editor_manager.py")
with open(os.path.join(BASE, "roop/img_editor/img_editor_manager.py"), encoding="utf-8") as f:
    mg_lines = f.readlines()
release_line = rttq_line = None
for i, line in enumerate(mg_lines, 1):
    if "self._release_local_gpu_before_comfy()" in line and i > 2500:
        release_line = i
        break
for i, line in enumerate(mg_lines, 1):
    if "run_top_tier_quality(" in line and "from " not in line and "import" not in line and "def " not in line:
        rttq_line = i
        break
check("T3 release local antes de run_top_tier_quality", release_line is not None and rttq_line is not None and release_line < rttq_line)

# T4/T5 — client threshold
fc = _ast_of("roop/img_editor/flux_edit_comfy_client.py")
with open(os.path.join(BASE, "roop/img_editor/flux_edit_comfy_client.py"), encoding="utf-8") as f:
    fc_lines = f.readlines()
low_lines = [i for i, l in enumerate(fc_lines, 1) if "_pre_vram_free <" in l]
vals = []
for i in low_lines:
    for tok in ("< 1500", "< 2200"):
        if tok in fc_lines[i - 1]:
            vals.append((i, tok))
check("T4 guard usa 2200", any(v[1] == "< 2200" for v in vals))
check("T5 sin restos de 1500 en guards", all(v[1] != "< 1500" for v in vals))

n_ok = sum(1 for _, ok in RESULTS if ok)
print(f"\n{n_ok}/{len(RESULTS)} OK")
raise SystemExit(0 if n_ok == len(RESULTS) else 1)
