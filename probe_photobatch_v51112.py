# -*- coding: utf-8 -*-
"""probe_photobatch_v51112.py — Verificación AST de los 3 fixes del batch de fotos.

1. FIX A: reintento de detección con preproceso UI (mismo pipeline que la UI)
   cuando valid_faces queda vacío en modo foto (is_real_video guard).
2. FIX B: fallback TARGET_FACES prueba el MEJOR match entre todos los destinos
   (no solo TARGET_FACES[0]).
3. FIX C: log [FOTO] SKIP en los 2 caminos de fallo silencioso (match débil y
   sin match) — el fallo ya no sale sin traza.
"""
import ast
import sys

PATH = "roop/ProcessMgr.py"

with open(PATH, "r", encoding="utf-8") as f:
    src = f.read()

tree = ast.parse(src)
results = []


def check(name, cond):
    results.append((name, bool(cond)))


# ---------- FIX A ----------
check("FIX A: reintento con detect_faces_robust presente",
      "detect_faces_robust" in src and "preprocess_image_for_detection" in src)
check("FIX A: guard solo modo foto (not is_real_video)",
      "not valid_faces and not is_real_video" in src)
check("FIX A: thresh 0.15 igual que la UI", "0.15" in src)
check("FIX A: log de caras recuperadas",
      "Reintento detección con preproceso UI" in src)
check("FIX A: skip con log cuando 0 caras en ambos pases",
      "0 caras detectadas (ni crudo ni preproceso)" in src)

# ---------- FIX B ----------
check("FIX B: busca mejor match entre todos los TARGET_FACES",
      "for _tf in target_faces:" in src)
check("FIX B: usa _calculate_similarity contra valid_faces",
      "for _vf in valid_faces:" in src)
check("FIX B: umbral 0.28 en fallback", "_best_sim > 0.28" in src)
check("FIX B: log del mejor match elegido",
      "mejor match de" in src and "destinos UI" in src)

# ---------- FIX C ----------
check("FIX C: log SKIP match debil contra referencia",
      "match débil contra su referencia" in src)
check("FIX C: log SKIP sin match entre destinos y caras",
      "sin match" in src and "foto original sin swap" in src)

# ---------- sanity estructural ----------
# El reintento debe estar ANTES del bloque de modo foto (que referencia valid_faces)
idx_retry = src.find("Reintento detección con preproceso UI")
idx_foto = src.find("MODO FOTO: Respetar si el usuario seleccion")
check("estructura: reintento antes del bloque MODO FOTO",
      0 < idx_retry < idx_foto)

# El guard del reintento debe estar antes de 'if not valid_faces:' hold-video
idx_skip = src.find("0 caras detectadas (ni crudo ni preproceso)")
idx_hold = src.find("if not faces_to_process:")
check("estructura: skip-log antes del hold de video", 0 < idx_skip < idx_hold)

# AST: confirmar que el bloque del reintento está dentro de process_frame
found_retry_in_process_frame = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "process_frame":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if "Reintento detección con preproceso UI" in sub.value:
                    found_retry_in_process_frame = True
check("AST: reintento vive dentro de process_frame", found_retry_in_process_frame)

found_skip_in_process_frame = False
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "process_frame":
        for sub in ast.walk(node):
            if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                if "0 caras detectadas (ni crudo ni preproceso)" in sub.value:
                    found_skip_in_process_frame = True
check("AST: skip-log vive dentro de process_frame", found_skip_in_process_frame)

ok = all(r for _, r in results)
for name, passed in results:
    print(f"{'OK ' if passed else 'FAIL'}  {name}")
print(f"\n{sum(r for _, r in results)}/{len(results)} checks OK")
sys.exit(0 if ok else 1)
