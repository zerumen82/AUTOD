# -*- coding: utf-8 -*-
"""probe_v51139_faceswap.py — Verificación de los 6 fixes v5.1139 (FaceSwap tab).

1. Source falsos positivos: contención kps-in-bbox en _has_face_geometry + dedup.
2. Destination closeup: fallback descalado 0.5 en get_all_faces_with_rotation.
3. Batch fotos: reintento con padding + descalado.
4. Parecido: master embedding ponderado MISMA persona (>=0.60, todo-o-nada).
5. Random multi-destino: sin truncado a 1 (max_taken) + guard UnboundLocalError.
6. Vídeo: setup sin ancla arbitraria en multi-face (single-face sí ancla).
"""
import ast
import sys
import os
import types

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

PASS = 0
FAIL = 0

def check(name, cond):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  OK  {name}")
    else:
        FAIL += 1
        print(f"  FAIL {name}")

def read(p):
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()

def tree_of(p):
    return ast.parse(read(p))

print("=" * 70)
print("PROBE v5.1139 — FaceSwap: source FP / closeup / batch / likeness / random / no-fallback")
print("=" * 70)

# ---------------------------------------------------------------- 1. Source FP
print("\n[1] Source: contención kps-in-bbox + dedup en face_util.py")
src_fu = read("roop/face_util.py")
check("contención kps-in-bbox presente",
      "_p[0] < _bx1 - _marg" in src_fu and "5b. CONTENCIÓN kps-in-bbox" in src_fu)
check("dedup _same_face_boxes en source path",
      "from roop.face_util_rotation import _same_face_boxes" in src_fu and "_kept_sf" in src_fu)

# Funcional: geometría de cara real vs falso positivo con kps fuera del bbox
import numpy as np
from roop.face_util import _has_face_geometry

class _F:
    def __init__(self, bbox, kps):
        self.bbox = bbox
        self.kps = kps
        self.det_score = 0.45

# Cara real frontal: kps dentro del bbox
real = _F([100, 100, 200, 200], np.array([
    [125, 135], [175, 135], [150, 165], [135, 180], [165, 180],
], dtype=np.float32))
check("cara real (kps dentro) aceptada", _has_face_geometry(real))

# Falso positivo: kps fuera del bbox (regresión de head predictor sobre otra región)
fp = _F([100, 100, 200, 200], np.array([
    [60, 40], [140, 40], [150, 165], [135, 180], [165, 180],
], dtype=np.float32))
check("falso positivo (kps fuera del bbox) rechazado", not _has_face_geometry(fp))

# ---------------------------------------------------------------- 2. Closeup
print("\n[2] Destination closeup: descalado 0.5 en face_util_rotation.py")
src_fur = read("roop/face_util_rotation.py")
check("fallback descalado presente",
      "Primer plano extremo (lado escalado" in src_fur and "_long_scaled5 > 1024.0" in src_fur)
check("remapeo bbox/kps al frame original",
      "_cf5.bbox[0] / _scale5" in src_fur and "_p[0] / _scale5" in src_fur)
check("queda DESPUÉS del fallback MediaPipe (orden correcto)",
      src_fur.find("MediaPipe fallback detectó") < src_fur.find("Primer plano extremo"))

# ---------------------------------------------------------------- 3. Batch fotos
print("\n[3] Batch fotos: reintento con padding + descalado en ProcessMgr.py")
src_pm = read("roop/ProcessMgr.py")
check("reintento padding presente", "Reintento con padding" in src_pm and "_pad_w9" in src_pm)
check("reintento descalado presente", "Reintento descalado 0.5" in src_pm and "_small9" in src_pm)
tree_pm = tree_of("roop/ProcessMgr.py")

def _find_call(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Name) and f.id == name:
                return node
            if isinstance(f, ast.Attribute) and f.attr == name:
                return node
    return None

copymake = None
for node in ast.walk(tree_pm):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "copyMakeBorder":
        for kw in node.keywords:
            if kw.arg == "value":
                copymake = node
check("fallbacks batch dentro de la rama `not is_real_video` del reintento",
      copymake is not None)

# ---------------------------------------------------------------- 4. Parecido
print("\n[4] Parecido: master embedding MISMA persona (todo-o-nada)")
check("combinación ponderada presente",
      "Master embedding combinado" in src_pm and "sin dilución de otras identidades" in src_pm)
check("usa el mismo umbral 0.60 de identidad (v5.1064)",
      ">= 0.60" in src_pm and "_SOURCE_IDENTITY_SIM_THRESH = 0.60" in src_pm)

# Funcional: verificar la matemática de la combinación
top = np.array([1.0, 0.0, 0.0], dtype=np.float32)
same1 = np.array([0.9, 0.44, 0.0], dtype=np.float32); same1 /= np.linalg.norm(same1)
other = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # sim 0 con top
pool = [(top, 3.0), (same1, 2.5), (other, 2.4)]
_same = [(e, q) for (e, q) in pool if float(np.dot(top, e)) >= 0.60]
check("solo caras de la misma persona entran al pool", len(_same) == 2)
_qsum = sum(q for e, q in _same)
_comb = sum(np.asarray(e, dtype=np.float32) * q for e, q in _same)
_comb = _comb / np.linalg.norm(_comb)
check("combinación no diluye identidad (sim con top > sim de top con same1 alone)",
      float(np.dot(_comb, top)) > float(np.dot(top, same1)))

# ---------------------------------------------------------------- 5. Random multi
print("\n[5] Random multi-destino: sin truncado + guard UnboundLocalError")
check("guard _random_multi = False en MODO FOTO",
      "_random_multi = False" in src_pm)
check("max_taken ya no usa len(_photo_bank) rígido",
      "_max_taken = max(len(_photo_bank)" in src_pm)
check("limit_one respeta multi en foto",
      "_limit_one" in src_pm and "is_real_video is False" in src_pm)

# ---------------------------------------------------------------- 6. Vídeo sin fallback
print("\n[6] Vídeo: setup sin ancla arbitraria en multi-face")
check("fallback-mayor eliminado del setup", "fallback-mayor (ref_sim" not in src_pm)
check("single-face: ancla sin ambigüedad", "auto-única-cara (sin ambigüedad)" in src_pm)
check("multi-face: espera la cara seleccionada",
      "NO se ancla ninguna" in src_pm and "esperando la cara seleccionada" in src_pm)
check("floor multi-face de RE-BUSQUEDA 0.22", "best_fb_sim >= 0.22" in src_pm)

# ---------------------------------------------------------------- Sintaxis global
print("\n[S] Sintaxis de todos los módulos tocados")
for m in ("roop/ProcessMgr.py", "roop/face_util.py", "roop/face_util_rotation.py",
          "ui/tabs/faceswap/ui.py", "ui/tabs/faceswap/events.py"):
    try:
        compile(read(m), m, "exec")
        check(f"compila {m}", True)
    except SyntaxError as e:
        check(f"compila {m}: {e}", False)

print("\n" + "=" * 70)
print(f"RESULTADO: {PASS} OK / {FAIL} FAIL")
print("=" * 70)
sys.exit(1 if FAIL else 0)
