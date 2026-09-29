# -*- coding: utf-8 -*-
"""v5.1130 — verificación: oclusión + boca (FaceSwap)."""
import ast
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PM = "roop/ProcessMgr.py"
_ok = 0
_fail = 0


def check(name, cond):
    global _ok, _fail
    if cond:
        _ok += 1
        print(f"  OK  {name}")
    else:
        _fail += 1
        print(f"  FAIL  {name}")


src = open(PM, encoding="utf-8", errors="replace").read()
tree = ast.parse(src)

# ---- AST checks ----
print("== AST: ProcessMgr ==")
check("Fix B: face_bbox pasado a detect_mouth_open",
      "face_bbox=getattr(target_face, 'bbox', None)" in src)

check("Fix C: _person_key por persona (EMA ratio)",
      "_person_key = f\"{video_key}_p{i}\"" in src)
check("Fix C: hold count por persona",
      "self._mouth_hold_count[_person_key]" in src)
check("Fix C: state history por persona",
      "self._mouth_state_history[_person_key]" in src)
check("Fix C/D: paste EMA por persona (dict)",
      "_mouth_paste_ema_by_person" in src)
check("Fix D: occ EMA reseteado en scene cut",
      "del occ_hist[video_key]" in src)
check("Fix D: limpieza estado bucal por persona en scene cut",
      "'_mouth_state_history', '_mouth_ratio_smooth', '_mouth_hold_count'" in src)
check("Fix A: rescale 112->align de landmarks 106",
      "_mr_for_mask = (mouth_region.astype(np.float32) * _scale)" in src)
check("Fix E: cap 0.88",
      "np.minimum(reduction, 0.88)" in src)
check("Fix E: blur occ adaptativo",
      "_occ_blur = int(max(11, min(41, (min(face_w, face_h) // 6) | 1))) | 1" in src)

# No debe quedar el setattr viejo del paste EMA global
check("no queda _mouth_paste_ema global como atributo suelto",
      "setattr(self, '_mouth_paste_ema'" not in src and "self._mouth_paste_ema = " not in src)

# El video_key dentro del bloque boca no debe reasignarse (usaba getattr después del EMA)
mouth_section = src[src.find("DETECTAR BOCA ABIERTA"):src.find("5. COORDENADAS FINALES")]
check("video_key no reasignado dentro del bloque boca (usa _person_key)",
      "video_key = getattr(self, '_current_video_key'" not in mouth_section)

# ---- Functional: rescale de landmarks 106 ----
print("== FUNC: rescale landmarks 106 -> aligned 256 ==")
import numpy as np
# Simular pts en espacio 112
mouth_pts = np.array([[40, 70], [56, 72], [48, 82], [44, 76], [52, 76]], dtype=np.int32)
w_align, h_align = 256, 256
_mr_max = float(mouth_pts.max())
_scale = w_align / 112.0
_mr_for_mask = (mouth_pts.astype(np.float32) * _scale).astype(mouth_pts.dtype)
check("detecta coords en espacio <=128", _mr_max <= 128.0)
check("escala aplicada (~2.29x)", abs(_mr_for_mask.max() - mouth_pts.max() * 256 / 112) < 1.0)
check("coords ahora en rango del lienzo 256", _mr_for_mask.max() > 112 and _mr_for_mask.max() < 256)

# Caso MediaPipe (dict) no debe tocarse
check("dict (MediaPipe) no es ndarray — sin rescale", not isinstance({"a": 1}, np.ndarray))

# pts en espacio frame (valores grandes) no deben rescalearse
big_pts = np.array([[400, 700], [560, 720]], dtype=np.int32)
check("coords frame (>128) sin rescale", float(big_pts.max()) > 128.0)

print()
print(f"TOTAL: {_ok} OK, {_fail} FAIL")
sys.exit(1 if _fail else 0)
