# -*- coding: utf-8 -*-
"""Probe v5.1153 — Segunda pasada en source + match inequívoco en fotos.

Parte A (AST): verifica la verificación de segunda pasada en face_util.extract_face_images
  y el gate de ambigüedad en ProcessMgr.process_frame (rama foto, no-random).

Parte B (funcional): simula la verificación de segunda pasada con un FACE_ANALYSER
  mockeado — cara real aceptada, falso positivo (manos/texto) rechazado.
"""
import ast
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

fails = []


def check(name, ok):
    print(("OK   " if ok else "FAIL ") + name)
    if not ok:
        fails.append(name)


# ---------------- PARTE A: AST ----------------
src = open("roop/face_util.py", encoding="utf-8").read()
tree = ast.parse(src)
extract = None
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef) and node.name == "extract_face_images":
        extract = node
raw_e = ast.get_source_segment(src, extract) or ""

check("A1: segunda pasada presente en extract_face_images",
      "Segunda pasada" in raw_e and "_faces_v = FACE_ANALYSER.get(_crop_v)" in raw_e)
check("A2: umbral sim >= 0.55 contra embedding original", ">= 0.55" in raw_e)
check("A3: gate min 40px", "min(_fw_v, _fh_v) < 40" in raw_e)
check("A4: usa el crop de la IMAGEN (img_rgb), no el bbox acolchado",
      "_crop_v = img_rgb[_y1v:_y2v, _x1v:_x2v]" in raw_e)
check("A5: log de falso positivo persistente", "falso positivo persistente" in raw_e)
check("A6: sin enumerar llamadas de usuario en el bloque de verificación (no es API pública)",
      "enumerate" not in raw_e.split("Segunda pasada")[1].split("v5.1139: dedup")[0])
check("A7: cero vocabulario de usuario en el fix (solo estructural)",
      all(w not in raw_e.lower() for w in ("nude", "nsfw", "naked")))

pm = open("roop/ProcessMgr.py", encoding="utf-8").read()
check("A8: gate de ambigüedad en fotos (diff < 0.03 → SKIP)",
      "match ambiguo" in pm and "sim1=" in pm and "< 0.03" in pm)
# El gate ambiguo debe estar en la rama else (no-random) del bloque `if _good:`:
# la línea `else:` más cercana ANTES del gate debe ser más cercana que el `if _random_multi:`.
_m = pm.find("match ambiguo")
_rm = pm.rfind("if _random_multi:", 0, _m)
_el = pm.rfind("else:", 0, _m)
check("A9: el skip de ambigüedad está en la rama else no-random (tras if _random_multi)",
      _rm != -1 and _el != -1 and _el > _rm)

# ---------------- PARTE B: funcional ----------------
class _Face:
    def __init__(self, bbox, emb, det=0.5):
        self.bbox = np.array(bbox, dtype=np.float32)
        self.embedding = emb
        self.det_score = det
        self.kps = np.zeros((5, 2), dtype=np.float32) + 50.0


def make_emb(seed):
    rng = np.random.RandomState(seed)
    e = rng.randn(512).astype(np.float32)
    return e / np.linalg.norm(e)


REAL_EMB = make_emb(1)
FAKE_EMB = make_emb(2)


class FakeAnalyserReal:
    """El crop ES una cara: re-detecta con la misma identidad."""
    def get(self, img):
        return [_Face([10, 10, 110, 110], REAL_EMB, det=0.65)]


class FakeAnalyserFP:
    """Falso positivo: el crop NO re-detecta la misma identidad."""
    def get(self, img):
        return [_Face([10, 10, 110, 110], FAKE_EMB, det=0.60)]


class FakeAnalyserNone:
    def get(self, img):
        return []


def second_pass(faces, analyser, img_hw=(300, 300), w=300, h=300):
    """Réplica exacta de la lógica del fix."""
    verified = []
    for _f in faces:
        _fw_v = _f.bbox[2] - _f.bbox[0]
        _fh_v = _f.bbox[3] - _f.bbox[1]
        if min(_fw_v, _fh_v) < 40:
            verified.append(_f)
            continue
        _x1v, _y1v, _x2v, _y2v = [int(max(0, v)) for v in _f.bbox]
        _x2v, _y2v = min(w, _x2v), min(h, _y2v)
        _crop_v = img_hw and np.zeros((_y2v - _y1v, _x2v - _x1v, 3), dtype=np.uint8)
        if _crop_v.size == 0:
            continue
        _faces_v = analyser.get(_crop_v)
        _emb_orig = getattr(_f, 'embedding', None)
        _ok_v = False
        if _faces_v and _emb_orig is not None:
            _emb_o = np.asarray(_emb_orig, dtype=np.float32).flatten()
            _no = np.linalg.norm(_emb_o)
            if _no > 0:
                _emb_o = _emb_o / _no
                for _fv in _faces_v:
                    _ev = getattr(_fv, 'embedding', None)
                    if _ev is None:
                        continue
                    _ev = np.asarray(_ev, dtype=np.float32).flatten()
                    _nv = np.linalg.norm(_ev)
                    if _nv > 0 and float(np.dot(_emb_o, _ev / _nv)) >= 0.55:
                        _ok_v = True
                        break
        if _ok_v:
            verified.append(_f)
    return verified


real = _Face([100, 100, 220, 240], REAL_EMB, det=0.55)
fp = _Face([40, 40, 180, 200], FAKE_EMB, det=0.45)
tiny = _Face([300, 300, 335, 335], FAKE_EMB, det=0.45)

out = second_pass([real, fp], FakeAnalyserReal())
check("B1: cara real aceptada + falso positivo (otra identidad en re-detect) rechazado",
      len(out) == 1 and out[0] is real)

out = second_pass([real], FakeAnalyserNone())
check("B2: crop sin re-detección → descartado", len(out) == 0)

out = second_pass([tiny], FakeAnalyserNone())
check("B3: cara <40px pasa sin segunda pasada (gates previos mandan)", len(out) == 1)

out = second_pass([real], FakeAnalyserFP())
check("B4: identidad distinta en re-detect (sim<0.55) → descartada",
      len(out) == 0)

# ---------------- Parte C: match inequívoco ----------------
def photo_match_logic(sims, thresh=0.28, ambiguity=0.03):
    good = sorted([s for s in sims if s > thresh], reverse=True)
    if not good:
        return "skip-debil"
    if len(good) >= 2 and good[0] - good[1] < ambiguity:
        return "skip-ambiguo"
    return "swap"


check("C1: sims 0.31/0.30/0.29 (ruido entre personas) → SKIP ambiguo",
      photo_match_logic([0.31, 0.30, 0.29]) == "skip-ambiguo")
check("C2: sims 0.62/0.30 → swap en la cara correcta",
      photo_match_logic([0.62, 0.30]) == "swap")
check("C3: una sola cara buena → swap",
      photo_match_logic([0.55]) == "swap")
check("C4: todas débiles → skip débil (comportamiento previo intacto)",
      photo_match_logic([0.15, 0.10]) == "skip-debil")

print()
if fails:
    print(f"probe_v51153_source_foto: {len(fails)} FAIL")
    sys.exit(1)
print("probe_v51153_source_foto: TODO OK")
