# -*- coding: utf-8 -*-
"""v5.1154b: verificar archivos EXACTOS que fallaron en el log 16:54 (padding + segunda pasada).
Con el codigo ACTUAL (det 0.14 + padding/downscale segunda pasada)."""
import sys, os
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roop.globals
if not hasattr(roop.globals, 'CFG') or roop.globals.CFG is None:
    class _CFG: pass
    roop.globals.CFG = _CFG()

from roop import face_util

# Archivos que fallaron en el log 16:54 (padding -> segunda pasada sim=0.000 n=0,
# y det_bajo < 0.22)
FALLOS_LOG = [
    "00000.jpg",           # padding, sim=0.000 n=0
    "00011.jpg",           # padding, sim=0.000 n=0
    "00213.jpg",           # padding, sim=0.000 n=0
    "00761.jpg",           # padding, sim=0.000 n=0
    "01467.jpg",           # padding, sim=0.000 n=0
    "[002497]_0.jpg",      # padding, sim=0.000 n=0
    "[002550]_0.jpg",      # padding, sim=0.000 n=0
    "[006802].jpg",        # padding, sim=0.000 n=0
    "[011638].jpg",        # directo, n=1 sim=0.000
    "00173.jpg",           # det=0.11 < 0.22 (cara 194x260 real?)
    "[002469]_0.jpg",      # det=0.12 < 0.22 (cara 159x183)
    "00831.jpg",           # det=0.19 < 0.22 (cara 226x401)
    "P1070467_0.jpg",      # det=0.19 < 0.22 (cara 188x212)
]

CARPETA = sys.argv[1] if len(sys.argv) > 1 else r"D:\PROJECTS\uee\FACES\MARETA"
ok, fail, missing = [], [], []
for name in FALLOS_LOG:
    fp = os.path.join(CARPETA, name)
    if not os.path.exists(fp):
        missing.append(name)
        continue
    try:
        faces = face_util.extract_face_images(fp, (False, True))
        if faces:
            det = float(getattr(faces[0], 'det_score', 0.0) or 0.0)
            ok.append((name, det, len(faces)))
            print(f"OK   {name}: {len(faces)} cara(s) det={det:.2f}")
        else:
            fail.append(name)
            print(f"FAIL {name}: 0 caras")
    except Exception as e:
        fail.append(name)
        print(f"ERR  {name}: {e}")

print(f"\n=== RESUMEN (codigo actual) ===")
print(f"  OK    : {len(ok)}/{len(FALLOS_LOG)}")
print(f"  FAIL  : {len(fail)} {fail}")
print(f"  missing: {len(missing)} {missing}")
