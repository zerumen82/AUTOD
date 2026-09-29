# -*- coding: utf-8 -*-
"""v5.1154: verificar la cadena REAL de source sobre la carpeta del usuario.
Cuenta caras aceptadas por extract_face_images(is_source_face=True).
"""
import sys, os, glob, time
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import roop.globals
# CFG minimo para la cadena de deteccion
if not hasattr(roop.globals, 'CFG') or roop.globals.CFG is None:
    class _CFG: pass
    roop.globals.CFG = _CFG()

from roop import face_util

CARPETA = sys.argv[1] if len(sys.argv) > 1 else r"D:\PROJECTS\uee\FACES\MARETA"
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # 0 = todas

files = sorted(
    glob.glob(os.path.join(CARPETA, "*.jpg")) +
    glob.glob(os.path.join(CARPETA, "*.jpeg")) +
    glob.glob(os.path.join(CARPETA, "*.png"))
)
if LIMIT:
    files = files[:LIMIT]

print(f"carpeta: {CARPETA} ({len(files)} archivos)")
t0 = time.time()
aceptadas = 0
rechazadas = 0
errores = 0
for i, fp in enumerate(files):
    try:
        faces = face_util.extract_face_images(fp, (False, True))  # is_source_face=True
        if faces:
            aceptadas += 1
        else:
            rechazadas += 1
    except Exception as e:
        errores += 1
        print(f"[ERR] {os.path.basename(fp)}: {e}")
    if (i + 1) % 100 == 0:
        print(f"  ... {i+1}/{len(files)} procesados ({time.time()-t0:.0f}s)")

dt = time.time() - t0
print(f"\n=== RESULTADO v5.1154 (cadena real, det>=0.14 + segunda pasada) ===")
print(f"  archivos        : {len(files)}")
print(f"  con cara aceptada: {aceptadas}")
print(f"  sin cara         : {rechazadas}")
print(f"  errores          : {errores}")
print(f"  tiempo           : {dt:.0f}s")
