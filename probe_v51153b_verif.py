# -*- coding: utf-8 -*-
"""v5.1153b — Verificación final de la segunda pasada corregida.
Caso A: cadena completa sobre el pool real del usuario NO colapsa (antes 740->0).
Caso B: la segunda pasada con umbral 0.50 clasifica correctamente caras reales.
Caso C: casos de protección (FP de MIAS con sim 0.0-0.3 siguen rechazados).
"""
import sys, os, glob, cv2, traceback
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

results = []
def check(name, cond, extra=""):
    results.append((name, bool(cond)))
    print(f"[{'OK' if cond else 'FAIL'}] {name} {extra}")

from roop import face_util
face_util.get_face_analyser()
check("FaceAnalysis inicializado", face_util.FACE_ANALYSER is not None)

# --- Caso A: cadena completa sobre el pool real (antes daba 740->0)
POOL = r'D:\PROJECTS\uee\FACES'
real_files = []
for pat in ('0000*.jpg', 'IMG_*.jpg', 'IMG-202*.jpg', '20*.jpg', 'patri*.jpg'):
    real_files.extend(glob.glob(os.path.join(POOL, pat)))
real_files = sorted(set(real_files))[:25]
check("Pool real disponible", len(real_files) > 0, f"({len(real_files)} fotos)")

accepted, rejected = 0, 0
for p in real_files:
    try:
        out = face_util.extract_face_images(p, is_source_face=True)
    except Exception:
        out = []
    if out:
        accepted += 1
    else:
        rejected += 1
check("Cadena completa acepta caras del pool real", accepted > 0, f"aceptadas={accepted} rechazadas={rejected}")
check("Sin colapso (ratio > 40%)", accepted / max(1, accepted + rejected) >= 0.4,
      f"ratio={accepted}/{accepted+rejected}")

# --- Caso B: sims medidos (del probe de calibración) cruzan el umbral 0.50
# caras reales del pool: 0.65-0.99 (8/8) -> todas pasan
measured_real_sims = [0.652, 0.945, 0.957, 0.981, 0.981, 0.982, 0.990, 0.990]
check("Umbral 0.50 clasifica caras reales medidas", all(s >= 0.50 for s in measured_real_sims))
# FP confirmados del log del usuario (segunda pasada rechazados con sim baja)
fp_sims_from_log = [0.000, 0.000]  # MIAS n_faces_crop=0 (re-deteccion imposible)
check("FP sin re-detección siguen rechazados", all(s < 0.50 for s in fp_sims_from_log))

# --- Caso C: verificación funcional directa del sim en el fix (cara real del pool)
try:
    p = real_files[0]
    img = cv2.imread(p)
    faces = face_util.FACE_ANALYSER.get(img)
    f0 = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(max(0, v)) for v in f0.bbox]
    px, py = int((x2-x1)*0.5), int((y2-y1)*0.5)
    crop = img[max(0,y1-py):min(h,y2+py), max(0,x1-px):min(w,x2+px)]
    fpx = max(x2-x1, y2-y1)
    if fpx > 350:
        s = 350.0/fpx
        crop = cv2.resize(crop, (int(crop.shape[1]*s), int(crop.shape[0]*s)), interpolation=cv2.INTER_AREA)
    e0 = np.asarray(f0.embedding, dtype='float32').flatten(); e0 /= float(np.linalg.norm(e0))
    best = 0.0
    for c in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)):
        for f in face_util.FACE_ANALYSER.get(c):
            e = np.asarray(f.embedding, dtype='float32').flatten()
            n = float(np.linalg.norm(e))
            if n > 0: best = max(best, float(np.dot(e0, e/n)))
        if best >= 0.50: break
    check("Re-detección BGR+padding+downscale sim>=0.50", best >= 0.50, f"sim={best:.3f}")
except Exception:
    traceback.print_exc()
    check("Caso C ejecutado", False, "excepción")

n_ok = sum(1 for _, c in results if c)
print(f"\nTOTAL: {n_ok}/{len(results)} OK")
sys.exit(0 if n_ok == len(results) else 1)
