# -*- coding: utf-8 -*-
"""Diagnóstico: por qué el source del usuario da pocas caras.
Reproduce los filtros y cuenta razones de rechazo sobre su carpeta real."""
import sys, os, glob, cv2, traceback
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roop import face_util
face_util.get_face_analyser()
FA = face_util.FACE_ANALYSER

SRC = sys.argv[1] if len(sys.argv) > 1 else r'D:\PROJECTS\uee\FACES\MIAS'
files = []
for pat in ('*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG'):
    files.extend(glob.glob(os.path.join(SRC, pat)))
files = sorted(set(files))
print(f"carpeta: {SRC} ({len(files)} archivos)")

stats = {
    'ok': 0, 'sin_deteccion': 0, 'det_bajo': 0, 'pequena': 0, 'aspecto': 0,
    'sin_geometria': 0, 'incompleta': 0, 'segunda_pasada': 0,
}
det_by_reject = {'det_bajo': [], 'pequena': [], 'incompleta': []}

def is_complete(f, w, h):
    try:
        return face_util._is_face_complete(f, w, h)
    except Exception:
        return True

for p in files[:200]:
    try:
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        faces = FA.get(img)
        if not faces:
            # reintentar con preproceso como hace el pipeline
            try:
                from roop.face_util import preprocess_image_for_detection
                pre = preprocess_image_for_detection(img)
                faces = FA.get(pre)
            except Exception:
                faces = []
        if not faces:
            stats['sin_deteccion'] += 1
            continue
        accepted_any = False
        for f in faces:
            det = float(getattr(f, 'det_score', 0.0) or 0.0)
            fw = float(f.bbox[2] - f.bbox[0]); fh = float(f.bbox[3] - f.bbox[1])
            aspect = fw / fh if fh > 0 else 0
            if det < 0.22:
                stats['det_bajo'] += 1
                det_by_reject['det_bajo'].append((det, fw, os.path.basename(p)))
                continue
            if not (fw > 24 and fh > 24):
                stats['pequena'] += 1
                det_by_reject['pequena'].append((det, fw, os.path.basename(p)))
                continue
            if not (0.25 < aspect < 3.5):
                stats['aspecto'] += 1
                continue
            if not face_util._has_face_geometry(f, max_tilt=45.0):
                stats['sin_geometria'] += 1
                continue
            if not is_complete(f, w, h):
                stats['incompleta'] += 1
                det_by_reject['incompleta'].append((det, fw, os.path.basename(p)))
                continue
            # segunda pasada (misma lógica del fix)
            x1, y1, x2, y2 = [int(max(0, v)) for v in f.bbox]
            px, py = int((x2 - x1) * 0.5), int((y2 - y1) * 0.5)
            crop = img[max(0, y1 - py):min(h, y2 + py), max(0, x1 - px):min(w, x2 + px)]
            if crop.size == 0:
                stats['segunda_pasada'] += 1
                continue
            fpx = max(x2 - x1, y2 - y1)
            if fpx > 350:
                s = 350.0 / fpx
                crop = cv2.resize(crop, (max(1, int(crop.shape[1]*s)), max(1, int(crop.shape[0]*s))), interpolation=cv2.INTER_AREA)
            e0 = np.asarray(getattr(f, 'embedding', None), dtype='float32').flatten() if getattr(f, 'embedding', None) is not None else None
            ok = False
            if e0 is not None and e0.size > 0:
                n0 = float(np.linalg.norm(e0))
                if n0 > 0:
                    e0 /= n0
                    fl = []
                    for c in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)):
                        try: fl = FA.get(c)
                        except Exception: fl = []
                        if fl: break
                    for fv in fl:
                        ev = getattr(fv, 'embedding', None)
                        if ev is None: continue
                        ev = np.asarray(ev, dtype='float32').flatten()
                        nv = float(np.linalg.norm(ev))
                        if nv > 0 and float(np.dot(e0, ev / nv)) >= 0.50:
                            ok = True; break
            if ok:
                accepted_any = True
            else:
                stats['segunda_pasada'] += 1
        if accepted_any:
            stats['ok'] += 1
    except Exception:
        traceback.print_exc()

print("\n=== RAZONES DE RECHAZO (por cara candidata) ===")
for k, v in stats.items():
    print(f"  {k:16s}: {v}")
print("\n=== MUESTRAS det_bajo ===")
for det, fw, name in sorted(det_by_reject['det_bajo'])[:10]:
    print(f"  det={det:.2f} w={fw:.0f} {name}")
print("=== MUESTRAS incompleta ===")
for det, fw, name in sorted(det_by_reject['incompleta'])[:10]:
    print(f"  det={det:.2f} w={fw:.0f} {name}")
print("=== MUESTRAS pequena ===")
for det, fw, name in sorted(det_by_reject['pequena'])[:10]:
    print(f"  det={det:.2f} w={fw:.0f} {name}")
