# -*- coding: utf-8 -*-
"""Mide sim(original vs re-detect) sobre fotos reales del pool del usuario
para calibrar el umbral de la segunda pasada v5.1153b."""
import sys, os, glob, cv2, traceback
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stderr.reconfigure(encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roop import face_util
face_util.get_face_analyser()
FA = face_util.FACE_ANALYSER

def best_sim(e0, fl):
    best = 0.0
    for f in fl or []:
        e = getattr(f, 'embedding', None)
        if e is None: continue
        e = np.asarray(e, dtype='float32').flatten()
        n = float(np.linalg.norm(e))
        if n > 0:
            best = max(best, float(np.dot(e0, e / n)))
    return best

def redo_sim(img, f0):
    """Re-detección en el crop con padding+downscale (misma lógica que el fix)."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [int(max(0, v)) for v in f0.bbox]
    px, py = int((x2 - x1) * 0.5), int((y2 - y1) * 0.5)
    c1x, c1y = max(0, x1 - px), max(0, y1 - py)
    c2x, c2y = min(w, x2 + px), min(h, y2 + py)
    crop = img[c1y:c2y, c1x:c2x]
    if crop.size == 0: return 0.0, 0
    fpx = max(x2 - x1, y2 - y1)
    if fpx > 350:
        s = 350.0 / fpx
        crop = cv2.resize(crop, (max(1, int(crop.shape[1] * s)), max(1, int(crop.shape[0] * s))), interpolation=cv2.INTER_AREA)
    e0 = np.asarray(f0.embedding, dtype='float32').flatten()
    n0 = float(np.linalg.norm(e0))
    if n0 <= 0: return 0.0, 0
    e0 /= n0
    fl = []
    for c in (crop, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)):
        try: fl = FA.get(c)
        except Exception: fl = []
        if fl: break
    return best_sim(e0, fl), len(fl)

sims = []
files = []
for pat in ('0000*.jpg', 'IMG_*.jpg', 'IMG-202*.jpg', '20*.jpg', 'P*.jpg'):
    files.extend(glob.glob(os.path.join(r'D:\PROJECTS\uee\FACES', pat)))
files = sorted(set(files))[:40]
print(f"muestra: {len(files)} fotos del pool real")
for p in files:
    try:
        img = cv2.imread(p)
        if img is None: continue
        faces = FA.get(img)
        if not faces: continue
        f0 = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
        det = float(getattr(f0, 'det_score', 0.0) or 0.0)
        if det < 0.22: continue
        s, nf = redo_sim(img, f0)
        sims.append((s, det, (f0.bbox[2]-f0.bbox[0]), os.path.basename(p), nf))
    except Exception:
        traceback.print_exc()

sims.sort()
print(f"\nn={len(sims)} caras validas (det>=0.22)")
if sims:
    vals = [s for s, *_ in sims]
    import statistics
    print(f"sim: min={min(vals):.3f} p10={vals[len(vals)//10]:.3f} med={statistics.median(vals):.3f} p90={vals[int(len(vals)*0.9)]:.3f} max={max(vals):.3f}")
    print("\ndetalle (sim, det, bbox_w, n_crop, file):")
    for s, det, bw, name, nf in sims[:8]:
        print(f"  {s:.3f} det={det:.2f} w={bw} n={nf} {name}")
    print("  ...")
    for s, det, bw, name, nf in sims[-8:]:
        print(f"  {s:.3f} det={det:.2f} w={bw} n={nf} {name}")
