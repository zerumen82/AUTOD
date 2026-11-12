# -*- coding: utf-8 -*-
"""Clasifica POR QUÉ se rechaza cada cara de origen (fallo 2).

Reproduce el filtro de `extract_face_images` sobre un banco real y separa los
motivos en sub-causas ACCIONABLES:

  det_bajo            - confianza por debajo del umbral (tamaño-dependiente)
  geometria           - kps no anatómicos
  incompleta_borde    - la cara está CORTADA por el borde de la foto (recorte
                        real: el usuario exige caras completas -> rechazo duro)
  incompleta_kps      - los kps caen fuera del bbox (marginal) con la cara
                        intacta en la foto -> rechazo FALSO, la cara es real
  region_dudosa       - la verificación independiente de MediaPipe no confirma
  segunda_pasada      - el crop no re-detecta la misma cara (falso positivo)

Uso:
    venv\\Scripts\\python.exe -m diagnostics.probe_source_reject_reasons_20261112 <DIR> [N]
"""
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np

import roop.face_util as fu
from roop.face_util import _has_face_geometry, _is_face_complete

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python -m diagnostics.probe_source_reject_reasons_20261112 <DIR> [N]")
bank = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 150

files = sorted(list(bank.glob("*.jpg")) + list(bank.glob("*.jpeg"))
               + list(bank.glob("*.JPG")) + list(bank.glob("*.png")))
sample = files[::max(1, len(files) // limit)][:limit]

RE_DET = "— cara descartada"
RE_GEO = "Sin geometría de cara"
RE_INC = "Cara INCOMPLETA"
RE_DOUBT = "Región dudosa descartada"
RE_SECOND = "NO re-detecta la misma cara"

reasons = Counter()
det_by_reason = {}
examples = {}

print(f"[PROBE] bank={bank} files={len(files)} sampled={len(sample)}", flush=True)

for path in sample:
    output = StringIO()
    with redirect_stdout(output):
        try:
            faces = fu.extract_face_images(str(path), (False, 0), is_source_face=True)
        except Exception:  # noqa: BLE001
            continue
    log = output.getvalue()
    if faces:
        reasons["ACEPTADA"] += 1
        continue
    # Reproducir el filtro para saber la sub-causa exacta.
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        reasons["ilegible"] += 1
        continue
    h, w = img.shape[:2]
    try:
        raw = fu.detect_faces_robust(img, fu.get_face_analyser())
    except Exception:  # noqa: BLE001
        raw = []
    sub = "desconocido"
    det_val = 0.0
    for f in raw:
        bbox = np.asarray(f.bbox, dtype=float)
        fw, fh = bbox[2] - bbox[0], bbox[3] - bbox[1]
        det = float(getattr(f, "det_score", 0.0) or 0.0)
        max_dim = max(fw, fh)
        min_det = 0.06 if max_dim >= 120 else 0.09 if max_dim >= 80 else \
            0.12 if max_dim >= 60 else 0.14
        if not (fw >= 40 and fh >= 40 and 0.25 < (fw / fh if fh else 0) < 3.5
                and det >= min_det):
            sub, det_val = "det_bajo", det
            break
        if not _has_face_geometry(f, max_tilt=45.0, relaxed=(det >= 0.50)):
            sub, det_val = "geometria", det
            break
        if not _is_face_complete(f, w, h, relaxed=(det >= 0.50)):
            # ¿Cortada por el borde o solo kps fuera?
            cut = (bbox[0] <= 1.0 or bbox[1] <= 1.0
                   or bbox[2] >= w - 1.0 or bbox[3] >= h - 1.0)
            sub = "incompleta_borde" if cut else "incompleta_kps"
            det_val = det
            break
    if sub == "desconocido":
        if RE_SECOND in log:
            sub = "segunda_pasada"
        elif RE_DOUBT in log:
            sub = "region_dudosa"
        elif RE_GEO in log:
            sub = "geometria"
        elif RE_INC in log:
            sub = "incompleta_borde"
        elif RE_DET in log:
            sub = "det_bajo"
        else:
            sub = "sin_deteccion"
    reasons[sub] += 1
    det_by_reason.setdefault(sub, []).append(det_val)
    examples.setdefault(sub, []).append(path.name)

print("\n[RESUMEN POR MOTIVO]", flush=True)
for key, value in sorted(reasons.items(), key=lambda kv: -kv[1]):
    dets = det_by_reason.get(key, [])
    det_txt = ""
    if dets:
        arr = np.asarray(dets, dtype=float)
        det_txt = f" det medio={arr.mean():.2f} min={arr.min():.2f} max={arr.max():.2f}"
    print(f"  {key}: {value}{det_txt}", flush=True)

print("\n[EJEMPLOS POR MOTIVO]", flush=True)
for key, names in sorted(examples.items()):
    print(f"  {key}: {', '.join(names[:8])}", flush=True)
print("[FIN]", flush=True)
