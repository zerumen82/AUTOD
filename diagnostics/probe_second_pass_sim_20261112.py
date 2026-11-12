# -*- coding: utf-8 -*-
"""Calibra el umbral de la verificación de segunda pasada (fallo 2).

Para cada archivo que el filtro RECHAZA por "región dudosa", mide el mejor
`s sim` que alcanza la verificación de segunda pasada (re-detección del crop
comparando embeddings). Con esos valores se puede elegir el umbral que
separa caras REALES de falsos positivos, en vez de elegirlo a ojo.

Uso:
    venv\\Scripts\\python.exe -m diagnostics.probe_second_pass_sim_20261112 <DIR> [N]
"""
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import cv2
import numpy as np

import roop.face_util as fu

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python -m diagnostics.probe_second_pass_sim_20261112 <DIR> [N]")
bank = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 150

files = sorted(list(bank.glob("*.jpg")) + list(bank.glob("*.jpeg"))
               + list(bank.glob("*.JPG")) + list(bank.glob("*.png")))
sample = files[::max(1, len(files) // limit)][:limit]

settings = fu.get_faceswap_config()
threshold = float(settings["source_identity_similarity_threshold"])
print(f"[PROBE] umbral configurado: {threshold}", flush=True)

rows = []
for path in sample:
    output = StringIO()
    with redirect_stdout(output):
        try:
            faces = fu.extract_face_images(str(path), (False, 0), is_source_face=True)
        except Exception:  # noqa: BLE001
            continue
    log = output.getvalue()
    if faces:
        continue
    if "Región dudosa" not in log and "NO re-detecta la misma cara" not in log:
        continue
    # Reproducir las detecciones para medir el sim de la segunda pasada.
    img = cv2.imread(str(path))
    if img is None:
        continue
    h, w = img.shape[:2]
    try:
        raw = fu.detect_faces_robust(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),
                                     fu.get_face_analyser())
    except Exception:  # noqa: BLE001
        continue
    cache = {}
    for f in raw:
        ok, sim, detail = fu._verify_source_face_identity(
            img, f, settings, threshold, smart_cache=cache)
        rows.append((path.name, ok, sim, detail))
        break

print("\n[ARCHIVO | OK | SIM | DETALLE]", flush=True)
for name, ok, sim, detail in sorted(rows, key=lambda r: -r[2]):
    print(f"  {name} | ok={ok} | sim={sim:.3f} | {detail}", flush=True)

sims = np.asarray([r[2] for r in rows], dtype=float)
if sims.size:
    print(f"\n[RESUMEN] n={sims.size} max={sims.max():.3f} "
          f"p90={np.percentile(sims, 90):.3f} mediana={np.median(sims):.3f} "
          f"min={sims.min():.3f}", flush=True)
    for cand in (0.40, 0.45, 0.50, 0.55, 0.60):
        print(f"  sim >= {cand:.2f}: {int((sims >= cand).sum())}/{sims.size}",
              flush=True)
print("[FIN]", flush=True)
