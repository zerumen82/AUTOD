# -*- coding: utf-8 -*-
"""A/B medido del filtro de caras de ORIGEN (fix v5.1335, punto 3).

Recorre un banco de fuentes y cuenta, por archivo:
  - caras ACEPTADAS
  - rechazos por motivo (det bajo, geometría, incompleta, región dudosa,
    segunda pasada)

El objetivo es ver si el filtro nuevo (umbral de confianza por tamaño +
geometría/completitud relajadas para det>=0.50 + residual acceptance) recupera
caras REALES sin introducir falsos positivos.

Uso:
    venv\\Scripts\\python.exe -m diagnostics.probe_source_bank_ab_20261112 <DIR> [N]
"""
import re
import sys
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from roop.face_util import extract_face_images

if len(sys.argv) < 2:
    raise SystemExit("Usage: python -m diagnostics.probe_source_bank_ab_20261112 <DIR> [N]")
bank = Path(sys.argv[1])
limit = int(sys.argv[2]) if len(sys.argv) > 2 else 200

files = sorted(list(bank.glob("*.jpg")) + list(bank.glob("*.jpeg"))
               + list(bank.glob("*.JPG")) + list(bank.glob("*.png")))
sample = files[::max(1, len(files) // limit)][:limit]

counts = Counter()
reason_of_rejection = Counter()

RE_DET = re.compile(r"det=([0-9.]+) < ([0-9.]+) — cara descartada")
RE_GEO = "Sin geometría de cara"
RE_INC = "Cara INCOMPLETA"
RE_DOUBT = "Región dudosa descartada"
RE_SECOND = "Segunda pasada"

print(f"[PROBE] bank={bank} files={len(files)} sampled={len(sample)}", flush=True)

for index, path in enumerate(sample, 1):
    output = StringIO()
    with redirect_stdout(output):
        try:
            faces = extract_face_images(str(path), (False, 0), is_source_face=True)
        except Exception as exc:  # noqa: BLE001
            counts["error"] += 1
            print(f"[PROBE] error en {path.name}: {exc}", flush=True)
            continue
    log = output.getvalue()
    if faces:
        counts["files_with_faces"] += 1
        counts["accepted_faces"] += len(faces)
    else:
        counts["files_without_faces"] += 1
    # Motivos de rechazo (por archivo, no por cara).
    if RE_DET.search(log):
        reason_of_rejection["det_bajo"] += 1
    if RE_GEO in log:
        reason_of_rejection["geometria"] += 1
    if RE_INC in log:
        reason_of_rejection["incompleta"] += 1
    if RE_DOUBT in log:
        reason_of_rejection["region_dudosa"] += 1
    if RE_SECOND in log:
        reason_of_rejection["segunda_pasada_fp"] += 1
    if "Recovery residual" in log:
        reason_of_rejection["residual_aceptado"] += 1
    if "Crop confirma identidad" in log:
        reason_of_rejection["crop_confirma_identidad"] += 1
    if index % 25 == 0:
        print(f"[PROBE] {index}/{len(sample)} {dict(counts)}", flush=True)

print("\n[RESUMEN]", flush=True)
for key, value in sorted(counts.items()):
    print(f"  {key}: {value}", flush=True)
print("[MOTIVOS DE RECHAZO (archivos)]", flush=True)
for key, value in sorted(reason_of_rejection.items()):
    print(f"  {key}: {value}", flush=True)
print("[FIN]", flush=True)
