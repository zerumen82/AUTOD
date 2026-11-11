# -*- coding: utf-8 -*-
"""Volcado por archivo del resultado del filtro de caras de origen.

Sirve para diffear ANTES/DESPUÉS de los fixes v5.1335 y saber QUÉ archivos
cambian, para validar visualmente que los recuperados son caras reales.

Uso:
    venv\\Scripts\\python.exe -m diagnostics.probe_source_bank_perfile_20261112 <DIR> <OUT.json> [N]
"""
import json
import sys
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from roop.face_util import extract_face_images

if len(sys.argv) < 3:
    raise SystemExit(
        "Usage: python -m diagnostics.probe_source_bank_perfile_20261112 <DIR> <OUT.json> [N]")
bank = Path(sys.argv[1])
out_path = Path(sys.argv[2])
limit = int(sys.argv[3]) if len(sys.argv) > 3 else 150

files = sorted(list(bank.glob("*.jpg")) + list(bank.glob("*.jpeg"))
               + list(bank.glob("*.JPG")) + list(bank.glob("*.png")))
sample = files[::max(1, len(files) // limit)][:limit]

records = {}
for path in sample:
    output = StringIO()
    with redirect_stdout(output):
        try:
            faces = extract_face_images(str(path), (False, 0), is_source_face=True)
            count = len(faces)
        except Exception:  # noqa: BLE001
            count = -1
    log = output.getvalue()
    records[path.name] = {
        "faces": count,
        "residual": "Recovery residual" in log,
        "det_bajo": "— cara descartada" in log,
        "geometria": "Sin geometría" in log,
        "incompleta": "Cara INCOMPLETA" in log,
        "region_dudosa": "Región dudosa" in log,
        "crop_confirma": "Crop confirma identidad" in log,
    }

out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
accepted = sum(1 for v in records.values() if v["faces"] > 0)
print(f"[PERFILE] archivos={len(records)} con_cara={accepted} "
      f"sin_cara={len(records) - accepted} -> {out_path}", flush=True)
