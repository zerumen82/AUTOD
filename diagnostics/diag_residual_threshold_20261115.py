"""Calibra el umbral de residual de un objeto nombrado (P4).

Mide, con el MISMO segmentador y la MISMA consulta que usa el editor
(`_measured_residual`), qué fracción de la zona medida sigue siendo el
objeto pedido DESPUÉS de la edición, en casos reales:

- `remove` del casco (LaMa, v5.1360): caso que la documentación considera
  bueno, con un resto de visor.
- `remove` de zapatos (2026-11-07): caso en el que el objeto SE QUEDÓ.

El umbral vigente es 40 % (`edit.residual_failure_fraction`). Este diagnóstico
dice qué mide la foto real y con qué número se separa un caso bueno de uno
malo.

Uso: venv\\Scripts\\python.exe diagnostics\\diag_residual_threshold_20261115.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CASES = (
    {
        "case": "remove_casco_lama",
        "before": "diagnostics/out/editor_camera_neutral_20261110/before.png",
        "after": "diagnostics/out/editor_categories_20261114/remove.png",
        "query": "black helmet",
    },
    {
        "case": "remove_zapatos",
        "before": "diagnostics/out/shoe_removal_variants_20261107/stack_input_cover_prod_20261107b.png",
        "after": "diagnostics/out/shoe_removal_variants_20261107/result_prod.png",
        "query": "shoes",
    },
)


def main() -> None:
    from roop.img_editor.clipseg_masker import get_clipseg_masker

    masker = get_clipseg_masker()
    ok, message = masker.initialize()
    print(f"[diag] CLIPSeg: {ok} {message}", flush=True)
    if not ok:
        raise SystemExit("CLIPSeg no disponible")
    masker.device = "cpu"
    rows = []
    try:
        masker.model.to("cpu")
        for case in CASES:
            before_path = ROOT / case["before"]
            after_path = ROOT / case["after"]
            if not before_path.is_file() or not after_path.is_file():
                print(f"[diag] falta {case['case']}")
                continue
            before = Image.open(before_path).convert("RGB")
            after = Image.open(after_path).convert("RGB")
            zone_mask = masker.generate_mask(before, case["query"])
            if zone_mask is None:
                print(f"[diag] zona no medida en {case['case']}")
                continue
            zone = np.asarray(
                zone_mask.convert("L").resize(before.size, Image.Resampling.NEAREST)
            ) > 127
            if after.size != before.size:
                after = after.resize(before.size, Image.Resampling.LANCZOS)
            residual_mask = masker.generate_mask(after, case["query"])
            residual = 0
            if residual_mask is not None:
                residual_arr = np.asarray(
                    residual_mask.convert("L").resize(
                        before.size, Image.Resampling.NEAREST)
                ) > 127
                residual = int((residual_arr & zone).sum())
            zone_px = max(1, int(zone.sum()))
            row = {
                "case": case["case"],
                "query": case["query"],
                "zone_px": zone_px,
                "residual_px": residual,
                "residual_fraction": round(residual / zone_px, 4),
                "threshold_now": 0.40,
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        target = ROOT / "diagnostics/out/residual_threshold_20261115.json"
        target.write_text(json.dumps(rows, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"[diag] escrito {target}")
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
