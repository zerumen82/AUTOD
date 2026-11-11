# -*- coding: utf-8 -*-
"""Validación real del fix v5.1334 (log 2026-11-11).

Repite los dos prompts que fallaron en el log usando la imagen de origen
de cada uno, en la sesión que tiene ComfyUI levantado. Si el usuario
quiere, ejecuta:
  venv/Scripts/python.exe diagnostics/probe_v51334_20261111.py

Escribe diagnostics/out/probe_v51334_20261111/ con antes/después,
máscara y summary.json.
"""
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image  # noqa: E402

from roop.img_editor.img_editor_manager import get_img_editor_manager  # noqa: E402
from roop.img_editor.instruction_verifier import measure_change  # noqa: E402

PROMPTS = [
    ("fallo_A_residual", "el hombre debe ir sin camiseta, sin pantalon",
     ROOT / "img_editor" / "edit_1794417889.png"),
    ("fallo_B_sin_editar", "el hombre debe ir con un top de mujer",
     ROOT / "img_editor" / "edit_1794417889.png"),
]

OUT = ROOT / "diagnostics" / "out" / "probe_v51334_20261111"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manager = get_img_editor_manager()
    results = []

    for case, prompt, source in PROMPTS:
        case_dir = OUT / case
        case_dir.mkdir(parents=True, exist_ok=True)
        image = Image.open(source).convert("RGB")
        image.save(case_dir / "before.png")

        print(f"[{case}] prompt={prompt!r} source={source.name}", flush=True)
        started = time.time()
        try:
            result, msg, mask = manager.generate_intelligent(
                image=image,
                prompt=prompt,
                engine="auto",
                use_rewriter=True,
            )
        except Exception:
            traceback.print_exc()
            results.append({"case": case, "error": traceback.format_exc()})
            continue
        elapsed = time.time() - started

        data = {
            "case": case,
            "prompt": prompt,
            "source": str(source),
            "elapsed_s": round(elapsed, 1),
            "message": msg,
            "engine": "auto",
            "compliance": getattr(manager, "_last_instruction_compliance", None),
            "mask": None,
            "result": None,
        }
        if mask is not None:
            mask.save(case_dir / "mask.png")
            data["mask"] = {
                "coverage": float(
                    (mask.convert("L").point(
                        lambda v: 255 if v >= 96 else 0)).getbbox() is not None
                ),
            }
        if result is None:
            print(f"[{case}] Sin imagen de resultado", flush=True)
        else:
            result.save(case_dir / "result.png")
            metrics = measure_change(image, result, mask)
            data["result"] = str(case_dir / "result.png")
            data["change"] = metrics
            print(f"[{case}] cambio={metrics}", flush=True)

        results.append(data)
        (case_dir / "summary.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    print(json.dumps(results, ensure_ascii=False), flush=True)
    ok = all(r.get("result") and not r.get("error") for r in results)
    print(f"\nRESULTADO: {'PASS' if ok else 'FAIL'} ({len(results)} casos)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
