"""Validación real (GPU) de la ruta de cámara y de la medida de cambio.

Caso del log 2026-11-10 01:01: «mejora el ANGULO» guardó una copia idéntica.
Con v5.1326/1327 debe enrutar a cámara y cambiar píxeles; si no cambia, el
editor debe decirlo en vez de guardar una copia.

Escribe `diagnostics/out/editor_camera_real_20261110/` con la salida, la
máscara y `summary.json`. Uso:
  venv/Scripts/python.exe diagnostics/real_editor_camera_20261110.py
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

PROMPT = "mejora el ANGULO"
SOURCE = ROOT / "img_editor" / "edit_1794268913.png"
OUT = ROOT / "diagnostics" / "out" / "editor_camera_real_20261110"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    image = Image.open(SOURCE).convert("RGB")
    image.save(OUT / "before.png")

    print(f"[REAL] prompt={PROMPT!r} source={SOURCE.name} size={image.size}", flush=True)
    manager = get_img_editor_manager()
    started = time.time()
    try:
        result, msg, mask = manager.generate_intelligent(
            image=image,
            prompt=PROMPT,
            engine="auto",
            use_rewriter=True,
        )
    except Exception:
        traceback.print_exc()
        return 1
    elapsed = time.time() - started

    data = {
        "prompt": PROMPT,
        "source": str(SOURCE),
        "elapsed_s": round(elapsed, 1),
        "message": msg,
        "engine": "auto",
        "compliance": getattr(manager, "_last_instruction_compliance", None),
        "mask": None,
        "result": None,
    }
    if mask is not None:
        mask.save(OUT / "mask.png")
        data["mask"] = {
            "coverage": float((mask.convert("L").point(lambda v: 255 if v >= 96 else 0)).getbbox() is not None),
        }
    if result is None:
        print("[REAL] Sin imagen de resultado", flush=True)
    else:
        result.save(OUT / "result.png")
        metrics = measure_change(image, result, mask)
        data["result"] = str(OUT / "result.png")
        data["change"] = metrics
        print(f"[REAL] cambio={metrics}", flush=True)

    (OUT / "summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(data, ensure_ascii=False), flush=True)
    return 0 if data.get("result") else 1


if __name__ == "__main__":
    sys.exit(main())
