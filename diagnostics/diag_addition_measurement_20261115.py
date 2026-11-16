"""Calibra la medición de «persona/objeto añadido» (P6/P8) con fotos reales.

Compara la foto de origen (una persona) con la salida de «Add another
astronaut beside the first one» (personas.png) y con la de «Add a small flag
beside the astronaut» (add.png): mide presencia del concepto y personas antes
y después para fijar umbrales que no den falsos FAIL.

Uso: venv\\Scripts\\python.exe diagnostics\\diag_addition_measurement_20261115.py
"""

import json
import sys
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.measured_verification import (  # noqa: E402
    concept_presence,
    person_measurement,
)

OUT = ROOT / "diagnostics/out/editor_categories_20261114"
SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"


def main() -> None:
    from roop.img_editor.clipseg_masker import get_clipseg_masker

    masker = get_clipseg_masker()
    ok, message = masker.initialize()
    print(f"[diag] CLIPSeg: {ok} {message}", flush=True)
    if not ok:
        raise SystemExit("CLIPSeg no disponible")
    masker.device = "cpu"
    try:
        masker.model.to("cpu")

        before = Image.open(SOURCE).convert("RGB")
        rows = []
        for name, target in (
            ("people", "astronaut"),
            ("add", "flag"),
            ("pose", "astronaut"),
        ):
            path = OUT / f"{name}.png"
            if not path.is_file():
                continue
            after = Image.open(path).convert("RGB")
            started = time.monotonic()
            before_people = person_measurement(masker, before)
            after_people = person_measurement(masker, after)
            presence_before = concept_presence(masker, before, target)
            presence_after = concept_presence(masker, after, target)
            row = {
                "case": name,
                "target": target,
                "people_before": before_people,
                "people_after": after_people,
                "presence_before": presence_before,
                "presence_after": presence_after,
                "presence_gain": round(
                    (presence_after or {}).get("presence", 0.0)
                    - (presence_before or {}).get("presence", 0.0), 4),
                "area_gain": round(
                    (presence_after or {}).get("area", 0.0)
                    - (presence_before or {}).get("area", 0.0), 4),
                "person_count_gain": (after_people or {}).get("count", 0)
                - (before_people or {}).get("count", 0),
                "person_area_gain": round(
                    (after_people or {}).get("area", 0.0)
                    - (before_people or {}).get("area", 0.0), 4),
                "seconds": round(time.monotonic() - started, 1),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        target_path = ROOT / "diagnostics/out/addition_measurement_20261115.json"
        target_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[diag] escrito {target_path}")
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
