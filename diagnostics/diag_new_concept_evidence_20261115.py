"""Calibra la medición de «objeto añadido» por evidencia NUEVA (P8) y
«persona añadida» por caras detectadas (P6).

La primera medición (heatmap global) no sirve: con una ruta de fotograma
completo la presencia del concepto baja en todos los casos medidos
(people -0,044; add -0,049; pose -0,172). Lo que sí discrimina es la
EVIDENCIA NUEVA: píxeles donde el concepto aparece en la salida y NO estaba
en la entrada. Y para «add» de una persona, el número de caras detectadas.

Uso: venv\\Scripts\\python.exe diagnostics\\diag_new_concept_evidence_20261115.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "diagnostics/out/editor_categories_20261114"
SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"


def new_evidence(masker, before: Image.Image, after: Image.Image, target: str) -> dict:
    heat_before = masker.probability(before, target)
    heat_after = masker.probability(after, target)
    if heat_before is None or heat_after is None:
        return {"available": False}
    before_map = np.asarray(heat_before, dtype=np.float32)
    after_map = np.asarray(heat_after, dtype=np.float32)
    if before_map.shape != after_map.shape:
        after_map = np.asarray(
            Image.fromarray((after_map * 255).astype(np.uint8)).resize(
                (before_map.shape[1], before_map.shape[0])),
            dtype=np.float32) / 255.0
    a = np.asarray(before.convert("RGB"), dtype=np.int16)
    b = np.asarray(after.convert("RGB"), dtype=np.int16)
    if a.shape != b.shape:
        b = np.asarray(
            Image.fromarray(b.astype(np.uint8)).resize(
                (a.shape[1], a.shape[0])), dtype=np.int16)
    changed = np.abs(a - b).max(axis=2) > 8
    fresh = (after_map > 0.4) & (before_map < 0.2)
    fresh_in_changed = fresh & changed
    total = float(fresh.size)
    return {
        "available": True,
        "fresh_area": round(float(fresh.sum()) / total, 5),
        "fresh_in_changed": round(float(fresh_in_changed.sum()) / total, 5),
        "peak_after": round(float(after_map.max()), 4),
        "peak_before": round(float(before_map.max()), 4),
        "changed_ratio": round(float(changed.mean()), 4),
    }


def face_count(image: Image.Image) -> int:
    try:
        import cv2
        from roop.face_util import get_face_analyser_cpu

        bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
        analyser = get_face_analyser_cpu()
        faces = analyser.get(bgr) or []
        return len(faces)
    except Exception as error:
        print(f"[diag] InsightFace falló: {error}")
        return -1


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
            ("remove", "helmet"),
        ):
            path = OUT / f"{name}.png"
            if not path.is_file():
                continue
            after = Image.open(path).convert("RGB")
            started = time.monotonic()
            row = {
                "case": name,
                "target": target,
                "evidence": new_evidence(masker, before, after, target),
                "faces_before": face_count(before),
                "faces_after": face_count(after),
                "seconds": round(time.monotonic() - started, 1),
            }
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
        target_path = ROOT / "diagnostics/out/new_concept_evidence_20261115.json"
        target_path.write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[diag] escrito {target_path}")
    finally:
        masker.unload()


if __name__ == "__main__":
    main()
