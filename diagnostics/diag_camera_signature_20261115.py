"""Diagnóstico P9: firma medible de cambio de cámara (antes/después).

Objetivo: la verificación actual (`_camera_change_score`) devolvió 0.00 en un
cambio de cámara REAL (Qwen Camera, ángulo alto) porque no encontró cara en la
imagen de salida y lo trató como "sin cambio". Aquí se mide qué señales SÍ
existen en el par antes/después para poder verificar el POV con evidencia.

Uso: venv\\Scripts\\python.exe diagnostics\\diag_camera_signature_20261115.py
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.top_tier_finish import _get_face_landmarks  # noqa: E402

OUT = ROOT / "diagnostics/out/editor_categories_20261114"
SOURCE = ROOT / "diagnostics/out/editor_camera_neutral_20261110/before.png"

# Índices de MediaPipe Face Mesh (468) usados para la firma:
FOREHEAD_TOP = 10      # borde superior del frente
BROW_LINE = (105, 334)  # cejas
CHIN = 152             # mentón
NOSE_TIP = 1
MOUTH = (13, 14)       # comisuras de la boca


def _polygon_area(points: np.ndarray) -> float:
    x = points[:, 0]
    y = points[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def _landmarks_signature(image: Image.Image) -> dict:
    points = _get_face_landmarks(image)
    if points is None:
        return {"available": False}
    forehead = points[FOREHEAD_TOP]
    brow = points[list(BROW_LINE)].mean(axis=0)
    chin = points[CHIN]
    mouth = points[list(MOUTH)].mean(axis=0)
    upper = float(abs(chin[1] - brow[1]))
    lower = float(abs(chin[1] - mouth[1]))
    total = float(abs(chin[1] - forehead[1]))
    return {
        "available": True,
        "forehead_over_total": float(abs(brow[1] - forehead[1]) / max(total, 1e-6)),
        "upper_over_lower": float(upper / max(lower, 1e-6)),
        "face_center_y": float(points[:, 1].mean()),
        "face_area": float(_polygon_area(points)),
        "roll": float(np.degrees(np.arctan2(
            points[list(BROW_LINE)][1][0] - points[list(BROW_LINE)][1][1],
            points[list(BROW_LINE)][0][0] - points[list(BROW_LINE)][0][1],
        ))),
    }


def _structure_correlation(before: Image.Image, after: Image.Image) -> float:
    import cv2

    size = (256, 256)
    maps = []
    for image in (before, after):
        gray = np.asarray(
            image.convert("L").resize(size, Image.Resampling.BILINEAR),
            dtype=np.float32,
        )
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge = np.sqrt(gx * gx + gy * gy)
        edge = edge - edge.min()
        norm = float(np.linalg.norm(edge))
        maps.append(edge / norm if norm > 0 else edge)
    return float(np.sum(maps[0] * maps[1]))


def main() -> None:
    before = Image.open(SOURCE).convert("RGB")
    rows = []
    for name in ("camera", "body", "clothing", "background", "pose"):
        path = OUT / f"{name}.png"
        if not path.is_file():
            continue
        after = Image.open(path).convert("RGB")
        if after.size != before.size:
            after = after.resize(before.size, Image.Resampling.LANCZOS)
        lm_before = _landmarks_signature(before)
        lm_after = _landmarks_signature(after)
        row = {
            "case": name,
            "landmarks_before": lm_before,
            "landmarks_after": lm_after,
            "structure_correlation": round(_structure_correlation(before, after), 4),
        }
        if lm_before.get("available") and lm_after.get("available"):
            row["delta_forehead_over_total"] = round(
                lm_after["forehead_over_total"] - lm_before["forehead_over_total"], 4)
            row["delta_face_center_y"] = round(
                lm_after["face_center_y"] - lm_before["face_center_y"], 4)
            row["delta_roll"] = round(lm_after["roll"] - lm_before["roll"], 2)
        else:
            lm_b = _get_face_landmarks(before)
            lm_a = _get_face_landmarks(after)
            if lm_b is not None and lm_a is not None:
                diff = float(np.linalg.norm(lm_b - lm_a, axis=1).mean())
                row["landmark_mean_diff"] = round(diff, 4)
                row["old_score"] = round(float(np.clip(diff / 0.1, 0.0, 1.0)), 4)
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    target = ROOT / "diagnostics/out/camera_signature_20261115.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[diag] escrito {target}")


if __name__ == "__main__":
    main()
