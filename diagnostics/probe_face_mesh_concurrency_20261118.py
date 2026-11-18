"""Ejercita la malla real compartida entre máscara y landmarks en varios hilos."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.top_tier_finish import _face_mesh_mask, _get_face_landmarks  # noqa: E402


def main():
    source = ROOT / "diagnostics" / "out" / "editor_camera_neutral_20261110" / "before.png"
    with Image.open(source) as opened:
        photo = opened.convert("RGB")
    bgr = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR)

    def run_one(index):
        if index % 2:
            mask = _face_mesh_mask(bgr, 1.0)
            return "mask", mask is not None and bool(np.any(mask > 0))
        landmarks = _get_face_landmarks(photo)
        return "landmarks", landmarks is not None and landmarks.shape == (468, 2)

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(run_one, range(48)))
    failures = [kind for kind, success in results if not success]
    assert not failures, f"Malla falló en {len(failures)}/48 llamadas: {failures}"
    print("PASS: 24 máscaras y 24 landmarks reales en 8 hilos; sin errores de grafo")


if __name__ == "__main__":
    main()
