# -*- coding: utf-8 -*-
"""Mide las señales de oclusión oral sobre fotos reales con swap real.

Ejecuta el pipeline REAL de FaceSwap sobre un conjunto de fotos de destino y
vuelca, por cada una, las señales que deciding si hay un objeto delante de la
boca: occ (crop-mean), rel (fracción de la boca cubierta), ring (anillo
peri-bucal) y diff_boca (si el swap YA cambió esa zona).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_mouth_occlusion_signals_20261112.py <DIR_FOTOS>
"""
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import roop.globals  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.face_util import extract_face_images  # noqa: E402
from roop.types import FaceSet  # noqa: E402

SOURCE = Path(r"D:\PROJECTS\uee\FACES\MARETA\P1010496.JPG")


def main():
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python -m diagnostics."
                         "probe_mouth_occlusion_signals_20261112 <DIR_FOTOS>")
    folder = Path(sys.argv[1])
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    src_faces = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)
    if not src_faces:
        print("sin cara de origen", flush=True)
        return 1
    source = src_faces[0][0]

    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        tgt_faces = extract_face_images(str(path), (False, 0))
        if not tgt_faces:
            print(f"{path.name}: sin cara", flush=True)
            continue
        target = tgt_faces[0][0]
        print(f"=== {path.name} ===", flush=True)
        try:
            roop.globals.enhancer_blend_factor = 0.70
            roop.globals.selected_enhancer = "GFPGAN"
            roop.globals.use_enhancer = True
            mgr = ProcessMgr()
            mgr.initialize([FaceSet([source])], [target], None,
                           face_swap_mode="selected_faces")
            mgr._is_single_photo = True
            mgr._process_face_swap_v21(source, target, frame.copy(), frame)
        except Exception as error:  # noqa: BLE001
            print(f"{path.name}: error {error}", flush=True)
    print("[FIN]", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
