# -*- coding: utf-8 -*-
"""Perfiles de canto: el gate de alineación NO debe rechazar caras reales.

v5.1339: el piso de `eye_dist/bbox_w` medía la separación ocular RELATIVA al
ancho del bbox. En un perfil de canto los dos ojos están casi superpuestos y
esa relación baja legítimamente a 0.02-0.16, así que el gate rechazaba
perfiles REALES y la foto salía SIN SWAP.

Aquí se ejecuta el swap REAL sobre un directorio de fotos y se comprueba que
la salida DIFIERE del original (es decir, hubo swap) y que el log NO emite
`ALIGN_REJECT` por kps incoherentes.

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_v51339_profile_align_gate_20261112.py <DIR_FOTOS>
"""
import sys
from contextlib import redirect_stdout
from io import StringIO
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
                         "probe_v51339_profile_align_gate_20261112 <DIR_FOTOS>")
    folder = Path(sys.argv[1])
    files = sorted(p for p in folder.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    src_faces = extract_face_images(str(SOURCE), (False, 0), is_source_face=True)
    if not src_faces:
        print("sin cara de origen", flush=True)
        return 1
    source = src_faces[0][0]

    failures = []
    checked = 0
    for path in files:
        frame = cv2.imread(str(path))
        if frame is None:
            continue
        tgt_faces = extract_face_images(str(path), (False, 0))
        if not tgt_faces:
            continue
        target = tgt_faces[0][0]
        output = StringIO()
        with redirect_stdout(output):
            try:
                roop.globals.enhancer_blend_factor = 0.70
                roop.globals.selected_enhancer = "GFPGAN"
                roop.globals.use_enhancer = True
                mgr = ProcessMgr()
                mgr.initialize([FaceSet([source])], [target], None,
                               face_swap_mode="selected_faces")
                mgr._is_single_photo = True
                result = mgr._process_face_swap_v21(source, target, frame.copy(), frame)
            except Exception as error:  # noqa: BLE001
                print(f"{path.name}: error {error}", flush=True)
                failures.append(f"{path.name}: excepción {error}")
                continue
        log = output.getvalue()
        align_reject = "kps incoherentes con bbox" in log
        changed = result is not None and not np.array_equal(
            np.asarray(result), np.asarray(frame))
        checked += 1
        if align_reject:
            failures.append(f"{path.name}: ALIGN_REJECT por kps incoherentes")
        if not changed:
            failures.append(f"{path.name}: salida idéntica al original (sin swap)")
        print(
            f"{'OK  ' if not align_reject and changed else 'FAIL'}: {path.name} "
            f"swap={'sí' if changed else 'NO'} align_reject={align_reject}",
            flush=True,
        )

    print(f"\n{checked - len(failures)}/{checked} fotos OK", flush=True)
    if failures:
        print("FALLAN:")
        for item in failures:
            print(f"  - {item}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
