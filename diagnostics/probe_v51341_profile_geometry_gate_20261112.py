# -*- coding: utf-8 -*-
"""Criterio ÚNICO de geometría kps↔bbox, invariante al yaw (v5.1341).

Problema: el gate clásico `0.05 < eye_dist/bbox_w < 1.20` estaba replicado en
CUATRO sitios de `roop/ProcessMgr.py`:
  1. `_remember_good_face_geom`   (histórico de kps sanos)
  2. `_recovered_coherent_kps`    (reconstrucción de kps degenerados)
  3. `_swap_from_saved_reference` (selección por archivo)
  4. gate de alineación de la foto

El piso mide la separación ocular RELATIVA al ancho del bbox, y esa relación
depende del yaw: en un PERFIL de canto los dos ojos están casi superpuestos y
baja legítimamente a 0.02-0.16. Los cuatro gates rechazaban así perfiles
REALES (medido 2026-11-12 sobre fotos reales del usuario).

La señal que sí distingue es el span VERTICAL de los kps (`vspan`): la
estructura ojos → nariz → boca se mantiene en frontal y en perfil, pero se
colapsa cuando el regresor devuelve los landmarks amontonados.

Este probe comprueba:
  - el helper con los valores MEDIDOS (perfiles reales y caso degenerado),
  - que los cuatro sitios usan el helper (un único criterio, sin umbrales
    duplicados por el código).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_v51341_profile_geometry_gate_20261112.py
"""
import inspect
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from roop.ProcessMgr import ProcessMgr  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""), flush=True)


def _kps(eye_x, eye_y, nose_y, mouth_x, mouth_y, size):
    """5 kps con la separación ocular y el span vertical pedidos."""
    import numpy as np
    return np.array([
        [size / 2 - eye_x, eye_y],
        [size / 2 + eye_x, eye_y],
        [size / 2, nose_y],
        [size / 2 - mouth_x, mouth_y],
        [size / 2 + mouth_x, mouth_y],
    ], dtype=np.float32)


def main():
    import numpy as np
    gate = ProcessMgr._coherent_face_geometry

    # --- Perfiles REALES medidos (eye_rel, vspan) -------------------------
    # 130947468 eye_rel 0.043 vspan 0.401 | 184262585 0.040/0.339
    # 57702125_010 0.022/0.370 | 57723678 0.076/0.381 | 868219247 0.157/0.263
    perfiles = [
        ("130947468", 0.043, 0.401),
        ("184262585", 0.040, 0.339),
        ("57702125_010", 0.022, 0.370),
        ("57723678", 0.076, 0.381),
        ("868219247", 0.157, 0.263),
    ]
    for nombre, eye_rel, vspan in perfiles:
        bw = 100.0
        bh = 100.0
        eye_sep = eye_rel * bw
        span_v = vspan * bh
        kps = np.array([
            [50 - eye_sep / 2, 50 - span_v / 2],
            [50 + eye_sep / 2, 50 - span_v / 2],
            [50, 50 - span_v / 6],
            [50 - eye_sep / 2, 50 + span_v / 2],
            [50 + eye_sep / 2, 50 + span_v / 2],
        ], dtype=np.float32)
        ok, er, vs = gate(kps, np.array([0, 0, bw, bh], dtype=np.float32))
        check(
            f"P1: perfil real {nombre} (eye_rel {eye_rel:.3f}, vspan {vspan:.3f}) es coherente",
            ok, f"eye_rel={er:.3f} vspan={vs:.3f}",
        )

    # --- Caso degenerado: kps amontonados ---------------------------------
    # log 1994859933.jpg: ojos a 17,6 px en una cara de 632 px (eye_rel 0.028)
    # y el span vertical colapsado.
    kps_deg = np.array([
        [100.0, 100.0], [101.0, 100.5], [100.5, 100.2],
        [100.0, 100.8], [101.0, 100.6],
    ], dtype=np.float32)
    ok, er, vs = gate(kps_deg, np.array([0, 0, 632, 632], dtype=np.float32))
    check(
        "P2: kps degenerados (vspan colapsado) se rechazan",
        not ok, f"eye_rel={er:.4f} vspan={vs:.4f}",
    )

    # --- kps con eyes juntos pero estructura vertical sana: perfil ------
    kps_perfil = _kps(2.0, 30.0, 45.0, 3.0, 62.0, 100)
    ok, er, vs = gate(kps_perfil, np.array([0, 0, 100, 100], dtype=np.float32))
    check(
        "P3: perfil con ojos juntos y span sano pasa",
        ok, f"eye_rel={er:.3f} vspan={vs:.3f}",
    )

    # --- eye_rel absurdo (kps dispersos) ---------------------------------
    kps_dispersos = np.array([
        [0.0, 0.0], [1000.0, 0.0], [500.0, 500.0],
        [0.0, 1000.0], [1000.0, 1000.0],
    ], dtype=np.float32)
    ok, er, vs = gate(kps_dispersos, np.array([0, 0, 100, 100], dtype=np.float32))
    check("P4: kps absurdos (eye_rel > 1.20) se rechazan", not ok, f"eye_rel={er:.3f}")

    # --- Un único criterio: los 4 sitios usan el helper ------------------
    source = inspect.getsource(ProcessMgr)
    body = source.split('"""')
    # Código (sin docstrings) para no contar menciones en comentarios.
    code = "\n".join(
        part for index, part in enumerate(body) if index % 2 == 0
    )
    uses = code.count("_coherent_face_geometry(")
    check(
        "P5: los 4 gates usan el helper único (>= 5 llamadas: def + 4 usos)",
        uses >= 5, f"{uses} apariciones en código",
    )
    # No debe quedar ningún umbral clásico replicado fuera del helper.
    legacy = re.findall(r"0\.05\s*<\s*_?eye_rel", code)
    check(
        "P6: no queda el umbral clásico replicado fuera del helper",
        len(legacy) <= 1,
        f"{len(legacy)} aparición(es) (solo dentro del helper se permite)",
    )

    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} contratos OK", flush=True)
    if failed:
        print("FALLAN:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
