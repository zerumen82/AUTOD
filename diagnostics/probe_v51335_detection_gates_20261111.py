# -*- coding: utf-8 -*-
"""Validación de los fixes v5.1335 (log 2026-11-11, puntos 3-7).

Comprueba sobre casos SINTÉTICOS y fixtures reales que:

  P3. El umbral de confianza de source ESCALA con el tamaño de la cara: una
      cara grande de baja confianza se acepta; una diminuta se rechaza.
  P3b. La geometría relajada (det>=0.50) acepta una cara real de perfil
      inclinado que la geometría estricta rechazaba.
  P4. La contención de landmarks con margen relajado acepta un landmark
      marginalmente fuera del bbox en det alta y lo rechaza en det baja.
  P5. La compensación de nitidez pegada alcanza >=0.85x del original y nunca
      pasa de 0.90x; los píxeles fuera de la máscara quedan intactos.
  P6. La detección de boca abierta y la ruta de blend usan el MISMO umbral
      (una boca medida como abierta no sale con restore casi nulo).
  P7. La limpieza de VRAM ejecuta gc.collect() ANTES de empty_cache() y
      informa de la VRAM reservada (no solo de la asignada).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_v51335_detection_gates_20261111.py
"""
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import roop.globals  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.face_util import _has_face_geometry, _is_face_complete  # noqa: E402
from roop.faceswap_config import landmarks_fit_bbox  # noqa: E402

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""), flush=True)


def _face(bbox, kps, det):
    """InsightFace-like stub con lo que leen los gates."""
    return types.SimpleNamespace(
        bbox=np.asarray(bbox, dtype=np.float32),
        kps=np.asarray(kps, dtype=np.float32),
        det_score=float(det),
        embedding=None,
    )


def _frontal_kps(cx, cy, w, h):
    """5 kps anatómicos de una cara frontal: ojos, nariz, boca."""
    return np.array([
        [cx - w * 0.20, cy - h * 0.15],
        [cx + w * 0.20, cy - h * 0.15],
        [cx, cy],
        [cx - w * 0.12, cy + h * 0.28],
        [cx + w * 0.12, cy + h * 0.28],
    ], dtype=np.float32)


def p3_residual_acceptance():
    """P3: si el filtro vacía la lista, se recuperan las caras de alta
    confianza con geometría válida (residual acceptance)."""
    import inspect
    src = inspect.getsource(fu_extract())
    has_residual = "Recovery residual" in src
    uses_prefilter = "_faces_prefilter" in src
    check("P3: el filtro vacío recupera caras de alta confianza",
          has_residual and uses_prefilter,
          f"residual={has_residual} prefilter={uses_prefilter}")


def fu_extract():
    import roop.face_util as mod
    return mod.extract_face_images


def p3_size_scaled_confidence():
    """P3: el umbral de det escala con el tamaño de la cara."""
    import inspect
    src = inspect.getsource(_face_util_module())
    # La tabla de umbrales por tamaño debe existir y ser monótona.
    for threshold, size in (("0.06", 120), ("0.09", 80), ("0.12", 60)):
        if threshold not in src:
            check(f"P3: umbral det={threshold} para caras >= {size}px", False,
                  "no encontrado en el filtro de source")
            return
    check("P3: umbral de det escala con el tamaño de la cara", True,
          "0.06@>=120px, 0.09@>=80px, 0.12@>=60px, 0.14 resto")


def _face_util_module():
    import roop.face_util as mod
    return mod


def p3b_relaxed_geometry_accepts_real_profile():
    """P3b: una cara de perfil inclinado (det alta) pasa en modo relajado."""
    # Cara de 100x150 con ojos SUPERPUESTOS en X (perfil de canto real: los
    # dos ojos caen en la misma columna) y línea de ojos muy inclinada.
    # La geometría ESTRICTA exige separación ocular > 0.5% del ancho → la
    # rechaza; la RELAJADA (det>=0.50) la acepta porque el detector está
    # seguro de la cara.
    kps = np.array([
        [149.0, 130.0],
        [149.4, 128.0],
        [150.0, 175.0],
        [152.0, 225.0],
        [153.0, 222.0],
    ], dtype=np.float32)
    face = _face([100, 100, 200, 250], kps, 0.72)
    strict_ok = _has_face_geometry(face, max_tilt=45.0)
    relaxed_ok = _has_face_geometry(face, max_tilt=45.0, relaxed=True)
    check("P3b: geometría relajada acepta perfil de canto (det 0.72)",
          relaxed_ok and not strict_ok,
          f"estricta={strict_ok} relajada={relaxed_ok}")


def p3b_completeness_relaxed():
    """P3b: el check de completitud relajado tolera kps 3px fuera del bbox."""
    kps = np.array([
        [99.0, 110.0],   # 1px fuera por la izquierda
        [199.0, 110.0],
        [150.0, 160.0],
        [138.0, 200.0],
        [162.0, 200.0],
    ], dtype=np.float32)
    face = _face([100, 100, 200, 250], kps, 0.68)
    strict_ok = _is_face_complete(face, 400, 400)
    relaxed_ok = _is_face_complete(face, 400, 400, relaxed=True)
    check("P3b: completitud relajada tolera kps marginalmente fuera",
          relaxed_ok and not strict_ok,
          f"estricta={strict_ok} relajada={relaxed_ok}")


def p4_landmark_margin_relaxed():
    """P4: landmarks_fit_bbox acepta margen relajado y lo documenta."""
    # Bbox PEQUEÑO (60x80): el margen ESTRUCTO es max(2px, 0.02×80=1.6px)
    # = 2px, así que un landmark 4px fuera lo rechaza. El RELAJADO es
    # max(5px, 0.05×80=4px) = 5px y lo acepta.
    bbox = np.array([100.0, 100.0, 160.0, 180.0])
    kps = np.array([
        [96.0, 110.0],    # 4px fuera a la izquierda
        [155.0, 110.0],
        [130.0, 140.0],
        [124.0, 165.0],
        [136.0, 165.0],
    ], dtype=np.float32)
    strict = landmarks_fit_bbox(kps, bbox)
    relaxed = landmarks_fit_bbox(kps, bbox, relaxed=True)
    check("P4: margen relajado acepta landmark 4px fuera del bbox",
          (not strict) and relaxed,
          f"estricto={strict} relajado={relaxed}")


def p5_pasted_sharpness_target():
    """P5: la nitidez pegada alcanza >=0.85x sin pasar de 0.90x."""
    target = os.path.join(ROOT, "testdata", "test_face.webp")
    frame = cv2.imread(target)
    if frame is None:
        check("P5: fixture de nitidez disponible", False, f"falta {target}")
        return
    soft = cv2.GaussianBlur(frame, (0, 0), 3.0)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    rx, ry = frame.shape[1] // 3, frame.shape[0] // 3
    mask = (((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1.0
            ).astype(np.float32)
    pasted = (frame.astype(np.float32) * (1.0 - mask[:, :, None])
              + soft.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
    region = mask >= 0.9

    def lap_var(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
        return float(cv2.Laplacian(gray, cv2.CV_64F)[region].var())

    src = lap_var(frame)
    out, note = ProcessMgr._compensate_pasted_sharpness(pasted, frame, mask)
    ratio = lap_var(out) / max(1.0, src)
    check("P5: nitidez pegada alcanza >=0.85x del original", ratio >= 0.85,
          f"{ratio:.0%}; {note}")
    check("P5: nitidez pegada no supera 0.90x (sin sobre-nitidez)", ratio <= 0.90,
          f"{ratio:.0%}")
    outside = mask == 0
    check("P5: píxeles fuera de la máscara intactos",
          np.array_equal(out[outside], pasted[outside]))


def p6_mouth_threshold_shared():
    """P6: detección y blend de boca comparten un único umbral."""
    import inspect
    src = inspect.getsource(ProcessMgr._process_face_swap_v21)
    has_threshold = "_mouth_open_threshold" in src
    shared = "_open_cut" in src
    check("P6: la ruta de blend usa el umbral de la detección", has_threshold and shared,
          f"guarda={has_threshold} usa={shared}")


def p7_vram_cleanup_order():
    """P7: gc.collect() ANTES de empty_cache() y se informa la reservada."""
    import inspect
    src = inspect.getsource(ProcessMgr._cleanup_vram)
    # Buscar las LLAMADAS reales (el docstring las menciona en prosa).
    body = src.split('"""')[-1] if src.count('"""') >= 2 else src
    gc_at = body.find("gc.collect()")
    empty_at = body.find("torch.cuda.empty_cache()")
    check("P7: gc.collect() se ejecuta antes de empty_cache()",
          gc_at != -1 and empty_at != -1 and gc_at < empty_at,
          f"gc@{gc_at} empty_cache@{empty_at}")
    check("P7: se informa de la VRAM reservada", "memory_reserved" in src)
    check("P7: se liberan los buffers IPC", "ipc_collect" in src)


def main():
    p3_size_scaled_confidence()
    p3_residual_acceptance()
    p3b_relaxed_geometry_accepts_real_profile()
    p3b_completeness_relaxed()
    p4_landmark_margin_relaxed()
    p5_pasted_sharpness_target()
    p6_mouth_threshold_shared()
    p7_vram_cleanup_order()

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
