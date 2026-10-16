"""H-FS-004: nitidez de la cara en fotos — A/B medido de la compensación.

Contratos:
  U1. Con un déficit sintético real (parche desenfocado, ~<0.6x del original),
      `ProcessMgr._compensate_face_sharpness` sube el parche a >=0.6x de la
      nitidez del original. (Pre-fix no existe el método → FAIL.)
  U2. Sin déficit (parche == original) devuelve el parche INTACTO (no
      sobre-nítido).
  U4-U7. `_compensate_pasted_sharpness` (v5.1204, marco tras el pegado):
      déficit sintético bajo máscara llega a >=0.6x sin pasar de 0.90x, los
      píxeles fuera de la máscara quedan byte a byte intactos y sin déficit
      devuelve el resultado intacto.
  I1. Integración real sobre fixtures no explícitos (test_face.webp +
      MARETA): el swap con la compensación activa deja la cara en >=0.6x de
      la nitidez del original dentro del bbox, y sin cambios fuera de la
      ventana. Además se mide el camino ANTIGUO (unsharp fijo) en la misma
      corrida — el método se hace fallar a propósito para que el llamador
      caiga al fallback — para ver el antes/después (informativo; aserción:
      la ruta nueva no es peor que la vieja -0.02).
  I4/I5. La misma A/B sobre una foto REAL no explícita del lote
      (07_original_frame_f1.png): el déficit del lote (parche 21 vs
      original 75) hace fallar la ruta vieja (<0.6x) y exige que la nueva
      llegue a >=0.6x mejorando >=5 puntos.

Sin GPU obligatoria; sin contenido explícito (fixtures de siempre).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_faceswap_photo_sharpness.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import roop.globals  # noqa: E402
from roop.ProcessMgr import ProcessMgr  # noqa: E402
from roop.face_util import extract_face_images  # noqa: E402
from roop.types import FaceSet  # noqa: E402

SOURCE = os.path.join(r"D:\PROJECTS\uee\FACES\MARETA", "P1010496.JPG")
TARGET = os.path.join(ROOT, "testdata", "test_face.webp")

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK  ' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""), flush=True)


def lap_var(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def lap_var_masked(img, region):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    L = cv2.Laplacian(gray, cv2.CV_64F)
    return float(L[region].var())


def pasted_unit_contracts():
    """U4-U6: unidad de _compensate_pasted_sharpness (marco, tras pegado)."""
    frame = cv2.imread(TARGET)
    assert frame is not None, f"falta fixture {TARGET}"
    soft = cv2.GaussianBlur(frame, (0, 0), 3.0)
    yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]]
    cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
    rx, ry = frame.shape[1] // 3, frame.shape[0] // 3
    mask = (((xx - cx) / max(1, rx)) ** 2 + ((yy - cy) / max(1, ry)) ** 2 <= 1.0
            ).astype(np.float32)
    pasted = (frame.astype(np.float32) * (1.0 - mask[:, :, None])
              + soft.astype(np.float32) * mask[:, :, None]).astype(np.uint8)
    region = mask >= 0.9
    src = lap_var_masked(frame, region)
    before = lap_var_masked(pasted, region)
    out, note = ProcessMgr._compensate_pasted_sharpness(pasted, frame, mask)
    ratio = lap_var_masked(out, region) / max(1.0, src)
    check(
        "U4: déficit pegado compensado a >=0.6x del original (marco)",
        ratio >= 0.6,
        f"{before:.0f} -> {lap_var_masked(out, region):.0f} de {src:.0f} ({ratio:.0%}); {note}",
    )
    check(
        "U5: sin sobrepasar el objetivo (<=0.90x) en la región pegada",
        ratio <= 0.90,
        f"ratio={ratio:.0%}",
    )
    outside = mask == 0
    same_outside = np.array_equal(out[outside], pasted[outside])
    check(
        "U6: píxeles fuera de la máscara intactos (byte a byte)",
        same_outside,
        f"{int(np.count_nonzero(out[outside] != pasted[outside]))} px distintos",
    )
    out2, note2 = ProcessMgr._compensate_pasted_sharpness(frame.copy(), frame, mask)
    check(
        "U7: sin déficit el resultado vuelve intacto",
        np.array_equal(out2, frame),
        note2,
    )


def unit_contracts():
    frame = cv2.imread(TARGET)
    assert frame is not None, f"falta fixture {TARGET}"
    # M identidad: el "original alineado" es el propio frame
    identity = np.eye(2, 3, dtype=np.float64)

    # U1: déficit sintético — parche desenfocado como el medido en el lote
    soft = cv2.GaussianBlur(frame, (0, 0), 3.0)
    before = lap_var(soft)
    src = lap_var(frame)
    out, note = ProcessMgr._compensate_face_sharpness(soft, frame, identity)
    ratio = lap_var(out) / max(1.0, src)
    check(
        "U1: déficit sintético compensado a >=0.6x del original",
        ratio >= 0.6,
        f"{before:.0f} -> {lap_var(out):.0f} de {src:.0f} ({ratio:.0%}); {note}",
    )
    check(
        "U3: sin sobrepasar el objetivo (<=0.90x en déficit sintético)",
        ratio <= 0.90,
        f"ratio={ratio:.0%} (sin retroceso llegaba a 105% → grano áspero)",
    )

    # U2: sin déficit → parche intacto
    out2, note2 = ProcessMgr._compensate_face_sharpness(frame.copy(), frame, identity)
    check(
        "U2: sin déficit el parche vuelve intacto",
        np.array_equal(out2, frame),
        note2,
    )


def _run_swap(frame, source, target, broken_compensation):
    """Ejecuta el swap real sobre `frame` con los fixtures dados.

    broken=True simula el camino ANTIGUO: el método de compensación alineada
    lanza y el llamador cae al unsharp fijo 1.10-1.15 (comportamiento
    pre-fix), y la compensación de nitidez pegada (v5.1204, post-warp en
    marco) también lanza → no aplica nada, como en pre-fix. Devuelve
    (resultado, ventana_de_pegado).
    """
    old_method = ProcessMgr.__dict__.get("_compensate_face_sharpness")
    old_pasted = ProcessMgr.__dict__.get("_compensate_pasted_sharpness")
    if broken_compensation:
        def _raise(*_a, **_k):
            raise RuntimeError("simulación pre-fix")
        ProcessMgr._compensate_face_sharpness = staticmethod(_raise)
        ProcessMgr._compensate_pasted_sharpness = staticmethod(_raise)
    try:
        roop.globals.enhancer_blend_factor = 0.70
        roop.globals.selected_enhancer = "GFPGAN"
        roop.globals.use_enhancer = True
        mgr = ProcessMgr()
        mgr.initialize([FaceSet([source])], [target], None, face_swap_mode="selected_faces")
        mgr._is_single_photo = True
        result = mgr._process_face_swap_v21(source, target, frame.copy(), frame)
        window = mgr._profile_head_window(frame.shape, target.bbox, is_profile=False)
        return result, window
    finally:
        ProcessMgr._compensate_face_sharpness = old_method
        ProcessMgr._compensate_pasted_sharpness = old_pasted


def integration_contract():
    frame = cv2.imread(TARGET)
    source_faces = extract_face_images(SOURCE, (False, 0), is_source_face=True)
    target_faces = extract_face_images(TARGET, (False, 0))
    assert source_faces and target_faces, "fixtures sin caras"
    source, target = source_faces[0][0], target_faces[0][0]

    x1, y1, x2, y2 = np.asarray(target.bbox, dtype=int)
    x1, y1 = max(0, x1), max(0, y1)
    orig_sharp = lap_var(frame[y1:y2, x1:x2])

    old_result, old_window = _run_swap(frame, source, target, broken_compensation=True)
    old_ratio = lap_var(old_result[y1:y2, x1:x2]) / max(1.0, orig_sharp)

    new_result, new_window = _run_swap(frame, source, target, broken_compensation=False)
    new_ratio = lap_var(new_result[y1:y2, x1:x2]) / max(1.0, orig_sharp)

    check(
        "I1: swap con compensación deja la cara en >=0.6x del original",
        new_ratio >= 0.6,
        f"original={orig_sharp:.0f}, viejo={old_ratio:.0%}, nuevo={new_ratio:.0%}",
    )
    check(
        "I2: la ruta nueva no empeora frente al unsharp fijo",
        new_ratio >= old_ratio - 0.02,
        f"viejo={old_ratio:.0%} nuevo={new_ratio:.0%}",
    )
    changed_outside = int(
        np.count_nonzero(cv2.absdiff(new_result, frame).max(axis=2)[new_window <= 0.001])
    )
    check(
        "I3: sin píxeles cambiados fuera de la ventana de pegado",
        changed_outside == 0,
        f"{changed_outside} px",
    )


def real_photo_contract():
    """I4/I5: la misma A/B sobre una foto REAL no explícita del lote
    (07_original_frame_f1.png preservado). Aquí el déficit es el del lote
    (parche 21 vs original 75 = 28% de la fuente): la ruta vieja cae por
    debajo de 0.6x y la nueva debe superar 0.6x con margen."""
    photo = os.path.join(
        ROOT, "diagnostics", "out", "halo_stages_20261013", "batch_evidence",
        "07_original_frame_f1.png",
    )
    if not os.path.exists(photo):
        check("I4: foto real f1 disponible", False, f"falta {photo}")
        return
    frame = cv2.imread(photo)
    src_faces = extract_face_images(SOURCE, (False, 0), is_source_face=True)
    tgt_faces = extract_face_images(photo, (False, 0))
    if not (src_faces and tgt_faces):
        check("I4: foto real f1 con caras detectadas", False, "sin caras")
        return
    source, target = src_faces[0][0], tgt_faces[0][0]
    x1, y1, x2, y2 = np.asarray(target.bbox, dtype=int)
    orig_sharp = lap_var(frame[y1:y2, x1:x2])

    old_res, _ = _run_swap(frame, source, target, broken_compensation=True)
    old_ratio = lap_var(old_res[y1:y2, x1:x2]) / max(1.0, orig_sharp)
    new_res, _ = _run_swap(frame, source, target, broken_compensation=False)
    new_ratio = lap_var(new_res[y1:y2, x1:x2]) / max(1.0, orig_sharp)

    check(
        "I4: foto real (déficit del lote) queda en >=0.6x del original",
        new_ratio >= 0.6,
        f"original={orig_sharp:.0f}, viejo={old_ratio:.0%} (fallaría), nuevo={new_ratio:.0%}",
    )
    check(
        "I5: la ruta nueva mejora la vieja en >=5 puntos en la foto real",
        new_ratio >= old_ratio + 0.05,
        f"viejo={old_ratio:.0%} nuevo={new_ratio:.0%}",
    )


def main():
    unit_contracts()
    pasted_unit_contracts()
    integration_contract()
    real_photo_contract()
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
