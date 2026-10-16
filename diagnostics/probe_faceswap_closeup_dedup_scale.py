"""Contrato: una cara grande no puede borrarse como «duplicada» de un
falso positivo diminuto contenido en su propio bbox.

Caso real 02457 (D:\\PROJECTS\\uee\\FACES\\MARETA\\02457.jpg): retrato de
perfil casi 90°. RetinaFace a det_size=1024 no ve la cara grande; la pasada
de rescate a 640 SÍ la detecta (358x653 px, det=0.47), pero el merge con
`_same_face_boxes` la descartaba como duplicada de una detección de
29x34 px (det=0.10) que cae dentro de su bbox → la foto se quedaba solo con
caras diminutas y el swap acababa en una persona de fondo.

Contratos:
  A. Misma cara detectada a dos escalas (1024 vs closeup 640) → sigue
     contando como duplicada (ratios medidos 1.04x y 1.59x de área).
  B. Detección minúcula DENTRO de una cara grande → NO es duplicada.
  C. `get_all_faces_smart(..., for_target=True, prefer_fast=False)` (la
     ruta real de foto de ProcessMgr) devuelve la cara grande principal de
     02457 (>30% del frame).

Uso:
    venv\\Scripts\\python.exe diagnostics\\probe_faceswap_closeup_dedup_scale.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

PHOTO_02457 = r"D:\PROJECTS\uee\FACES\MARETA\02457.jpg"

RESULTS = []


def check(name, ok, detail=""):
    RESULTS.append((name, ok, detail))
    print(f"{'OK ' if ok else 'FAIL'}: {name}" + (f" — {detail}" if detail else ""))


def contracts_same_face_boxes():
    from roop.face_util_rotation import _same_face_boxes

    # A. Misma cara a dos escalas (medido en 02457: bg y tercera cara)
    check(
        "A1: misma cara 1024-vs-640 (ratio 1.04x) cuenta como duplicada",
        _same_face_boxes([508, 600, 555, 664], [509, 602, 555, 665]),
    )
    check(
        "A2: misma cara 1024-vs-640 (ratio 1.59x) cuenta como duplicada",
        _same_face_boxes([428, 584, 457, 618], [423, 574, 457, 620]),
    )
    # Duplicado clásico casi idéntico
    check(
        "A3: cajas casi idénticas cuentan como duplicadas",
        _same_face_boxes([100, 100, 300, 300], [104, 103, 302, 299]),
    )

    # B. Falso positivo diminuto dentro de la cara grande NO la borra
    check(
        "B1: mini-detección 29x34 dentro de la cara grande NO es duplicada",
        not _same_face_boxes([97, 28, 455, 681], [428, 584, 457, 618]),
        "ratio de área 237x",
    )
    check(
        "B2: cara pequeña en otra zona NO es duplicada",
        not _same_face_boxes([97, 28, 455, 681], [508, 600, 555, 664]),
    )


def contract_product_path():
    from roop.face_util_rotation import get_all_faces_smart

    if not os.path.exists(PHOTO_02457):
        check("C: ruta de foto devuelve la cara grande de 02457", False,
              f"foto no disponible: {PHOTO_02457}")
        return

    import cv2
    frame = cv2.imread(PHOTO_02457)
    h, w = frame.shape[:2]
    faces = get_all_faces_smart(
        frame, min_score=None, for_target=True, prefer_fast=False, include_180=True,
    )
    areas = [((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) / (w * h) for f in faces]
    biggest = max(areas) if areas else 0.0
    check(
        "C: ruta de foto devuelve la cara grande principal de 02457 (>30% del frame)",
        biggest > 0.30,
        f"caras={len(faces)}, área máxima={biggest:.3f}",
    )


def contract_ui_gallery():
    """D: la galería de selección de la UI (`extract_face_images`, rama
    target) debe ofrecer la cara grande principal, no solo la persona de
    fondo. Antes del fix: detected_count=1 (47x64 px, 0.5% del frame) en el
    probe de composición con esta misma foto."""
    from roop.face_util import extract_face_images

    if not os.path.exists(PHOTO_02457):
        check("D: galería UI devuelve la cara grande principal de 02457", False,
              f"foto no disponible: {PHOTO_02457}")
        return

    results = extract_face_images(PHOTO_02457)
    boxes = [face_obj.bbox for face_obj, _ in results]
    areas = [((b[2] - b[0]) * (b[3] - b[1])) / (768 * 768) for b in boxes]
    biggest = max(areas) if areas else 0.0
    check(
        "D: galería UI devuelve la cara grande principal de 02457 (>30% del frame)",
        biggest > 0.30,
        f"caras={len(results)}, área máxima={biggest:.3f}",
    )


def main():
    print(f"Foto caso real: {PHOTO_02457} (existe={os.path.exists(PHOTO_02457)})")
    contracts_same_face_boxes()
    contract_product_path()
    contract_ui_gallery()
    failed = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} contratos OK")
    if failed:
        print("FALLAN:")
        for n in failed:
            print(f"  - {n}")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
