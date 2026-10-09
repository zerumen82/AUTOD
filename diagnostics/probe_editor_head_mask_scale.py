"""Mide la escala real de _head_mask_from_faces en la foto 800x533 de la sesion
edit_1790845755 y comprueba cuanto cubre la elipse de cabeza.

La mascara enviada a ComfyUI en esa sesion media 26,8 % del cuadro con un bbox
de 432x810 (35 % x 97 % del alto) en un solo componente: eso no es una cabeza,
es cabeza + torso + ventana. Aqui se reconstruye la mascara con los
multiplicadores reales del codigo (rx=fw*0.78, ry=fh*1.05, lobulo de nuca
rx=fw*0.40 ry=fh*0.42 desplazado fw*0.48, y GaussianBlur(31,31)) sobre las
cajas de cara detectadas, y se mide cuanto ocupa.

No requiere GPU ni ComfyUI: solo InsightFace con los modelos locales.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

IMAGE = "ui/tob/ComfyUI/input/fast_1790845644.png"  # 1248x832, la foto que entro
FULL_MASK = "ui/tob/ComfyUI/input/mask_1790845644.png"


def ellipse_mask(shape, faces, rx_mul, ry_mul, nape=True, blur=31):
    h, w = shape
    head = np.zeros((h, w), dtype=np.uint8)
    for f in faces:
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        fw, fh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        cx = int(round((x1 + x2) * 0.5))
        cy = int(round(y1 + fh * 0.22))
        rx = max(2, int(round(fw * rx_mul)))
        ry = max(2, int(round(fh * ry_mul)))
        cv2.ellipse(head, (cx, cy), (rx, ry), 0, 0, 360, 255, -1)
        if nape:
            kps = np.asarray(getattr(f, "kps", None), dtype=np.float32)
            if kps.shape[0] >= 3 and kps.shape[1] >= 2:
                eye_cx = float((kps[0, 0] + kps[1, 0]) * 0.5)
                nose = float(kps[2, 0] - eye_cx)
                if abs(nose) >= fw * 0.04:
                    back = -1 if nose > 0 else 1
                    ncx = int(round(cx + back * fw * 0.48))
                    ncy = int(round(y1 + fh * 0.42))
                    nrx = max(2, int(round(fw * 0.40)))
                    nry = max(2, int(round(fh * 0.42)))
                    cv2.ellipse(head, (ncx, ncy), (nrx, nry), 0, 0, 360, 255, -1)
    if blur:
        head = cv2.GaussianBlur(head, (blur, blur), 0)
    return head


def report(tag, head, shape):
    h, w = shape
    core = head >= 160
    cov = float(core.mean())
    ys, xs = np.where(core)
    if not len(xs):
        print(f"{tag}: sin nucleo")
        return cov
    bw, bh = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
    print(f"{tag}: cobertura={cov:.1%}  bbox={int(bw)}x{int(bh)} "
          f"= {bw / w:.0%} x {bh / h:.0%} del cuadro")
    return cov


def main():
    from insightface.app import FaceAnalysis

    img = Image.open(IMAGE).convert("RGB")
    arr = np.asarray(img)
    fa = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"],
        root=os.path.dirname(os.path.abspath(".")),
    )
    fa.prepare(ctx_id=-1, det_size=(1024, 1024), det_thresh=0.5)
    faces = fa.get(arr[:, :, ::-1])
    print(f"caras detectadas: {len(faces)}  foto {img.size}")

    shape = arr.shape[:2]
    for i, f in enumerate(faces):
        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        fw, fh = x2 - x1, y2 - y1
        print(f"  cara{i}: bbox=({int(x1)},{int(y1)})-{int(x2)}x{int(y2)} "
              f"fw={fw:.0f} fh={fh:.0f}  fw/iw={fw / img.size[0]:.1%} fh/ih={fh / img.size[1]:.1%}")

    print("\n-- codigo actual (rx=0.78, ry=1.05, nuca, blur31) --")
    cur = ellipse_mask(shape, faces, 0.78, 1.05)
    report("actual", cur, shape)

    print("\n-- candidatos --")
    for rx_mul, ry_mul, nape, blur in (
        (0.78, 1.05, True, 31),   # actual
        (0.62, 0.80, True, 31),   # cabeza contenida
        (0.58, 0.72, True, 21),
        (0.55, 0.68, False, 21),
    ):
        m = ellipse_mask(shape, faces, rx_mul, ry_mul, nape=nape, blur=blur)
        report(f"rx={rx_mul} ry={ry_mul} nuca={nape} blur={blur}", m, shape)

    if os.path.exists(FULL_MASK):
        real = np.asarray(Image.open(FULL_MASK).convert("L"))
        print(f"\n-- mascara real de la sesion: {real.shape} --")
        report("real", real, shape)


if __name__ == "__main__":
    main()
