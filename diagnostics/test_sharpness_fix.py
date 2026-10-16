# -*- coding: utf-8 -*-
"""Experimento offline: compensacion de nitidez del parche de paste.

Contexto (log real): el parche llega de 1 MP escalado x3.51 y mide
Laplacian var 16 frente a 402 del recorte original (y 28 vs 244 en el
refino). El unsharp actual (sigma 1.5, una pasada, tope 0.90) no alcanza.

Prueba, sobre un recorte real de la foto (zona NO pegada de la edicion
previa, equivalente a source):

  A) actual:  1 pasada sigma1.5, fuerza min(0.9, 0.5*(src/ed-1))
  B) candidato: multi-pasada (1.0, 2.0, 4.0) con tope por pasada,
     para hasta 70% de la nitidez fuente.

Mide: nitidez final, MAE vs fuente (halo/ruido) y sobrepaso maximo.
"""
import numpy as np
import cv2
from PIL import Image

SRC_EDIT = r"D:\PROJECTS\AUTOAUTO\img_editor\edit_1791586497.png"
# Fuente: fuera de la zona pegada (x>2600 a escala 6395) la edicion es la
# foto original.
BOX = (3000, 900, 3000 + 1700, 900 + 1700)
SCALE = 3.51


def lapvar(a):
    g = np.clip(a, 0, 255).astype(np.uint8) if a.dtype != np.uint8 else a
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def soften(arr, scale):
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(8, int(w / scale)), max(8, int(h / scale))),
                       interpolation=cv2.INTER_AREA)
    back = cv2.resize(small, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return back


def sharpen_current(arr, src_sharp, ed_sharp):
    if not (src_sharp > 1.0 and ed_sharp < src_sharp * 0.80):
        return arr, ed_sharp, None
    strength = float(min(0.90, 0.50 * (src_sharp / max(1.0, ed_sharp) - 1.0)))
    if strength < 0.05:
        return arr, ed_sharp, strength
    blur = cv2.GaussianBlur(arr, (0, 0), 1.5)
    out = cv2.addWeighted(arr, 1.0 + strength, blur, -strength, 0)
    return out, lapvar(out), strength


def sharpen_multiscale(arr, src_sharp, ed_sharp, target_ratio=0.70,
                       passes=((1.0, 0.6), (2.0, 0.6), (4.0, 0.5)),
                       rounds=2):
    trail = [ed_sharp]
    used = []
    for _ in range(rounds):
        for sigma, cap in passes:
            if ed_sharp >= src_sharp * target_ratio:
                break
            strength = float(min(cap, 0.50 * (src_sharp / max(1.0, ed_sharp) - 1.0)))
            if strength < 0.05:
                continue
            blur = cv2.GaussianBlur(arr, (0, 0), sigma)
            arr = cv2.addWeighted(arr, 1.0 + strength, blur, -strength, 0)
            ed_sharp = lapvar(arr)
            trail.append(ed_sharp)
            used.append((sigma, strength))
    return arr, ed_sharp, (used, trail)


def main():
    photo = np.asarray(Image.open(SRC_EDIT).convert("RGB").crop(BOX),
                       dtype=np.float32)
    src_sharp = lapvar(photo)
    patch = soften(photo, SCALE)
    ed_sharp = lapvar(patch)
    print(f"fuente BOX={BOX} nitidez={src_sharp:.0f} | "
          f"parche escalado x{SCALE} nitidez={ed_sharp:.0f} "
          f"ratio={ed_sharp/src_sharp:.3f}")

    gray = photo.mean(axis=2)
    mu = cv2.boxFilter(gray, -1, (9, 9))
    mu2 = cv2.boxFilter(gray * gray, -1, (9, 9))
    std = np.sqrt(np.maximum(mu2 - mu * mu, 0))
    flat = std < 5.0

    def evaluate(tag, img, used):
        sharp = lapvar(img)
        mae = float(np.abs(img - photo).mean())
        dev = np.abs(img - photo).mean(axis=2)
        print(f"  [{tag:28s}] nitidez={sharp:5.0f} ({sharp/src_sharp:3.0%}) "
              f"MAE={mae:5.2f} plano media={dev[flat].mean():4.2f} "
              f"p99={np.percentile(dev[flat], 99):5.2f} "
              f"pasadas={len(used)}")
        return sharp

    a, sa, wa = sharpen_current(patch.copy(), src_sharp, ed_sharp)
    evaluate("A actual (sigma1.5 x1)", a, [wa] if wa else [])

    print("  variantes B (multi):")
    for target in (0.45, 0.55, 0.65, 0.70):
        for caps in ((0.5, 0.5, 0.4), (0.6, 0.6, 0.5)):
            passes = tuple((s, c) for s, c in zip((1.0, 2.0, 4.0), caps))
            b, sb, info = sharpen_multiscale(
                patch.copy(), src_sharp, ed_sharp,
                target_ratio=target, passes=passes,
            )
            used, trail = info
            tag = f"t{target:.2f} caps{caps[0]}"
            evaluate(tag, b, used)
    # trayectoria de la varieta elegible mas cercana a t0.55/0.6
    b, sb, info = sharpen_multiscale(
        patch.copy(), src_sharp, ed_sharp, target_ratio=0.55,
        passes=((1.0, 0.5), (2.0, 0.5), (4.0, 0.4)),
    )
    print("  trayectoria t0.55 caps0.5: " + "->".join(
        f"{v:.0f}" for v in info[1]))


if __name__ == "__main__":
    main()
