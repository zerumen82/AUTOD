# -*- coding: utf-8 -*-
"""Mide si el feather nuevo (0.02 -> ~68px en recorte 3428) surtio efecto.

Metodo: la zona pegada es la UNICA que difiere entre OLD (edit_1791586497,
feather 6px) y NEW (edit_1791587926, feather ~68px); fuera ambas son la
misma foto. El perfil de diferencia |OLD-NEW| por columna/fila se desvanece
en el borde con el ancho del feather:

  OLD (radio 6px reales -> 1.5px a escala 1600): rampa corta.
  NEW (radio ~68px reales -> ~17px a escala 1600): rampa ancha.

Ademas, dentro de NEW se mide el paso de luminancia y de nitidez en el mismo
borde para decir si el escalon persiste.

Solo numeros.
"""
import numpy as np
import cv2
from PIL import Image

OLD = r"D:\PROJECTS\AUTOAUTO\img_editor\edit_1791586497.png"
NEW = r"D:\PROJECTS\AUTOAUTO\img_editor\edit_1791587926.png"
W = 1600


def load(path):
    im = Image.open(path).convert("RGB")
    h = int(W * im.height / im.width)
    im = im.resize((W, h), Image.Resampling.BOX)
    return np.asarray(im, dtype=np.float32).mean(axis=2)


def ramp_width(prof, x_cross, direction, half=45):
    """Ancho 10%-90% de la rampa alrededor del cruce de 50% del meseta."""
    lo = max(0, x_cross - half)
    hi = min(len(prof), x_cross + half)
    seg = prof[lo:hi]
    plateau = float(np.median(prof[max(0, x_cross - 60):x_cross + 60]))
    if plateau < 1e-3:
        return float("nan")
    target = plateau * 0.5
    # recorrer hacia fuera del centro hasta cruzar el 50%
    # direction=+1: cruce en el borde DERECHO (la prof cae hacia fuera)
    idx50 = None
    rng = range(len(seg) - 1, -1, -1) if direction > 0 else range(len(seg))
    xs = [lo + i for i in rng]
    # encontrar el indice donde la prof cruza target viniendo del meseta
    # meseta dentro de la caja: buscar el 50% mas cercano al centro
    center = x_cross - lo
    order = sorted(range(len(seg)), key=lambda i: abs(i - center))
    for i in order:
        if seg[i] <= target:
            idx50 = i
            break
    if idx50 is None:
        return float("nan")
    # de ahi, caminar hacia fuera hasta el 90% y hasta el 10%
    def walk(step_dir):
        i = idx50
        while 0 <= i + step_dir < len(seg):
            i += step_dir
            if seg[i] >= plateau * 0.9 or seg[i] <= plateau * 0.1:
                break
        return i
    out_dir = 1 if xs[idx50] > center else -1
    i90 = idx50
    while 0 <= i90 + out_dir < len(seg) and seg[i90] < plateau * 0.9:
        i90 += out_dir
    i10 = idx50
    while 0 <= i10 - out_dir < len(seg) and seg[i10] > plateau * 0.1:
        i10 -= out_dir
    return abs(i90 - i10)


def find_crossings(prof, m=40):
    """Devuelve (x_izq, x_der) donde el perfil cruza el 50% de su meseta."""
    seg = prof[m:-m]
    plateau = float(np.median(seg[seg > np.percentile(seg, 60)]))
    if plateau < 0.5:
        return None, None
    half = plateau * 0.5
    xs = np.arange(m, len(prof) - m)
    above = seg >= half
    idx = np.where(above)[0]
    if len(idx) == 0:
        return None, None
    return int(xs[idx[0]]), int(xs[idx[-1]])


def lap_var(a):
    return float(cv2.Laplacian(np.clip(a, 0, 255).astype(np.uint8),
                               cv2.CV_64F).var())


def edge_inside_new(a, axis, pos, lo, hi, band=4):
    if axis == 0:
        inner = a[lo:hi, pos:pos + 3 * band]
        outer = a[lo:hi, pos - 3 * band:pos]
        near_i = a[lo:hi, pos:pos + band]
        near_o = a[lo:hi, pos - band:pos]
    else:
        inner = a[pos:pos + 3 * band, lo:hi]
        outer = a[pos - 3 * band:pos, lo:hi]
        near_i = a[pos:pos + band, lo:hi]
        near_o = a[pos - band:pos, lo:hi]
    step = float(near_i.mean() - near_o.mean())
    li, lo_ = lap_var(inner), lap_var(outer)
    return step, li, lo_


def main():
    a_old, a_new = load(OLD), load(NEW)
    d = np.abs(a_old - a_new)
    col = d.mean(axis=0)
    row = d.mean(axis=1)
    print(f"[diff] media global={d.mean():.3f}")
    x1, x2 = find_crossings(col)
    y1, y2 = find_crossings(row)
    print(f"[zona pegada estimada] x[{x1}-{x2}] y[{y1}-{y2}] "
          f"(meseta col={np.median(col[x1+20:x2-20]):.2f}, "
          f"row={np.median(row[y1+20:y2-20]):.2f})")
    if x1 is None or y1 is None:
        print("sin zona detectada")
        return
    # Rampas OLD vs NEW: perfil de diferencia en cada borde
    for name, prof, pos in (("V izq", col, x1), ("V der", col, x2),
                            ("H arr", row, y1), ("H abj", row, y2)):
        # para el borde izquierdo la meseta esta a la derecha; usar la
        # diferencia contra la meseta de la propia zona
        w_old = ramp_width(col if name.startswith("V") else row, pos, 0)
        print(f"  {name} pos={pos}: ancho_rampa(diff OLD-NEW)~{w_old:.1f}px "
              f"perfil[-8..+8]="
              f"{np.round((col if name.startswith('V') else row)[pos-8:pos+9], 2)}")
    # Paso y nitidez DENTRO de NEW en los 4 bordes
    for tag, axis, pos, lo, hi in (
        ("V izq", 0, x1, y1, y2), ("V der", 0, x2, y1, y2),
        ("H arr", 1, y1, x1, x2), ("H abj", 1, y2, x1, x2),
    ):
        step, li, lo_ = edge_inside_new(a_new, axis, pos, lo, hi)
        step_o, li_o, lo_o = edge_inside_new(a_old, axis, pos, lo, hi)
        print(f"  [borde {tag} x/y={pos}] NEW: paso={step:+.2f} "
              f"nitidez in={li:.0f} out={lo_:.0f} | "
              f"OLD: paso={step_o:+.2f} nitidez in={li_o:.0f} out={lo_o:.0f}")


if __name__ == "__main__":
    main()
