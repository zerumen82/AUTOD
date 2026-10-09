"""probe_editor_bald_sharpness_seam.py

Mide la costura de nitidez de la integración local en edit_1790808153.png.
La generación se hizo a 1088x960 sobre un crop de 3674x3276 de la foto
original 6503x4263. Si el paste escribe esa generación interpolada, dentro
de la máscara la nitidez cae ~3,4x respecto a la foto original de alrededor.

Nota: fast_1790808102.png es el crop local enviado al modelo (1088x960);
la salida es la foto completa 6503x4263. Para medir la costura usamos la
SALIDA (que integra generación + original) y comparamos nitidez dentro vs
fuera de la máscara proyectada a la resolución original.
"""

import os
import sys
import cv2
import numpy as np

MASK_PATH = os.path.join("ui", "tob", "ComfyUI", "input", "mask_1790808102.png")
OUT_PATH = os.path.join("img_editor", "edit_1790808153.png")


def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


for p in (MASK_PATH, OUT_PATH):
    if not os.path.exists(p):
        print(f"FALTA {p}")
        sys.exit(1)

msk = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)
out = cv2.imread(OUT_PATH)

h_out, w_out = out.shape[:2]
# La máscara corresponde al crop local 3674x3276 en el offset de la ventana.
# El log: contexto focal 25%, crop=32.7% de la foto. El crop 3674x3276 es el
# 32.7% del area. Recuperamos el offset desde la relación de tamaños:
# mask_1790808102 es el work_mask 1088x960 ya escalado? No: work_mask es el
# crop 3674x3276 reducido por el cliente a 1088x960. La máscara guardada en
# input/ es la misma resolución 1088x960 (ya comprobado: mask es 1088x960).
# El box exacto de la ventana no se guarda, pero el _crop_focal_context cap
# 0.35 y el tamaño del crop permiten estimarlo. Para la costura basta saber
# que la ventana cubre el centro-izquierda; proyectamos la máscara al
# rectángulo estimado del crop: escala = sqrt(3674*3276/(1088*960)) ~= 3.38.
# Sin el box real, usamos un indicador global: repartir la salida en celdas
# y comparar nitidez celda a celda contra la distancia a la máscara.
print(f"salida: {w_out}x{h_out}")
print(f"mask  : {msk.shape[1]}x{msk.shape[0]}")

# Aproximación robusta: nitidez de la salida en franjas horizontales y
# comparación con la fracción de máscara por franja. Si la generación
# interpolada está solo dentro de la máscara, las franjas con más máscara
# tienen menos nitidez neta.
mask_frac_rows = []
sharp_rows = []
rows = 12
h_m, w_m = msk.shape
for i in range(rows):
    y0m, y1m = h_m * i // rows, h_m * (i + 1) // rows
    frac = float((msk[y0m:y1m] >= 160).mean())
    mask_frac_rows.append(frac)
print("fraccion de mascara por franja (arriba->abajo):",
      " ".join(f"{f:.2f}" for f in mask_frac_rows))

# Nitidez de la salida completa y de su version reducida 1088x960
out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
print(f"nitidez salida completa: {lap_var(out_gray):.0f}")
out_small = cv2.resize(out, (1088, 960), interpolation=cv2.INTER_AREA)
print(f"nitidez salida a 1088x960: {lap_var(cv2.cvtColor(out_small, cv2.COLOR_BGR2GRAY)):.0f}")

# La foto original (25.7MP) en zonas sin generacion deberia tener nitidez alta.
# Celdas 16x16: nitidez vs fraccion de mascara local (mask escalada al canvas).
msk_full = cv2.resize(msk, (w_out, h_out), interpolation=cv2.INTER_NEAREST)
cell = 256
gh, gw = h_out // cell, w_out // cell
sharp_grid = np.zeros((gh, gw))
frac_grid = np.zeros((gh, gw))
for gy in range(gh):
    for gx in range(gw):
        ys, ye = gy * cell, (gy + 1) * cell
        xs, xe = gx * cell, (gx + 1) * cell
        sharp_grid[gy, gx] = lap_var(out_gray[ys:ye, xs:xe])
        frac_grid[gy, gx] = float((msk_full[ys:ye] >= 160).mean())
hi_mask = frac_grid >= 0.5
lo_mask = frac_grid <= 0.05
print(f"celdas mayormente EN mascara (n={hi_mask.sum()}): nitidez mediana {np.median(sharp_grid[hi_mask]):.0f}")
print(f"celdas FUERA de mascara (n={lo_mask.sum()}): nitidez mediana {np.median(sharp_grid[lo_mask]):.0f}")
if hi_mask.sum() and lo_mask.sum():
    ratio = np.median(sharp_grid[hi_mask]) / max(1e-6, np.median(sharp_grid[lo_mask]))
    print(f"ratio nitidez dentro/fuera: {ratio:.2f} (1.0 = sin costura; <<1 = generacion borrosa pegada)")

# Guardar el recorte de la cabeza ya generado por el probe anterior ya existe;
# aqui guardamos un par de parches 512px dentro y fuera para inspeccion visual
ys, xs = np.where(msk_full >= 160)
if len(xs):
    cy, cx = int(ys.mean()), int(xs.mean())
    patch_in = out[max(0, cy - 256):cy + 256, max(0, cx - 256):cx + 256]
    cv2.imwrite("diagnostics/editor_bald_patch_inside_1790808153.png", patch_in)
    px = min(w_out - 512, max(0, int(cx + (w_out - cx) / 2)))
    patch_out = out[0:512, px:px + 512]
    cv2.imwrite("diagnostics/editor_bald_patch_outside_1790808153.png", patch_out)
    print("parches dentro/fuera guardados en diagnostics/editor_bald_patch_*")
