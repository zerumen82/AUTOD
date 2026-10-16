"""probe_editor_bald_seam_localize.py

Localiza la ventana del crop (3674x3276) dentro de la salida 6503x4263 por
template matching con la entrada fast_1790808102.png (que ES el contenido de
esa ventana a 1088x960), y mide nitidez dentro vs fuera de la mascara
proyectada. Con eso se cierra si el paste escribio la generacion interpolada
3.38x dentro de la mascara sobre una foto mucho mas nitida.
"""

import os
import sys
import cv2
import numpy as np

FAST = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1790808102.png")
MASK = os.path.join("ui", "tob", "ComfyUI", "input", "mask_1790808102.png")
OUT = os.path.join("img_editor", "edit_1790808153.png")

for p in (FAST, MASK, OUT):
    if not os.path.exists(p):
        if p == OUT:
            print(f"SKIP: artefacto historico ausente ({OUT}); "
                  "la salida de la sesion 2026-10-01 no esta en el repo")
            sys.exit(0)
        print(f"FALTA {p}")
        sys.exit(1)

fast = cv2.imread(FAST)
msk = cv2.imread(MASK, cv2.IMREAD_GRAYSCALE)
out = cv2.imread(OUT)
h_out, w_out = out.shape[:2]

# La ventana es 3674x3276 (del log). Escala de generacion: 1088/3674
scale = 1088.0 / 3674.0
print(f"escala ventana->generacion: {scale:.4f} (interpolacion x{1/scale:.2f})")

# Reducir la salida al mismo factor y buscar la entrada en ella
out_small = cv2.resize(out, (int(round(w_out * scale)), int(round(h_out * scale))),
                       interpolation=cv2.INTER_AREA)
res = cv2.matchTemplate(out_small, fast, cv2.TM_CCOEFF_NORMED)
_, max_val, _, max_loc = cv2.minMaxLoc(res)
px, py = max_loc
print(f"match: pos=({px},{py}) conf={max_val:.3f} (out_small {out_small.shape[1]}x{out_small.shape[0]})")

if max_val < 0.5:
    print("Confianza baja: la ventana no encaja; abortar")
    sys.exit(1)

# Box real en la salida
inv = 1.0 / scale
bx1, by1 = int(px * inv), int(py * inv)
bw, bh = int(1088 * inv), int(960 * inv)
bx2, by2 = min(w_out, bx1 + bw), min(h_out, by1 + bh)
print(f"ventana real: ({bx1},{by1})-({bx2},{by2}) {bx2-bx1}x{by2-by1}")

# Proyectar la mascara al box real
msk_full = cv2.resize(msk, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_NEAREST)
canvas_mask = np.zeros((h_out, w_out), dtype=np.uint8)
canvas_mask[by1:by2, bx1:bx2] = msk_full
inside = canvas_mask >= 160
print(f"cobertura de mascara en la foto: {inside.mean()*100:.1f}%")

out_gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)


def lap_var_roi(roi):
    return float(cv2.Laplacian(roi, cv2.CV_64F).var())


# Nitidez dentro vs fuera en celdas 256
cell = 256
gh, gw = h_out // cell, w_out // cell
sharp = np.zeros((gh, gw))
fr = np.zeros((gh, gw))
for gy in range(gh):
    for gx in range(gw):
        ys, ye = gy * cell, (gy + 1) * cell
        xs, xe = gx * cell, (gx + 1) * cell
        sharp[gy, gx] = lap_var_roi(out_gray[ys:ye, xs:xe])
        fr[gy, gx] = float(inside[ys:ye, xs:xe].mean())
hi = fr >= 0.5
lo = (fr <= 0.05)
print(f"celdas EN mascara (n={int(hi.sum())}): nitidez mediana {np.median(sharp[hi]):.0f}")
print(f"celdas FUERA mascara (n={int(lo.sum())}): nitidez mediana {np.median(sharp[lo]):.0f}")
if hi.sum() and lo.sum():
    r = np.median(sharp[hi]) / max(1e-6, np.median(sharp[lo]))
    print(f"RATIO nitidez dentro/fuera: {r:.2f}")
    print("  ~1.0 sin costura | <<1 generacion interpolada mas borrosa que el original")

# Nitidez de la entrada fast (el contenido de la ventana a 1088x960) vs la
# misma region de la salida reducida a 1088x960 (lo que quedo escrito)
win_out_small = cv2.resize(out[by1:by2, bx1:bx2], (1088, 960), interpolation=cv2.INTER_AREA)
outside_win_mask = msk < 160
g_fast = cv2.cvtColor(fast, cv2.COLOR_BGR2GRAY)
g_win = cv2.cvtColor(win_out_small, cv2.COLOR_BGR2GRAY)
if outside_win_mask.any():
    d_out = np.abs(win_out_small.astype(np.int16) - fast.astype(np.int16)).max(axis=2)
    print(f"diff medio FUERA de mascara dentro de la ventana: {d_out[outside_win_mask].mean():.2f}")
    print(f"  (0 = foto original conservada fuera de la mascara)")
print(f"nitidez entrada fast (ventana 1088x960): {lap_var_roi(g_fast):.0f}")
print(f"nitidez salida ventana reducida 1088x960: {lap_var_roi(g_win):.0f}")

# Parche visual: cabeza dentro de la mascara a resolucion original
ys, xs = np.where(inside)
if len(xs):
    cy, cx = int(ys.mean()), int(xs.mean())
    patch = out[max(0, cy - 400):cy + 400, max(0, cx - 400):cx + 400]
    cv2.imwrite("diagnostics/editor_bald_patch_head_fullres.png", patch)
    # mismo parche de la foto original (entrada fast upscaled al box)
    fast_up = cv2.resize(fast, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_LANCZOS4)
    patch_in = fast_up[max(0, cy - by1 - 400):cy - by1 + 400, max(0, cx - bx1 - 400):cx - bx1 + 400]
    cv2.imwrite("diagnostics/editor_bald_patch_head_inputup.png", patch_in)
    print("parches full-res guardados: editor_bald_patch_head_{fullres,inputup}.png")
