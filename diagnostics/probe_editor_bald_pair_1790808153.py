"""probe_editor_bald_pair_1790808153.py

Mide el par real de la sesión 2026-10-01 00:37:
- entrada: ui/tob/ComfyUI/input/fast_1790808102.png (+ mask_1790808102.png)
- salida : img_editor/edit_1790808153.png

Instrucción del usuario: «el hombre debe ser calvo».
Métricas: tamaños, diff global, diff dentro/fuera de la ventana integrada,
y heurística de pelo calvo (pixels oscuros tipo pelo en la parte alta del sujeto).
"""

import os
import sys
import numpy as np
import cv2

IN_PATH = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1790808102.png")
MASK_PATH = os.path.join("ui", "tob", "ComfyUI", "input", "mask_1790808102.png")
OUT_PATH = os.path.join("img_editor", "edit_1790808153.png")

ok = True


def need(path):
    global ok
    if not os.path.exists(path):
        print(f"FALTA {path}")
        ok = False
        return None
    return path


inp = cv2.imread(need(IN_PATH))
msk = cv2.imread(need(MASK_PATH), cv2.IMREAD_GRAYSCALE)
out = cv2.imread(need(OUT_PATH))

if not ok:
    sys.exit(1)

print(f"entrada: {inp.shape[1]}x{inp.shape[0]}")
print(f"salida : {out.shape[1]}x{out.shape[0]}")
print(f"mask   : {msk.shape[1]}x{msk.shape[0]}" if msk is not None else "mask   : None")

if out.shape[:2] != inp.shape[:2]:
    out_r = cv2.resize(out, (inp.shape[1], inp.shape[0]), interpolation=cv2.INTER_LANCZOS4)
    print(f"[AVISO] salida remuestreada a {inp.shape[1]}x{inp.shape[0]} para comparar (la entrega interpoló)")
else:
    out_r = out

diff = np.abs(out_r.astype(np.int16) - inp.astype(np.int16)).max(axis=2)
print(f"diff medio global: {diff.mean():.2f} niveles | píxeles >12: {(diff > 12).mean() * 100:.1f}%")

if msk is not None and msk.shape[:2] == inp.shape[:2]:
    inside = msk >= 96
    if inside.any():
        print(f"mask cobertura: {inside.mean() * 100:.1f}% | diff medio dentro: {diff[inside].mean():.2f}")
    outside = ~inside
    if outside.any():
        print(f"diff medio FUERA de la máscara: {diff[outside].mean():.2f} | píxeles fuera >12: {(diff[outside] > 12).mean() * 100:.2f}%")

# Heurística de calvicie: la cabeza del hombre está en la región medida.
# Pelo oscuro = pixels con luminancia baja en la franja superior del sujeto.
# Sin saber dónde está exactamente, medimos la franja superior 45% de la máscara.
if msk is not None and msk.shape[:2] == inp.shape[:2]:
    ys, xs = np.where(msk >= 96)
    if len(ys):
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
        print(f"bbox máscara: x=[{x0},{x1}] y=[{y0},{y1}] ({x1-x0}x{y1-y0})")
        # cabeza = tercio superior del bbox de máscara
        head_y1 = y0 + (y1 - y0) // 3
        head_mask = np.zeros_like(msk, dtype=bool)
        head_mask[y0:head_y1, x0:x1] = True
        head_mask &= msk >= 96
        for name, img in (("entrada", inp), ("salida", out_r)):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            dark = (gray[head_mask] < 70).mean() * 100
            sat = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 1][head_mask]
            print(f"{name}: px oscuros (<70) en zona cabeza: {dark:.1f}% | saturación media: {sat.mean():.1f}")
        # diff en la cabeza
        print(f"diff medio en zona cabeza: {diff[head_mask].mean():.2f}")

# Guardar recortes para inspección visual posterior
diag_dir = "diagnostics"
head_crop_in = inp[max(0, y0 - 20):y0 + (y1 - y0) // 3 + 20, max(0, x0 - 20):x1 + 20]
head_crop_out = out_r[max(0, y0 - 20):y0 + (y1 - y0) // 3 + 20, max(0, x0 - 20):x1 + 20]
cv2.imwrite(os.path.join(diag_dir, "editor_bald_head_in_1790808102.png"), head_crop_in)
cv2.imwrite(os.path.join(diag_dir, "editor_bald_head_out_1790808153.png"), head_crop_out)
cv2.imwrite(os.path.join(diag_dir, "editor_bald_diff_1790808153.png"),
            np.clip(diff * 4, 0, 255).astype(np.uint8))
print("Recortes guardados en diagnostics/editor_bald_head_{in,out}_*.png y diff ×4")
