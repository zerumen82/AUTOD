"""probe_editor_bald_sharpness_fix_check.py

Verifica el fix de nitidez de _integrate_local_generation con los datos
reales de la sesión 1790808102/1790808153: reconstruye el escenario
(source = salida original con el parche sustituido por la entrada escalada
con LANCZOS, igual que el paste) y ejecuta la integración real del manager
con y sin compensación, midiendo el ratio de nitidez dentro/fuera.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.getcwd())

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

# Ventana localizada por el probe anterior
scale = 1088.0 / 3674.0
out_small = cv2.resize(out, (int(round(w_out * scale)), int(round(h_out * scale))),
                       interpolation=cv2.INTER_AREA)
res = cv2.matchTemplate(out_small, fast, cv2.TM_CCOEFF_NORMED)
_, _, _, max_loc = cv2.minMaxLoc(res)
px, py = max_loc
inv = 1.0 / scale
bx1, by1 = int(px * inv), int(py * inv)
bw, bh = int(1088 * inv), int(960 * inv)
bx2, by2 = min(w_out, bx1 + bw), min(h_out, by1 + bh)
box = (bx1, by1, bx2, by2)
print(f"ventana: {box} ({bx2-bx1}x{by2-by1})")

from PIL import Image
from roop.img_editor.img_editor_manager import ImgEditorManager

manager = ImgEditorManager.__new__(ImgEditorManager)

source = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
generated = Image.fromarray(cv2.cvtColor(fast, cv2.COLOR_BGR2RGB))
mask_img = Image.fromarray(msk)

result = manager._integrate_local_generation(source, generated, mask_img, box)
result_bgr = cv2.cvtColor(np.asarray(result), cv2.COLOR_RGB2BGR)

msk_full = cv2.resize(msk, (bx2 - bx1, by2 - by1), interpolation=cv2.INTER_NEAREST)
canvas_mask = np.zeros((h_out, w_out), dtype=np.uint8)
canvas_mask[by1:by2, bx1:bx2] = msk_full
inside = canvas_mask >= 160
outside = canvas_mask < 40

g = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)


def lap_roi(mask_bool):
    ys, xs = np.where(mask_bool)
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    return float(cv2.Laplacian(g[y0:y1, x0:x1], cv2.CV_64F).var())


in_sharp = lap_roi(inside)
out_sharp = lap_roi(outside)
print(f"DESPUES del fix: nitidez dentro={in_sharp:.0f} fuera={out_sharp:.0f} ratio={in_sharp/max(1e-6,out_sharp):.2f}")

# Referencia: el resultado ORIGINAL sin fix (edit_1790808153.png)
g_old = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
ys, xs = np.where(inside)
y0, y1 = ys.min(), ys.max() + 1
x0, x1 = xs.min(), xs.max() + 1
old_in = float(cv2.Laplacian(g_old[y0:y1, x0:x1], cv2.CV_64F).var())
print(f"ANTES (salida real sin fix): nitidez dentro={old_in:.0f} fuera={out_sharp:.0f} ratio={old_in/max(1e-6,out_sharp):.2f}")

cv2.imwrite("diagnostics/editor_bald_head_fixed_check.png",
            result_bgr[max(0, by1):by2, max(0, bx1):bx2])
print("recorte de la ventana tras el fix: diagnostics/editor_bald_head_fixed_check.png")
