"""probe_editor_bald_outside_detail.py

¿Se conservó la foto original fuera de la ventana integrada?
Compara nitidez (varianza Laplaciana) y diff en franjas del canvas lejos
de la máscara medida. Si se integró bien: diff~0 y nitidez igual a la
entrada. Si se entregó la generación 1088x960 interpolada a 6503x4263:
nitidez colapsada y diff alto en todo el marco.
"""

import os
import sys
import cv2
import numpy as np

IN_PATH = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1790808102.png")
MASK_PATH = os.path.join("ui", "tob", "ComfyUI", "input", "mask_1790808102.png")
OUT_PATH = os.path.join("img_editor", "edit_1790808153.png")


def lap_var(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


for p in (IN_PATH, MASK_PATH, OUT_PATH):
    if not os.path.exists(p):
        print(f"FALTA {p}")
        sys.exit(1)

inp = cv2.imread(IN_PATH)
out = cv2.imread(OUT_PATH)
msk = cv2.imread(MASK_PATH, cv2.IMREAD_GRAYSCALE)

h_in, w_in = inp.shape[:2]
h_out, w_out = out.shape[:2]
print(f"entrada {w_in}x{h_in} | salida {w_out}x{h_out}")

# Foto original de referencia: buscar por tamaño en la carpeta de la UI no es
# trivial; la entrada fast_ es su versión reducida. La salida a resolución
# original (6503x4263) solo puede venir de la foto original integrada.
# Como no conocemos la foto original, comparamos nitidez de la SALIDA con la
# de la entrada re-escalada a 6503x4263 (lo que la integración debería haber
# conservado en las zonas originales, ~nítidas).

out_small = cv2.resize(out, (w_in, h_in), interpolation=cv2.INTER_AREA)
out_small_up = cv2.resize(out_small, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)
inp_up = cv2.resize(inp, (w_out, h_out), interpolation=cv2.INTER_LANCZOS4)

# Franjas lejos de la máscara (bbox máscara x=[185,733] y=[57,779] a 1088x960)
sx, sy = w_out / w_in, h_out / h_in
zones = {
    "izquierda (x<170)": (0, 170, 0, h_in),
    "derecha (x>750)": (750, w_in, 0, h_in),
    "abajo (y>800)": (0, w_in, 800, h_in),
}
for name, (x0, x1, y0, y1) in zones.items():
    X0, X1, Y0, Y1 = int(x0 * sx), int(x1 * sx), int(y0 * sy), int(y1 * sy)
    in_z = cv2.cvtColor(inp_up[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    out_z = cv2.cvtColor(out[Y0:Y1, X0:X1], cv2.COLOR_BGR2GRAY)
    # diff a escala de entrada: bajar la franja de la salida a su tamaño original
    out_z_small = cv2.resize(out[Y0:Y1, X0:X1], (x1 - x0, y1 - y0),
                             interpolation=cv2.INTER_AREA).astype(np.int16)
    in_z_small = inp[y0:y1, x0:x1].astype(np.int16)
    diff = float(np.abs(out_z_small - in_z_small).mean())
    print(
        f"{name}: nitidez in={lap_var(in_z):.0f} out={lap_var(out_z):.0f} "
        f"(ratio out/in={lap_var(out_z) / max(1e-6, lap_var(in_z)):.2f}) diff={diff:.1f}"
    )

# ¿La salida entera es la generación 1088x960 interpolada?
out_ds = cv2.resize(out, (w_in, h_in), interpolation=cv2.INTER_AREA)
diff_full = np.abs(out_ds.astype(np.int16) - inp.astype(np.int16)).max(axis=2)
print(f"\ndiff medio global (salida reducida vs entrada): {diff_full.mean():.2f}")

# Mapa de diff por celdas 8x8 para localizar qué zonas conservaron el original
cell = 8
gh, gw = h_in // cell, w_in // cell
grid = diff_full[: gh * cell, : gw * cell].reshape(gh, cell, gw, cell).mean(axis=(1, 3))
kept = (grid < 8).sum()
print(f"celdas 8x8 con diff<8 (original conservado): {kept}/{gh * gw} ({kept / (gh * gw) * 100:.1f}%)")
np.save(os.path.join("diagnostics", "editor_bald_diffgrid_1790808153.npy"), grid)
print("Grid guardado en diagnostics/editor_bald_diffgrid_1790808153.npy")
