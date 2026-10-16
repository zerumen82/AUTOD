"""Cobertura por región de las máscaras despachadas por el probe de pared.

Mide, sobre los calcos dispatch_mask_*.jpg, qué fracción de cada cuadro,
tramo de pared y cuerpo queda DENTRO de la máscara que recibiría el modelo.
Sirve para decidir el alcance del cierre estructural (F3) con datos y no a
ojo: los cuadros deben quedar dentro para poder eliminarse; las personas,
fuera.
"""

import numpy as np
from PIL import Image

# Regiones de la foto de producción (1280x816), a ojo sobre los calcos.
REGIONS = {
    "cuadro_izq (village)": (30, 0, 250, 170),
    "cuadro_centro (meadow)": (400, 0, 560, 190),
    "cuadro_bw_dcha": (660, 0, 850, 140),
    "pared_izq_libre": (0, 200, 60, 400),
    "pared_entre": (560, 60, 650, 260),
    "cara_hombre": (260, 220, 400, 400),
    "cara_mujer": (770, 90, 870, 200),
    "brazo_mujer_alzado": (520, 240, 760, 300),
    "sofa_inferior": (150, 620, 560, 800),
}


def red_fraction(path):
    arr = np.asarray(Image.open(path).convert("RGB")).astype(int)
    red = (arr[:, :, 0] > 200) & (arr[:, :, 1] < 70) & (arr[:, :, 2] < 70)
    print(f"\n{path}  (rojo total = {red.mean():.1%})")
    for name, (x0, y0, x1, y1) in REGIONS.items():
        patch = red[y0:y1, x0:x1]
        print(f"  {name:26s} {patch.mean():6.1%}")


if __name__ == "__main__":
    for index in (1, 5, 6):
        red_fraction(f"diagnostics/out/dispatch_mask_{index}.jpg")
