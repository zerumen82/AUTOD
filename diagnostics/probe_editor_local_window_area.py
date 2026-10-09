"""Comprueba que ventana local produce _appearance_work_region antes y despues
del arreglo del suelo de 1 MP, con la foto y la mascara reales de la sesion
edit_1790845755 (800x533, un solo rostro que ocupa el 52 % del alto).

Usa los metodos reales del manager sin inicializarlo.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))

PHOTO_SIZE = (800, 533)          # log: "Muestra la foto completa 800x533"
FULL_MASK = "ui/tob/ComfyUI/input/mask_1790845644.png"   # 1248x832 (escala LongCat)


def window_for(manager, image, mask):
    work, work_mask, box = manager._appearance_work_region(image, mask)
    return work.size, box


def main():
    from roop.img_editor.img_editor_manager import ImgEditorManager

    manager = object.__new__(ImgEditorManager)

    # La mascara guardada esta a 1248x832; el trabajo real ocurre a 800x533.
    mask = Image.open(FULL_MASK).convert("L").resize(PHOTO_SIZE, Image.Resampling.BILINEAR)
    image = Image.new("RGB", PHOTO_SIZE, (120, 120, 120))

    arr = np.asarray(mask)
    core = arr >= 160
    ys, xs = np.where(core)
    print(f"foto {PHOTO_SIZE}  photo_area={PHOTO_SIZE[0] * PHOTO_SIZE[1]:,}")
    print(f"mask core: cobertura={core.mean():.1%}  "
          f"bbox={int(xs.max() - xs.min() + 1)}x{int(ys.max() - ys.min() + 1)}")

    size, box = window_for(manager, image, mask)
    w, h = size
    frac = (w * h) / float(PHOTO_SIZE[0] * PHOTO_SIZE[1])
    print(f"\n_appearance_work_region -> ventana {w}x{h} box={box}")
    print(f"  ventana = {frac:.1%} de la foto   (antes del arreglo era 100 %)")
    if frac >= 0.98:
        print("  VEREDICTO: sigue siendo la foto entera")
    elif frac >= 0.80:
        print("  VEREDICTO: ventana casi total, la localizacion sigue sin efecto")
    else:
        print("  VEREDICTO: ventana local real")


if __name__ == "__main__":
    main()
