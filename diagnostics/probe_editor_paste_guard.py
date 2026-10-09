"""Verifica la contención del pegado en _integrate_local_generation.

Contrato vigente (MEMORY 2026-10-01): el guard de altura por forma (60 %
del alto) fue retirado porque era tan hardcodeado como la elipse que
reemplazaba, y vetaba retratos legítimos al 97 % del alto. La protección
pasó a ser MEDIR el cambio y recortarlo al sujeto ya localizado con
`limit_mask`, así que aquí se verifica esa propiedad:

1. Un recorte de cabeza legítimo (~25 % del alto) sí se pega.
2. Una zona medida que abarca casi todo el alto, pegada con limit_mask
   sobre el sujeto, NO escribe un solo píxel fuera de ese sujeto
   (edit_1790845755.png: el fallo era componer la foto entera
   —rectángulo translúcido y doble cabeza—, no el hecho de tener
   97 % de alto).
3. La rama keep_measured_mask=True (paso único medido) integra sin
   crashear: antes leía measured_zone sin asignarla y la ruta rápida
   moría con UnboundLocalError.

Caso 2 sin limit_mask se reporta como referencia, sin aserción: sin
sujeto que acotar no hay propiedad que exigir.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath("."))


def build(photo_size, mask_box, shift_px):
    """Foto, mascara y una 'generacion' desplazada `shift_px` a la derecha."""
    rng = np.random.default_rng(7)
    arr = rng.integers(90, 130, (photo_size[1], photo_size[0], 3), dtype=np.uint8)
    photo = Image.fromarray(arr)

    w, h = photo_size
    m = np.zeros((h, w), np.uint8)
    x1, y1, x2, y2 = mask_box
    m[y1:y2, x1:x2] = 255
    mask = Image.fromarray(m, "L")

    # salida "generada": la misma textura corrida
    gen = Image.fromarray(np.roll(arr, shift_px, axis=1))
    return photo, gen, mask


def limit_image(limit_box, size):
    arr = np.zeros((size[1], size[0]), np.uint8)
    x1, y1, x2, y2 = limit_box
    arr[y1:y2, x1:x2] = 255
    return Image.fromarray(arr, "L")


def changed_mask(photo, out, threshold=8):
    a = np.asarray(photo, np.int16)
    b = np.asarray(out, np.int16)
    return np.abs(a - b).max(axis=2) > threshold


def main():
    from roop.img_editor.img_editor_manager import ImgEditorManager
    manager = object.__new__(ImgEditorManager)

    # 1) recorte de cabeza legitimo: ~25 % del alto -> debe pegar
    photo, gen, mask = build((800, 533), (600, 150, 780, 285), shift_px=3)
    out = manager._integrate_local_generation(photo, gen, mask, (0, 0, 800, 533))
    changed = float(changed_mask(photo, out).mean())
    print(f"recorte de cabeza (25 % alto): pegado, cambio={changed:.1%} del cuadro")
    assert changed > 0.001, "un recorte de cabeza debe pegarse"

    # 2) zona medida al 97 % del alto (el caso real) pegada sobre un sujeto
    #    acotado: debe editar dentro y no escribir ni un píxel fuera.
    subject_box = (600, 120, 920, 560)
    photo, gen, mask = build((1248, 832), (544, 0, 976, 810), shift_px=40)
    out = manager._integrate_local_generation(
        photo, gen, mask, (0, 0, 1248, 832),
        limit_mask=limit_image(subject_box, photo.size),
    )
    changed = changed_mask(photo, out)
    inside = np.zeros(changed.shape, bool)
    x1, y1, x2, y2 = subject_box
    inside[y1:y2, x1:x2] = True
    leak = float(changed[~inside].mean())
    edited_inside = float(changed[inside].mean())
    print(
        f"zona al 97 % del alto con sujeto acotado: "
        f"dentro={edited_inside:.1%} del sujeto, fuga fuera={leak:.6%}"
    )
    assert edited_inside > 0.01, (
        f"el sujeto pedido debe editarse, solo cambió {edited_inside:.1%}"
    )
    assert leak == 0.0, f"no debe escribir fuera del sujeto, fugó {leak:.6%}"

    # referencia informativa: sin limit_mask no hay sujeto que acotar
    photo, gen, mask = build((1248, 832), (544, 0, 976, 810), shift_px=40)
    out = manager._integrate_local_generation(
        photo, gen, mask, (0, 0, 1248, 832),
    )
    print(
        f"[ref] sin limit_mask: cambio={float(changed_mask(photo, out).mean()):.1%} "
        f"del cuadro (sin sujeto acotado no se exige contención)"
    )

    # 3) la misma zona con keep_measured_mask (paso único medido, la rama
    #    que antes leía measured_zone sin asignarla y crasheaba con
    #    UnboundLocalError → «Ruta rápida falló»). Esa bandera pega la
    #    máscara medida entera e ignora limit_mask («a measured request is
    #    already the zone to paste»), así que la contención se exige contra
    #    la máscara medida, no contra el sujeto.
    photo, gen, mask = build((1248, 832), (544, 0, 976, 810), shift_px=40)
    try:
        out = manager._integrate_local_generation(
            photo, gen, mask, (0, 0, 1248, 832),
            limit_mask=limit_image(subject_box, photo.size),
            keep_measured_mask=True,
        )
    except Exception as error:  # noqa: BLE001 — el fallo ES lo que se mide
        raise AssertionError(
            f"keep_measured_mask=True no debe crashear: {type(error).__name__}: {error}"
        ) from error
    changed_kmm = changed_mask(photo, out)
    measured = np.asarray(mask) >= 96
    leak_kmm = float(changed_kmm[~measured].mean())
    print(
        f"keep_measured_mask=True: cambio="
        f"{float(changed_kmm.mean()):.1%} del cuadro, "
        f"fuga fuera de la máscara medida={leak_kmm:.6%}"
    )
    assert float(changed_kmm[measured].mean()) > 0.01, (
        "la zona medida pedida debe componerse"
    )
    assert leak_kmm == 0.0, (
        f"keep_measured_mask solo pega dentro de la máscara medida, fugó {leak_kmm:.6%}"
    )

    print(
        "OK: el recorte legítimo pega, la zona amplia no escribe fuera del "
        "sujeto y la rama measured no crashea ni se sale de su máscara"
    )


if __name__ == "__main__":
    main()
