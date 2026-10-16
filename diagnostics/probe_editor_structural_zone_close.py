"""Zona estructural cerrada: los objetos entre los parches entran en la máscara (v5.1209).

«Quita los cuadros» mide la PARED entre los cuadros; los cuadros quedan como
huecos y exteriores de los parches, y FastEdit no toca nada fuera de la
máscara (0 % medido el 15/10), así que no se podía eliminar nada. Este probe
fija el contrato de `_closed_structural_zone`: huecos interiores rellenados,
nada fuera de la propia caja de la zona y personas restadas.
"""

import numpy as np
from PIL import Image
from unittest.mock import patch

from roop.img_editor.img_editor_manager import ImgEditorManager


SIZE = (200, 150)


def ring_zone():
    """Pared medida con un cuadro (hueco) encerrado dentro."""
    pixels = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    pixels[20:80, 20:140] = 255
    pixels[35:65, 50:90] = 0
    return Image.fromarray(pixels, "L")


def person_mask():
    """Persona que roza el borde de la zona (no la domina)."""
    pixels = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    pixels[20:80, 100:140] = 255
    return Image.fromarray(pixels, "L")


def main():
    manager = ImgEditorManager()
    image = Image.new("RGB", SIZE, (200, 200, 200))

    # 0) La zona medida sobrevive entera: el cierre solo AÑADE.
    zone = ring_zone()
    before = np.asarray(zone.convert("L")) > 96
    closed = manager._closed_structural_zone(image, zone, faces=())
    assert closed is not None, "el cierre devolvió None sin personas"
    after = np.asarray(closed.convert("L")) > 96
    lost = before & ~after
    assert not lost.any(), (
        f"el cierre PERDIÓ {int(lost.sum())} px de la zona medida "
        "(la máscara que llega al modelo no puede encoger)"
    )
    print("OK: el cierre añade sin perder la zona medida")

    # 1) El hueco interior (el cuadro entre los parches) queda dentro.
    hole = after[40:60, 55:85]
    assert hole.all(), "el cuadro encerrado entre los parches no entró en la máscara"
    print("OK: hueco interior de la zona rellenado")

    # 2) Nada fuera de la caja de la zona medida.
    ys, xs = np.nonzero(before)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    added = after & ~before
    outside = added.copy()
    outside[y0:y1, x0:x1] = False
    assert not outside.any(), "el cierre escribió fuera de la caja de la zona"
    print("OK: el cierre no sale de la caja de la zona medida")

    # 3) Las personas se restan del cierre (superficie con roce de borde).
    person = person_mask()
    overlap_ratio = float((((np.asarray(person) > 96) & before).sum()) / before.sum())
    assert 0.02 <= overlap_ratio < 0.55, overlap_ratio
    # _background_mask_from_people devuelve FONDO claro y persona oscuro.
    background_px = np.full((SIZE[1], SIZE[0]), 255, np.uint8)
    background_px[np.asarray(person) > 96] = 0
    with patch.object(manager, "_get_clipseg_masker", return_value=object()), \
            patch.object(
                manager, "_background_mask_from_people",
                return_value=Image.fromarray(background_px, "L"),
            ):
        trimmed = manager._closed_structural_zone(image, zone, faces=[object()])
    trimmed_arr = np.asarray(trimmed.convert("L")) > 96
    assert not (trimmed_arr & (np.asarray(person) > 96)).any(), (
        "la persona quedó dentro de la máscara cerrada"
    )
    assert trimmed_arr[40:60, 55:85].all(), "el cierre se perdió al restar a la persona"
    print("OK: la persona roza la zona y se resta sin perder el cierre")

    # 4) Caja dominada por personas: la eliminación no es de objetos y la
    #    zona medida se conserva tal cual.
    dominant = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    dominant[20:60, 20:140] = 255
    background_dominant = np.full((SIZE[1], SIZE[0]), 255, np.uint8)
    background_dominant[np.asarray(dominant) > 96] = 0
    with patch.object(manager, "_get_clipseg_masker", return_value=object()), \
            patch.object(
                manager, "_background_mask_from_people",
                return_value=Image.fromarray(background_dominant, "L"),
            ):
        kept = manager._closed_structural_zone(image, zone, faces=[object()])
    assert np.array_equal(np.asarray(kept.convert("L")) > 96, before), (
        "una caja con más del 55 % de personas no debe sustituir a la zona medida"
    )
    print("OK: la caja dominada por personas conserva la zona medida")

    # 4) Solo la eliminación estructural porta el hecho (la compuerta del caller).
    remove = manager._prompt_authority_facts(
        "quita los cuadros de la pared", "remove the pictures from the wall",
    )
    assert remove.get("structural_remove"), remove
    add = manager._prompt_authority_facts(
        "añade una lámpara a la mesa", "Add a lamp to the table",
    )
    assert not add.get("structural_remove"), add
    recolor = manager._prompt_authority_facts(
        "las paredes son de color gris", "the walls are grey",
    )
    assert not recolor.get("structural_remove"), recolor
    appearance = manager._prompt_authority_facts("el hombre es calvo", "the man is bald")
    assert not appearance.get("structural_remove"), appearance
    print("OK: structural_remove se enciende solo en add/remove medido")

    print("PASS: cierre de zona estructural (v5.1209)")


if __name__ == "__main__":
    main()
