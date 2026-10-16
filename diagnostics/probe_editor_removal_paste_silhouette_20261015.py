"""v5.1212 — Fix G sin generación: el material retirado fuera de la silueta se pega.

Casos (foto sintética, sin contenido real; el problema se reproduce solo
con geometría de máscaras):
  1. pelo (material oscuro) que cruza el borde de la silueta contra la
     cortina: la generación lo quita y el pegado debe escribirlo también
     fuera de la silueta — si no, queda el arco del pelo viejo sobre el
     fondo (edit_1792096645 y edit_1792096853, 15/10);
  2. un parche del fondo que el modelo reescribe, lejos y con otro color
     ORIGINAL, no entra: cambio fuerte no basta sin el color del material
     medido.

Contrato medido (sin formas ni palabras):
  - la zona de cambio mide delta >= umbral Otsu (el mismo de dentro),
    conectada al cambio retenido en el sujeto, y el color ORIGINAL de
    esos píxeles igual al del material retirado (mediana del original
    bajo el cambio medido).
  - el pegado (`_integrate_local_generation`) escribe esos píxeles.

Cliente falsificado: no se genera ninguna imagen.
"""

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager

CURTAIN = (180, 180, 185)
SKIN = (222, 172, 150)
HAIR = (58, 44, 38)
ROOM_REWRITE = (96, 112, 140)


def _scene():
    w, h = 320, 240
    px = np.zeros((h, w, 3), dtype=np.uint8)
    px[:, :] = CURTAIN
    # Persona: rectángulo que llega al borde inferior.
    subject = np.zeros((h, w), dtype=bool)
    subject[60:, 120:220] = True
    px[subject] = SKIN
    # Pelo: elipse que cruza la silueta (x 120..220) contra la cortina.
    yy, xx = np.mgrid[0:h, 0:w]
    hair = ((xx - 170.0) / 70.0) ** 2 + ((yy - 60.0) / 34.0) ** 2 <= 1.0
    px[hair] = HAIR
    original = Image.fromarray(px)

    # La generación quita el pelo: dentro de la persona -> piel, fuera -> cortina.
    gen_px = px.copy()
    gen_px[hair & subject] = SKIN
    gen_px[hair & ~subject] = CURTAIN
    # El modelo reescribe además un trozo lejano de la cortina (otro color).
    control = np.zeros((h, w), dtype=bool)
    control[10:45, 10:60] = True
    gen_px[control] = ROOM_REWRITE
    generated = Image.fromarray(gen_px)

    mask = Image.fromarray((subject.astype(np.uint8)) * 255, "L")
    outside_hair = hair & ~subject
    return original, generated, mask, outside_hair, control


def main():
    manager = ImgEditorManager()
    original, generated, mask, outside_hair, control = _scene()

    zone = manager._changed_zone_on_subject(generated, original, mask)
    assert zone is not None, "sin zona de cambio medida"
    zone_arr = np.asarray(zone) >= 160

    kept_out = float(zone_arr[outside_hair].mean())
    assert kept_out > 0.9, (
        "el material retirado fuera de la silueta no entra en la zona "
        f"({kept_out:.0%}); queda el pelo viejo dibujado sobre la cortina"
    )
    assert not (zone_arr & control).any(), (
        "un fondo reescrito de otro color original entró en la zona"
    )
    print(f"zona: pelo fuera de silueta {kept_out:.0%}; control fuera")

    # El pegado escribe esos píxeles: la cortina queda sin pelo.
    full = manager._integrate_local_generation(
        original, generated, mask, (0, 0, 320, 240),
    )
    out = np.asarray(full).astype(int)
    src = np.asarray(original).astype(int)
    diff_out = np.abs(out[outside_hair] - src[outside_hair]).max(axis=1)
    assert float((diff_out > 24).mean()) > 0.9, (
        "el pegado siguió dejando el material viejo fuera de la silueta"
    )
    assert int(np.abs(out[control] - src[control]).max()) <= 24, (
        "el pegado escribió un fondo reescrito lejano"
    )
    print("PROBE OK — material retirado fuera de la silueta pegado; fondo reescrito fuera")


if __name__ == "__main__":
    main()
