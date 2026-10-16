"""v5.1211 — guards sintéticos sin generación (Fix C y Fix D).

Fix C (`_close_recolor_halo`): en un recolor de superficie, el hueco de
medición que rodea a una persona es pared. Antes sobrevivía como el muro
ORIGINAL cálido dentro de una habitación recién gris, y se leía como un HALO
naranja alrededor de la persona (edit_1792086138 / edit_1792086220, 15/10:
los píxeles sin tocar medían idénticos a la foto previa —217,157,134—
mientras el resto de la pared pasaba de durazno 223,157 a gris 137,138).

Fix D (`_degenerate_faces_mask`): una cara que el generado colapsa a negro
(edit_1792086220: mediana de luma 6 frente a 193 de la fuente en los mismos
píxeles) se excluye del pegado y la cara original se conserva; una cara que
solo se movió, envejeció o se recoloreó —o una más clara— no se toca.
"""

from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager

H, W = 640, 480


def _wall_scene():
    """Pared + persona + hueco de medición alrededor + objeto lejano."""
    yy, xx = np.ogrid[:H, :W]
    person = ((xx - 240) ** 2) / (90.0 ** 2) + ((yy - 400) ** 2) / (140.0 ** 2) <= 1
    # El segmentador deja fuera el anillo de rebote junto a la persona.
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))  # radio 15
    gap = (cv2.dilate(person.astype(np.uint8), kern) > 0) & ~person
    # Un objeto sobre la pared, lejos de cualquier persona (no es halo).
    obj = ((xx - 60) ** 2 + (yy - 60) ** 2) <= 30.0 ** 2

    region = np.ones((H, W), dtype=bool)
    region[person] = False
    region[gap] = False
    region[obj] = False
    assert 0.12 <= region.mean() <= 0.90, region.mean()

    mask = Image.fromarray(region.astype(np.uint8) * 255, "L")
    # Convención de _background_mask_from_people: 255 = FONDO (sin personas).
    background = Image.fromarray((~person).astype(np.uint8) * 255, "L")
    faces = [SimpleNamespace(bbox=[150.0, 260.0, 330.0, 540.0])]
    return mask, background, faces, person, gap, obj


def test_fix_c():
    manager = ImgEditorManager()
    mask, background, faces, person, gap, obj = _wall_scene()
    img = Image.new("RGB", (W, H), (180, 180, 180))

    with (patch.object(manager, "_get_clipseg_masker", return_value=object()),
          patch.object(ImgEditorManager, "_background_mask_from_people",
                       staticmethod(lambda image, masker, f: background))):
        out = manager._close_recolor_halo(img, mask, faces)
    arr = np.asarray(out.convert("L")) >= 96

    gap_filled = float((arr & gap).sum()) / float(gap.sum())
    print(f"    hueco junto a la persona rellenado: {gap_filled:.1%}")
    assert gap_filled >= 0.60, f"el halo no se cerró ({gap_filled:.1%})"
    assert not (arr & person).any(), "la persona entró en la zona de pared"
    assert not (arr & obj).any(), "un objeto lejos de las personas fue repintado"

    # Sin caras medidas no hay ancla de personas: la máscara queda como está.
    with (patch.object(manager, "_get_clipseg_masker", return_value=object()),
          patch.object(ImgEditorManager, "_background_mask_from_people",
                       staticmethod(lambda image, masker, f: background))):
        untouched = manager._close_recolor_halo(img, mask, [])
    assert np.array_equal(np.asarray(untouched), np.asarray(mask)), \
        "sin caras la zona no debe cambiar"
    assert manager._close_recolor_halo(img, None, faces) is None
    print("    Fix C OK: halo junto a personas cerrado; objeto lejano y personas fuera")


def test_fix_d():
    manager = ImgEditorManager()
    src = np.full((240, 240, 3), 100, dtype=np.uint8)
    src[60:140, 60:140] = 190
    source = Image.fromarray(src)
    faces = [SimpleNamespace(bbox=[60.0, 60.0, 140.0, 140.0])]

    collapsed = src.copy()
    collapsed[60:140, 60:140] = 6  # edit_1792086220: luma 6 vs 193 de la fuente
    bright = src.copy()
    bright[60:140, 60:140] = 245  # cara más clara: no es colapso

    m_bad = manager._degenerate_faces_mask(
        source, Image.fromarray(collapsed), faces,
    )
    assert m_bad is not None, "el colapso a negro no se detectó"
    m = np.asarray(m_bad.convert("L")) >= 128
    core = m[70:130, 70:130]
    assert core.mean() >= 0.60, f"la máscara no cubre la cara degradada ({core.mean():.1%})"
    far = np.zeros_like(m)
    far[10:230, 10:230] = True
    assert not (m & ~far).any(), "la máscara se desborda de la vecindad de la cara"
    assert manager._degenerate_faces_mask(source, source, faces) is None, \
        "una cara intacta marcó como degenerada"
    assert manager._degenerate_faces_mask(
        source, Image.fromarray(bright), faces,
    ) is None, "una cara más clara marcada como degenerada"
    print("    Fix D OK: colapso negro excluido; intacta y más clara intactas")


def main():
    test_fix_c()
    test_fix_d()
    print("PASS: guards de halo y de cara degenerada")


if __name__ == "__main__":
    main()
