"""CPU dispatch checks for a measured recolor of part of a subject.

These masks stand in for segmenter responses. They test routing and abstention,
not whether CLIPSeg finds the right pixels in a real photograph.
"""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


SIZE = (160, 120)
# The measured surface is one flat color; everything beyond it is a
# different-colored neighbour (curtain). v5.1225: a whole-surface recolor
# follows its own color until the color changes, so the scene needs a real
# color boundary — a single flat frame is one surface as far as the
# measurement can tell.
PHOTO_ARR = np.zeros((SIZE[1], SIZE[0], 3), np.float32)
PHOTO_ARR[:] = np.array([130, 120, 110], np.float32)
PHOTO_ARR[:, 90:] = np.array([60, 90, 160], np.float32)
PHOTO = Image.fromarray(PHOTO_ARR.astype(np.uint8))
NEIGHBOR = np.zeros((SIZE[1], SIZE[0]), bool)
NEIGHBOR[:, 90:] = True


def rectangle(box):
    pixels = np.zeros((SIZE[1], SIZE[0]), np.uint8)
    x1, y1, x2, y2 = box
    pixels[y1:y2, x1:x2] = 255
    return Image.fromarray(pixels, "L")


class Segmenter:
    def __init__(self, subject):
        self.subject = subject

    def generate_mask(self, _image, _query):
        return self.subject


def main():
    manager = ImgEditorManager()
    woman = rectangle((50, 10, 120, 110))
    hair = rectangle((68, 10, 100, 32))
    face = SimpleNamespace(bbox=(70, 35, 100, 65))
    wall = rectangle((0, 0, 50, 120))

    # A found part stays smaller than its person. Neither whole-surface
    # continuation nor person subtraction belongs to this dispatch.
    with (
        patch.object(manager, "_subject_queries", return_value=["the woman"]),
        patch.object(manager, "_get_clipseg_masker", return_value=Segmenter(woman)),
        patch.object(manager, "_request_region_on_subject", return_value=hair),
        patch.object(manager, "_continue_measured_region", side_effect=AssertionError("continued whole person")),
    ):
        passes = manager._measured_region_passes(
            PHOTO, "La mujer debe tener el pelo rojo", "The woman must have red hair",
            surface_recolor=True, faces=[face],
        )
    assert len(passes) == 1 and passes[0]["requested_part"]
    assert passes[0]["measured_request"]
    assert np.array_equal(np.asarray(passes[0]["mask"]), np.asarray(hair))
    assert np.array_equal(np.asarray(passes[0]["person"]), np.asarray(woman))
    print("OK: recolor usa solo la parte medida y conserva el sujeto como límite")

    # If the subject contains a detected face but no requested part is found,
    # returning [] prevents the old whole-person repaint.
    with (
        patch.object(manager, "_subject_queries", return_value=["the woman"]),
        patch.object(manager, "_get_clipseg_masker", return_value=Segmenter(woman)),
        patch.object(manager, "_request_region_on_subject", return_value=None),
        patch.object(manager, "_continue_measured_region", side_effect=AssertionError("repainted whole person")),
    ):
        missing = manager._measured_region_passes(
            PHOTO, "La mujer debe tener el pelo rojo", "The woman must have red hair",
            surface_recolor=True, faces=[face],
        )
    assert missing == []
    print("OK: parte ausente en persona detiene el recolor")

    # v5.1214: a measured surface that only BLEEDS over a face edge is not
    # that person. The old guard fired on any face touch (15% of the bbox),
    # so the measured wall behind a full-body person was discarded as «a
    # person without its part» and never painted. A face largely inside the
    # region is still a person: the abstention below keeps that contract.
    bleeding_wall = rectangle((0, 0, 80, 120))  # covers ~33% of the face bbox
    with (
        patch.object(manager, "_subject_queries", return_value=["the wall"]),
        patch.object(manager, "_get_clipseg_masker", return_value=Segmenter(bleeding_wall)),
        patch.object(manager, "_request_region_on_subject", return_value=None),
        patch.object(manager, "_continue_measured_region", return_value=bleeding_wall) as continued,
    ):
        bleeding = manager._measured_region_passes(
            PHOTO, "las paredes deben ser grises", "the walls must be grey",
            surface_recolor=True, faces=[face],
        )
    assert len(bleeding) == 1 and not bleeding[0].get("requested_part")
    # v5.1225: the whole-surface recolor follows its own color to where the
    # color changes. The measured fragment stays inside the zone, the
    # different-colored neighbour stays out, and the fragment-only
    # continuation that used to end the flow no longer runs.
    zone = np.asarray(bleeding[0]["mask"]) >= 96
    fragment = np.asarray(bleeding_wall) >= 96
    assert (zone & fragment).sum() == fragment.sum()
    assert float((zone & NEIGHBOR).sum()) / int(NEIGHBOR.sum()) <= 0.05
    assert not continued.called
    print("OK: superficie que roza una cara no se descarta como persona")

    # The abstention still holds for a real person: the face sits inside the
    # measured region and no requested part was found.
    assert manager._region_contains_a_face(
        np.asarray(woman), [face], SIZE[0], SIZE[1],
    )
    assert not manager._region_contains_a_face(
        np.asarray(bleeding_wall), [face], SIZE[0], SIZE[1],
    )
    assert not manager._region_contains_a_face(
        np.asarray(wall), [face], SIZE[0], SIZE[1],
    )
    print("OK: la contención de la cara dentro de la región separa persona de superficie")

    # A wall has no facial overlap; the same color-following applies without
    # the person guards interfering.
    with (
        patch.object(manager, "_subject_queries", return_value=["the wall"]),
        patch.object(manager, "_get_clipseg_masker", return_value=Segmenter(wall)),
        patch.object(manager, "_request_region_on_subject", return_value=None),
        patch.object(manager, "_continue_measured_region", return_value=wall) as continued,
    ):
        background = manager._measured_region_passes(
            PHOTO, "las paredes deben ser grises", "the walls must be grey",
            surface_recolor=True, faces=[face],
        )
    assert len(background) == 1 and not background[0].get("requested_part")
    # v5.1225: same flood for a surface with no face overlap; the neighbour
    # of another color still stays out.
    zone = np.asarray(background[0]["mask"]) >= 96
    fragment = np.asarray(wall) >= 96
    assert (zone & fragment).sum() == fragment.sum()
    assert float((zone & NEIGHBOR).sum()) / int(NEIGHBOR.sum()) <= 0.05
    assert not continued.called
    print("OK: superficie sin persona sigue su color y deja al vecino fuera")

    # Structural region edits never acquire the recolor-only part flag. They
    # retain the ordinary path, including eligibility for face-repair later.
    with (
        patch.object(manager, "_subject_queries", return_value=["the object"]),
        patch.object(manager, "_get_clipseg_masker", return_value=Segmenter(woman)),
        patch.object(manager, "_request_region_on_subject", side_effect=AssertionError("part lookup in structural edit")),
        patch.object(manager, "_continue_measured_region", return_value=woman) as continued,
    ):
        structural = manager._measured_region_passes(
            PHOTO, "Remove the object", "Remove the object",
            surface_recolor=False, faces=[face],
        )
    assert len(structural) == 1 and not structural[0].get("requested_part")
    continued.assert_called_once()
    print("OK: add/remove mantiene la ruta estructural y su guard posterior")


if __name__ == "__main__":
    main()
