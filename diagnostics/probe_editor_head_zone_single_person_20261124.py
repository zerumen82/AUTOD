#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 1 — la zona de cabeza de una edición de apariencia no puede editar a otra persona.

Reproduce el caso del 2026-11-24 («el hombre debe ser calvo», dos caras):
la máscara del sujeto cubre a las dos personas; antes del arreglo la zona era
(head1 | head2) & sujeto, así que el agujero de generación y el pegado
incluían la cabeza ajena. Tras el arreglo solo entra la cabeza medida.
Sin ComfyUI, sin GPU, sin reiniciar la UI.
"""
import os
import sys

import cv2
import numpy as np
from PIL import Image
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402


def main() -> int:
    manager = ImgEditorManager.__new__(ImgEditorManager)
    width, height = 600, 800
    photo = Image.new("RGB", (width, height), (128, 128, 128))
    # Dos caras reales separadas; la de la izquierda es la más confiable.
    first = SimpleNamespace(bbox=(200, 150, 330, 300), det_score=0.91)
    second = SimpleNamespace(bbox=(360, 140, 490, 290), det_score=0.82)
    faces = [first, second]

    subject = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(subject, (265, 225), (105, 125), 0, 0, 360, 255, -1)
    cv2.ellipse(subject, (425, 215), (105, 125), 0, 0, 360, 255, -1)
    subject_mask = Image.fromarray(subject, "L")

    head_first = np.asarray(
        manager._head_mask_from_faces(photo, [first]).convert("L")
    ) >= 96
    head_second = np.asarray(
        manager._head_mask_from_faces(photo, [second]).convert("L")
    ) >= 96

    # Reproducción de la causa: la unión antigua sí contiene la otra cabeza.
    old_zone = (head_first | head_second) & (np.asarray(subject_mask) >= 128)
    leaked_before = int((old_zone & head_second).sum())
    assert leaked_before > 1000, (
        "la reproducción antigua debería filtrar la otra cabeza; "
        f"filtró {leaked_before} px"
    )
    print(f"ANTES: otra cabeza dentro de la zona = {leaked_before} px (FAIL esperado)")

    # Arreglo: una sola cabeza medida, sin la ajena.
    zone_img = manager._head_zone_for_subject(
        photo, subject_mask, faces, announce=False,
    )
    assert zone_img is not None, "la zona medida no se pudo construir"
    zone = np.asarray(zone_img.convert("L")) >= 128
    leaked_after = int((zone & head_second).sum())
    kept = int((zone & head_first).sum())
    print(f"DESPUES: otra cabeza dentro de la zona = {leaked_after} px")
    print(f"DESPUES: cabeza elegida conservada = {kept} px")
    assert kept >= 48, "la cabeza elegida desapareció de la zona"
    assert leaked_after < 50, (
        "la otra persona sigue entrando en la zona de edición: "
        f"{leaked_after} px"
    )

    # El sujeto amplio ilumina las dos personas y el desempate anterior
    # escogía la detección más confiable. La parte solicitada señala la otra.
    requested = Image.fromarray(head_second.astype(np.uint8) * 255, "L")
    requested_zone = manager._head_zone_for_subject(
        photo, subject_mask, faces, announce=False, request_mask=requested,
    )
    assert requested_zone is not None, "se perdió la cabeza de la parte pedida"
    selected = np.asarray(requested_zone.convert("L")) >= 128
    selected_pixels = int((selected & head_second).sum())
    wrong_pixels = int((selected & head_first).sum())
    assert selected_pixels >= 48 and wrong_pixels < 50, (
        f"la parte pidió la segunda cabeza, pero entraron {wrong_pixels} px "
        f"de la primera y solo {selected_pixels} px de la segunda"
    )
    print(f"Parte pedida en la segunda cara: {selected_pixels} px; ajena: {wrong_pixels} px")

    # Una sola cara: la zona sigue existiendo (no se rompe el caso simple).
    only = manager._head_zone_for_subject(
        photo, Image.fromarray(
            ((head_first.astype(np.uint8)) * 255), "L",
        ), [first], announce=False,
    )
    assert only is not None and int((np.asarray(only) >= 128).sum()) >= 48
    print("Una sola cara: zona mantenida OK")
    print("PASS — Fase 1: la edición de apariencia ya no toma la otra cabeza")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
