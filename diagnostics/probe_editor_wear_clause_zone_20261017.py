"""Contrato v5.1221: una cláusula que ABRE con sintagma determinante nombra
a su sujeto aunque el verbo no esté en el patrón de predicado.

Log real 13:47 (edit_1792237624, «la mujer no lleva medias, ni zapatos»):
'wears'/'lleva' no casa con _PREDICATE_RE (is/has/go/está/tiene…),
_visual_subject_description devolvió "", nunca se propuso «la mujer» al
segmentador → zonas medidas=0 → la foto entera salió del modelo en un pase
sin verificación, y la pierna que cruza hacia el hombre salió con tono y
textura de media (medido: 169/109/96 uniforme frente a su brazo 180/134/130).
Sin vocabulario de prendas: solo gramática (el extractor de sintagmas
_determiner_phrases ya existente); el segmentador y la compuerta de cara
siguen decidiendo si esa persona existe en la foto.
"""
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


class _Face:
    def __init__(self, bbox):
        self.bbox = bbox


class _WomanMasker:
    """El segmentador solo enciende a la mujer; el resto no existe."""

    def generate_mask(self, image, prompt, threshold=0.4):
        arr = np.zeros((image.height, image.width), np.uint8)
        text = (prompt or "").lower()
        if "woman" in text or "mujer" in text:
            arr[10:130, 4:90] = 255
        return Image.fromarray(arr, "L")


class _EmptyMasker:
    def generate_mask(self, image, prompt, threshold=0.4):
        return Image.fromarray(
            np.zeros((image.height, image.width), np.uint8), "L",
        )


class FakeClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(
            image=Image.new("RGB", kwargs["image"].size, "red"),
        ), "OK (stub)"


def group_flow(manager):
    """La cláusula del usuario real mide persona: pase local, no foto entera."""
    photo = Image.new("RGB", (240, 160), "white")
    woman_face = _Face((16, 18, 40, 44))
    client = FakeClient()
    console = io.StringIO()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=object()),
          patch("roop.face_util.detect_faces_robust", return_value=[woman_face]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_WomanMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the woman wears no stockings, no shoes"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          contextlib.redirect_stdout(console)):
        output, message, _ = manager.generate_intelligent(
            image=photo, prompt="la mujer no lleva medias, ni zapatos",
            engine="longcat",
        )
    assert output is not None and output.size == photo.size, message
    generations = [args for name, args in client.calls if name == "generate"]
    assert len(generations) == 1, client.calls
    call = generations[0]
    # Contrato: la persona medida es la zona; la foto entera NO sale.
    assert call["mask_image"] is not None, message
    assert call["keep_source_pixels"] is True, message
    assert call["no_kontext"] is False, message
    assert "sin zona medida, la foto entera sale del modelo" not in message, message
    assert "zonas medidas pegadas" in message, message
    src = np.asarray(photo.convert("RGB"))
    out = np.asarray(output.convert("RGB"))
    assert (out[5, 200] == src[5, 200]).all(), out[5, 200]
    assert not (out[70, 50] == src[70, 50]).all(), out[70, 50]
    console_text = console.getvalue()
    assert "Sujeto 1/1 medido" in console_text, console_text
    print("OK: «no lleva» mide a la mujer y el pase compuesto es local, no foto entera")


def group_unit(manager):
    """Solo gramática: el sintagma inicial propone sujeto; nada inventado."""
    assert manager._visual_subject_description(
        "the woman wears no stockings",
    ) == "the woman"
    assert manager._visual_subject_description("la mujer no lleva medias") == "la mujer"
    # Sin sintagma inicial no se inventa sujeto.
    assert manager._visual_subject_description("no shoes") == ""
    assert manager._visual_subject_description("ni zapatos") == ""
    # El camino de predicado existente queda intacto.
    assert manager._visual_subject_description("the woman is bald") == "the woman"
    assert manager._visual_subject_description(
        "the woman must go without stockings",
    ) == "the woman"
    # Un imperativo empieza por verbo: no nombra sujeto.
    assert manager._visual_subject_description("Add a person near the doorway") == ""
    queries = manager._subject_queries(
        "la mujer no lleva medias, ni zapatos",
        "the woman wears no stockings, no shoes",
    )
    assert queries == ["the woman", "la mujer"], queries
    print("OK: el sintagma inicial nombra al sujeto aunque el verbo no esté en el patrón")


def group_guard(manager):
    """Sin segmentación ni cara que confirme, la cláusula no inventa ancla."""
    photo = Image.new("RGB", (240, 160), "white")
    client = FakeClient()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.face_util.get_face_analyser", return_value=None),
          patch("roop.face_util.detect_faces_robust", return_value=[]),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          patch.object(manager, "_get_clipseg_masker", return_value=_EmptyMasker()),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="the woman wears no stockings, no shoes"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client)):
        output, message, _ = manager.generate_intelligent(
            image=photo, prompt="la mujer no lleva medias, ni zapatos",
            engine="longcat",
        )
    assert output is not None, message
    generations = [args for name, args in client.calls if name == "generate"]
    assert generations and generations[0]["mask_image"] is None, message
    assert "sin zona medida" in message, message
    print("OK: sin medida que confirme la cláusula sigue siendo global (sin ancla inventada)")


def main():
    manager = ImgEditorManager()
    failures = []
    for group in (group_flow, group_unit, group_guard):
        try:
            group(manager)
        except AssertionError as error:
            failures.append(f"{group.__name__}: {error}")
            print(f"ROJO: {group.__name__}: {error}")
    if failures:
        raise SystemExit(f"{len(failures)} grupo(s) en rojo")
    print("TODO VERDE")


if __name__ == "__main__":
    main()
