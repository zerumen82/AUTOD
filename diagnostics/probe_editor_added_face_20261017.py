"""Contrato: una cara que el modelo pinta donde la foto no tiene ninguna dispara el segundo intento.

Log real 2026-10-17 10:23 (`ui_start_20261017_031334`): «el hombre debe ser
calvo» sobre foto 2847x4263 (`edit_1792225412`). El análisis de arranque
imprimió 2 caras (la de la entrada: hombre + mujer) y el modelo pintó una
TERCERA cara sobre la coronilla del hombre mientras obedecía el calvo. El
pegado por cambio medido la conserva porque textura nueva mide como cambio
real; ni el no-op ni el fantasma la cogen (contenido nuevo: la correlación
HF se rompe, justo lo contrario de un resto desvanecido). Una persona que
nadie pidió es medible como Fix D mide una cara colapsada: cada cara del
resultado debe caer cerca de una cara de la entrada.

Contratos:
 1. Emparejamiento (unidad): una cara del resultado cuyo centro queda a más
    de 0.6 de su propia altura de TODO centro de cara de la entrada devuelve
    máscara; una cara desplazada normalmente, la ausencia de caras de
    referencia o la ausencia de detecciones no (conservador: sin referencia
    no se juzga).
 2. Flujo: Turbo entrega una cara añadida -> el retry salta con el prompt de
    «exactly the same people» y la consola lo anuncia antes del render; si el
    retry persiste con la cara, esa región se restaura de la foto original.
 3. Rojo demostrado: con el detector desactivado no hay retry: la cara ajena
    se entregaba tal cual (estado previo).
"""
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def _photo(width=400, height=600, seed=7):
    """Foto texturizada: el cambio medido y la firma HF exigen textura."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 120.0)
    base += rng.normal(0, 34, (height, width, 3))
    base[:, : width // 2] += 38.0
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _replaced(img, seed=11):
    """Reemplazo real en la mitad derecha: cambio medible, no fantasma."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    arr[:, w // 2:] = np.clip(128 + rng.normal(0, 34, (h, w // 2, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _face(x1, y1, x2, y2, score=0.93):
    return SimpleNamespace(bbox=[x1, y1, x2, y2], det_score=score)


def _rect_mask(width=400, height=600, rect=(220, 80, 380, 180)):
    arr = np.zeros((height, width), dtype=np.uint8)
    x1, y1, x2, y2 = rect
    arr[y1:y2, x1:x2] = 255
    return Image.fromarray(arr, "L")


def _check_matching_contracts():
    manager = ImgEditorManager()
    source = _photo()
    reference = [_face(120, 180, 240, 340), _face(260, 380, 360, 500)]

    # [1] Cara añadida: dos caras de la entrada + una tercera sobre la
    #     coronilla (lejos de todo centro de referencia) -> mascara.
    with patch.object(
        ImgEditorManager, "_detect_edit_faces",
        return_value=reference + [_face(140, 60, 250, 190)],
    ):
        mask = manager._added_face_mask(source, source.copy(), reference)
    assert mask is not None, "la cara lejana de toda referencia debe rechazarse"
    assert mask.size == source.size
    pixels = np.asarray(mask)
    assert pixels[125, 195] >= 200, (
        f"el centro de la cara añadida debe ir en la mascara: {pixels[125, 195]}"
    )
    assert pixels[260, 180] < 8, "la cara de la fuente no debe tocarse"
    assert pixels[440, 310] == 0, "la otra cara de la fuente no debe tocarse"

    # [2] Cara desplazada dentro de la tolerancia (0.6 de su altura): una
    #     edición normal que movió la cara no es una persona nueva.
    shifted = [_face(160, 230, 280, 390), _face(260, 380, 360, 500)]
    with patch.object(ImgEditorManager, "_detect_edit_faces",
                      return_value=shifted):
        assert manager._added_face_mask(source, source.copy(), reference) is None, (
            "una cara movida dentro de la tolerancia no es añadida"
        )

    # [3] Sin caras de referencia no se juzga (ruta de cámara, foto sin
    #     caras): conservador, la ausencia de baseline no acusa a nadie.
    with patch.object(
        ImgEditorManager, "_detect_edit_faces",
        return_value=reference + [_face(140, 60, 250, 190)],
    ):
        assert manager._added_face_mask(source, source.copy(), []) is None

    # [4] Sin detecciones en el resultado no hay nada que rechazar.
    with patch.object(ImgEditorManager, "_detect_edit_faces", return_value=[]):
        assert manager._added_face_mask(source, source.copy(), reference) is None

    print("OK: cara lejana a toda referencia rechazada; cara desplazada, "
          "sin referencia y sin detecciones no")


class AddedFaceThenCleanClient:
    """Primer render: cambio real. Segundo: mismo cambio (segunda pasada)."""

    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        attempt = sum(name == "generate" for name, _ in self.calls)
        base = kwargs["image"].convert("RGB")
        return (SimpleNamespace(image=_replaced(base, seed=11 + attempt)),
                f"OK attempt {attempt}")


def _run_flow(manager, client, added_face_effect):
    def fake_finish(image, original, **kwargs):
        return image, "acabado simulado", {"scores": {}}

    log = io.StringIO()
    with (patch("roop.img_editor.img_editor_manager.ImgEditorManager"
                "._added_face_mask", side_effect=added_face_effect),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="Remove the hair"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit",
                side_effect=fake_finish),
          contextlib.redirect_stdout(log)):
        output, message, _ = manager.generate_intelligent(
            image=_photo(), prompt="Quita el pelo", engine="longcat",
            quality_mode=True, enhance_tier="4k",
            quality_preserve_faces=False,
        )
    return output, log.getvalue()


def _check_added_face_triggers_retry():
    manager = ImgEditorManager()
    client = AddedFaceThenCleanClient()
    mask = _rect_mask()
    output, console = _run_flow(manager, client, [mask, None])

    assert output is not None
    attempts = [call for name, call in client.calls if name == "generate"]
    assert len(attempts) == 2, (
        f"la cara añadida debe disparar el segundo intento; "
        f"intentos: {len(attempts)}"
    )
    prompt_2 = attempts[1]["prompt"]
    assert "exactly the same people" in prompt_2, prompt_2
    assert "no face may be added" in prompt_2, prompt_2
    assert "no face may be added" not in attempts[0]["prompt"]
    assert "Cara añadida por el modelo" in console, console
    assert "Segundo intento aplicado:" in console, console
    assert "se restaura de la foto original" not in console, console
    print("OK: cara añadida -> segundo intento Turbo con las mismas personas "
          "y consola anunciandolo antes del render")


def _check_persistent_added_face_restored():
    """El retry vuelve a pintar la cara: esa región vuelve a la foto original."""
    manager = ImgEditorManager()
    client = AddedFaceThenCleanClient()
    mask = _rect_mask()
    source = _photo()
    output, console = _run_flow(manager, client, lambda *a, **k: mask)

    attempts = [call for name, call in client.calls if name == "generate"]
    assert len(attempts) == 2, f"intentos: {len(attempts)}"
    assert "se restaura de la foto original" in console, console

    out = np.asarray(output.convert("RGB"))
    src = np.asarray(source.convert("RGB"))
    assert np.array_equal(out[100:170, 240:360], src[100:170, 240:360]), (
        "la región de la cara añadida debe quedar restaurada de la foto original"
    )
    assert not np.array_equal(out[300:380, 240:360], src[300:380, 240:360]), (
        "el resto del cambio medido debe conservarse"
    )
    print("OK: cara persistente tras el retry -> región restaurada de la "
          "foto original, el resto del cambio se queda")


def _check_no_detector_no_retry():
    """Estado rojo previo: sin el detector, la cara ajena se entregaba tal cual."""
    manager = ImgEditorManager()
    client = AddedFaceThenCleanClient()
    _, console = _run_flow(manager, client, lambda *a, **k: None)

    attempts = [call for name, call in client.calls if name == "generate"]
    assert len(attempts) == 1, (
        "sin detector no debe haber segundo intento (reproduccion del defecto)"
    )
    assert "Cara añadida por el modelo" not in console
    print("OK: sin detector no hay retry (la cara ajena se entregaba; "
          "defecto reproducido)")


def main():
    _check_matching_contracts()
    _check_added_face_triggers_retry()
    _check_persistent_added_face_restored()
    _check_no_detector_no_retry()
    print("GRUPOS OK")


if __name__ == "__main__":
    main()
