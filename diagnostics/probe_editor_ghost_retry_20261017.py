"""Contrato: el resto fantasma del material original dispara el segundo intento.

Log real 2026-10-17 01:49 (`ui_start_20261017_014714`): «EL HOMBRE ES CALVO»
sobre foto 718x600. El modelo no reemplazó el pelo, lo desvaneció hacia el
fondo a opacidad parcial; la cáscara resultante sobre el cuero cabelludo
medió cambio (no era no-op) así que se pegó tal cual y quedó el halo (la
cáscara midió el tono de la pared: [141,143,156] vs pared [135,140,154]).

Contratos:
 1. Firma medida del fantasma (unidad): dentro de los píxeles cambiados, la
    estructura de alta frecuencia del original sigue legible (correlación
    alta) mientras su amplitud cae. Un reemplazo real reescribe la textura
    (correlación rota). No-op y regeneración de escena no cuentan.
 2. Flujo: Turbo devuelve un fantasma -> el retry salta con el prompt de
    reemplazo completo («translucent ghost») y la consola lo ANUNCIA antes
    del segundo render. Con el detector desactivado (estado rojo previo) el
    retry no salta: el halo se entregaba tal cual.
"""
import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


def _textured_photo(width=192, height=128, seed=5):
    """Foto con textura en TODA la zona que luego cambia: el detector exige
    estructura HF midedible (std>=1) en los pixeles cambiados."""
    rng = np.random.default_rng(seed)
    base = np.full((height, width, 3), 120.0)
    base += rng.normal(0, 34, (height, width, 3))
    base[:, : width // 2] += 38.0  # region estructurada: mitad izq mas clara
    return Image.fromarray(np.clip(base, 0, 255).astype(np.uint8))


def _ghosted(img, alpha=0.45, toward=255.0):
    """El material original se DESVANECE hacia el fondo: misma estructura,
    amplitud de contraste reducida a alpha. Firma del defecto 01:49."""
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    arr[:, w // 2:] = arr[:, w // 2:] * alpha + toward * (1 - alpha)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _replaced(img, seed=11):
    """Reemplazo real: textura NUEVA opaca en la misma region."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(img.convert("RGB")).astype(np.float32)
    h, w = arr.shape[:2]
    arr[:, w // 2:] = np.clip(128 + rng.normal(0, 34, (h, w // 2, 3)), 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def _scene_replaced(img, seed=13):
    rng = np.random.default_rng(seed)
    arr = rng.normal(128, 34, (*np.asarray(img).shape[:2], 3))
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _check_detector_contracts():
    detector = ImgEditorManager._edit_leaves_a_ghost
    source = _textured_photo()

    # [1] Fantasma: cambio real + estructura retenida + amplitud caida.
    assert detector(source, _ghosted(source)) is True, (
        "el desvanecimiento del material original debe detectarse como fantasma"
    )

    # [2] Reemplazo real: la textura nueva rompe la correlacion.
    assert detector(source, _replaced(source)) is False, (
        "un reemplazo con textura nueva no es fantasma"
    )

    # [3] No-op: sin cambio medible no hay fantasma que discutir.
    assert detector(source, source.copy()) is False

    # [4] Regeneracion de escena: casi todo el encuadre cambia, el techo de
    #     cobertura impide hablar de resto fantasma.
    assert detector(source, _scene_replaced(source)) is False
    print("OK: fantasma medido (correlacion alta + amplitud caida) frente a "
          "reemplazo real, no-op y escena completa")


class GhostThenReplaceClient:
    """Primer render: fantasma (desvanecido). Segundo: reemplazo opaco."""

    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        attempt = sum(name == "generate" for name, _ in self.calls)
        base = kwargs["image"].convert("RGB")
        result = _ghosted(base) if attempt == 1 else _replaced(base)
        return SimpleNamespace(image=result), f"OK attempt {attempt}"


def _check_ghost_triggers_retry():
    manager = ImgEditorManager()
    source = _textured_photo()
    client = GhostThenReplaceClient()

    def fake_finish(image, original, **kwargs):
        return image, "acabado simulado", {"scores": {}}

    log = io.StringIO()
    with (patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="Remove the hair"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit",
                side_effect=fake_finish),
          contextlib.redirect_stdout(log)):
        output, message, _ = manager.generate_intelligent(
            image=source, prompt="Quita el pelo", engine="longcat",
            quality_mode=True, enhance_tier="4k",
            quality_preserve_faces=False,
        )
    assert output is not None
    attempts = [call for name, call in client.calls if name == "generate"]
    assert len(attempts) == 2, (
        f"el fantasma debe disparar el segundo intento; intentos: {len(attempts)}"
    )
    assert "translucent ghost" in attempts[1]["prompt"], attempts[1]["prompt"]
    assert "fully opaque" in attempts[1]["prompt"]
    console = log.getvalue()
    assert "Resto fantasma del material original" in console, console
    assert "Segundo intento aplicado:" in console, console
    print("OK: fantasma -> segundo intento Turbo con reemplazo completo y "
          "consola anunciandolo antes del render")


def _check_no_detector_no_retry():
    """Estado rojo previo: sin el detector, el fantasma se entregaba tal cual."""
    manager = ImgEditorManager()
    source = _textured_photo()
    client = GhostThenReplaceClient()

    def fake_finish(image, original, **kwargs):
        return image, "acabado simulado", {"scores": {}}

    log = io.StringIO()
    with (patch("roop.img_editor.img_editor_manager.ImgEditorManager"
                "._edit_leaves_a_ghost", return_value=False),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="Remove the hair"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.top_tier_finish.run_quality_enhance_after_edit",
                side_effect=fake_finish),
          contextlib.redirect_stdout(log)):
        manager.generate_intelligent(
            image=source, prompt="Quita el pelo", engine="longcat",
            quality_mode=True, enhance_tier="4k",
            quality_preserve_faces=False,
        )
    attempts = [call for name, call in client.calls if name == "generate"]
    assert len(attempts) == 1, (
        "sin detector no debe haber segundo intento (reproduccion del defecto)"
    )
    assert "Resto fantasma" not in log.getvalue()
    print("OK: sin detector no hay retry (el halo se entregaba; defecto reproducido)")


def main():
    _check_detector_contracts()
    _check_ghost_triggers_retry()
    _check_no_detector_no_retry()
    print("GRUPOS OK")


if __name__ == "__main__":
    main()
