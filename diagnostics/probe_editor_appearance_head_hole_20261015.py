"""v5.1211 — Fix F sin generación: cabeza como agujero para apariencia medida.

Entrada REAL: img_editor/edit_1792085759.png (400×600, la salida del render 1
del 15/10 que SIGUIÓ CON PELO pese a «el hombre debe ser calvo»: la parte que
CLIPSeg mide para «bald» es solo la coronilla, el agujero de generación era
esa mancha y el pelo fuera del agujero no pudo quitarse).

Contrato:
  - la parte pedida mide CONTENIDA en la cabeza de la persona medida ⇒ el
    agujero de generación pasa a ser la cabeza completa;
  - el pegado es el cambio medido dentro de esa cabeza («cambio medido ∩
    permitido»), no la máscara entera;
  - una parte fuera de la cabeza no se amplía.

Cliente falsificado: no se genera ninguna imagen.
"""

import contextlib
import io
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from roop.img_editor.img_editor_manager import ImgEditorManager


class RecorderClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (64, 40), "red")), "OK (stub)"


def main():
    source = Image.open("img_editor/edit_1792085759.png").convert("RGB")
    manager = ImgEditorManager()

    client = RecorderClient()
    buffer = io.StringIO()
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
          patch.object(manager, "_release_local_gpu_before_comfy"),
          contextlib.redirect_stdout(buffer)):
        output, message, mask = manager.generate_intelligent(
            image=source, prompt="el hombre debe ser calvo",
            engine="longcat", use_rewriter=True,
            auto_upscale=False, enhance_faces=False,
        )
    log = buffer.getvalue()
    print(log)
    print(f"mensaje final: {message!r}")

    assert output is not None, message
    generates = [args for name, args in client.calls if name == "generate"]
    assert generates, (message, "la apariencia no llegó a generación")
    assert "mide dentro de la cabeza" in log, \
        ("la parte medida no se amplió a la cabeza; el agujero seguía en la "
         "mancha de coronilla y el pelo quedaba fuera")

    first = generates[0]
    zone = first.get("mask_image")
    assert zone is not None, "generación sin máscara"
    assert first.get("keep_source_pixels") is True, "la zona no quedó anclada"
    cov_zone = float((np.asarray(zone.convert("L")) > 96).mean())

    # La coronilla sola es <1% de la foto; la cabeza de la persona mide más.
    masker = manager._get_clipseg_masker()
    bald = masker.generate_mask(source, "bald")
    cov_bald = (
        float((np.asarray(bald.convert("L")) > 96).mean())
        if bald is not None else 0.0
    )
    print(f"    coronilla medida por CLIPSeg: {cov_bald:.2%}; "
          f"zona despachada: {cov_zone:.2%}")
    if cov_bald > 0.002:
        assert cov_zone >= 2.0 * cov_bald, \
            (cov_zone, cov_bald, "la zona despachada no amplía a la cabeza")

    print("PASS: la apariencia de cabeza mide agujero en la cabeza y pegado por cambio medido")


if __name__ == "__main__":
    main()
