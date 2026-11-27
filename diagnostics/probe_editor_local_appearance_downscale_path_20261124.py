#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fase 2 — la ruta real acota la imagen de trabajo antes de ComfyUI.

Fuerza el tope a 200 px sobre la foto del probe de calvo (1138x561) y
comprueba que el cliente recibe una imagen acotada y los pasos de la receta
LongCat vigente. Sin ComfyUI real ni reinicio de la UI.
"""
import io
import os
import sys
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from roop.face_util import get_face_analyser_cpu  # noqa: E402
from roop.img_editor.gen_prompt_config import get_longcat_model_config  # noqa: E402
from roop.img_editor.img_editor_manager import ImgEditorManager  # noqa: E402

SRC = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1792015141.png")


class CaptureClient:
    def __init__(self):
        self.calls = []

    def load(self, version):
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(image=Image.new("RGB", (32, 20), "red")), "OK (stub)"


def main() -> int:
    manager = ImgEditorManager()
    client = CaptureClient()
    photo = Image.open(SRC).convert("RGB")
    buffer = io.StringIO()
    with (
        redirect_stdout(buffer),
        patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
        patch("roop.face_util.get_face_analyser", new=lambda *a, **k: get_face_analyser_cpu()),
        patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0),
        patch.object(manager, "_release_local_gpu_before_comfy"),
        patch("roop.img_editor.prompt_translator.translate_prompt",
              new=lambda text, *a, **k: text),
        patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
              return_value=client),
        patch.object(manager, "_local_appearance_quality", return_value=(28, 200)),
    ):
        manager.generate_intelligent(
            image=photo, prompt="Man must be bald.", engine="longcat",
        )
    log = buffer.getvalue()
    print("\n".join(
        line for line in log.splitlines()
        if "acotada" in line or "una pasada" in line or "una cara" in line
    ))
    assert "Imagen de trabajo acotada" in log, "no se acotó la imagen de trabajo"
    generates = [call for call in client.calls]
    assert generates, "el cliente no recibió ninguna generación"
    images = [call["image"] for call in generates if call.get("image") is not None]
    assert images, "no se capturó imagen de trabajo"
    # El primer pase es el que viene del recorte medido y del codificador
    # Qwen VL. Los pases de refino focal posteriores construyen su propia
    # ventana pequeña y no pasan por este tope.
    first_side = max(images[0].size)
    biggest = max(max(image.size) for image in images)
    print(f"primer pase enviado: {first_side} px (tope forzado 200)")
    print(f"lado largo mayor de todos los pases: {biggest} px")
    assert first_side <= 200, f"el primer pase recibió una imagen de {first_side} px"
    steps = [int(call["num_inference_steps"]) for call in generates
             if "num_inference_steps" in call]
    expected_steps = int(get_longcat_model_config()["steps"])
    assert steps and min(steps) >= expected_steps, (
        f"pasos por debajo de la receta {expected_steps}: {steps}"
    )
    print(f"pasos enviados: {steps}")
    print("PASS — Fase 2: tope de resolución aplicado en la ruta real")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
