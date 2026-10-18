"""Image-only editor inputs, no mask canvas, and compatible Android endpoints.

Real Gradio components, preprocessing and handler routing. Only image inference
is substituted in the handler check; no models or GPU are used.
"""
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import os, sys

sys.path.insert(0, os.getcwd())

import gradio as gr
import numpy as np
from PIL import Image

from ui.tabs.img_editor import events
from ui.tabs.img_editor.ui import build_img_editor_ui


def main():
    with gr.Blocks() as app:
        ui = build_img_editor_ui()
        events.wire_events(ui)
    assert isinstance(ui["input_img"], gr.Image)
    config = app.get_config_file()
    components = {component["id"]: component for component in config["components"]}
    assert not any(component["type"] == "imageeditor" for component in components.values())
    endpoints = {dependency["api_name"]: dependency
                 for dependency in config["dependencies"]}
    for name, count in (("edit", 8), ("edit_options", 9), ("edit_select", 13)):
        endpoint = endpoints[name]
        assert len(endpoint["inputs"]) == count, endpoint
        assert components[endpoint["inputs"][0]]["type"] == "image", name
    print("PASS: entrada Image sin lienzo de máscaras; contratos 13/9/8 intactos")

    pixels = np.random.default_rng(18).integers(0, 256, (32, 48, 3), dtype=np.uint8)
    source = Image.fromarray(pixels)
    transferred = ui["input_img"].preprocess(ui["input_img"].postprocess(source))
    assert np.array_equal(np.asarray(transferred), pixels), "transferencia con pérdida"
    print("PASS: transferencia PNG conserva todos los píxeles de la imagen")

    received = []

    def generate(**kwargs):
        received.append(kwargs)
        return kwargs["image"], "OK", None

    manager = SimpleNamespace(generate_intelligent=generate,
                              _last_instruction_compliance=None,
                              _release_local_gpu_before_comfy=lambda: None)
    with tempfile.TemporaryDirectory(prefix="editor_api_") as directory:
        with (
            patch.object(events, "get_img_editor_manager", return_value=manager),
            patch.object(events, "get_img_editor_output_dir", return_value=directory),
            patch.object(events, "_cleanup_vram"),
        ):
            result, status, mask = events.on_generate(
                transferred, "make the jacket red", "kontext", 0,
                False, 1, False, "hd",
            )
        assert result is not None and "Guardado" in status, status
        assert len(list(Path(directory).glob("*.png"))) == 1
    assert len(received) == 1 and mask is None
    assert received[0]["mask_image"] is None
    assert received[0]["mask_mode"] == "global"
    assert np.array_equal(np.asarray(received[0]["image"]), pixels)
    print("PASS: Kontext recibe la foto original, sin máscara manual")


if __name__ == "__main__":
    main()
