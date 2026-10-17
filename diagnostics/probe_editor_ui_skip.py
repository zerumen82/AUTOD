"""probe_editor_ui_skip.py — contrato vigente de una petición focal ambigua.

Historia: con el código de 2026-10-01 «EL HOMBRE DEBE SER CALVO» se omitía
sin generar (CLIPSeg no medía la calvicie y la UI devolvía la foto igual con
«no se pudo localizar»). Desde el fix de cabeza-anclada (MEMORY 2026-10-12)
la petición SÍ genera: la cabeza del sujeto medido es la zona del pase
local, anclada (no_kontext=False, nodo 11), y solo esa zona se pega — la
foto entera nunca se regenera.

Este probe ejercita la ruta REAL de la UI (`on_generate`) y el manager real
con la generación Turbo simulada (sin GPU):

1. la petición genera (no se omite);
2. el cambio está acotado a una fracción local de la foto;
3. el workflow va anclado (no_kontext=False) y con máscara local.

Detección de caras y CLIPSeg corren reales en CPU; solo la inferencia
Turbo, el traductor de red y el verificador VLM están simulados.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.getcwd())

from ui.tabs.img_editor.events import on_generate, wire_events  # noqa: E402


SOURCE = os.path.join("ui", "tob", "ComfyUI", "input", "fast_1790331799.png")


class FakeClient:
    """Intercepta load/generate y registra el workflow despachado."""

    def __init__(self):
        self.calls = []

    def load(self, version):
        self.calls.append(("load", version))
        return True, "OK (stub)"

    def generate(self, **kwargs):
        self.calls.append(("generate", kwargs))
        return SimpleNamespace(image=Image.new("RGB", (256, 256), "red")), "OK (stub)"


def _check_wiring_concurrency():
    """Todos los listeners de on_generate deben ser de concurrencia ilimitada.

    Medido 2026-10-16 (gradio 6.13.0): los listeners de API sin
    concurrency_limit quedaban con la concurrencia por defecto (1) y un
    segundo envío mientras la primera petición corría se ENCOLABA y se
    ejecutaba tras ella sin rechazo («procesado dos veces», 18:56/19:00 de
    la sesión real). Con None el envío en caliente corre en paralelo y la
    guarda de on_generate lo rechaza en voz alta.
    """

    class FakeComponent:
        def __init__(self, name):
            self.name = name
            self.clicks = []
            self.changes = []

        def click(self, fn=None, inputs=None, outputs=None, **kwargs):
            self.clicks.append({"fn": fn, "inputs": inputs, **kwargs})
            return self

        def change(self, fn=None, inputs=None, outputs=None, **kwargs):
            self.changes.append({"fn": fn, **kwargs})
            return self

    keys = (
        "input_img", "prompt", "engine", "denoise", "autoregressive",
        "auto_iterations", "quality_mode", "quality_options", "enhance_tier",
        "quality_preserve_faces", "quality_deposter", "quality_upscale",
        "quality_color", "quality_realism", "output_img", "status",
        "mask_preview", "bt_use_as_input", "bt_open_folder",
    )
    ui_comp = {key: FakeComponent(key) for key in keys}
    for name in ("gen_btn", "legacy_options_btn", "legacy_api_btn"):
        ui_comp[name] = FakeComponent(name)
    wire_events(ui_comp)
    for btn_name, api in (
        ("gen_btn", "edit_select"),
        ("legacy_options_btn", "edit_options"),
        ("legacy_api_btn", "edit"),
    ):
        clicks = ui_comp[btn_name].clicks
        assert clicks, f"{btn_name} no se conectó en wire_events"
        click = clicks[0]
        assert click.get("api_name") == api, click
        assert click.get("concurrency_limit") is None, (
            f"{btn_name} sin concurrencia ilimitada: un envío en caliente se "
            f"encolaría y se ejecutaría sin rechazo ({click})"
        )
    print("OK: los tres listeners de on_generate corren en paralelo y la "
          "guarda rechaza los envíos en caliente")


def main():
    _check_wiring_concurrency()
    if not os.path.exists(SOURCE):
        print(f"FALTA {SOURCE}")
        sys.exit(1)

    source = Image.open(SOURCE).convert("RGB").resize((800, 533), Image.Resampling.LANCZOS)
    client = FakeClient()
    verifier = SimpleNamespace(
        verify=lambda *_args, **_kw: {"compliant": True, "source": "probe_stub"}
    )
    with (patch("ui.tabs.comfy_launcher.is_comfyui_running", return_value=True),
          patch("roop.img_editor.prompt_translator.translate_prompt",
                return_value="Man must be bald"),
          patch("roop.img_editor.flux_edit_comfy_client.get_flux_edit_comfy_client",
                return_value=client),
          patch("roop.img_editor.instruction_verifier.get_instruction_verifier",
                return_value=verifier),
          patch("roop.img_editor.img_editor_manager.get_vram_gb", return_value=8.0)):
        output, status, _mask = on_generate(
            source, "EL HOMBRE DEBE SER CALVO", "longcat", 0,
            False, 1, False, "hd",
        )

    assert output is not None, f"la peticion no genero: {status}"
    assert "Guardado" in status, status

    generations = [kwargs for name, kwargs in client.calls if name == "generate"]
    assert generations, f"sin llamadas de generacion: {client.calls}"
    first = generations[0]
    assert first["no_kontext"] is False, (
        f"la peticion focal debe ir ANCLADA (no_kontext=False): {first.get('no_kontext')}"
    )
    assert first.get("mask_image") is not None, (
        "el pase local debe llevar mascara (zona de la cabeza)"
    )

    a = np.asarray(source, dtype=np.int16)
    b = np.asarray(output.convert("RGB"), dtype=np.int16)
    assert a.shape == b.shape, (a.shape, b.shape)
    changed = (np.abs(a - b).max(axis=2) > 12).mean() * 100
    assert changed < 30.0, (
        f"el cambio {changed:.1f}% del canvas no esta acotado: "
        "la foto entera se regenero en vez de la cabeza"
    )
    assert changed > 0.5, (
        f"el cambio {changed:.1f}% es nulo: la peticion no aplico nada"
    )

    print(f"OK: la peticion focal ambiguia genera anclada "
          f"(mask/no_kontext ok, {len(generations)} pasada(s), "
          f"cambio {changed:.1f}% del canvas)")
    print(f"status={status}")


if __name__ == "__main__":
    main()
