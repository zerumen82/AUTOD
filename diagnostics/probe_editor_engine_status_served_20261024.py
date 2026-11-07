"""Serve the actual Editor: unique motor choices, API and classical CPU color.

No inference doubles and no diffusion render. This certifies the served UI,
input/output and absence of a false model/verification claim, not Kontext image
quality. The diagnostic closes only its own temporary HTTP server.
"""
from __future__ import annotations

import json
import socket
import sys
import time
from pathlib import Path

import gradio as gr
import numpy as np
import pydantic.v1  # noqa: F401 -- initialize before Gradio background threads
import requests
from gradio_client import Client, handle_file
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roop.img_editor.edit_execution import current_execution  # noqa: E402
from ui.tabs.img_editor import events, state  # noqa: E402
from ui.tabs.img_editor.ui import build_img_editor_ui  # noqa: E402


def main():
    out = ROOT / "diagnostics/out/editor_engine_status_20261024" / str(time.time_ns())
    out.mkdir(parents=True, exist_ok=False)
    yy, xx = np.indices((128, 192))
    pixels = np.stack((70 + xx % 90, 60 + yy % 80, 65 + (xx + yy) % 60), axis=-1)
    source = out / "source.png"
    Image.fromarray(pixels.astype(np.uint8)).save(source)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with gr.Blocks() as app:
        ui = build_img_editor_ui()
        events.wire_events(ui)
    app.queue()
    client = None
    try:
        app.launch(server_name="127.0.0.1", server_port=port, share=False,
                   prevent_thread_lock=True, allowed_paths=[str(out)], quiet=True, show_error=True)
        url = f"http://127.0.0.1:{port}"
        response = requests.get(f"{url}/config", timeout=10)
        response.raise_for_status()
        config = response.json()
        (out / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        component = next(entry for entry in config["components"] if entry["id"] == ui["engine"]._id)
        choices = component["props"]["choices"]
        values = [value for _label, value in choices]
        assert len(values) == len(set(values)), choices
        assert values.count("kontext") == 1
        assert "no se sustituye" in component["props"]["info"]
        endpoints = {entry["api_name"]: entry for entry in config["dependencies"]}
        for name, count in (("edit", 8), ("edit_options", 9), ("edit_select", 13), ("edit_region", 14)):
            assert len(endpoints[name]["inputs"]) == count, name
        client = Client(url, verbose=False)
        rejected, rejection, _mask = client.predict(
            handle_file(str(source)), "", "kontext", 0, False, 1, False, "hd", api_name="/edit")
        assert rejected is None and rejection == "Escribe un prompt", rejection
        started = time.monotonic()
        result, message, _mask = client.predict(
            handle_file(str(source)), "", "kontext", 0, False, 1, True, "hd",
            False, False, False, True, False, api_name="/edit_select")
        assert result is not None, message
        with Image.open(result) as image:
            assert image.size == (192, 128)
            image.convert("RGB").save(out / "result.png")
        assert "Guardado" in message and "color" in message, message
        assert "Motor ejecutado: Kontext" not in message, "classical CPU color claimed model inference"
        assert "señal positiva" not in message, "no-prompt color claimed semantic verification"
        assert not state.is_generating and current_execution() is None
        metrics = {
            "scope": "Real HTTP UI + CPU classical color; no image-model inference or quality certification",
            "url": url, "choices": choices, "status": message,
            "empty_prompt_rejection": rejection, "elapsed_s": time.monotonic() - started,
            "source": str(source), "result": str(out / "result.png"),
            "api_inputs": {name: len(endpoints[name]["inputs"])
                           for name in ("edit", "edit_options", "edit_select", "edit_region")},
        }
        (out / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
        print(f"EVIDENCE {out}", flush=True)
    finally:
        if client is not None:
            client.close()
        app.close()


if __name__ == "__main__":
    main()
