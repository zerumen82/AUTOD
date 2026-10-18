#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico (v3): capturar el workflow EXACTO que envia el cliente de Generate a
NoobAI (sin parchear el prompt) y volver a renderizarlo fuera del cliente.

Motivo: con los mismos parametros (768x512, 12 pasos, seed 4242, euler/normal,
v_prediction+zsnr), el workflow minimo da mean 21 / std 24 pero la muestra cruda
que guarda ComfyUI via el cliente da mean 3.7 / std 5.9 (negra). Lo unico que
queda por comparar es el texto de prompt/negative que arma el cliente.
"""
import io
import os
import sys
import time
from unittest.mock import patch

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient  # noqa: E402
from roop.comfy_workflows import get_comfyui_url  # noqa: E402

PROMPT = "a wooden cabin in a pine forest, morning light, detailed"


def capture(alias, inject_nsfw, seed, steps):
    client = FluxGenComfyClient()
    client.load(alias)
    captured = []

    def fake_post(url, **kwargs):
        captured.append(kwargs["json"]["prompt"])
        raise RuntimeError("capturado")

    with patch.object(client, "is_available", return_value=True), \
            patch("roop.img_editor.flux_gen_comfy_client.requests.post", side_effect=fake_post):
        try:
            client.generate(PROMPT, width=768, height=512, seed=seed, steps=steps,
                            lora_name="None", inject_nsfw=inject_nsfw)
        except Exception:
            pass
    return (captured[0] if captured else None), client


def render(workflow, tag):
    base = get_comfyui_url()
    response = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:300]}"
    prompt_id = response.json().get("prompt_id")
    deadline = time.time() + 600
    while time.time() < deadline:
        entry = requests.get(f"{base}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return f"ERROR COMFYUI: {str(status.get('messages'))[:400]}"
            for node in entry.get("outputs", {}).values():
                for image in node.get("images", []):
                    params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                              "type": image.get("type", "output")}
                    data = requests.get(f"{base}/view", params=params, timeout=60).content
                    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
                    return (f"mean={pixels.mean():6.1f} std={pixels.std():6.1f} "
                            f"mediana={np.percentile(pixels, 50):5.0f} [{tag}]")
        time.sleep(2)
    return "timeout"


def main():
    for inject_nsfw in (True, False):
        for steps, seed in ((12, 4242), (28, 4242)):
            wf, _client = capture("noobai_vpred", inject_nsfw, seed, steps)
            if wf is None:
                print(f"FALLO captura inject_nsfw={inject_nsfw} steps={steps}")
                continue
            pos = wf.get("6", {}).get("inputs", {}).get("text", "")
            neg = wf.get("7", {}).get("inputs", {}).get("text", "")
            print(f"--- inject_nsfw={inject_nsfw} steps={steps} seed={seed}")
            print(f"    POS({len(pos)}): {pos[:220]}")
            print(f"    NEG({len(neg)}): {neg[:220]}")
            print(f"    {render(wf, f'nsfw={inject_nsfw} steps={steps} seed={seed}')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
