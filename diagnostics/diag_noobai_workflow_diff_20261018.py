#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico (v4): diff real entre el workflow del cliente y el workflow minimo.

Medido: mismo prompt, mismo negative, mismos parametros (768x512, 12 pasos, seed
4242, cfg 5.0, euler/normal), pero la muestra cruda del cliente es negra
(mean 4.2 / std 5.8) y la del workflow minimo no (mean 21.0 / std 24.3). Este
script vuelca los dos JSON y los normaliza para ver exactamente que difiere.
"""
import json
import os
import sys
import time
from unittest.mock import patch

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.img_editor.flux_gen_comfy_client import FluxGenComfyClient  # noqa: E402
from roop.comfy_workflows import get_comfyui_url  # noqa: E402

CKPT = "ILUS_noobai_xl_vpred_1.safetensors"
PROMPT = "a wooden cabin in a pine forest, morning light, detailed"
NEGATIVE = ("worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, "
            "extra fingers, fewer fingers, text, watermark, signature, jpeg artifacts, sketch")
W, H, STEPS, CFG, SEED = 768, 512, 12, 5.0, 4242


def client_workflow():
    client = FluxGenComfyClient()
    client.load("noobai_vpred")
    captured = []

    def fake_post(url, **kwargs):
        # Guardamos el payload COMPLETO, no solo el prompt: si el cliente envia
        # extra_data/client_id, eso no aparece en el prompt y no lo habriamos visto.
        captured.append(kwargs["json"])
        raise RuntimeError("capturado")

    with patch.object(client, "is_available", return_value=True), \
            patch("roop.img_editor.flux_gen_comfy_client.requests.post", side_effect=fake_post):
        try:
            client.generate(PROMPT, width=W, height=H, seed=SEED, steps=STEPS, lora_name="None")
        except Exception:
            pass
    return captured[0]


def minimal_workflow():
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "5": {"class_type": "ModelSamplingDiscrete",
              "inputs": {"model": ["1", 0], "sampling": "v_prediction", "zsnr": True}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["5", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["2", 0], "seed": SEED, "steps": STEPS, "cfg": CFG,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["7", 0], "filename_prefix": "diag"}},
    }


def normalize(workflow):
    """Convierte referencias a nodos en etiquetas de rol para poder comparar."""
    roles = {}
    for node_id, node in workflow.items():
        kind = node.get("class_type", "")
        if kind == "CheckpointLoaderSimple":
            roles[node_id] = "CKPT"
        elif kind == "EmptyLatentImage":
            roles[node_id] = "LATENT"
        elif kind == "CLIPTextEncode":
            roles.setdefault(node_id, "CLIPTEXT")
        elif kind == "ModelSamplingDiscrete":
            roles[node_id] = "SAMPLING"
        elif kind == "KSampler":
            roles[node_id] = "KSAMPLER"
        elif kind == "VAEDecode":
            roles[node_id] = "VAEDECODE"
        elif kind == "SaveImage":
            roles[node_id] = "SAVE"
        else:
            roles[node_id] = f"OTRO:{kind}"

    def ref(value):
        if isinstance(value, list) and len(value) == 2 and str(value[0]) in roles:
            return [roles[str(value[0])], value[1]]
        return value

    out = {}
    for node_id, node in workflow.items():
        kind = node.get("class_type", "")
        inputs = {k: ref(v) for k, v in node.get("inputs", {}).items()}
        if kind == "CLIPTextEncode":
            inputs = {"text": inputs.get("text", ""), "clip": inputs.get("clip")}
        out[roles[node_id] + ("_pos" if kind == "CLIPTextEncode" and inputs.get("text") == PROMPT else
                              "_neg" if kind == "CLIPTextEncode" else "")] = {"class": kind, "inputs": inputs}
    return out


def render(workflow, tag):
    base = get_comfyui_url()
    response = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    prompt_id = response.json().get("prompt_id")
    deadline = time.time() + 600
    while time.time() < deadline:
        entry = requests.get(f"{base}/history/{prompt_id}", timeout=30).json().get(prompt_id)
        if entry:
            if entry.get("status", {}).get("status_str") == "error":
                return "ERROR: " + str(entry["status"].get("messages"))[:300]
            for node in entry.get("outputs", {}).values():
                for image in node.get("images", []):
                    params = {"filename": image["filename"], "subfolder": image.get("subfolder", ""),
                              "type": image.get("type", "output")}
                    data = requests.get(f"{base}/view", params=params, timeout=60).content
                    from PIL import Image
                    import io
                    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
                    return f"{tag}: mean={pixels.mean():6.1f} std={pixels.std():6.1f} mediana={np.percentile(pixels, 50):5.0f}"
        time.sleep(2)
    return f"{tag}: timeout"


def main():
    payload = client_workflow()
    client_wf = payload.get("prompt")
    mini_wf = minimal_workflow()
    folder = os.path.join(ROOT, ".autodeep_temp")
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "workflow_cliente_noobai.json"), "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False)
    extras = {k: v for k, v in payload.items() if k != "prompt"}
    print(f"Payload del cliente (ademas de 'prompt'): {json.dumps(extras, ensure_ascii=False)[:400]}")
    print()

    a, b = normalize(client_wf), normalize(mini_wf)
    print("== Diff de nodos normalizados (cliente vs minimo) ==")
    diff = False
    for key in sorted(set(a) | set(b)):
        if a.get(key) != b.get(key):
            diff = True
            print(f"[{key}]")
            print(f"   cliente: {json.dumps(a.get(key), ensure_ascii=False)}")
            print(f"   minimo : {json.dumps(b.get(key), ensure_ascii=False)}")
    if not diff:
        print("Sin diferencias tras normalizar")

    print()
    print("== Render de los dos workflows, en el mismo orden ==")
    print(render(mini_wf, "minimo "))
    print(render(client_wf, "cliente"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
