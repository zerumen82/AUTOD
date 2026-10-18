#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico (v5): que configuracion de muestreo deja de salir negra NoobAI v-pred
de forma REPRODUCIBLE.

Medido hasta ahora (mismo checkpoint, mismos parametros, resultados distintos entre
sesiones):
- 12 pasos + nodo v_prediction+zsnr=true  -> negra (mean 4.2 / std 5.8), varias veces.
- 12 pasos + sin nodo (auto)              -> mean 21.0 / std 24.3 en una sesion.
- 28 pasos + sin nodo (auto)              -> mean 69.6 / std 77.3 en una sesion.
- 28 pasos + nodo v_pred+zsnr=true        -> mean 19.5 / std 43.3 en otra sesion.
Y el diff de JSON demuestra que cliente y workflow minimo son identicos salvo el
nombre de fichero. Por eso aqui se repite cada variante dos veces seguidas en la
misma sesion: sin repeticiones no se puede distinguir un arreglo de un suerte.
"""
import io
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.comfy_workflows import get_comfyui_url  # noqa: E402

CKPT = "ILUS_noobai_xl_vpred_1.safetensors"
PROMPT = "a wooden cabin in a pine forest, morning light, detailed"
NEGATIVE = ("worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, "
            "extra fingers, fewer fingers, text, watermark, signature, jpeg artifacts, sketch")
W, H, STEPS, CFG, SEED = 768, 512, 30, 5.0, 4242

VARIANTS = [
    ("sin_nodo", None, False),
    ("zsnr_false", "v_prediction", False),
    ("zsnr_true", "v_prediction", True),
]


def build(name, sampling, zsnr):
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["2", 0], "seed": SEED, "steps": STEPS, "cfg": CFG,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": f"vp5_{name}"}},
    }
    if sampling:
        wf["5"] = {"class_type": "ModelSamplingDiscrete",
                   "inputs": {"model": ["1", 0], "sampling": sampling, "zsnr": zsnr}}
        wf["6"]["inputs"]["model"] = ["5", 0]
    return wf


def render(name, workflow):
    base = get_comfyui_url()
    response = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
    if response.status_code != 200:
        return f"HTTP {response.status_code}: {response.text[:200]}"
    prompt_id = response.json().get("prompt_id")
    deadline = time.time() + 900
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
                    pixels = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"), dtype=np.float32)
                    return (f"mean={pixels.mean():6.1f} std={pixels.std():6.1f} "
                            f"mediana={np.percentile(pixels, 50):5.0f} "
                            f"p95={np.percentile(pixels, 95):5.0f}")
        time.sleep(2)
    return "timeout"


def main():
    print(f"== NoobAI v-pred: {STEPS} pasos, {W}x{H}, cfg {CFG}, euler/normal, seed {SEED} ==")
    for repeticion in (1, 2):
        for name, sampling, zsnr in VARIANTS:
            print(f"{name:12s} rep{repeticion}: {render(name, build(name, sampling, zsnr))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
