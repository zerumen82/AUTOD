#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico (v6): separar 'el modelo esta mal' de 'le estoy preguntando mal'.

Medido en v5 (30 pasos, 768x512, cfg 5.0, euler/normal, seed 4242): las tres
variantes de muestreo dan lo mismo y de forma repetible, pero oscuro
(mean ~20, mediana 5, p95 ~101): imagen negra con destellos. Sin embargo, en otra
sesion ese mismo seed a 28 pasos dio mean 69.6 / std 77.3.

Hipotesis: NoobAI es un modelo de personajes anime entrenado con tags de booru. El
prompt del probe ("a wooden cabin in a pine forest, morning light") es fotorreal y
fuera de su dominio, y un seed concreto puede dar composicion negra. Aqui se
cruza prompt (bosque vs personaje) x varios seeds, todo con la config real.
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
NEGATIVE = ("worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, "
            "extra fingers, fewer fingers, text, watermark, signature, jpeg artifacts, sketch")
W, H, STEPS, CFG = 768, 512, 30, 5.0

PROMPTS = {
    "bosque": "a wooden cabin in a pine forest, morning light, detailed",
    "personaje": "masterpiece, best quality, 1girl, silver hair, blue eyes, school uniform, classroom",
}
SEEDS = [4242, 123456, 777]


def build(prompt_key, seed):
    return {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": W, "height": H, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPTS[prompt_key], "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["2", 0], "seed": seed, "steps": STEPS, "cfg": CFG,
            "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": f"vp6_{prompt_key}_{seed}"}},
    }


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
                            f"mediana={np.percentile(pixels, 50):5.0f} p95={np.percentile(pixels, 95):5.0f}")
        time.sleep(2)
    return "timeout"


def main():
    print(f"== NoobAI v-pred sin nodo extra: {STEPS} pasos, {W}x{H}, cfg {CFG} ==")
    for prompt_key in PROMPTS:
        for seed in SEEDS:
            etiqueta = f"{prompt_key:10s} seed={seed:7d}"
            print(f"{etiqueta}: {render(prompt_key, build(prompt_key, seed))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
