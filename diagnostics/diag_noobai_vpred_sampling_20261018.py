#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostico (v2): por que ILUS_noobai_xl_vpred_1.safetensors sale negra por el
cliente de Generate si el mismo checkpoint sale bien en el workflow minimo.

Hecho medido:
- El checkpoint SI es v-pred (2516 tensores y marca `v_pred`; Illustrious eps
  tiene 2515 y ninguna). Safetensors sin metadatos.
- Workflow minimo 512x512 / 8 pasos / seed 777: imagen real en las 5 variantes
  (eps, v_pred+zsnr true, v_pred+zsnr false, dpmpp), con std 40-53. Las tres
  primeras salieron IDENTICAS, senal de que ComfyUI ya autodetecta v-pred.
- Cliente de Generate 768x512 / 12 pasos / seed 4242 / euler+normal + nodo
  ModelSamplingDiscrete v_prediction+zsnr=true: NEGRA (mean 3.7, std 5.9).

Este probe reproduce los parametros EXACTOS del cliente y va variando una sola
cosa por render (zsnr, nodo, seed, pasos) para ver que combinacion da negra.
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
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "noobai_vpred_sampling_20261018")
PROMPT = "masterpiece, best quality, (1girl, silver hair, rooftop at sunset:1.35)"
NEGATIVE = ("worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, "
            "extra fingers, fewer fingers, text, watermark, signature, jpeg artifacts, sketch")

WIDTH, HEIGHT, CFG = 768, 512, 5.0

# (nombre, sampling(None=sin nodo), zsnr, sampler, scheduler, steps, seed)
VARIANTS = [
    ("cliente_exacto", "v_prediction", True, "euler", "normal", 12, 4242),
    ("zsnr_false", "v_prediction", False, "euler", "normal", 12, 4242),
    ("sin_nodo_eps", None, False, "euler", "normal", 12, 4242),
    ("seed_777", "v_prediction", True, "euler", "normal", 12, 777),
    ("pasos_28", "v_prediction", True, "euler", "normal", 28, 4242),
]


def build(variant):
    _name, sampling, zsnr, sampler, scheduler, steps, seed = variant
    wf = {
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": CKPT}},
        "2": {"class_type": "EmptyLatentImage",
              "inputs": {"width": WIDTH, "height": HEIGHT, "batch_size": 1}},
        "3": {"class_type": "CLIPTextEncode", "inputs": {"text": PROMPT, "clip": ["1", 1]}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["1", 1]}},
        "6": {"class_type": "KSampler", "inputs": {
            "model": ["1", 0], "positive": ["3", 0], "negative": ["4", 0],
            "latent_image": ["2", 0], "seed": seed, "steps": steps, "cfg": CFG,
            "sampler_name": sampler, "scheduler": scheduler, "denoise": 1.0}},
        "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["1", 2]}},
        "8": {"class_type": "SaveImage",
              "inputs": {"images": ["7", 0], "filename_prefix": f"vpdiag_{_name}"}},
    }
    if sampling:
        wf["5"] = {"class_type": "ModelSamplingDiscrete",
                   "inputs": {"model": ["1", 0], "sampling": sampling, "zsnr": zsnr}}
        wf["6"]["inputs"]["model"] = ["5", 0]
    return wf


def render(name, workflow):
    base = get_comfyui_url()
    started = time.time()
    response = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=30)
    if response.status_code != 200:
        return None, f"HTTP {response.status_code}: {response.text[:200]}"
    prompt_id = response.json().get("prompt_id")
    deadline = time.time() + 600
    while time.time() < deadline:
        history = requests.get(f"{base}/history/{prompt_id}", timeout=30).json()
        entry = history.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                return None, str(status.get("messages"))[:300]
            for node in entry.get("outputs", {}).values():
                for image in node.get("images", []):
                    params = {"filename": image["filename"],
                              "subfolder": image.get("subfolder", ""),
                              "type": image.get("type", "output")}
                    data = requests.get(f"{base}/view", params=params, timeout=60).content
                    return Image.open(io.BytesIO(data)).convert("RGB"), f"{time.time() - started:.0f}s"
        time.sleep(2)
    return None, "timeout"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for variant in VARIANTS:
        name = variant[0]
        image, note = render(name, build(variant))
        if image is None:
            print(f"FALLO   {name:20s} {note}")
            continue
        pixels = np.asarray(image, dtype=np.float32)
        state = "OK    " if pixels.std() > 20 else "NEGRA "
        image.save(os.path.join(OUT_DIR, f"v2_{name}.png"))
        print(f"{state} {name:20s} mean={pixels.mean():6.1f} std={pixels.std():6.1f} ({note})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
