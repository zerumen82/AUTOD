#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe de FLUX.1 Kontext dev en el Editor (2026-10-18): TEXTOS + IMG2IMG.

Tras el probe de censura (Kontext = 0% censura, mejor fidelidad), aqui se mide
la otra mitad de un buen editor. Cuatro casos sobre la misma referencia:

1. texto_nuevo        — cartel de neón 'HOTEL' que no existía (texto, denoise 1,0)
2. texto_jacket       — palabra 'PARIS' sobre la chaqueta (texto sobre textura)
3. img2img_suave_040  — luz de atardecer (transformación sutil, denoise 0,40)
4. img2img_fuerte_080 — vestido rojo de noche (transformación fuerte, 0,80)

Se usa el cliente real del Editor. El juicio de legibilidad/fidelidad es
visual (se leen los PNG); aqui solo se dejan metricas basicas + tiempo.
"""
import os
import sys
import time

import numpy as np
import requests
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient  # noqa: E402
from roop.img_editor.flux_gen_comfy_client import get_comfyui_url  # noqa: E402

REF = os.path.join(ROOT, "diagnostics", "out", "gen_uncensored_20261018", "lustify4.png")
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "edit_text_20261018")

MODELO = "flux1-kontext-dev-Q4_K_M.gguf"

# Cada caso lleva su denoise: 1,0 = edicion por instruccion (Kontext es modelo
# de EDICION, no img2img: el denoise efectivo lo decide la rama del cliente);
# 0,40 y 0,80 = img2img puro sobre el latente de la foto para medir la gama
# completa (sutil -> fuerte) que pide el usuario.
CASOS = [
    ("texto_nuevo",
     "add a bright neon sign hanging on the brick wall on the right that reads "
     "'HOTEL' in glowing red capital letters, everything else unchanged", 1.0),
    ("texto_jacket",
     "write the word 'PARIS' in white capital letters across the chest of the "
     "denim jacket, same pose, same face, everything else unchanged", 1.0),
    ("img2img_suave_040",
     "golden hour sunset light, warm tones, cinematic mood, same person, "
     "same pose, same clothes", 0.40),
    ("img2img_fuerte_080",
     "the woman now wears an elegant red evening dress, same face, "
     "same pose, same street background", 0.80),
]

STEPS = 20
CFG = 3.5


def cola_ocupada():
    try:
        queue = requests.get(f"{get_comfyui_url()}/queue", timeout=5).json()
        return len(queue.get("queue_running", [])) + len(queue.get("queue_pending", []))
    except Exception:
        return -1


def main():
    # Espera en vez de abortar (doctrina del probe de censura v2: un probe que
    # aborta con la cola ocupada muere si otro proceso usa ComfyUI).
    esperados = 0
    while "--force" not in sys.argv:
        ocupada = cola_ocupada()
        if ocupada <= 0:
            break
        esperados += 15
        print(f"ESPERA cola ocupada ({ocupada} trabajos): duermo 15s "
              f"(llevamos {esperados}s)", flush=True)
        if esperados > 1800:
            print("FALLO cola ocupada mas de 30 min: me retiro")
            return 1
        time.sleep(15)

    if not os.path.isfile(REF):
        print(f"FALLO sin imagen de referencia: {REF}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    reference = Image.open(REF).convert("RGB")
    reference = reference.resize((768, int(768 * reference.height / reference.width)))

    only = []
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = [a.strip() for a in sys.argv[idx + 1].split(",") if a.strip()]

    for etiqueta, instruccion, denoise in CASOS:
        if only and etiqueta not in only:
            continue
        client = FluxEditComfyClient()
        started_load = time.time()
        loaded, mensaje = client.load(MODELO)
        if not loaded:
            print(f"FALLO {etiqueta}: load: {mensaje}")
            continue
        load_s = time.time() - started_load
        started = time.time()
        try:
            result, note = client.generate(
                reference, instruccion,
                num_inference_steps=STEPS, guidance_scale=CFG,
                seed=7777, denoise=denoise,
            )
        except Exception as error:
            print(f"FALLO {etiqueta}: {type(error).__name__}: {error}")
            continue
        elapsed = time.time() - started
        image = getattr(result, "image", None) if result is not None else None
        if image is None:
            print(f"FALLO {etiqueta}: sin imagen ({note})")
            continue
        path = os.path.join(OUT_DIR, f"{etiqueta}.png")
        image.save(path)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        print(f"OK   {etiqueta:18s} d={denoise:.2f} load={load_s:.0f}s gen={elapsed:.0f}s "
              f"std={pixels.std():5.1f} p95={np.percentile(pixels, 95):5.0f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
