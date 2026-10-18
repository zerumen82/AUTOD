#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe de POV en el Editor (2026-10-18): FLUX.1 Kontext dev cambiando angulo.

Tras censura/textos/img2img, aqui se mide la ultima palanca que pide el usuario:
CAMBIAR EL PUNTO DE VISTA. Un cambio de POV no esta en la imagen de origen, asi
que hay que darle al modelo margen (denoise 0,70-0,75 — el terreno que el probe
img2img marco como "reinterpreta la escena"). El usuario indico que el parecido
de cara NO es prioritario aqui: lo que se juzga es que el angulo cambie de verdad
y la escena siga siendo creible.

Tres casos sobre la misma referencia:
1. pov_trasera   — camara detras, sobre el hombro (denoise 0,70)
2. pov_baja      — contrapicado dramatico desde abajo (denoise 0,75)
3. pov_cenital   — picado desde arriba (denoise 0,75)

Cliente real del Editor; el juicio del angulo es visual (se leen los PNG).
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
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "edit_pov_20261018")

MODELO = "flux1-kontext-dev-Q4_K_M.gguf"

CASOS = [
    ("pov_trasera_070",
     "camera now positioned behind the woman, over-the-shoulder view looking "
     "down the street, we see her from behind, same denim jacket, same street",
     0.70),
    ("pov_baja_075",
     "dramatic low angle shot from below looking up at the woman and the "
     "buildings, same street, same denim jacket",
     0.75),
    ("pov_cenital_075",
     "high angle shot from above looking down at the woman standing on the "
     "cobblestone street, same street, same denim jacket",
     0.75),
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
    # Espera en vez de abortar (doctrina de esta serie de probes).
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
                seed=31337, denoise=denoise,
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
