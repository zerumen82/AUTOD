#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Probe de MASCARAS en el Editor (2026-10-18): Kontext vs LongCat Full.

Pregunta del usuario (2026-10-18): "revisa lo de las mascaras, porque si cambio
no quiero volver atras". Auditoria en codigo + aqui confirmacion de despacho:

- flux_edit_comfy_client.py l.546-556: mask_image se DESCARTA para
  kontext/klein ("una mascara de inpaint no puede aplicarse aqui... Ignorada
  siempre en klein/kontext") — el EDIT arranca de latente vacio con la imagen
  completa como referencia.
- Para el resto de ramas (LongCat/Qwen/flux) la mascara llega al workflow:
  SetLatentNoiseMask ("foto") o VAEEncodeForInpaint ("inpaint").

Este probe no mide calidad visual (eso lo juzga quien mira la imagen): mide
DESPACHO — que el print de "mascara omitida" sale en Kontext y que el workflow
de LongCat declara mask=inpaint/foto con la mascara colgada.

Mismo retrato de referencia; mascara = mitad DERECHA (fondo/calle). La
instruccion pide cambiar ZONA ENMASCARADA (edificio nocturno), asi que ademas
se ve si el cambio se queda dentro de la mitad derecha.
"""
import os
import sys
import time

import numpy as np
import requests
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.img_editor.flux_edit_comfy_client import FluxEditComfyClient  # noqa: E402
from roop.img_editor.flux_gen_comfy_client import get_comfyui_url  # noqa: E402

REF = os.path.join(ROOT, "diagnostics", "out", "gen_uncensored_20261018", "lustify4.png")
OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "edit_mask_20261018")

INSTRUCCION = ("make the buildings and the street in the right half of the image "
               "a dark night scene with blue tones, left half unchanged, "
               "same person, same pose")

CASOS = [
    ("kontext_mascara", "flux1-kontext-dev-Q4_K_M.gguf", 20, 3.5),
    ("longcat_mascara", "LongCat-Image-Edit-Q4_K_S.gguf", 35, 2.5),
]

DENOISE = 1.0


def cola_ocupada():
    try:
        queue = requests.get(f"{get_comfyui_url()}/queue", timeout=5).json()
        return len(queue.get("queue_running", [])) + len(queue.get("queue_pending", []))
    except Exception:
        return -1


def main():
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

    # Mascara = mitad derecha (blanco = zona a editar).
    mask = Image.new("L", reference.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.rectangle([reference.width // 2, 0, reference.width, reference.height], fill=255)

    only = []
    if "--only" in sys.argv:
        idx = sys.argv.index("--only")
        if idx + 1 < len(sys.argv):
            only = [a.strip() for a in sys.argv[idx + 1].split(",") if a.strip()]

    for etiqueta, modelo, steps, cfg in CASOS:
        if only and etiqueta not in only:
            continue
        client = FluxEditComfyClient()
        started_load = time.time()
        loaded, mensaje = client.load(modelo)
        if not loaded:
            print(f"FALLO {etiqueta}: load: {mensaje}")
            continue
        load_s = time.time() - started_load
        started = time.time()
        try:
            result, note = client.generate(
                reference, INSTRUCCION,
                num_inference_steps=steps, guidance_scale=cfg,
                seed=9090, denoise=DENOISE, mask_image=mask,
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
        # Diferencia izquierda/derecha: si la mascara manda, la mitad izquierda
        # debe parecerse a la referencia y la derecha cambiar.
        ref_px = np.asarray(reference, dtype=np.float32)
        mitad = reference.width // 2
        diff_izq = np.abs(pixels[:, :mitad] - ref_px[:, :mitad]).mean()
        diff_der = np.abs(pixels[:, mitad:] - ref_px[:, mitad:]).mean()
        print(f"OK   {etiqueta:18s} load={load_s:.0f}s gen={elapsed:.0f}s "
              f"diff_izq={diff_izq:5.1f} diff_der={diff_der:5.1f} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
