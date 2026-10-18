#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prueba de humo de los checkpoints sin censura descargados el 2026-10-18.

Fase 1 (sin GPU): contrato del workflow por alias. Comprueba que load() acepta el
  alias y que el workflow que se enviaria a ComfyUI usa el cargador correcto
  (CheckpointLoaderSimple para SDXL, UnetLoaderGGUF + CLIPLoaderGGUF/T5 para
  Chroma), y los valores de config/model_configs.json (steps, sampler, scheduler,
  cfg), incluido que NoobAI V-Pred no lleve Karras ni lleve el nodo v-pred
  redundante (el checkpoint trae la clave `v_pred` y ComfyUI la autodetecta).

Fase 2 (GPU real, --real): carga cada modelo en ComfyUI y renderiza UNA imagen
  768x512, con los pasos de su propia configuracion y con un prompt de SU
  dominio (medido el 2026-10-18: NoobAI da imagen negra en los 3 seeds con un
  prompt fotorreal de bosque e imagen real en los 3 con un prompt de personaje;
  probar un modelo de ilustracion con un prompt fotorreal seria una prueba rota).
  No mide calidad: mide que los pesos cargan, muestrean y decodifican. La
  comprobacion automatica de salida es std>20 y p95>120, umbrales tomados de lo
  medido (prompts rotos daban p95 48-101, imagenes reales 206-255); lo unico que
  un probe puede afirmar sin ojos. La validacion visual es del usuario.
"""
import os
import sys
import time
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from roop.img_editor.flux_gen_comfy_client import (  # noqa: E402
    FluxGenComfyClient,
    get_comfyui_url,
)

OUT_DIR = os.path.join(ROOT, "diagnostics", "out", "gen_uncensored_20261018")

CASES = [
    # (alias, fichero, es_sdxl, prompt de prueba)
    # El prompt importa: medido el 2026-10-18, NoobAI (modelo de personajes anime
    # entrenado con tags de booru) da imagen negra en los 3 seeds probados con un
    # prompt fotorreal de bosque, y imagen real en los 3 con un prompt de personaje.
    # Probar un modelo de ilustracion con un prompt fotorreal seria una prueba rota,
    # no un modelo roto.
    ("lustify4", "REAL_lustify4_v40.safetensors", True,
     "photorealistic portrait of a woman with brown hair, denim jacket, "
     "city street, natural light, detailed skin"),
    ("biglove_xl4", "REAL_biglove_xl4.safetensors", True,
     "photorealistic portrait of a man with a beard, leather jacket, "
     "studio light, detailed skin"),
    ("wai_illustrious_v17", "ILUS_wai_illustrious_v17.safetensors", True,
     "masterpiece, best quality, 1girl, silver hair, blue eyes, school uniform, classroom"),
    ("noobai_vpred", "ILUS_noobai_xl_vpred_1.safetensors", True,
     "masterpiece, best quality, 1girl, silver hair, blue eyes, school uniform, classroom"),
    ("illustrious_xl_v2", "ILUS_illustrious_xl_v2.safetensors", True,
     "masterpiece, best quality, 1girl, silver hair, blue eyes, school uniform, classroom"),
    ("chroma", "DIT_chroma1-hd_q4_k_m.gguf", False,
     "a wooden cabin in a pine forest, morning light, detailed"),
]


def capture_workflow(alias, prompt):
    """Construye el workflow real sin enviarlo y devuelve (wf, conf, cliente)."""
    client = FluxGenComfyClient()
    loaded, message = client.load(alias)
    if not loaded:
        return None, None, message
    captured = []

    def fake_post(url, **kwargs):
        captured.append(kwargs["json"]["prompt"])
        raise RuntimeError("capturado")

    with patch.object(client, "is_available", return_value=True), \
            patch.object(client, "_prepare_prompt_intelligent", return_value=(prompt, {})), \
            patch.object(client, "_prepare_prompt", side_effect=lambda p: p), \
            patch("roop.img_editor.flux_gen_comfy_client.requests.post", side_effect=fake_post):
        try:
            client.generate(prompt, width=768, height=512, seed=1234, lora_name="None")
        except Exception:
            pass
    if not captured:
        return None, None, "sin workflow capturado"
    return captured[0], client._get_model_conf(), ""


def phase1(cases=CASES):
    print("== Fase 1: contrato del workflow (sin GPU) ==")
    failures = 0
    for alias, expected_file, is_sdxl, prompt in cases:
        wf, conf, error = capture_workflow(alias, prompt)
        if wf is None:
            print(f"FALLO {alias}: {error}")
            failures += 1
            continue
        problems = []
        loader = wf.get("2", {}).get("class_type")
        loader_inputs = wf.get("2", {}).get("inputs", {})
        if is_sdxl:
            if loader != "CheckpointLoaderSimple":
                problems.append(f"cargador {loader}")
            if loader_inputs.get("ckpt_name") != expected_file:
                problems.append(f"ckpt_name {loader_inputs.get('ckpt_name')}")
        else:
            if loader != "UnetLoaderGGUF":
                problems.append(f"cargador {loader}")
            if loader_inputs.get("unet_name") != expected_file:
                problems.append(f"unet_name {loader_inputs.get('unet_name')}")
            if wf.get("3", {}).get("class_type") != "CLIPLoaderGGUF":
                problems.append(f"clip {wf.get('3', {}).get('class_type')}")
            if wf.get("3", {}).get("inputs", {}).get("type") != "chroma":
                problems.append("tipo de encoder distinto de chroma")
            if not any(n.get("class_type") == "T5TokenizerOptions" for n in wf.values()):
                problems.append("falta T5TokenizerOptions")
            if any(n.get("class_type") == "FluxGuidance" for n in wf.values()):
                problems.append("FluxGuidance no corresponde a Chroma")
        sampler = wf.get("8", {}).get("inputs", {})
        for key, expected in (("sampler_name", conf["sampler"]), ("scheduler", conf["scheduler"]),
                              ("steps", conf["steps"])):
            if sampler.get(key) != expected:
                problems.append(f"{key}={sampler.get(key)} (config {expected})")
        if sampler.get("cfg") != conf["cfg"]:
            problems.append(f"cfg={sampler.get('cfg')} (config {conf['cfg']})")
        if alias == "noobai_vpred" and sampler.get("scheduler") == "karras":
            problems.append("v-pred con Karras")
        if conf.get("v_prediction"):
            # NoobAI trae la clave `v_pred`, ComfyUI la autodetecta y el nodo
            # override sobra (medido: imagen identica con y sin el).
            node = next((n for n in wf.values() if n.get("class_type") == "ModelSamplingDiscrete"), None)
            if node:
                problems.append("nodo ModelSamplingDiscrete redundante (el checkpoint ya trae `v_pred`)")
            elif sampler.get("model") != ["2", 0]:
                problems.append(f"KSampler no toma el modelo del checkpoint: {sampler.get('model')}")
        state = "OK   " if not problems else "FALLO"
        if problems:
            failures += 1
        print(f"{state} {alias:22s} cargador={loader} cfg={sampler.get('cfg')} "
              f"sampler={sampler.get('sampler_name')}/{sampler.get('scheduler')} steps={sampler.get('steps')}")
        for problem in problems:
            print(f"      {problem}")
    return failures


def phase2(cases=CASES):
    print("== Fase 2: render real en ComfyUI (pasos de la config, 768x512) ==")
    import requests

    base = get_comfyui_url()
    try:
        if requests.get(f"{base}/system_stats", timeout=5).status_code != 200:
            print("FALLO ComfyUI no responde")
            return 1
    except Exception as error:
        print(f"FALLO ComfyUI: {error}")
        return 1

    os.makedirs(OUT_DIR, exist_ok=True)
    failures = 0
    for alias, _expected_file, _is_sdxl, prompt in cases:
        client = FluxGenComfyClient()
        loaded, message = client.load(alias)
        if not loaded:
            print(f"FALLO {alias}: load fallo: {message}")
            failures += 1
            continue
        started = time.time()
        try:
            # Sin steps: usa los de config/model_configs.json, que es como se
            # renderiza de verdad (recortarlos hace salir oscuro y mentir).
            result, gen_message = client.generate(
                prompt, width=768, height=512, seed=4242, lora_name="None"
            )
        except Exception as error:
            print(f"FALLO {alias}: {type(error).__name__}: {error}")
            failures += 1
            continue
        elapsed = time.time() - started
        # generate() devuelve GenResult (con .image), no una PIL.Image.
        image = getattr(result, "image", None)
        if image is None:
            print(f"FALLO {alias}: sin imagen ({gen_message})")
            failures += 1
            continue
        path = os.path.join(OUT_DIR, f"{alias}.png")
        image.save(path)
        pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
        std = float(pixels.std())
        p95 = float(np.percentile(pixels, 95))
        # std solo no basta: una imagen casi negra con destellos daba std 34.
        # Medido el 2026-10-18: prompts rotos daban p95 48-101, imagenes reales 206-255.
        ok = std > 20.0 and p95 > 120.0
        state = "OK   " if ok else "SOSPECHOSO"
        if not ok:
            failures += 1
        print(f"{state} {alias:22s} {image.size} std={std:5.1f} p95={p95:5.0f} "
              f"en {elapsed:.0f}s -> {path}")
    return failures


def main():
    only = []
    if "--only" in sys.argv:
        index = sys.argv.index("--only")
        if index + 1 < len(sys.argv):
            only = [a.strip() for a in sys.argv[index + 1].split(",") if a.strip()]
    cases = [c for c in CASES if not only or c[0] in only]
    failures = phase1(cases)
    if "--real" in sys.argv:
        failures += phase2(cases)
    else:
        print("(fase 2 omitida: pasa --real para render GPU)")
    print(f"== RESULTADO: {'OK' if failures == 0 else str(failures) + ' fallos'} ==")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

