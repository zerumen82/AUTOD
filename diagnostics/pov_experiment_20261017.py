"""Aislar por qué la ruta de cámara de producción no mueve el POV.

Dos corridas con la MISMA entrada y la MISMA pose que la petición real de
2026-10-17 16:50 («cambia el pov»), cambiando UN solo delta frente al
control de producción (AutoAutoQwenCamera_00004_, que NO movió la cámara):

  B (scale):  + FluxKontextImageScale -> ~1 MP antes de VAEEncode/image1
              (igual que la workflow de referencia que sí funcionó el 28/09)
  C (clipfp8): CLIP nativo fp8 (qwen_2.5_vl_7b_fp8_scaled, con torre visual)
              en vez de CLIPLoaderGGUF Q4_K_M, sin escalar (768 px igual
              que producción)

Métrica: correlación de Pearson en gris 256x170 entre entrada y salida.
Control de producción (sin movimiento) arriba; la oficina del 28/09
(con movimiento real) como referencia de qué números da un cambio real.
CPU puro para medir; la GPU solo la usa ComfyUI por petición HTTP.
"""
import io
import os
import sys
import time

import numpy as np
import requests
from PIL import Image


COMFY = "http://127.0.0.1:8188"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(
    ROOT, ".autodeep_temp",
    "96deeeb19739ab8cfbd8ee1d280cb98f125b33b33a6ac8b57bfa139e502b5dfc",
    "image.webp",
)
CONTROL = os.path.join(ROOT, "ui", "tob", "ComfyUI", "output",
                       "AutoAutoQwenCamera_00004_.png")
POSE = "<sks> front-right quarter view elevated shot medium shot"
SEED = 424242
UNET = "qwen-image-edit-2511-Q2_K.gguf"
ANGLE_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CLIP_GGUF = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
CLIP_FP8 = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
VAE = "qwen_image_vae.safetensors"


def correlation(a_path, b_path):
    def gray(path):
        im = Image.open(path).convert("L").resize((256, 170), Image.Resampling.LANCZOS)
        return np.asarray(im, np.float64).ravel()
    x, y = gray(a_path), gray(b_path)
    x, y = x - x.mean(), y - y.mean()
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    return float((x * y).sum() / denom) if denom else 0.0


def workflow(variant, image_name):
    clip_node = (
        {"class_type": "CLIPLoader", "inputs": {
            "clip_name": CLIP_FP8, "type": "qwen_image", "device": "default"}}
        if variant == "clipfp8"
        else {"class_type": "CLIPLoaderGGUF", "inputs": {
            "clip_name": CLIP_GGUF, "type": "qwen_image"}}
    )
    # B: la referencia oficial escala a ~1 MP y usa ese tensor en VAEEncode
    # e image1; producción mete el resize a 768 px directo.
    scale_to = "9" if variant == "scale" else "1"
    nodes = {
        "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
        "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["2", 0], "lora_name": ANGLE_LORA, "strength_model": 0.9}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
            "model": ["3", 0], "lora_name": LIGHTNING_LORA, "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {
            "model": ["4", 0], "shift": 3.1}},
        "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}},
        "7": clip_node,
        "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["7", 0], "prompt": POSE, "vae": ["8", 0],
            "image1": [scale_to, 0]}},
        "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
            "clip": ["7", 0], "prompt": "", "vae": ["8", 0],
            "image1": [scale_to, 0]}},
        "13": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero"}},
        "14": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
            "conditioning": ["12", 0], "reference_latents_method": "index_timestep_zero"}},
        "15": {"class_type": "KSampler", "inputs": {
            "model": ["6", 0], "positive": ["13", 0], "negative": ["14", 0],
            "latent_image": ["10", 0], "seed": SEED, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
        "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}},
        "17": {"class_type": "SaveImage", "inputs": {
            "images": ["16", 0], "filename_prefix": "AutoAutoPOVExp"}},
    }
    if variant == "scale":
        nodes["9"] = {"class_type": "FluxKontextImageScale", "inputs": {
            "image": ["1", 0]}}
        nodes["10"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": ["9", 0], "vae": ["8", 0]}}
    else:
        # Producción: resize manual a 768 px ANTES de subir (igual que el
        # cliente); VAEEncode e image1 leen el LoadImage directo.
        nodes["10"] = {"class_type": "VAEEncode", "inputs": {
            "pixels": ["1", 0], "vae": ["8", 0]}}
    return nodes


def wait_idle(timeout_s=900):
    started = time.time()
    while time.time() - started < timeout_s:
        try:
            queue = requests.get(f"{COMFY}/queue", timeout=8).json()
        except Exception as exc:
            print(f"[exp] cola no responde ({exc}); reintento", flush=True)
            time.sleep(5)
            continue
        if not queue.get("queue_running") and not queue.get("queue_pending"):
            print(f"[exp] cola libre tras {time.time() - started:.0f}s", flush=True)
            return True
        print(f"[exp] ocupada (running={len(queue.get('queue_running', []))} "
              f"pending={len(queue.get('queue_pending', []))}); espero", flush=True)
        time.sleep(10)
    return False


def production_resize(image):
    # Misma receta del cliente: lado largo 768, múltiplo de 16, LANCZOS.
    scale = 768.0 / max(image.size)
    width = max(16, round(image.width * scale / 16) * 16)
    height = max(16, round(image.height * scale / 16) * 16)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def run_variant(variant):
    print(f"[exp] === variante {variant} ===", flush=True)
    if not wait_idle():
        print("[exp] la cola no quedó libre; se omite", flush=True)
        return None
    source = Image.open(INPUT).convert("RGB")
    if variant == "clipfp8":
        source = production_resize(source)  # único delta: el CLIP
    buf = io.BytesIO()
    source.save(buf, format="PNG")
    buf.seek(0)
    uploaded = requests.post(f"{COMFY}/upload/image", files={
        "image": (f"pov_exp_{variant}.png", buf, "image/png")}, timeout=60)
    uploaded.raise_for_status()
    image_name = uploaded.json()["name"]
    nodes = workflow(variant, image_name)
    queued = requests.post(f"{COMFY}/prompt", json={"prompt": nodes}, timeout=30)
    queued.raise_for_status()
    payload = queued.json()
    if "prompt_id" not in payload:
        print(f"[exp] rechazado: {payload}", flush=True)
        return None
    print(f"[exp] encolado {payload['prompt_id']}", flush=True)
    started = time.time()
    while time.time() - started < 700:
        history = requests.get(f"{COMFY}/history/{payload['prompt_id']}",
                               timeout=15).json()
        if payload["prompt_id"] in history:
            entry = history[payload["prompt_id"]]
            status = entry.get("status", {})
            if status.get("status_str") == "error":
                print(f"[exp] ERROR {status}", flush=True)
                return None
            images = entry.get("outputs", {}).get("17", {}).get("images", [])
            if images:
                meta = images[0]
                raw = requests.get(f"{COMFY}/view", params={
                    "filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
                    "type": meta.get("type", "output")}, timeout=60)
                raw.raise_for_status()
                target = os.path.join(ROOT, "diagnostics", "out",
                                      f"pov_exp_{variant}_20261017.png")
                with open(target, "wb") as fh:
                    fh.write(raw.content)
                secs = time.time() - started
                corr = correlation(INPUT, target)
                print(f"[exp] OK {variant} ({secs:.0f}s) -> {target} "
                      f"correlacion={corr:.4f}", flush=True)
                return target
        time.sleep(4)
    print(f"[exp] timeout {variant}", flush=True)
    return None


def main():
    control = correlation(INPUT, CONTROL)
    print(f"[exp] control produccion (sin movimiento): correlacion={control:.4f}",
          flush=True)
    office = correlation(
        os.path.join(ROOT, "diagnostics", "img_editor_scene_office_final.png"),
        os.path.join(ROOT, "diagnostics", "editor_viewpoint_qwen_multiangle.png"),
    )
    print(f"[exp] referencia oficina 28/09 (movimiento real): correlacion={office:.4f}",
          flush=True)
    results = {}
    for variant in ("scale", "clipfp8"):
        results[variant] = run_variant(variant)
    print("[exp] RESUMEN", flush=True)
    print(f"[exp]   control produccion (igual POV): {control:.4f}", flush=True)
    print(f"[exp]   referencia oficina (POV movido): {office:.4f}", flush=True)
    for variant, target in results.items():
        if target:
            print(f"[exp]   {variant}: {correlation(INPUT, target):.4f} -> {target}",
                  flush=True)
        else:
            print(f"[exp]   {variant}: SIN RESULTADO", flush=True)
    if all(results.values()):
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
