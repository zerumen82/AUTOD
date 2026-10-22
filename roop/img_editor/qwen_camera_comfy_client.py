"""Image-to-image camera changes with Qwen Image Edit 2511 and the angle LoRA."""

import io
import os
import re
import time

import requests
from PIL import Image

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image
from roop.img_editor.flux_edit_comfy_client import (
    GenResult, _queue_is_idle, _server_vram_free_mb, _sync_unload,
    get_models_base,
)
from roop.img_editor.nlp.semantic_analyzer import camera_pose_config, score_camera_pose


UNET = "qwen-image-edit-2511-Q2_K.gguf"
ANGLE_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CLIP = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ = "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
VAE = "qwen_image_vae.safetensors"
# v5.1222: la LoRA de ángulos necesita ~1 MP para REENCUADRAR. Medido
# 2026-10-17 (diagnostics/out/pov_experiments_20261017.png): con la
# entrada directa a 768 px la salida RECONSTRUÍA la escena sin mover la
# cámara (runs reales 00:11 / 02:38 / 16:53: correlación 0.83 frente a
# 0.35 de un movimiento real) y con FluxKontextImageScale (~1 MP, igual
# que la workflow de referencia del autor — oficina 28/09 sí se movió)
# la misma foto/pose/semilla salió REENCUADRADA (correlación -0.08) en
# 197 s sobre esta 8 GB: sin coste de tiempo ni de VRAM.
# El único tope de la SUBIDA acota decode/envío de fotos gigantes; el
# lienzo lo decide node9 y es constante (~1 MP) para cualquier entrada,
# así que el presupuesto de la ruta (stall 300 s / run 420 s) se mantiene.
UPLOAD_SIDE = 2048

# Presupuestos de la ruta de cámara, medidos en esta máquina (2026-10-16,
# con MUSICIA/ACE-Step compartiendo el equipo): la carga fría + prep tardó
# 202 s en llegar al KSampler (job interrumpido con los 14 nodos previos ya
# ejecutados) y una recarga limpia documenta 164 s hasta el primer paso.
# El stall viejo de 150 s mataba el job justo antes de muestrear y la foto
# salía sin cambiar el POV.
# - stall: tiempo máximo desde el encolado hasta el PRIMER paso de muestreo.
CAMERA_FIRST_STEP_STALL_S = 300
# - timeout: carga + 4 pasos Lightning + decode/save (recorrido frío
#   histórico completo: 129,8 s en 2026-10-01; con el equipo cargado, ~350 s).
CAMERA_RUN_TIMEOUT_S = 420
# - tope de la ruta entera (ambos intentos); solo se reintenta si queda
#   presupuesto para un ciclo razonable.
CAMERA_ROUTE_BUDGET_S = 720
CAMERA_RETRY_MIN_S = 300


def camera_pose(raw_prompt: str, translated_prompt: str = ""):
    """Pick one of the angle LoRA's 96 trained poses by measured concepts.

    Each axis takes its highest-scoring trained phrase against the
    CAMERA_POSES concept anchors (same _score_texts engine as the rest of
    the editor; no word lists in code). The designed defaults only apply
    when a whole axis stays below the measurement signal."""
    scores = score_camera_pose(f"{raw_prompt} {translated_prompt}")
    signal = camera_pose_config()["signal_threshold"]

    def _best(axis):
        phrases = scores[axis]
        phrase = max(phrases, key=phrases.get)
        return phrase, phrases[phrase]

    azimuth, az_score = _best("azimuth")
    elevation, el_score = _best("elevation")
    distance, di_score = _best("distance")
    if az_score < signal:
        if el_score >= signal:
            # Un encuadre solo de altura mantiene el frente (previa).
            azimuth = "front view"
        else:
            # Genérico sin medida: movimiento visible (default diseñado).
            azimuth = "front-right quarter view"
    if el_score < signal:
        elevation = ("eye-level shot" if azimuth != "front-right quarter view"
                     else "elevated shot")
    if di_score < signal:
        distance = "medium shot"
    return f"<sks> {azimuth} {elevation} {distance}"


def remaining_edit(raw_prompt: str, translated_prompt: str):
    """Remove only independent camera clauses; retain any mixed clause intact."""
    source = translated_prompt or raw_prompt
    clauses = re.split(r"\s*(?:[,;]|\band\b|\by\b)\s*", source, flags=re.I)
    camera_terms = re.compile(
        r"\b(?:pov|camera|angle|viewpoint|perspective|view|angulo|camara|vista|cenital|picado)\b",
        re.I,
    )
    other_actions = re.compile(
        r"\b(?:clothes|clothing|wear|dress|shirt|wall|color|hair|"
        r"body|anatomy|pose|sit|sitting|stand|standing|kneel|kneeling|"
        r"face|expression|hand|arm|leg|foot|feet|"
        r"remove|add|replace|change.*(?:clothes|wall|hair|person)|ropa|pared|pelo|"
        r"quitar|anadir|cambiar.*(?:ropa|pared|pelo|persona))\b", re.I,
    )
    kept = [c.strip() for c in clauses if c.strip() and
            (not camera_terms.search(c) or other_actions.search(c))]
    return ", ".join(kept)


class QwenCameraComfyClient:
    def available(self):
        base = get_models_base()
        files = (("diffusion_models", UNET), ("loras", ANGLE_LORA),
                 ("loras", LIGHTNING_LORA), ("text_encoders", CLIP),
                 ("text_encoders", MMPROJ), ("vae", VAE))
        missing = [name for subdir, name in files
                   if not os.path.isfile(os.path.join(base, subdir, name))]
        if missing:
            return False, "Faltan modelos Qwen: " + ", ".join(missing)
        try:
            url = get_comfyui_url()
            for node in ("UnetLoaderGGUF", "CLIPLoaderGGUF"):
                response = requests.get(f"{url}/object_info/{node}", timeout=5)
                response.raise_for_status()
                if node not in response.json():
                    return False, f"ComfyUI no tiene {node}"
        except Exception as exc:
            return False, f"ComfyUI no disponible: {exc}"
        return True, "OK"

    @staticmethod
    def _workflow(image_name, prompt, seed):
        return {
            "1": {"class_type": "LoadImage", "inputs": {"image": image_name}},
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": UNET}},
            "3": {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": ["2", 0], "lora_name": ANGLE_LORA, "strength_model": 0.9}},
            "4": {"class_type": "LoraLoaderModelOnly", "inputs": {
                "model": ["3", 0], "lora_name": LIGHTNING_LORA, "strength_model": 1.0}},
            "5": {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": ["4", 0], "shift": 3.1}},
            "6": {"class_type": "CFGNorm", "inputs": {"model": ["5", 0], "strength": 1.0}},
            "7": {"class_type": "CLIPLoaderGGUF", "inputs": {
                "clip_name": CLIP, "type": "qwen_image"}},
            "8": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
            # v5.1222: el escalado ~1 MP de la workflow de referencia del
            # autor; sin él la LoRA reconstruía la escena en vez de
            # reencuadrarla (medido, ver UPLOAD_SIDE). VAEEncode e image1
            # salen de este nodo.
            "9": {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}},
            "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["9", 0], "vae": ["8", 0]}},
            "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                "clip": ["7", 0], "prompt": prompt, "vae": ["8", 0], "image1": ["9", 0]}},
            "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                "clip": ["7", 0], "prompt": "", "vae": ["8", 0], "image1": ["9", 0]}},
            "13": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
                "conditioning": ["11", 0], "reference_latents_method": "index_timestep_zero"}},
            "14": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {
                "conditioning": ["12", 0], "reference_latents_method": "index_timestep_zero"}},
            "15": {"class_type": "KSampler", "inputs": {
                "model": ["6", 0], "positive": ["13", 0], "negative": ["14", 0],
                "latent_image": ["10", 0], "seed": int(seed), "steps": 4, "cfg": 1.0,
                "sampler_name": "euler", "scheduler": "simple", "denoise": 1.0}},
            "16": {"class_type": "VAEDecode", "inputs": {"samples": ["15", 0], "vae": ["8", 0]}},
            "17": {"class_type": "SaveImage", "inputs": {
                "images": ["16", 0], "filename_prefix": "AutoAutoQwenCamera"}},
        }

    def generate(self, image: Image.Image, pose: str, seed=None,
                 progress_callback=None, cancel_check=None):
        ready, note = self.available()
        if not ready:
            return None, note
        if cancel_check and cancel_check():
            return None, "Cancelado"
        url = get_comfyui_url()
        started = time.time()
        last_message = "Qwen Camera no terminó"
        for attempt in (1, 2):
            if cancel_check and cancel_check():
                return None, "Cancelado"
            remaining = CAMERA_ROUTE_BUDGET_S - (time.time() - started)
            if attempt == 2 and remaining < CAMERA_RETRY_MIN_S:
                # El reintento solo vale si cabe un ciclo razonable; sin
                # presupuesto la cola se libera sola y no se descarga nada.
                print(
                    f"[QwenCamera] Sin presupuesto para el reintento "
                    f"(quedan {remaining:.0f}s de {CAMERA_ROUTE_BUDGET_S}s)",
                    flush=True,
                )
                return None, last_message
            free_mb, _ = _server_vram_free_mb(url)
            if attempt == 2:
                if not _queue_is_idle(url):
                    return None, last_message
                print("[QwenCamera] Carga sin paso 1; se descarga y se reintenta una vez", flush=True)
                _sync_unload(url, "reintento Qwen Camera tras carga sin muestreo")
            elif free_mb is not None and free_mb < 5600 and _queue_is_idle(url):
                _sync_unload(url, f"cambio a Qwen Camera ({free_mb:.0f}MB libres)")
            timeout = min(CAMERA_RUN_TIMEOUT_S, remaining)
            stall = min(CAMERA_FIRST_STEP_STALL_S, timeout - 60)
            result, message = self._generate_once(
                url, image, pose, seed, progress_callback, cancel_check, started,
                stall_timeout=stall, timeout=timeout)
            if result is not None:
                return result, message
            last_message = message
            if attempt == 1 and str(message).startswith("Sin muestreo"):
                continue
            return None, message
        return None, last_message

    def _generate_once(self, url, image, pose, seed, progress_callback, cancel_check, started,
                       stall_timeout=CAMERA_FIRST_STEP_STALL_S, timeout=CAMERA_RUN_TIMEOUT_S):
        try:
            buffer = io.BytesIO()
            camera_input = image.convert("RGB")
            # v5.1222: el lienzo lo decide node9 (FluxKontextImageScale,
            # ~1 MP — medido en UPLOAD_SIDE). Aquí solo se acota el envío
            # de fotos gigantes; una foto que ya cabe viaja intacta.
            scale = min(1.0, UPLOAD_SIDE / max(camera_input.size))
            if scale < 1.0:
                width = max(16, round(camera_input.width * scale / 16) * 16)
                height = max(16, round(camera_input.height * scale / 16) * 16)
                camera_input = camera_input.resize((width, height), Image.Resampling.LANCZOS)
            width, height = camera_input.size
            camera_input.save(buffer, format="PNG")
            buffer.seek(0)
            uploaded = requests.post(f"{url}/upload/image", files={
                "image": ("autoauto_camera.png", buffer, "image/png")}, timeout=60)
            uploaded.raise_for_status()
            image_name = uploaded.json()["name"]
            workflow = self._workflow(image_name, pose, seed if seed is not None else int(time.time()))
            queued = requests.post(f"{url}/prompt", json={"prompt": workflow}, timeout=30)
            queued.raise_for_status()
            payload = queued.json()
            if "prompt_id" not in payload:
                return None, f"ComfyUI rechazó el workflow: {payload}"
            print(f"[QwenCamera] {pose} {width}x{height}", flush=True)
            meta, message = wait_for_comfy_image(
                url, payload["prompt_id"], timeout=timeout, steps_hint=4,
                progress_callback=progress_callback, cancel_check=cancel_check,
                stall_timeout=stall_timeout)
            if meta is None:
                return None, message
            response = requests.get(f"{url}/view", params={
                "filename": meta["filename"], "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output")}, timeout=90)
            response.raise_for_status()
            output = Image.open(io.BytesIO(response.content)).convert("RGB")
            return GenResult(output, time.time() - started), (
                f"OK ({time.time() - started:.0f}s, {camera_input.width}x{camera_input.height})")
        except Exception as exc:
            return None, f"Qwen Camera falló: {exc}"
