"""Image-to-image camera changes with Qwen Image Edit 2511 and the angle LoRA."""

import io
import os
import re
import time
import unicodedata

import requests
from PIL import Image

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image
from roop.img_editor.flux_edit_comfy_client import (
    GenResult, _queue_is_idle, _server_vram_free_mb, _sync_unload,
    get_models_base,
)


UNET = "qwen-image-edit-2511-Q2_K.gguf"
ANGLE_LORA = "qwen-image-edit-2511-multiple-angles-lora.safetensors"
LIGHTNING_LORA = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
CLIP = "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf"
MMPROJ = "Qwen2.5-VL-7B-Instruct-mmproj-f16.gguf"
VAE = "qwen_image_vae.safetensors"
# 768 px on the long side finished a cold Qwen run on this 8 GB GPU in
# 129.8 s. 1152 px only finished after the weights were already resident.
CAMERA_SIDE = 768


def _fold(value):
    return "".join(c for c in unicodedata.normalize("NFD", value.lower())
                   if unicodedata.category(c) != "Mn")


def camera_pose(raw_prompt: str, translated_prompt: str = ""):
    """Map a requested camera position to one of the LoRA's 96 trained poses."""
    text = _fold(f"{raw_prompt} {translated_prompt}")
    back = bool(re.search(r"\b(back|behind|rear|espalda|detras|desde atras|opposite|opuesto)\b", text))
    left = bool(re.search(r"\b(left|izquierda|izquierdo)\b", text))
    right = bool(re.search(r"\b(right|derecha|derecho)\b", text))
    if back and left:
        azimuth = "back-left quarter view"
    elif back and right:
        azimuth = "back-right quarter view"
    elif back:
        azimuth = "back view"
    elif left:
        azimuth = "left side view" if re.search(
            r"\b(side|lateral|perfil|profile|desde la izquierda|from the left)\b", text
        ) else "front-left quarter view"
    elif right:
        azimuth = "right side view" if re.search(
            r"\b(side|lateral|perfil|profile|desde la derecha|from the right)\b", text
        ) else "front-right quarter view"
    elif re.search(r"\b(side|lateral|perfil|profile)\b", text):
        azimuth = "right side view"
    elif re.search(r"\b(front|frente|frontal)\b", text):
        azimuth = "front view"
    else:
        # Vertical camera requests keep the front azimuth. A generic "change
        # POV" needs a visible move, so use a quarter view instead.
        azimuth = "front view" if re.search(
            r"\b(overhead|top.down|bird.?s.eye|cenital|picado|high.angle|"
            r"elevated|elevado|desde arriba|desde lo alto|low.angle|"
            r"contrapicado|desde abajo)\b", text) else "front-right quarter view"

    if re.search(r"\b(overhead|top.down|bird.?s.eye|cenital|desde arriba|desde lo alto|picado|high.angle)\b", text):
        elevation = "high-angle shot"
    elif re.search(r"\b(elevated|elevado|desde arriba|desde lo alto)\b", text):
        elevation = "elevated shot"
    elif re.search(r"\b(low.angle|from below|contrapicado|desde abajo)\b", text):
        elevation = "low-angle shot"
    elif re.search(r"\b(eye.level|nivel de los ojos)\b", text):
        elevation = "eye-level shot"
    else:
        elevation = "elevated shot" if azimuth == "front-right quarter view" else "eye-level shot"

    if re.search(r"\b(close.?up|primer plano|cerca|cercano)\b", text):
        distance = "close-up"
    elif re.search(r"\b(wide|full.body|plano general|cuerpo entero|lejos|alejado)\b", text):
        distance = "wide shot"
    else:
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
            "10": {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["8", 0]}},
            "11": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                "clip": ["7", 0], "prompt": prompt, "vae": ["8", 0], "image1": ["1", 0]}},
            "12": {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {
                "clip": ["7", 0], "prompt": "", "vae": ["8", 0], "image1": ["1", 0]}},
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
            free_mb, _ = _server_vram_free_mb(url)
            if attempt == 2:
                if not _queue_is_idle(url):
                    return None, last_message
                _sync_unload(url, "reintento Qwen Camera tras carga sin muestreo")
            elif free_mb is not None and free_mb < 5600 and _queue_is_idle(url):
                _sync_unload(url, f"cambio a Qwen Camera ({free_mb:.0f}MB libres)")
            result, message = self._generate_once(
                url, image, pose, seed, progress_callback, cancel_check, started)
            if result is not None:
                return result, message
            last_message = message
            if attempt == 1 and str(message).startswith("Sin muestreo"):
                print("[QwenCamera] Carga sin paso 1; se descarga y se reintenta una vez", flush=True)
                continue
            return None, message
        return None, last_message

    def _generate_once(self, url, image, pose, seed, progress_callback, cancel_check, started):
        try:
            buffer = io.BytesIO()
            camera_input = image.convert("RGB")
            # The old FluxKontextImageScale expanded every input to ~1 MP,
            # including mobile photos uploaded at 800 px. That made Qwen
            # unpredictable on 8 GB and ignored the client's 768 px cap.
            # Use an explicit, modest render size even for smaller uploads.
            scale = CAMERA_SIDE / max(camera_input.size)
            width = max(16, round(camera_input.width * scale / 16) * 16)
            height = max(16, round(camera_input.height * scale / 16) * 16)
            camera_input = camera_input.resize((width, height), Image.Resampling.LANCZOS)
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
                url, payload["prompt_id"], timeout=240, steps_hint=4,
                progress_callback=progress_callback, cancel_check=cancel_check,
                stall_timeout=150)
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
