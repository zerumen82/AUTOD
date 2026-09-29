#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boogu Image Edit ComfyUI Client
Asume modelo GGUF Boogu Image Edit con nodos GGUF estándar.
"""

import os, sys, json, time, requests, io
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_base():
    return os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models")

@dataclass
class GenResult:
    image: Image.Image
    time_taken: float = 0.0

class BooguEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._boogu_version = None
        self._clip_name = None
        self._clip_type = None
        self._vae_name = None
        self._is_gguf_clip = False
        self._dual_clip = False
        self._clip_name2 = None

    def is_available(self):
        try:
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _model_exists(self, subdir: str, name: str) -> bool:
        path = os.path.join(get_models_base(), subdir, name)
        exists = os.path.exists(path)
        if not exists:
            # GGUF models can be in unet/ instead of diffusion_models/
            if subdir == "diffusion_models" and name.endswith(".gguf"):
                alt_path = os.path.join(get_models_base(), "unet", name)
                if os.path.exists(alt_path):
                    return True
        return exists

    def load(self, *args, **kwargs) -> Tuple[bool, str]:
        # Determine version from args or kwargs
        if args:
            version = args[0]
        else:
            version = kwargs.get('flux_version') or kwargs.get('boogu_version') or "boogu-image-edit-Q2_K.gguf"
        comfy_url = get_comfyui_url()
        if not self.is_available():
            return False, f"ComfyUI no responde en {comfy_url}. ¿Está iniciado?"

        # Verify GGUF nodes
        try:
            _oi = requests.get(f"{comfy_url}/object_info/UnetLoaderGGUF", timeout=5)
            if _oi.status_code != 200 or "UnetLoaderGGUF" not in _oi.json():
                raise ValueError("sin UnetLoaderGGUF")
        except Exception:
            return False, (
                f"El ComfyUI de {comfy_url} NO tiene nodos GGUF (falta UnetLoaderGGUF). "
                "Seguramente es otra instalación (p.ej. puerto 8189). Cerrá ese ComfyUI "
                "y reiniciá el servidor para que use el propio (puerto 8188)."
            )

        # Avoid re-checking if already loaded same version
        if self._loaded and getattr(self, '_boogu_version', None) == version:
            return True, f"Boogu {version} ya cargado (reutilizado)"

        # Determine clip and vae based on model name (heuristic)
        # For now assume Boogi uses same as flux2-klein-4b: qwen3-4b-abl-q4_0.gguf clip and flux2_vae
        # If model name contains "qwen" maybe use qwen clip.
        clip_name = "qwen3-4b-abl-q4_0.gguf"
        clip_type = "flux"
        vae_name = "flux2_vae.safetensors"
        self._dual_clip = False
        self._is_gguf_clip = True
        self._clip_name2 = None

        # Override if known patterns
        if "t5" in version.lower():
            # dual clip with t5
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
            vae_name = "ae.safetensors"
        elif "longcat" in version.lower():
            clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            clip_type = "longcat_image"
            vae_name = "ae.safetensors"
        elif "qwen" in version.lower() and "image" in version.lower():
            clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            clip_type = "qwen_image"
            vae_name = "Qwen_Image-VAE.safetensors"

        checks = [
            ("diffusion_models", version, f"Modelo Boogu {version}"),
            ("text_encoders", clip_name, f"CLIP {clip_name}"),
        ]
        if self._dual_clip:
            checks.append(("text_encoders", self._clip_name2, f"CLIP2 {self._clip_name2}"))
        checks.append(("vae", vae_name, f"VAE {vae_name}"))

        missing = []
        for subdir, fname, label in checks:
            if not self._model_exists(subdir, fname):
                missing.append(f"  - {label}: ui/tob/ComfyUI/models/{subdir}/{fname}")

        if missing:
            return False, "Modelos faltantes:\n" + "\n".join(missing)

        self._boogu_version = version
        self._clip_name = clip_name
        self._clip_type = clip_type
        self._vae_name = vae_name
        self._loaded = True
        return True, f"Boogu {version} listo"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.75,
        mask_image: Optional[Image.Image] = None,
        lora_name: Optional[str] = None,
        lora_strength: float = 1.0,
        progress_callback=None,
        reference_latents_method: str = "offset",
        grow_mask_by: int = 4,
        no_kontext: bool = False,
        drop_edit_image_ref: bool = False,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        drop_edit_image_ref = bool(drop_edit_image_ref or kwargs.get("drop_edit_image_ref", False))
        no_kontext = bool(no_kontext or kwargs.get("no_kontext", False))
        ref_method = reference_latents_method or "offset"
        if no_kontext:
            ref_method = "none"

        # Prompt negativo base
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, low resolution"

        # Determine if it's a LongCat/Kontext/Qwen/Klein type for size limits
        is_longcat = "longcat" in self._boogu_version.lower()
        is_kontext = "kontext" in self._boogu_version.lower()
        is_qwen = "qwen" in self._boogu_version.lower() and "image" in self._boogu_version.lower()
        is_klein = "klein" in self._boogu_version.lower()

        # Image preprocessing
        t0 = time.time()
        image = image.convert("RGB")
        w, h = image.size
        new_w, new_h = (w // 64) * 64, (h // 64) * 64
        if new_w < 64: new_w = 64
        if new_h < 64: new_h = 64
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)

        # Max dimension based on model type
        max_dim = 1024 if (is_longcat or is_kontext) else (640 if is_qwen else 768)
        if new_w > max_dim or new_h > max_dim:
            image.thumbnail((max_dim, max_dim), Image.LANCZOS)
            new_w, new_h = image.size
            new_w, new_h = (new_w // 64) * 64, (new_h // 64) * 64
            image = image.resize((new_w, new_h), Image.LANCZOS)

        iname = f"boogu_input_{int(t0)}.png"
        buf = io.BytesIO(); image.save(buf, "PNG"); buf.seek(0)
        comfy_url = get_comfyui_url()
        r = requests.post(f"{comfy_url}/upload/image", files={"image": (iname, buf, "image/png")})
        if r.status_code != 200:
            return None, f"Error subiendo imagen: {r.status_code}"

        mname = None
        if mask_image:
            mname = f"mask_{int(t0)}.png"
            mbuf = io.BytesIO(); mask_image.convert("L").resize((new_w, new_h)).save(mbuf, "PNG"); mbuf.seek(0)
            requests.post(f"{comfy_url}/upload/image", files={"image": (mname, mbuf, "image/png")})

        # Build workflow similar to FluxEdit but without complex branching; assume standard img2img with reference latents
        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": iname, "upload": "image"}},
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": self._boogu_version, "device": "default"}},
        }

        # Clip loader
        if self._dual_clip:
            wf["3"] = {
                "class_type": "DualCLIPLoaderGGUF",
                "inputs": {"clip_name1": self._clip_name, "clip_name2": self._clip_name2, "type": self._clip_type}
            }
            clip_output = ["3", 0]
        elif self._is_gguf_clip:
            wf["3"] = {
                "class_type": "CLIPLoaderGGUF",
                "inputs": {"clip_name": self._clip_name, "type": self._clip_type, "device": "default"}
            }
            clip_output = ["3", 0]
        else:
            wf["3"] = {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": self._clip_name, "type": self._clip_type, "device": "default"}
            }
            clip_output = ["3", 0]

        # VAE
        wf["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}}

        # LoRAs
        last_model = ["2", 0]
        last_clip = clip_output
        if lora_name and lora_name != "None":
            wf["20"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": last_model,
                    "clip": last_clip,
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength
                }
            }
            last_model = ["20", 0]
            last_clip = ["20", 1]

        # Text encode
        wf["5"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": last_clip}}
        wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
        positive_input = ["5", 0]
        negative_input = ["6", 0]

        # Reference latents method (for img2img)
        # We'll use FluxKontextMultiReferenceLatentMethod if not no_kontext and not drop_edit_image_ref
        # but for simplicity we just pass latent from VAEEncode
        wf["7"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}}
        wf["8"] = {"class_type": "KSampler", "inputs": {
            "model": last_model,
            "positive": positive_input,
            "negative": negative_input,
            "latent_image": ["7", 0],
            "seed": seed or int(t0) % 1000000,
            "steps": num_inference_steps,
            "cfg": guidance_scale,
            "sampler_name": "euler_ancestral",
            "scheduler": "simple",
            "denoise": denoise
        }}
        wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}}
        wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "BooguEdit"}}

        # Queue
        r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:200])
            return None, f"ComfyUI rechazó el prompt: {err}"

        pid = r.json().get("prompt_id")
        if not pid:
            return None, "ComfyUI no devolvió prompt_id"

        def _on_progress(prog):
            if progress_callback:
                progress_callback(prog)
            elif int(prog.get("elapsed") or 0) % 15 == 0 and prog.get("phase"):
                print(f"[BooguEdit] {prog.get('phase')} ({int(prog.get('elapsed') or 0)}s)", flush=True)

        img_meta, wait_msg = wait_for_comfy_image(
            comfy_url,
            pid,
            timeout=3600,
            steps_hint=num_inference_steps,
            progress_callback=_on_progress,
            cancel_check=kwargs.get("cancel_check"),
        )
        if img_meta is None:
            return None, wait_msg

        res = requests.get(f"{comfy_url}/view?filename={img_meta['filename']}", timeout=30)
        if res.status_code != 200:
            return None, f"Error descargando imagen: {res.status_code}"

        elapsed = time.time() - t0
        pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
        return GenResult(image=pil_img, time_taken=elapsed), f"OK ({elapsed:.0f}s)"

    def unload(self):
        self._loaded = False

_client = None

def get_boogu_edit_comfy_client() -> BooguEditComfyClient:
    global _client
    if _client is None:
        _client = BooguEditComfyClient()
    return _client

def is_boogu_edit_available() -> bool:
    c = get_boogu_edit_comfy_client()
    return c.is_available()