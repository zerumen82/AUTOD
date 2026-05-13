#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, requests, io
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

from roop.comfy_workflows import get_comfyui_url

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_base():
    return os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models")

@dataclass
class GenResult:
    image: Image.Image
    time_taken: float = 0.0

class FluxEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._flux_version = None

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
            print(f"[FluxClient] FALTA MODELO: {path}")
        return exists

    def load(self, flux_version="flux2-klein-4b-Q4_K_S.gguf") -> Tuple[bool, str]:
        comfy_url = get_comfyui_url()
        if not self.is_available():
            return False, f"ComfyUI no responde en {comfy_url}. ¿Está iniciado?"


        clip_map = {
            "flux2-klein-4b-Q4_K_S.gguf": ("qwen_3_4b_fp4_flux2.safetensors", "flux2"),
            "flux-2-klein-base-4b-Q4_K_S.gguf": ("qwen3-4b-abl-q4_0.gguf", "flux2"),
            "flux1-schnell-Q4_K_S.gguf": ("t5-v1_1-xxl-encoder-Q4_K_S.gguf", "flux"),
            "flux1-dev-Q4_K.gguf": ("t5-v1_1-xxl-encoder-Q4_K_S.gguf", "flux"),
        }
        self._dual_clip = False
        self._clip_name2 = None

        if flux_version == "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf":
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
        elif flux_version not in clip_map:
            return False, f"Versión FLUX no soportada: {flux_version}"
        else:
            clip_name, clip_type = clip_map[flux_version]

        checks = [
            ("diffusion_models", flux_version, f"Modelo FLUX {flux_version}"),
            ("text_encoders", clip_name, f"CLIP {clip_name}"),
        ]
        if self._dual_clip:
            checks.append(("text_encoders", self._clip_name2, f"CLIP2 {self._clip_name2}"))
        vae_name = "flux2_vae.safetensors" if "flux2" in flux_version else "ae.safetensors"
        checks.append(("vae", vae_name, f"VAE {vae_name}"))

        missing = []
        for subdir, fname, label in checks:
            if not self._model_exists(subdir, fname):
                missing.append(f"  - {label}: ui/tob/ComfyUI/models/{subdir}/{fname}")

        if missing:
            return False, "Modelos faltantes:\n" + "\n".join(missing)

        self._flux_version = flux_version
        self._clip_name = clip_name
        self._clip_type = clip_type
        self._vae_name = vae_name
        self._loaded = True
        return True, f"FLUX {flux_version} listo"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.60,
        mask_image: Optional[Image.Image] = None,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        t0 = time.time()
        w, h = image.size
        new_w, new_h = (w // 64) * 64, (h // 64) * 64
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
        if new_w > 768 or new_h > 768:
            image.thumbnail((768, 768), Image.LANCZOS)
            new_w, new_h = image.size

        iname = f"fast_{int(t0)}.png"
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

        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": iname, "upload": "image"}},
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": self._flux_version, "device": "default"}},
            "3": {"class_type": "DualCLIPLoaderGGUF" if self._dual_clip else "CLIPLoader", "inputs": {"clip_name1": self._clip_name, "clip_name2": self._clip_name2, "type": self._clip_type} if self._dual_clip else {"clip_name": self._clip_name, "type": self._clip_type, "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "low quality, blurry, pixelated, low resolution, JPEG artifacts", "clip": ["3", 0]}},
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["2", 0], "positive": ["6", 0], "negative": ["7", 0],
                    "latent_image": ["5", 0], "seed": seed or int(t0) % 1000000,
                    "steps": num_inference_steps, "cfg": guidance_scale,
                    "sampler_name": "euler_ancestral", "scheduler": "normal", "denoise": denoise
                }
            },
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "FastEdit"}}
        }

        if mname:
            wf["12"] = {"class_type": "LoadImage", "inputs": {"image": mname, "upload": "image"}}
            wf["13"] = {"class_type": "ImageToMask", "inputs": {"image": ["12", 0], "channel": "red"}}
            wf["5"] = {"class_type": "VAEEncodeForInpaint", "inputs": {"pixels": ["1", 0], "vae": ["4", 0], "mask": ["13", 0], "grow_mask_by": 6}}
        else:
            wf["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}}

        r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:200])
            return None, f"ComfyUI rechazó el prompt: {err}"

        pid = r.json().get("prompt_id")
        if not pid:
            return None, "ComfyUI no devolvió prompt_id"

        last_progress = 0
        while True:
            elapsed = time.time() - t0
            r = requests.get(f"{comfy_url}/history/{pid}")
            if r.status_code == 200 and pid in r.json():
                hist = r.json()[pid]
                if "outputs" in hist and "10" in hist["outputs"] and hist["outputs"]["10"].get("images"):
                    img_data = hist["outputs"]["10"]["images"][0]
                    res = requests.get(f"{comfy_url}/view?filename={img_data['filename']}")
                    if res.status_code == 200:
                        elapsed = time.time() - t0
                        return GenResult(image=Image.open(io.BytesIO(res.content)), time_taken=elapsed), f"OK ({elapsed:.0f}s)"
                if hist.get("status") == "failed":
                    error_info = hist.get("error", "Error desconocido en ComfyUI")
                    return None, f"ComfyUI error: {error_info}"
                if "outputs" in hist and not hist["outputs"]:
                    if int(elapsed) > last_progress:
                        print(f"[FluxClient] ⏳ {int(elapsed)}s esperando...", flush=True)
                        last_progress = int(elapsed)


            if elapsed > 3600:
                return None, f"Timeout 60min - ComfyUI no completó la generación"
            time.sleep(1)

_client = None
def get_flux_edit_comfy_client() -> FluxEditComfyClient:
    global _client
    if _client is None:
        _client = FluxEditComfyClient()
    return _client