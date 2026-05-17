#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, requests, io
import numpy as np
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

class FluxGenComfyClient:
    """Cliente para generación pura (txt2img) en ComfyUI usando Flux"""
    def __init__(self):
        self._loaded = False
        self._flux_version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf" # Usar el preferido del usuario

    def is_available(self):
        try:
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _clean_prompt(self, prompt: str) -> str:
        """Limpia y traduce el prompt de forma similar al editor"""
        translations = {
            "mujer": "woman",
            "hombre": "man",
            "chica": "girl",
            "chico": "boy",
            "paisaje": "landscape",
            "retrato": "portrait",
            "realista": "realistic",
            "foto": "photo",
            "playa": "beach",
            "ciudad": "city",
            "montaña": "mountain",
            "bosque": "forest",
            "noche": "night",
            "dia": "day",
            "vikingo": "viking",
            "guerrero": "warrior"
        }
        
        cleaned = (prompt or "").lower().strip()
        for es, en in translations.items():
            cleaned = cleaned.replace(es, en)
            
        # Limpiar partículas
        stop_words = [" a ", " la ", " las ", " el ", " los ", " de ", " un ", " una ", " con ", " en "]
        for word in stop_words:
            cleaned = cleaned.replace(word, " ")
            
        return " ".join(cleaned.split())

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = 8,
        guidance_scale: float = 3.5,
        seed: int = None,
        width: int = 512,
        height: int = 768,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:
        
        comfy_url = get_comfyui_url()
        if not self.is_available():
            return None, "ComfyUI no está disponible"

        t0 = time.time()
        
        # Limpiar y optimizar prompt
        clean_p = self._clean_prompt(prompt)
        final_prompt = f"{clean_p}, realistic, high quality, 8k, photorealistic"
        print(f"[GenFlux] Prompt optimizado: {final_prompt}")
        
        # Seleccionar modelo y clips
...
            "6": {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {
                    "clip": ["3", 0], 
                    "prompt": final_prompt, 
                    "vae": ["4", 0], 
                    "image1": None, "image2": None, "image3": None
                }
            },
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}},
            "11": {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}},
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["17", 0], "positive": ["11", 0], "negative": ["7", 0],
                    "latent_image": ["1", 0], "seed": seed or int(t0) % 1000000,
                    "steps": steps, "cfg": 1.0, # Turbo usa CFG 1.0
                    "sampler_name": "euler_ancestral", "scheduler": "simple", "denoise": 1.0
                }
            },
            "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}},
            "10": {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "GenFlux"}}
        }

        try:
            r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
            if r.status_code != 200:
                return None, f"Error: {r.text[:100]}"
            
            pid = r.json().get("prompt_id")
            while True:
                r = requests.get(f"{comfy_url}/history/{pid}")
                if r.status_code == 200 and pid in r.json():
                    hist = r.json()[pid]
                    if "outputs" in hist and "10" in hist["outputs"]:
                        img_data = hist["outputs"]["10"]["images"][0]
                        res = requests.get(f"{comfy_url}/view?filename={img_data['filename']}")
                        if res.status_code == 200:
                            pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
                            return GenResult(image=pil_img, time_taken=time.time()-t0), "OK"
                time.sleep(1)
                if time.time() - t0 > 600: return None, "Timeout (600s)"
        except Exception as e:
            return None, str(e)

_gen_client = None
def get_flux_gen_client() -> FluxGenComfyClient:
    global _gen_client
    if _gen_client is None:
        _gen_client = FluxGenComfyClient()
    return _gen_client
