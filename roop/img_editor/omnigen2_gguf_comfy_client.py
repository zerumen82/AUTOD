#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniGen2 GGUF ComfyUI Client
Usa nodos estándar de ComfyUI + ComfyUI-GGUF
"""

import os, sys, json, time, requests, io
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

COMFY = "http://127.0.0.1:8188"

@dataclass
class GenResult:
    image: Image.Image
    time_taken: float = 0.0

class OmniGen2ComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}

    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False

    def get_model_paths(self) -> dict:
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        return {
            "model": "omnigen2_fp16.safetensors",
            "encoder": "qwen_2.5_vl_fp16.safetensors",
            "vae": "ae.safetensors",
        }

    def check_models(self) -> Tuple[bool, str]:
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        model_path = os.path.join(base, "diffusion_models", "omnigen2_fp16.safetensors")
        encoder_path = os.path.join(base, "text_encoders", "qwen_2.5_vl_fp16.safetensors")
        
        # VAE (buscar alternativas)
        vae_options = ["ae.safetensors", "vae-ft-mse-840000-ema-pruned.safetensors", "qwen_image_vae.safetensors"]
        vae_found = None
        for v in vae_options:
            vp = os.path.join(base, "vae", v)
            if os.path.exists(vp):
                vae_found = v
                break
        
        missing = []
        if not os.path.exists(model_path):
            missing.append("model: omnigen2_fp16.safetensors")
        if not os.path.exists(encoder_path):
            missing.append("encoder: qwen_2.5_vl_fp16.safetensors")
        
        if missing:
            return False, f"Modelos faltantes: {', '.join(missing)}"
        
        self._model_paths = self.get_model_paths()
        if vae_found:
            self._model_paths["vae"] = vae_found
        self._model_paths["encoder"] = "qwen_2.5_vl_fp16.safetensors"
        return True, f"Modelos OK (Encoder: fp16 safetensors, VAE: {self._model_paths['vae']})"

    def load(self, progress_callback=None) -> Tuple[bool, str]:
        if self._loaded:
            return True, "OmniGen2 GGUF listo"
        
        if not self.is_available():
            return False, "ComfyUI no disponible"
        
        ok, msg = self.check_models()
        if not ok:
            return False, msg
        
        self._loaded = True
        return True, "OmniGen2 GGUF listo"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 5.0,
        seed: int = None,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado"

        t0 = time.time()

        if seed is None:
            seed = int(time.time() * 1000) % 1000000000

        # Upload imagen
        img_buffer = io.BytesIO()
        image.save(img_buffer, format='PNG')
        img_bytes = img_buffer.getvalue()
        
        files = {'image': ('input.png', img_bytes, 'image/png')}
        upload_resp = requests.post(
            f"{COMFY}/upload/image",
            files=files,
            data={"subfolder": "", "type": "input"}
        )
        
        if upload_resp.status_code != 200:
            return None, f"Error upload: {upload_resp.status_code}"
        
        input_image = upload_resp.json()["name"]

        # Workflow usando nodos estándar de ComfyUI (no GGUF):
        # - CheckpointLoaderStandard (para safetensors fp16)
        # - VAELoader (estándar)
        # - KSampler (estándar)
        
        workflow = {
            # 1. Load checkpoint (model + clip)
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {
                    "ckpt_name": "omnigen2_fp16.safetensors"
                }
            },
            # 2. Load VAE
            "2": {
                "class_type": "VAELoader",
                "inputs": {
                    "vae_name": self._model_paths["vae"]
                }
            },
            # 3. Load input image
            "3": {
                "class_type": "LoadImage",
                "inputs": {
                    "image": input_image,
                    "choose file to upload": "image"
                }
            },
            # 4. Encode image to latent (img2img)
            "4": {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["3", 0],
                    "vae": ["2", 0]
                }
            },
            # 5. Positive prompt (CLIP del checkpoint output 1)
            "5": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": prompt,
                    "clip": ["1", 1]
                }
            },
            # 6. Negative prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {
                    "text": "low quality, worst quality, blurry, deformed, bad anatomy, ugly, cartoon, 3d render",
                    "clip": ["1", 1]
                }
            },
            # 7. KSampler (model del checkpoint output 0)
            "7": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["4", 0],
                    "seed": seed,
                    "steps": num_inference_steps,
                    "cfg": guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.75
                }
            },
            # 8. VAE Decode
            "8": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["7", 0],
                    "vae": ["2", 0]
                }
            },
            # 9. Save
            "9": {
                "class_type": "SaveImage",
                "inputs": {
                    "images": ["8", 0],
                    "filename_prefix": "omnigen2"
                }
            }
        }

        print(f"[OmniGen2] Enviando workflow...", flush=True)
        
        queue_resp = requests.post(
            f"{COMFY}/prompt",
            json={"prompt": workflow}
        )
        
        print(f"[OmniGen2] Response: {queue_resp.status_code}", flush=True)
        
        if queue_resp.status_code != 200:
            return None, f"Error: {queue_resp.status_code} - {queue_resp.text[:200]}"
        
        result = queue_resp.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return None, f"No prompt_id: {result}"
        
        print(f"[OmniGen2] Prompt ID: {prompt_id}", flush=True)
        
        # Wait for completion
        max_wait = 300
        while time.time() - t0 < max_wait:
            time.sleep(0.5)
            hist_resp = requests.get(f"{COMFY}/history/{prompt_id}")
            
            if hist_resp.status_code == 200:
                hist = hist_resp.json()
                if prompt_id in hist:
                    status = hist[prompt_id].get("status", {})
                    
                    if status.get("completed"):
                        output_node = hist[prompt_id].get("outputs", {}).get("9", {})
                        images = output_node.get("images", [])
                        
                        if images:
                            img_resp = requests.get(
                                f"{COMFY}/view",
                                params={"filename": images[0]["filename"], "type": "output"}
                            )
                            
                            result_img = Image.open(io.BytesIO(img_resp.content))
                            elapsed = time.time() - t0
                            
                            return GenResult(image=result_img, time_taken=elapsed), f"OK ({elapsed:.1f}s)"
                    
                    if status.get("error"):
                        return None, f"Error: {status['error']}"
        
        return None, "Timeout (5 min)"

    def unload(self):
        self._loaded = False


_omnigen2_client = None

def get_omnigen2_comfy_client():
    global _omnigen2_client
    if _omnigen2_client is None:
        _omnigen2_client = OmniGen2ComfyClient()
    return _omnigen2_client