#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OmniGen2 ComfyUI Client - fp16 con Qwen externo + preload
"""

import os, sys, json, time, requests, io
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass
import threading

COMFY = "http://127.0.0.1:8188"

@dataclass
class GenResult:
    image: Image.Image
    time_taken: float = 0.0

class OmniGen2ComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}
        self._preload_done = False

    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False

    def get_model_paths(self) -> dict:
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        return {
            "model": "omnigen2_fp16.safetensors",
            "qwen": "qwen_2.5_vl_fp16.safetensors",
            "vae": "ae.safetensors",
        }

    def check_models(self) -> Tuple[bool, str]:
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        model_path = os.path.join(base, "diffusion_models", "omnigen2_fp16.safetensors")
        qwen_path = os.path.join(base, "text_encoders", "qwen_2.5_vl_fp16.safetensors")
        
        if not os.path.exists(model_path):
            return False, "Falta: omnigen2_fp16.safetensors"
        if not os.path.exists(qwen_path):
            return False, "Falta: qwen_2.5_vl_fp16.safetensors"
        
        vae_options = ["ae.safetensors", "vae-ft-mse-840000-ema-pruned.safetensors"]
        vae_found = "ae.safetensors"
        for v in vae_options:
            vp = os.path.join(base, "vae", v)
            if os.path.exists(vp):
                vae_found = v
                break
        
        self._model_paths = self.get_model_paths()
        self._model_paths["vae"] = vae_found
        return True, f"Modelos OK"

    def _do_preload(self):
        """Pre-carga el modelo en background cuando carga ComfyUI"""
        if not self.is_available() or self._preload_done:
            return
        
        print("[OmniGen2] Pre-cargando modelo en background...", flush=True)
        
        # Workflow mínimo válido para forzar la carga de todos los modelos
        preload_workflow = {
            "1": {
                "class_type": "UNETLoader", 
                "inputs": {
                    "unet_name": self._model_paths["model"],
                    "weight_dtype": "default"
                }
            },
            "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": self._model_paths["qwen"], "type": "omnigen2"}},
            "3": {"class_type": "VAELoader", "inputs": {"vae_name": self._model_paths["vae"]}},
            "4": {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 1}},
            "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "p", "clip": ["2", 0]}},
            "6": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["5", 0],
                    "negative": ["5", 0],
                    "latent_image": ["4", 0],
                    "seed": 1,
                    "steps": 1,
                    "cfg": 1.0,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": 0.0
                }
            },
            "7": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["3", 0]}},
            "8": {"class_type": "PreviewImage", "inputs": {"images": ["7", 0]}}
        }
        
        try:
            resp = requests.post(f"{COMFY}/prompt", json={"prompt": preload_workflow}, timeout=30)
            if resp.status_code == 200:
                print("[OmniGen2] [OK] Pre-carga iniciada (UNET+CLIP+VAE)", flush=True)
                self._preload_done = True
            else:
                print(f"[OmniGen2] Pre-carga rechazada: {resp.text}", flush=True)
        except Exception as e:
            print(f"[OmniGen2] Pre-carga error: {e}", flush=True)

    def load(self, progress_callback=None) -> Tuple[bool, str]:
        if self._loaded:
            return True, "OmniGen2 listo"
        
        if not self.is_available():
            return False, "ComfyUI no disponible"
        
        ok, msg = self.check_models()
        if not ok:
            return False, msg
        
        # Iniciar pre-carga en background
        threading.Thread(target=self._do_preload, daemon=True).start()
        
        self._loaded = True
        return True, "OmniGen2 listo (preload en background)"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 12,
        guidance_scale: float = 4.0,
        seed: int = None,
        width: int = 1024,
        height: int = 1024,
        denoise: float = 0.5,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado"

        t0 = time.time()

        if seed is None:
            seed = int(time.time() * 1000) % 1000000000

        actual_steps = min(num_inference_steps, 12)
        
        # Reducir resolución para evitar OOM en 8GB
        w, h = image.size
        max_size = 512
        if w > max_size or h > max_size:
            scale = max_size / max(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            new_w = (new_w // 16) * 16
            new_h = (new_h // 16) * 16
            image = image.resize((new_w, new_h), Image.LANCZOS)
            print(f"[OmniGen2] Redimensionando a {new_w}x{new_h} para 8GB")
        
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

        # Workflow: modelo fp16 + Qwen fp16 como CLIP
        workflow = {
            # 1. Load UNET (OmniGen2 fp16)
            "1": {
                "class_type": "UNETLoader",
                "inputs": {
                    "unet_name": self._model_paths["model"],
                    "weight_dtype": "default"
                }
            },
            # 2. Load Qwen (CLIP externo)
            "2": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": self._model_paths["qwen"], "type": "omnigen2"}
            },
            # 3. Load VAE
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": self._model_paths["vae"]}
            },
            # 4. Load input image
            "4": {
                "class_type": "LoadImage",
                "inputs": {"image": input_image, "choose file to upload": "image"}
            },
            # 5. Encode to latent
            "5": {
                "class_type": "VAEEncode",
                "inputs": {"pixels": ["4", 0], "vae": ["3", 0]}
            },
            # 6. Positive prompt
            "6": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": ["2", 0]}
            },
            # 7. Negative prompt
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": "low quality, worst quality, blurry", "clip": ["2", 0]}
            },
            # 8. KSampler
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["1", 0],
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "seed": seed,
                    "steps": actual_steps,
                    "cfg": guidance_scale,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise
                }
            },
            # 9. VAE Decode
            "9": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["8", 0], "vae": ["3", 0]}
            },
            # 10. Save
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9", 0], "filename_prefix": "omnigen2"}
            }
        }

        print(f"[OmniGen2] Generando con {actual_steps} pasos...", flush=True)
        
        queue_resp = requests.post(f"{COMFY}/prompt", json={"prompt": workflow})

        print(f"[OmniGen2] Response: {queue_resp.status_code}", flush=True)
        
        if queue_resp.status_code != 200:
            return None, f"Error: {queue_resp.status_code}"
        
        result = queue_resp.json()
        prompt_id = result.get("prompt_id")
        if not prompt_id:
            return None, f"No prompt_id"
        
        print(f"[OmniGen2] Prompt ID: {prompt_id}", flush=True)
        
        max_wait = 3600  # Aumentado a 1 hora para generaciones lentas en 8GB
        last_node = None
        
        while time.time() - t0 < max_wait:
            time.sleep(2)
            try:
                # 1. Verificar progreso detallado
                prog_resp = requests.get(f"{COMFY}/prompt")
                if prog_resp.status_code == 200:
                    prog_data = prog_resp.json()
                    exec_info = prog_data.get("exec_info", {})
                    queue_remaining = exec_info.get("queue_remaining", 0)
                    
                    # Intentar obtener el nodo actual si nuestro prompt está ejecutándose
                    # (Nota: ComfyUI no siempre expone el nodo exacto por REST de forma simple sin WebSockets, 
                    # pero el historial nos dirá qué nodos ya terminaron)
                
                # 2. Verificar historial para ver qué ha terminado
                hist_resp = requests.get(f"{COMFY}/history/{prompt_id}")
                
                if hist_resp.status_code == 200:
                    hist = hist_resp.json()
                    if prompt_id in hist:
                        h = hist[prompt_id]
                        status = h.get("status", {})
                        
                        # Ver qué nodos han terminado
                        completed_nodes = list(h.get("outputs", {}).keys())
                        if "8" in completed_nodes and last_node != "8":
                            print("[OmniGen2]  Sampler completado. Iniciando decodificación VAE (esto puede tardar)...", flush=True)
                            last_node = "8"
                        elif "9" in completed_nodes and last_node != "9":
                            print("[OmniGen2] [IMAGE] Decodificación VAE completada. Guardando imagen...", flush=True)
                            last_node = "9"

                        if status.get("completed"):
                            output_node = h.get("outputs", {}).get("10", {})
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
            except Exception as e:
                print(f"[OmniGen2] Poll error: {e}", flush=True)
        
        return None, f"Timeout ({max_wait}s)"

    def unload(self):
        self._loaded = False


_omnigen2_client = None

def get_omnigen2_comfy_client():
    global _omnigen2_client
    if _omnigen2_client is None:
        _omnigen2_client = OmniGen2ComfyClient()
    return _omnigen2_client