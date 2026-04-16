#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Z-Image Turbo ComfyUI Client
GGUF version optimized for consumer GPU (~6GB VRAM with Q4)
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

class ZImageEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}
        self._zimage_version = "q4"

    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False

    def get_model_paths(self, version: str = "q4") -> dict:
        """Rutas para Z-Image Turbo GGUF"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        # Modelos disponibles: Q4_K_M (~5GB), Q4_K_S, Q5, Q6, Q8
        version_map = {
            "q3": "z_image_turbo-Q3_K_S.gguf",
            "q4": "z_image_turbo-Q4_K_M.gguf",  # Recomendado para 8GB
            "q5": "z_image_turbo-Q5_K_M.gguf",
            "q6": "z_image_turbo-Q6_K.gguf",
            "q8": "z_image_turbo-Q8_0.gguf"
        }
        
        return {
            "zimage_unet": os.path.join(base, "unet", version_map.get(version, "z_image_turbo-Q4_K_M.gguf")),
            "zimage_clip": os.path.join(base, "text_encoders", "qwen_3_4b.safetensors"),
            "zimage_vae": os.path.join(base, "vae", "ae.safetensors"),
            "zimage_version": version
        }
    
    def check_models(self, version: str = "q4") -> Tuple[bool, str]:
        """Verifica que los modelos existan"""
        paths = self.get_model_paths(version)
        missing = []
        
        for name, path in paths.items():
            if name != "zimage_version" and not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            return False, f"Modelos faltantes:\n" + "\n".join(missing)
        
        self._model_paths = paths
        self._zimage_version = version
        return True, "Modelos OK"

    def load(self, progress_callback=None, zimage_version: str = "q4") -> Tuple[bool, str]:
        """Carga los modelos y verifica disponibilidad"""
        if self._loaded and getattr(self, '_zimage_version', None) == zimage_version:
            return True, f"Z-Image Turbo ({zimage_version}) listo"
        
        if self._loaded and getattr(self, '_zimage_version', None) != zimage_version:
            self._loaded = False
            self._model_paths = {}
        
        if not self.is_available():
            return False, "ComfyUI no disponible en 127.0.0.1:8188"
        
        ok, msg = self.check_models(zimage_version)
        if not ok:
            return False, msg
        
        self._zimage_version = zimage_version
        self._loaded = True
        return True, f"Z-Image Turbo ({zimage_version}) listo"

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 20,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.75,
        width: int = 1024,
        height: int = 1024,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        t0 = time.time()
        version = getattr(self, '_zimage_version', 'q4')
        
        # Optimizado para velocidad
        if num_inference_steps > 25:
            num_inference_steps = 25
        
        print(f"[ZImageEdit] Generando ({num_inference_steps} pasos, denoise={denoise})...", flush=True)
        print(f"[ZImageEdit] ✅ Modelo GGUF Z-Image Turbo ({version}) (~5-6GB)", flush=True)
        print(f"[ZImageEdit] ⏱️ Tiempo estimado: ~{num_inference_steps * 0.5:.0f} min", flush=True)
        
        try:
            # 1. Redimensionar imagen
            w, h = image.size
            new_w = (w // 64) * 64
            new_h = (h // 64) * 64
            if new_w < 64: new_w = 64
            if new_h < 64: new_h = 64
            if new_w != w or new_h != h:
                print(f"[ZImageEdit] Redimensionando {w}x{h} → {new_w}x{new_h}", flush=True)
                image = image.resize((new_w, new_h), Image.LANCZOS)

            # 2. Upload imagen
            iname = f"zimage_input_{int(t0)}.png"
            buf = io.BytesIO()
            image.save(buf, "PNG")
            buf.seek(0)
            
            r = requests.post(f"{COMFY}/upload/image", 
                            files={"image": (iname, buf, "image/png")})
            if r.status_code != 200:
                raise Exception(f"Upload failed: {r.text}")
            
            # Nombres de archivos
            unet_name = os.path.basename(self._model_paths["zimage_unet"])
            clip_name = os.path.basename(self._model_paths.get("zimage_clip", "Qwen3-4B-UD-Q5_K_XL.gguf"))
            vae_name = os.path.basename(self._model_paths.get("zimage_vae", "ae.safetensors"))
            
            print(f"[ZImageEdit] Modelos: unet={unet_name}, clip={clip_name} (qwen3), vae={vae_name}", flush=True)

            # Workflow Z-Image para img2img
            workflow = {
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": iname, "upload": "image"}
                },
                "2": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": unet_name, "device": "default"}
                },
                "3": {
                    "class_type": "CLIPLoader",
                    "inputs": {"clip_name": clip_name, "type": "qwen_image"}
                },
                "4": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": vae_name}
                },
                "5": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}
                },
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": prompt, "clip": ["3", 0]}
                },
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": "low quality, blurry, bad anatomy, deformed, ugly, watermark", "clip": ["3", 0]}
                },
                "8": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["2", 0],
                        "positive": ["6", 0],
                        "negative": ["7", 0],
                        "latent_image": ["5", 0],
                        "seed": seed or int(t0) % 1000000,
                        "steps": num_inference_steps,
                        "cfg": guidance_scale,
                        "sampler_name": "euler_ancestral",  # Más rápido para img2img
                        "scheduler": "simple",  # Más rápido que "normal"
                        "denoise": denoise
                    }
                },
                "9": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["8", 0], "vae": ["4", 0]}
                },
                "10": {
                    "class_type": "SaveImage",
                    "inputs": {"images": ["9", 0], "filename_prefix": "zimage_edit"}
                }
            }

            # Ejecutar workflow
            r = requests.post(f"{COMFY}/prompt", json={"workflow": workflow})
            if r.status_code != 200:
                raise Exception(f"Workflow failed: {r.text}")
            
            prompt_id = r.json().get("prompt_id")
            if not prompt_id:
                raise Exception(f"No prompt_id: {r.text}")

            # Poll para resultado
            for _ in range(300):  # 5 min max
                time.sleep(1)
                r = requests.get(f"{COMFY}/history/{prompt_id}")
                if r.status_code == 200:
                    data = r.json()
                    if prompt_id in data:
                        outputs = data[prompt_id].get("outputs", {})
                        for node_id, node_out in outputs.items():
                            if "images" in node_out:
                                for im in node_out["images"]:
                                    vr = requests.get(
                                        f"{COMFY}/view?filename={im['filename']}&subfolder={im.get('subfolder', '')}"
                                    )
                                    if vr.status_code == 200:
                                        elapsed = time.time() - t0
                                        print(f"[ZImageEdit] OK en {elapsed:.1f}s", flush=True)
                                        return (
                                            GenResult(
                                                image=Image.open(io.BytesIO(vr.content)),
                                                time_taken=elapsed
                                            ),
                                            f"OK ({elapsed:.1f}s)"
                                        )
                        if data[prompt_id].get("status") == "failed":
                            error_msg = data[prompt_id].get("runtime_object", {}).get("prompt", {}).get("errors", [])
                            raise Exception(f"ComfyUI error: {error_msg}")

            raise Exception("Timeout (>5 min)")
            
        except Exception as e:
            print(f"[ZImageEdit] ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def unload(self):
        self._loaded = False

_client = None

def get_zimage_edit_comfy_client() -> ZImageEditComfyClient:
    global _client
    if _client is None:
        _client = ZImageEditComfyClient()
    return _client

def is_zimage_edit_available() -> bool:
    c = get_zimage_edit_comfy_client()
    return c.is_available()