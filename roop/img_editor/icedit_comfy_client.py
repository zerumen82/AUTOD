#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ICEdit ComfyUI Client - Motor de edición de imágenes con ICEdit (Nunchaku)
Instrucciones: https://github.com/River-Zhang/ICEdit
VRAM: ~4-6GB (nunchaku)
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

class ICEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}
    
    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False
    
    def get_model_paths(self) -> dict:
        """Rutas de modelos para FLUX Fill Dev"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        return {
            # FLUX Fill Dev GGUF (gpustack)
            "flux_gguf": os.path.join(base, "diffusion_models", "flux1-fill-dev-q4_k_m.gguf"),
            
            # Text encoders (NO-GGUF, safetensors)
            "clip_l": os.path.join(base, "text_encoders", "clip_l.safetensors"),
            "t5": os.path.join(base, "text_encoders", "t5xxl_fp8.safetensors"),
            
            # ICEdit LoRA
            "lora": os.path.join(base, "loras", "ICEdit-normal-LoRA.safetensors"),
            
            # VAE (flux autoencoder)
            "vae": os.path.join(base, "vae", "ae.safetensors"),
        }
    
    def check_models(self) -> Tuple[bool, str]:
        """Verifica que los modelos existan"""
        paths = self.get_model_paths()
        missing = []
        
        for name, path in paths.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            return False, f"Modelos faltantes:\n" + "\n".join(missing)
        
        self._model_paths = paths
        return True, "Modelos OK"
    
    def load(self, progress_callback=None) -> Tuple[bool, str]:
        """Carga los modelos y verifica disponibilidad"""
        if self._loaded:
            return True, "ICEdit listo"
        
        # Verificar ComfyUI
        if not self.is_available():
            return False, "ComfyUI no disponible en 127.0.0.1:8188"
        
        # Verificar modelos
        ok, msg = self.check_models()
        if not ok:
            return False, msg
        
        self._loaded = True
        return True, "ICEdit listo (nunchaku, ~4-6GB)"
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        seed: int = None,
        lora_strength: float = 1.0,
        width: int = 512,
        height: int = 512,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:
        """Genera imagen editada con ICEdit"""
        
        if not self._loaded:
            return None, "No cargado - llama a load() primero"
        
        t0 = time.time()
        print(f"[ICEdit] Generando ({num_inference_steps} pasos)...", flush=True)
        
        try:
            # 1. Upload imagen
            iname = f"icedit_input_{int(t0)}.png"
            buf = io.BytesIO()
            image.save(buf, "PNG")
            buf.seek(0)
            
            r = requests.post(f"{COMFY}/upload/image", 
                            files={"image": (iname, buf, "image/png")})
            if r.status_code != 200:
                raise Exception(f"Upload failed: {r.text}")
            
            # 2. Construir workflow usando nodos oficiales ICEdit
            # El pre-prompt "A diptych with..." se agrega automáticamente en InContextEditInstruction
            
            # Nombres de archivos (solo nombre, no ruta completa)
            flux_name = os.path.basename(self._model_paths["flux_gguf"])
            clip_l_name = os.path.basename(self._model_paths["clip_l"])
            t5_name = os.path.basename(self._model_paths["t5"])
            lora_name = os.path.basename(self._model_paths["lora"])
            vae_name = os.path.basename(self._model_paths["vae"])
            
            print(f"[ICEdit] Modelos: flux={flux_name}, clip_l={clip_l_name}, t5={t5_name}, lora={lora_name}, vae={vae_name}", flush=True)
            
            # Workflow usando nodos oficiales de ICEdit-ComfyUI-official
            # Estos nodos automáticamente manejan el pre-prompt y diptych
            workflow = {
                # Load image
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": iname, "upload": "image"}
                },
                # DiptychCreate - crear imagen diptych (izquierda=derecha=original)
                "2": {
                    "class_type": "DiptychCreate",
                    "inputs": {"image_input": ["1", 0], "image": "None"}
                },
                # Load FLUX GGUF
                "3": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": flux_name}
                },
                # Load CLIPs (DualCLIPLoader)
                "4": {
                    "class_type": "DualCLIPLoader",
                    "inputs": {
                        "clip_name1": clip_l_name,
                        "clip_name2": t5_name,
                        "type": "flux"
                    }
                },
                # Load VAE
                "5": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": vae_name}
                },
                # In-Context Edit Instruction - pre-prompt automático
                "6": {
                    "class_type": "InContextEditInstruction",
                    "inputs": {
                        "editText": prompt,
                        "clip": ["4", 0]
                    }
                },
                # ICEdit Conditioning
                "7": {
                    "class_type": "ICEFConditioning",
                    "inputs": {
                        "In_context": ["6", 0],
                        "negative": ["9", 0],  # se填充
                        "vae": ["5", 0],
                        "diptych": ["2", 0],
                        "maskDiptych": ["2", 1]
                    }
                },
                # Load LoRA
                "8": {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": ["3", 0],
                        "lora_name": lora_name,
                        "strength_model": lora_strength
                    }
                },
                # Negative prompt
                "9": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {
                        "clip": ["4", 0],
                        "text": "low quality, blurry, bad anatomy, deformed, ugly, watermark, text, signature"
                    }
                },
                # KSampler
                "10": {
                    "class_type": "KSampler",
                    "inputs": {
                        "model": ["8", 0],
                        "positive": ["7", 0],
                        "negative": ["7", 1],
                        "latent_image": ["7", 2],
                        "seed": seed or int(t0) % 1000000,
                        "steps": num_inference_steps,
                        "cfg": guidance_scale,
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": 1.0
                    }
                },
                # VAE Decode
                "11": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["10", 0], "vae": ["5", 0]}
                },
                # Save image
                "12": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["11", 0],
                        "filename_prefix": "ICEdit_output"
                    }
                }
            }
            
            # 3. Queue workflow
            r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow})
            if r.status_code != 200:
                raise Exception(f"Queue failed: {r.text}")
            
            pid = r.json()["prompt_id"]
            print(f"[ICEdit] Prompt queued: {pid}", flush=True)
            
            # 4. Wait for completion
            max_wait = 600  # 10 minutos
            for i in range(max_wait):
                time.sleep(1)
                r = requests.get(f"{COMFY}/history/{pid}")
                
                if r.status_code == 200:
                    data = r.json()
                    if pid in data:
                        outputs = data[pid].get("outputs", {})
                        
                        # Buscar imagen en outputs
                        for node_id, node_out in outputs.items():
                            if "images" in node_out:
                                for im in node_out["images"]:
                                    vr = requests.get(
                                        f"{COMFY}/view?filename={im['filename']}&subfolder={im.get('subfolder', '')}"
                                    )
                                    if vr.status_code == 200:
                                        elapsed = time.time() - t0
                                        print(f"[ICEdit] OK en {elapsed:.1f}s", flush=True)
                                        return (
                                            GenResult(
                                                image=Image.open(io.BytesIO(vr.content)),
                                                time_taken=elapsed
                                            ),
                                            f"OK ({elapsed:.1f}s)"
                                        )
            
            raise Exception("Timeout (>10 min)")
            
        except Exception as e:
            print(f"[ICEdit] ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def unload(self):
        self._loaded = False

# Singleton
_client = None

def get_icedit_comfy_client() -> ICEditComfyClient:
    global _client
    if _client is None:
        _client = ICEditComfyClient()
    return _client

def is_icedit_available() -> bool:
    c = get_icedit_comfy_client()
    return c.is_available()