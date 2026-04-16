#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qwen Image Edit ComfyUI Client
Usa los nodos oficiales de ComfyUI-QwenImageEdit con VAE convertido
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

class QwenEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}

    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False

    def get_model_paths(self, qwen_version: str = "q3") -> dict:
        """Rutas para Qwen Image Edit GGUF (optimizado para GPU 8GB)"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"

        # Seleccionar modelo según versión
        if qwen_version == "q2":
            qwen_model = "Qwen_Image_Edit-Q2_K.gguf"  # ~7GB VRAM, más rápido
        else:
            qwen_model = "Qwen_Image_Edit-Q3_K_M.gguf"  # ~10GB VRAM, mejor calidad

        return {
            "qwen_image": os.path.join(base, "diffusion_models", qwen_model),
            "qwen_clip": os.path.join(base, "text_encoders", "qwen_2.5_vl_7b_fp8_scaled.safetensors"),
            "vae": os.path.join(base, "vae", "qwen_image_vae.safetensors"),
            "qwen_version": qwen_version
        }
    
    def check_models(self, qwen_version: str = "q3") -> Tuple[bool, str]:
        """Verifica que los modelos existan"""
        paths = self.get_model_paths(qwen_version)
        missing = []
        
        for name, path in paths.items():
            if name != "qwen_version" and not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            return False, f"Modelos faltantes:\n" + "\n".join(missing)
        
        self._model_paths = paths
        self._qwen_version = qwen_version
        return True, "Modelos OK"
    
    def load(self, progress_callback=None, qwen_version: str = "q3") -> Tuple[bool, str]:
        """Carga los modelos y verifica disponibilidad"""
        if self._loaded and getattr(self, '_qwen_version', None) == qwen_version:
            return True, f"Qwen Image Edit ({qwen_version}) listo"
        
        # Reset si cambia versión
        if self._loaded and getattr(self, '_qwen_version', None) != qwen_version:
            self._loaded = False
            self._model_paths = {}
        
        if not self.is_available():
            return False, "ComfyUI no disponible en 127.0.0.1:8188"
        
        ok, msg = self.check_models(qwen_version)
        if not ok:
            return False, msg
        
        self._qwen_version = qwen_version
        self._loaded = True
        return True, f"Qwen Image Edit ({qwen_version}) listo"
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.75,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        t0 = time.time()
        qwen_version = getattr(self, '_qwen_version', 'q3')
        
        # Optimizado para máxima velocidad
        # Q2 = ~8GB VRAM, más rápido
        # Q3 = ~10GB VRAM, mejor calidad
        if num_inference_steps > 8:
            num_inference_steps = 8  # Máximo para velocidad (euler_ancestral + simple = más rápido)
        
        version_label = "Q2_K (8GB, rápido)" if qwen_version == "q2" else "Q3_K_M (10GB, calidad)"
        print(f"[QwenEdit] Generando ({num_inference_steps} pasos, denoise={denoise})...", flush=True)
        print(f"[QwenEdit] ✅ Modelo GGUF {version_label} (autoregresivo)", flush=True)
        print(f"[QwenEdit] ⏱️ Tiempo estimado: ~{num_inference_steps * 1.5:.0f} min", flush=True)
        
        try:
            # 1. Redimensionar imagen a múltiplos de 64 (requerido por Qwen)
            w, h = image.size
            new_w = (w // 64) * 64
            new_h = (h // 64) * 64
            if new_w < 64: new_w = 64
            if new_h < 64: new_h = 64
            if new_w != w or new_h != h:
                print(f"[QwenEdit] Redimensionando {w}x{h} → {new_w}x{new_h} (múltiplo de 64)", flush=True)
                image = image.resize((new_w, new_h), Image.LANCZOS)

            # 2. Upload imagen
            iname = f"qwen_input_{int(t0)}.png"
            buf = io.BytesIO()
            image.save(buf, "PNG")
            buf.seek(0)
            
            r = requests.post(f"{COMFY}/upload/image", 
                            files={"image": (iname, buf, "image/png")})
            if r.status_code != 200:
                raise Exception(f"Upload failed: {r.text}")
            
            # Nombres de archivos
            qwen_name = os.path.basename(self._model_paths["qwen_image"])
            qwen_clip_name = os.path.basename(self._model_paths["qwen_clip"])
            vae_name = os.path.basename(self._model_paths["vae"])

            qwen_label = "Q2_K" if qwen_version == "q2" else "Q3_K_M"
            print(f"[QwenEdit] Modelos: qwen={qwen_name} (GGUF {qwen_label}), clip={qwen_clip_name}, vae={vae_name} (3D→2D)", flush=True)

            # Workflow usando Qwen Image con VAE 2D de FLUX
            workflow = {
                # Load image
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": iname, "upload": "image"}
                },
                # Load Qwen Image GGUF (UNET cuantizado)
                "2": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": qwen_name, "device": "default"}
                },
                # Load Qwen CLIP fp8 (text encoder)
                "3": {
                    "class_type": "CLIPLoader",
                    "inputs": {"clip_name": qwen_clip_name, "type": "qwen_image"}
                },
                # Load Qwen VAE (maneja 3D->2D conversion)
                "4": {
                    "class_type": "QwenVAELoader",
                    "inputs": {"vae_name": vae_name}
                },
                # Encode image - necesario para edición de imagen
                "5": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}
                },
                # Text encode (positive) con nodo de Qwen Image Edit - pasa la imagen de referencia
                "6": {
                    "class_type": "TextEncodeQwenImageEdit",
                    "inputs": {"clip": ["3", 0], "prompt": prompt, "vae": ["4", 0], "image": ["1", 0]}
                },
                # Negative prompt (sin imagen de referencia)
                "7": {
                    "class_type": "TextEncodeQwenImageEdit",
                    "inputs": {
                        "clip": ["3", 0],
                        "prompt": "low quality, blurry, bad anatomy, deformed, ugly",
                        "vae": ["4", 0],
                        "image": None
                    }
                },
                # KSampler
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
                        "denoise": denoise  # img2img: 0.75 permite edición manteniendo estructura
                    }
                },
                # VAE Decode (con --lowvram, ComfyUI usa tiled automático si hay OOM)
                "9": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["8", 0], "vae": ["4", 0]}
                },
                # Save image
                "10": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["9", 0],
                        "filename_prefix": "QwenEdit_output"
                    }
                }
            }
            
            # Queue workflow
            r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow})
            if r.status_code != 200:
                raise Exception(f"Queue failed: {r.text}")
            
            pid = r.json()["prompt_id"]
            print(f"[QwenEdit] Prompt queued: {pid}", flush=True)
            
            # Wait for completion
            # Optimizado: euler_ancestral + simple scheduler = más rápido
            max_wait = 1800  # 30 minutos (reducido de 45 por optimización)
            last_comfy_check = 0
            
            for i in range(max_wait):
                time.sleep(1)
                
                # Mostrar progreso cada 2 minutos
                if i > 0 and i % 120 == 0:
                    elapsed_min = i / 60
                    print(f"[QwenEdit] ⏳ Procesando... {elapsed_min:.1f} min transcurridos (lowvram mode)", flush=True)
                
                # Verificar que ComfyUI sigue vivo cada 10 segundos
                if i - last_comfy_check >= 10:
                    try:
                        health = requests.get(f"{COMFY}/system_stats", timeout=15)
                        if health.status_code != 200:
                            raise Exception("ComfyUI process murió durante el procesamiento")
                        last_comfy_check = i
                    except requests.exceptions.ConnectionError:
                        raise Exception("ComfyUI se cerró inesperadamente - posible error de VRAM")
                    except requests.exceptions.ReadTimeout:
                        pass
                
                try:
                    r = requests.get(f"{COMFY}/history/{pid}", timeout=30)
                except requests.exceptions.ConnectionError:
                    raise Exception("Conexión con ComfyUI perdida durante el procesamiento")
                except requests.exceptions.ReadTimeout:
                    continue
                
                if r.status_code == 200:
                    data = r.json()
                    if pid in data:
                        outputs = data[pid].get("outputs", {})
                        
                        # Verificar si hubo errores en el procesamiento
                        status = data[pid].get("status", {})
                        if status.get("status_str") == "error":
                            error_msg = status.get("messages", ["Error desconocido en ComfyUI"])
                            if isinstance(error_msg, list) and len(error_msg) > 0:
                                if isinstance(error_msg[0], list):
                                    error_msg = [str(m) for sublist in error_msg for m in sublist]
                                else:
                                    error_msg = [str(m) for m in error_msg]
                            raise Exception(f"ComfyUI error: {'; '.join(error_msg)}")

                        for node_id, node_out in outputs.items():
                            if "images" in node_out:
                                for im in node_out["images"]:
                                    vr = requests.get(
                                        f"{COMFY}/view?filename={im['filename']}&subfolder={im.get('subfolder', '')}"
                                    )
                                    if vr.status_code == 200:
                                        elapsed = time.time() - t0
                                        print(f"[QwenEdit] OK en {elapsed:.1f}s", flush=True)
                                        return (
                                            GenResult(
                                                image=Image.open(io.BytesIO(vr.content)),
                                                time_taken=elapsed
                                            ),
                                            f"OK ({elapsed:.1f}s)"
                                        )
            
            raise Exception("Timeout (>10 min)")
            
        except Exception as e:
            print(f"[QwenEdit] ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def unload(self):
        self._loaded = False

_client = None

def get_qwen_edit_comfy_client() -> QwenEditComfyClient:
    global _client
    if _client is None:
        _client = QwenEditComfyClient()
    return _client

def is_qwen_edit_available() -> bool:
    c = get_qwen_edit_comfy_client()
    return c.is_available()