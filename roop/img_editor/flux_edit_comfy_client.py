#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX Image Edit ComfyUI Client
Usa FLUX fp8 para edición de imágenes con instrucciones en lenguaje natural
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

class FluxEditComfyClient:
    def __init__(self):
        self._loaded = False
        self._model_paths = {}
    
    def is_available(self):
        try:
            return requests.get(f"{COMFY}/system_stats", timeout=3).status_code == 200
        except:
            return False
    
    def get_model_paths(self) -> dict:
        """Rutas para FLUX GGUF (optimizado para GPU 8GB)"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"

        return {
            # FLUX UNet GGUF Q4_K (~3.5GB VRAM)
            "flux": os.path.join(base, "diffusion_models", "flux1-dev-Q4_K.gguf"),

            # FLUX text encoders - usar versión GGUF para reducir VRAM
            "clip_l": os.path.join(base, "text_encoders", "clip_l.safetensors"),
            "t5xxl": os.path.join(base, "text_encoders", "t5-v1_1-xxl-encoder-Q8_0.gguf"),

            # VAE estándar
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
            return True, "FLUX Image Edit listo"
        
        if not self.is_available():
            return False, "ComfyUI no disponible en 127.0.0.1:8188"
        
        ok, msg = self.check_models()
        if not ok:
            return False, msg
        
        self._loaded = True
        return True, "FLUX Image Edit listo"
    
    def generate(
        self,
        image: Image.Image,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.75,  # Nuevo parámetro para controlar la intensidad de edición
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:
        
        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        # NOTA: FLUX Dev en RTX 3060 Ti (8GB) con lowvram:
        # - Cada paso tarda ~2m45s debido al CPU offloading secuencial
        # - 8 pasos = ~22 min, pero con buena calidad de edición
        # - Para pruebas rápidas, usar 4-6 pasos (~11-16 min)
        # - Para producción, considerar GPU con 12GB+ VRAM

        t0 = time.time()
        print(f"[FluxEdit] Generando ({num_inference_steps} pasos, denoise={denoise})...", flush=True)
        print(f"[FluxEdit] [OK] Modelos GGUF optimizados para GPU 8GB", flush=True)
            print(f"[FluxEdit] Tiempo estimado: ~{num_inference_steps * 2.75:.0f} min (modo lowvram)", flush=True)
        
        # Negative prompt por defecto
        negative_prompt = "low quality, worst quality, bad quality, jpeg artifacts, blurry, out of focus, poorly drawn, bad anatomy, deformed, disfigured, mutated, extra limbs"
        
        # FLUX Schnell usa 8 pasos, Dev usa 25
        # Para img2img con GPU de 8GB en lowvram, usar máximo 6 pasos (rápido) o 8 (calidad)
        # 6 pasos = ~16 min, 8 pasos = ~22 min
        if num_inference_steps > 8:
            num_inference_steps = 8  # Balance calidad/velocidad para lowvram
        
        try:
            # 1. Redimensionar imagen a múltiplos de 64 (requerido por VAE)
            w, h = image.size
            new_w = (w // 64) * 64
            new_h = (h // 64) * 64
            if new_w < 64: new_w = 64
            if new_h < 64: new_h = 64
            if new_w != w or new_h != h:
                print(f"[FluxEdit] Redimensionando {w}x{h} → {new_w}x{new_h} (múltiplo de 64)", flush=True)
                image = image.resize((new_w, new_h), Image.LANCZOS)

            # 2. Upload imagen
            iname = f"flux_input_{int(t0)}.png"
            buf = io.BytesIO()
            image.save(buf, "PNG")
            buf.seek(0)
            
            r = requests.post(f"{COMFY}/upload/image", 
                            files={"image": (iname, buf, "image/png")})
            if r.status_code != 200:
                raise Exception(f"Upload failed: {r.text}")
            
            # Nombres de archivos
            flux_name = os.path.basename(self._model_paths["flux"])
            clip_l_name = os.path.basename(self._model_paths["clip_l"])
            t5_name = os.path.basename(self._model_paths["t5xxl"])
            vae_name = os.path.basename(self._model_paths["vae"])
            
            print(f"[FluxEdit] Modelos: flux={flux_name}, clip_l={clip_l_name}, t5={t5_name}, vae={vae_name}", flush=True)
            
            # FLUX GGUF workflow - nodos directos de ComfyUI-GGUF
            workflow = {
                # 1. Cargar imagen de entrada
                "1": {
                    "class_type": "LoadImage",
                    "inputs": {"image": iname, "upload": "image"}
                },
                # 2. GGUF Unet Loader
                "2": {
                    "class_type": "UnetLoaderGGUF",
                    "inputs": {"unet_name": flux_name, "device": "default"}
                },
                # 3. CLIPLoader para FLUX - usar nodos GGUF del custom node ComfyUI-GGUF
                # DualCLIPLoaderGGUF carga clip_l + t5xxl ambos en formato GGUF/safetensors
                "3": {
                    "class_type": "DualCLIPLoaderGGUF",
                    "inputs": {"clip_name1": clip_l_name, "clip_name2": t5_name, "type": "flux"}
                },
                # 4. Load VAE
                "4": {
                    "class_type": "VAELoader",
                    "inputs": {"vae_name": vae_name}
                },
                # 5. VAEEncode para inpaint
                "5": {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}
                },
                # 6. CLIPTextEncode para positivo
                "6": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": prompt, "clip": ["3", 0]}
                },
                # 7. CLIPTextEncode para negativo
                "7": {
                    "class_type": "CLIPTextEncode",
                    "inputs": {"text": negative_prompt, "clip": ["3", 0]}
                },
                # 8. KSampler
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
                        "sampler_name": "euler",
                        "scheduler": "normal",
                        "denoise": denoise  # img2img: 0.75 permite edición manteniendo estructura original
                    }
                },
                # 9. VAEDecode (con --lowvram, ComfyUI hace tiled automático si OOM)
                "9": {
                    "class_type": "VAEDecode",
                    "inputs": {"samples": ["8", 0], "vae": ["4", 0]}
                },
                # 10. SaveImage
                "10": {
                    "class_type": "SaveImage",
                    "inputs": {
                        "images": ["9", 0],
                        "filename_prefix": "FluxEdit_output"
                    }
                }
            }
            
            # Queue workflow
            r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow})
            if r.status_code != 200:
                raise Exception(f"Queue failed: {r.text}")
            
            pid = r.json()["prompt_id"]
            print(f"[FluxEdit] Prompt queued: {pid}", flush=True)
            
            # Wait for completion
            # FLUX en lowvram con 8 pasos = ~22 min, con margen = 45 min
            max_wait = 2700  # 45 minutos
            last_comfy_check = 0
            
            for i in range(max_wait):
                time.sleep(1)
                
                # Mostrar progreso cada 2 minutos
                if i > 0 and i % 120 == 0:
                    elapsed_min = i / 60
                    print(f"[FluxEdit] Procesando... {elapsed_min:.1f} min transcurridos (lowvram mode)", flush=True)
                
                # Verificar que ComfyUI sigue vivo cada 10 segundos (no cada 5, para no interferir)
                if i - last_comfy_check >= 10:
                    try:
                        # Timeout largo: ComfyUI puede estar ocupado con VAE decode en lowvram
                        health = requests.get(f"{COMFY}/system_stats", timeout=15)
                        if health.status_code != 200:
                            raise Exception("ComfyUI process murió durante el procesamiento")
                        last_comfy_check = i
                    except requests.exceptions.ConnectionError:
                        raise Exception("ComfyUI se cerró inesperadamente - posible error de VRAM")
                    except requests.exceptions.ReadTimeout:
                        # ComfyUI está ocupado (VAE decode, etc.), no es error real
                        pass
                
                try:
                    # Timeout largo: history puede tardar si ComfyUI está haciendo VAE decode
                    r = requests.get(f"{COMFY}/history/{pid}", timeout=30)
                except requests.exceptions.ConnectionError:
                    raise Exception("Conexión con ComfyUI perdida durante el procesamiento")
                except requests.exceptions.ReadTimeout:
                    # ComfyUI sigue procesando (VAE decode lento), continuar
                    continue
                
                if r.status_code == 200:
                    data = r.json()
                    if pid in data:
                        outputs = data[pid].get("outputs", {})
                        
                        # Verificar si hubo errores en el procesamiento
                        status = data[pid].get("status", {})
                        if status.get("status_str") == "error":
                            error_msg = status.get("messages", ["Error desconocido en ComfyUI"])
                            # error_msg puede ser lista de listas o lista de strings
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
                                        print(f"[FluxEdit] OK en {elapsed:.1f}s", flush=True)
                                        return (
                                            GenResult(
                                                image=Image.open(io.BytesIO(vr.content)),
                                                time_taken=elapsed
                                            ),
                                            f"OK ({elapsed:.1f}s)"
                                        )
            
            raise Exception("Timeout (>10 min)")
            
        except Exception as e:
            print(f"[FluxEdit] ERROR: {str(e)}", flush=True)
            import traceback
            traceback.print_exc()
            return None, str(e)
    
    def unload(self):
        self._loaded = False

_client = None

def get_flux_edit_comfy_client() -> FluxEditComfyClient:
    global _client
    if _client is None:
        _client = FluxEditComfyClient()
    return _client

def is_flux_edit_available() -> bool:
    c = get_flux_edit_comfy_client()
    return c.is_available()
