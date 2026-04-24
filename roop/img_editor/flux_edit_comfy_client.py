#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLUX Image Edit ComfyUI Client
Usa FLUX fp8 para edición de imágenes con instrucciones en lenguaje natural
"""

import os, sys, json, time, requests, io, shutil
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
    
    def get_model_paths(self, flux_version="flux2-klein-4b-Q4_K_S.gguf") -> dict:
        """Rutas para FLUX GGUF (optimizado para GPU 8GB)"""
        base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
        
        # Asegurar extensión .gguf
        if not flux_version.endswith(".gguf"):
            flux_version += ".gguf"

        is_klein = ("klein" in flux_version.lower()) or ("flux2" in flux_version.lower())

        if is_klein:
            # FLUX.2/Klein usa CLIPLoader(type=flux2) con Qwen (no DualCLIP de FLUX.1)
            qwen_clip = os.path.join(base, "text_encoders", "qwen_3_4b.safetensors")
            if not os.path.exists(qwen_clip):
                qwen_clip = os.path.join(base, "text_encoders", "qwen_3_4b_fp4_flux2.safetensors")
            flux2_vae = os.path.join(base, "vae", "flux2_vae.safetensors")
            if not os.path.exists(flux2_vae):
                flux2_vae = os.path.join(base, "vae", "flux2-vae.safetensors")

            return {
                "flux": os.path.join(base, "diffusion_models", flux_version),
                "clip_flux2": qwen_clip,
                "vae": flux2_vae,
            }

        return {
            # FLUX UNet GGUF dinámico
            "flux": os.path.join(base, "diffusion_models", flux_version),

            # FLUX.1 text encoders - usar versión GGUF para reducir VRAM
            "clip_l": os.path.join(base, "text_encoders", "clip_l.safetensors"),
            "t5xxl": os.path.join(base, "text_encoders", "t5-v1_1-xxl-encoder-Q4_K_S.gguf"),

            # VAE estándar
            "vae": os.path.join(base, "vae", "ae.safetensors"),
        }
    
    def check_models(self, flux_version="flux2-klein-4b-Q4_K_S.gguf") -> Tuple[bool, str]:
        """Verifica que los modelos existan"""
        paths = self.get_model_paths(flux_version)
        missing = []
        
        for name, path in paths.items():
            if not os.path.exists(path):
                missing.append(f"{name}: {path}")
        
        if missing:
            return False, f"Modelos faltantes:\n" + "\n".join(missing)

        # Fix de integridad para FLUX.1 (clip_l en carpeta clip)
        if "clip_l" in paths:
            try:
                base = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models"
                clip_folder_file = os.path.join(base, "clip", "clip_l.safetensors")
                text_encoder_file = paths["clip_l"]
                if os.path.exists(clip_folder_file):
                    if os.path.getsize(clip_folder_file) == 0 and os.path.getsize(text_encoder_file) > 0:
                        shutil.copy2(text_encoder_file, clip_folder_file)
                        print("[FluxEdit] Reparado models/clip/clip_l.safetensors (0 bytes -> archivo válido)", flush=True)
            except Exception as e:
                print(f"[FluxEdit] Warning reparando clip_l: {e}", flush=True)
        
        self._model_paths = paths
        return True, "Modelos OK"
    
    def load(self, progress_callback=None, flux_version="flux2-klein-4b-Q4_K_S.gguf", **kwargs) -> Tuple[bool, str]:
        """Carga los modelos y verifica disponibilidad"""
        # Si ya está cargado un modelo diferente, forzar recarga de rutas
        if self._loaded and self._model_paths.get("flux", "").endswith(flux_version):
            return True, "FLUX Image Edit listo"
        
        if not self.is_available():
            return False, "ComfyUI no disponible en 127.0.0.1:8188"
        
        ok, msg = self.check_models(flux_version)
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
        denoise: float = 0.75,
        mask_image: Optional[Image.Image] = None,  # NUEVO: Soporte para máscara
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:
        
        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        t0 = time.time()
        print(f"[FluxEdit] Generando ({num_inference_steps} pasos, denoise={denoise})...", flush=True)
        if mask_image:
            print("[FluxEdit] 🎭 Usando máscara para inpainting selectivo", flush=True)
        
        # Negative prompt por defecto
        negative_prompt = "low quality, worst quality, bad quality, jpeg artifacts, blurry, out of focus, poorly drawn, bad anatomy, deformed, disfigured, mutated, extra limbs"
        
        if num_inference_steps > 8:
            num_inference_steps = 8
        
        try:
            requested_w = kwargs.get("target_width")
            requested_h = kwargs.get("target_height")
            if requested_w and requested_h:
                w, h = int(requested_w), int(requested_h)
            else:
                w, h = image.size

            new_w = (w // 64) * 64
            new_h = (h // 64) * 64
            if new_w < 64: new_w = 64
            if new_h < 64: new_h = 64
            
            if new_w != image.size[0] or new_h != image.size[1]:
                image = image.resize((new_w, new_h), Image.LANCZOS)
            
            # Ajustar máscara al mismo tamaño si existe
            if mask_image:
                mask_image = mask_image.convert("L").resize((new_w, new_h), Image.NEAREST)

            # 2. Upload imagen y máscara
            iname = f"flux_input_{int(t0)}.png"
            buf = io.BytesIO()
            image.save(buf, "PNG")
            buf.seek(0)
            
            r = requests.post(f"{COMFY}/upload/image", files={"image": (iname, buf, "image/png")})
            if r.status_code != 200: raise Exception(f"Upload failed: {r.text}")

            mname = None
            if mask_image:
                mname = f"flux_mask_{int(t0)}.png"
                mbuf = io.BytesIO()
                mask_image.save(mbuf, "PNG")
                mbuf.seek(0)
                rm = requests.post(f"{COMFY}/upload/image", files={"image": (mname, mbuf, "image/png")})
                if rm.status_code != 200: raise Exception(f"Mask Upload failed: {rm.text}")
            
            # Nombres de archivos
            flux_name = os.path.basename(self._model_paths["flux"])
            clip_l_name = os.path.basename(self._model_paths.get("clip_l", "")) if "clip_l" in self._model_paths else None
            t5_name = os.path.basename(self._model_paths.get("t5xxl", "")) if "t5xxl" in self._model_paths else None
            clip_flux2_name = os.path.basename(self._model_paths.get("clip_flux2", "")) if "clip_flux2" in self._model_paths else None
            vae_name = os.path.basename(self._model_paths["vae"])
            
            is_schnell = "schnell" in flux_name.lower()
            is_klein = "klein" in flux_name.lower() or "flux2" in flux_name.lower()

            actual_guidance = max(1.5, min(3.0, guidance_scale)) if is_klein else (1.0 if is_schnell else guidance_scale)
            actual_steps = max(4, min(12, num_inference_steps))
            actual_denoise = max(0.15, min(0.95, denoise))

            # Construir Workflow Dinámico
            workflow = {}
            workflow["1"] = {"class_type": "LoadImage", "inputs": {"image": iname, "upload": "image"}}
            workflow["2"] = {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": flux_name, "device": "default"}}
            
            if is_klein:
                workflow["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": clip_flux2_name, "type": "flux2", "device": "default"}}
            else:
                workflow["3"] = {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": clip_l_name, "clip_name2": t5_name, "type": "flux"}}
            
            workflow["4"] = {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}}
            
            if mname:
                # MODO INPAINT: Usar VAE Encode for Inpaint
                workflow["12"] = {"class_type": "LoadImage", "inputs": {"image": mname, "upload": "image"}}
                workflow["5"] = {
                    "class_type": "VAEEncodeForInpaint",
                    "inputs": {
                        "pixels": ["1", 0],
                        "vae": ["4", 0],
                        "mask": ["12", 0],
                        "grow_mask_by": 6
                    }
                }
            else:
                workflow["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["1", 0], "vae": ["4", 0]}}

            workflow["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["3", 0]}}
            workflow["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": ["3", 0]}}
            
            sampling_model = ["2", 0]
            if not is_klein:
                workflow["11"] = {
                    "class_type": "ModelSamplingFlux",
                    "inputs": {
                        "model": ["2", 0],
                        "max_shift": 1.15, "base_shift": 0.5,
                        "width": new_w // 8, "height": new_h // 8
                    }
                }
                sampling_model = ["11", 0]

            workflow["8"] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": sampling_model,
                    "positive": ["6", 0],
                    "negative": ["7", 0],
                    "latent_image": ["5", 0],
                    "seed": seed or int(t0) % 1000000,
                    "steps": actual_steps,
                    "cfg": actual_guidance,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": actual_denoise
                }
            }
            
            workflow["9"] = {
                "class_type": "VAEDecodeTiled",
                "inputs": {
                    "samples": ["8", 0], "vae": ["4", 0],
                    "tile_size": 512, "overlap": 64, "temporal_size": 64, "temporal_overlap": 8
                }
            }
            workflow["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "FluxEdit_output"}}
            
            # Queue workflow
            r = requests.post(f"{COMFY}/prompt", json={"prompt": workflow})
            if r.status_code != 200:
                raise Exception(f"Queue failed: {r.text}")
            
            pid = r.json()["prompt_id"]
            print(f"[FluxEdit] Prompt queued: {pid}", flush=True)
            
            # Wait for completion
            # FLUX en lowvram con 8 pasos = ~22 min, con margen amplio para VAE decode = 90 min
            max_wait = 5400  # 90 minutos
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
                    # Timeout MUY largo: history puede tardar mucho si ComfyUI está haciendo VAE decode pesado
                    r = requests.get(f"{COMFY}/history/{pid}", timeout=300)
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
            
            raise Exception(f"Timeout total excedido ({max_wait/60:.0f} min)")
            
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
