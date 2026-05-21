#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import time
import requests
import io
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
    """Cliente para generacion pura (txt2img) en ComfyUI usando LongCat"""
    def __init__(self):
        self._loaded = False
        self._flux_version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"  # Usar el preferido del usuario

    def is_available(self):
        try:
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _model_exists(self, subdir: str, name: str) -> bool:
        path = os.path.join(get_models_base(), subdir, name)
        return os.path.exists(path)

    def _rewrite_prompt(self, prompt: str) -> Tuple[str, float]:
        """Usa el LLM para traducir y analizar el prompt semánticamente"""
        try:
            from roop.img_editor.prompt_rewriter import get_prompt_rewriter
            rewriter = get_prompt_rewriter()
            analysis = rewriter.rewrite(prompt)
            enhanced = analysis.get("prompt", prompt)
            magnitude = analysis.get("magnitude", 0.5)
            print(f"[GenFlux] Semantic boost: {enhanced[:60]}... (Mag: {magnitude})")
            return enhanced, magnitude
        except Exception as e:
            print(f"[GenFlux] Rewriter error, using original: {e}")
            return prompt, 0.5

    def _resolve_models(self) -> Optional[Tuple[str, str, str]]:
        """Resuelve los nombres de modelo, CLIP y VAE desde ComfyUI o modelo local."""
        comfy_url = get_comfyui_url()
        flux_version = self._flux_version

        # CLIP para LongCat
        clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        clip_type = "longcat_image"

        # VAE: flux2 para modelos flux2/flux-2/LongCat, ae.safetensors para otros
        if "flux2" in flux_version or "flux-2" in flux_version or "LongCat" in flux_version:
            vae_name = "flux2_vae.safetensors"
        else:
            vae_name = "ae.safetensors"

        # Verificar que los archivos existen
        checks = [
            ("diffusion_models", flux_version, f"Modelo {flux_version}"),
            ("text_encoders", clip_name, f"CLIP {clip_name}"),
            ("vae", vae_name, f"VAE {vae_name}"),
        ]
        missing = []
        for subdir, fname, label in checks:
            path = os.path.join(get_models_base(), subdir, fname)
            if not os.path.exists(path):
                missing.append(f"  - {label}: {path}")
        if missing:
            return None, "Modelos faltantes:\n" + "\n".join(missing)

        return (flux_version, clip_name, clip_type, vae_name), None

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        width: int = 512,
        height: int = 768,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        comfy_url = get_comfyui_url()
        if not self.is_available():
            return None, "ComfyUI no esta disponible"

        start = time.time()

        # Dimensiones multiplo de 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        # Resolver modelos
        resolved, err = self._resolve_models()
        if not resolved:
            return None, err
        flux_version, clip_name, clip_type, vae_name = resolved

        # 1. Análisis Semántico y Traducción Dinámica
        final_prompt, magnitude = self._rewrite_prompt(prompt)
        
        # 2. Resolución Dinámica de Parámetros (si no se pasan explícitamente)
        is_longcat = "LongCat" in flux_version
        is_longcat_turbo = "Turbo" in flux_version
        
        if steps is None:
            # Escalar pasos: 8 a 24 según magnitud
            steps = int(8 + (magnitude * 16))
            if is_longcat_turbo: steps = min(steps, 12)
            
        if guidance_scale is None:
            # Escalar CFG: 3.0 a 7.5
            guidance_scale = 3.0 + (magnitude * 4.5)
            if is_longcat_turbo: guidance_scale = 1.0

        cfg = guidance_scale
        denoise = 1.0

        print(f"[GenFlux] Dynamic Params: Mag={magnitude:.2f}, Steps={steps}, CFG={cfg:.1f}")

        # Construir workflow desde cero (txt2img)
        # LongCat Turbo inserta ModelSamplingAuraFlow (17) entre UNET (2) y KSampler (8)
        if is_longcat_turbo:
            wf = {
                "17": {
                    "class_type": "ModelSamplingAuraFlow",
                    "inputs": {"model": ["2", 0], "shift": 3.1}
                },
            }
        else:
            wf = {}

        # Node IDs:
        #  1 = EmptyLatentImage             (ruido inicial LATENT)
        #  2 = UnetLoaderGGUF               (modelo UNET)
        #  3 = CLIPLoader                   (CLIP, type=longcat_image)
        #  4 = VAELoader                    (VAE)
        #  6 = TextEncodeQwenImageEditPlus  (prompt positivo)
        #  7 = CLIPTextEncode               (prompt negativo)
        #  8 = KSampler                     (muestreo desde ruido)
        #  9 = VAEDecode                    (latent -> imagen)
        # 10 = SaveImage                    (guardar resultado)
        # 11 = FluxKontextMultiReferenceLatentMethod (metodo LongCat)
        # 17 = ModelSamplingAuraFlow      (solo si LongCat Turbo)
        wf.update({
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {
                    "width": width,
                    "height": height,
                    "batch_size": 1
                }
            },
            "2": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": flux_version}
            },
            "3": {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": clip_name, "type": clip_type}
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name}
            },
            "6": {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {
                    "clip": ["3", 0],
                    "prompt": final_prompt,
                    "vae": ["4", 0],
                    "image1": None,
                    "image2": None,
                    "image3": None
                }
            },
            "7": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": negative_prompt, "clip": ["3", 0]}
            },
            "11": {
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}
            },
            "8": {
                "class_type": "KSampler",
                "inputs": {
                    "model": ["17", 0] if is_longcat_turbo else ["2", 0],
                    "positive": ["11", 0],
                    "negative": ["7", 0],
                    "latent_image": ["1", 0],
                    "seed": seed or int(time.time()) % 1000000,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler_ancestral",
                    "scheduler": "simple",
                    "denoise": denoise
                }
            },
            # VAEDecode estándar (no Tiled) para máxima compatibilidad con imágenes estáticas
            # VAEDecodeTiled con params temporales es para video y causa errores en 8GB
            "9": {
                "class_type": "VAEDecode",
                "inputs": {
                    "samples": ["8", 0], 
                    "vae": ["4", 0]
                }
            },
            "10": {
                "class_type": "SaveImage",
                "inputs": {"images": ["9", 0], "filename_prefix": "GenFlux"}
            },
        })

        if is_longcat:
            print(
                "[GenFlux] LongCat workflow: "
                f"cfg={cfg}, denoise={denoise}, "
                f"model={flux_version}, "
                f"nodes={sorted(wf.keys(), key=lambda x: int(x) if x.isdigit() else 99)}",
                flush=True,
            )

        try:
            r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
            if r.status_code != 200:
                return None, f"ComfyUI rechazo el prompt: {r.text[:200]}"

            pid = r.json().get("prompt_id")
            deadline = time.time() + 600
            while time.time() < deadline:
                r = requests.get(f"{comfy_url}/history/{pid}")
                if r.status_code == 200 and pid in r.json():
                    hist = r.json()[pid]
                    if "outputs" in hist and "10" in hist["outputs"]:
                        img_data = hist["outputs"]["10"]["images"][0]
                        res = requests.get(f"{comfy_url}/view?filename={img_data['filename']}")
                        if res.status_code == 200:
                            pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
                            return GenResult(image=pil_img, time_taken=time.time()-start), "OK"
                time.sleep(1)
            return None, "Timeout (600s)"
        except Exception as e:
            return None, str(e)

_gen_client = None
def get_flux_gen_client() -> FluxGenComfyClient:
    global _gen_client
    if _gen_client is None:
        _gen_client = FluxGenComfyClient()
    return _gen_client
