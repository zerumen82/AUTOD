#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, json, time, requests, io
import numpy as np
from typing import Optional, Tuple
from PIL import Image
from dataclasses import dataclass

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image

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
        self._is_longcat_turbo = False

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

        # Avoid re-checking and "reloading" logs if already loaded the same version
        if self._loaded and getattr(self, '_flux_version', None) == flux_version:
            return True, f"FLUX {flux_version} ya cargado (reutilizado)"

        clip_map = {
            "flux2-klein-4b-Q4_K_S.gguf": ("qwen_3_4b_fp4_flux2.safetensors", "flux2"),
            "flux-2-klein-base-4b-Q4_K_S.gguf": ("qwen3-4b-abl-q4_0.gguf", "flux2"),
            "flux1-schnell-Q4_K_S.gguf": ("t5-v1_1-xxl-encoder-Q4_K_S.gguf", "flux"),
            "flux1-dev-Q4_K.gguf": ("t5-v1_1-xxl-encoder-Q4_K_S.gguf", "flux"),
        }
        self._dual_clip = False
        self._is_gguf_clip = False
        self._clip_name2 = None

        if flux_version == "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf":
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
        elif flux_version == "flux1-dev-Q2_K.gguf":
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
        elif flux_version == "LongCat-Image-Edit-Turbo-Q4_K_S.gguf":
            clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            clip_type = "longcat_image"
            self._dual_clip = False
            self._is_gguf_clip = False
            self._clip_name2 = None
            self._is_longcat_turbo = True
        elif flux_version == "LongCat-Image-Edit-Q4_K_S.gguf":
            clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            clip_type = "longcat_image"
            self._dual_clip = False
            self._is_gguf_clip = False
            self._clip_name2 = None
            self._is_longcat_turbo = False
        elif flux_version in (
            "ggml-model-Q4_K_M.gguf",
            "flux1-fill-dev-q4_k_m.gguf",
            "flux1-fill-dev-Q4_K.gguf",
        ) or "fill" in flux_version.lower():
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
            self._is_longcat_turbo = False
        else:
            clip_name, clip_type = clip_map[flux_version]
            if clip_name.endswith(".gguf"):
                self._is_gguf_clip = True

        checks = [
            ("diffusion_models", flux_version, f"Modelo FLUX {flux_version}"),
            ("text_encoders", clip_name, f"CLIP {clip_name}"),
        ]
        if self._dual_clip:
            checks.append(("text_encoders", self._clip_name2, f"CLIP2 {self._clip_name2}"))
        
        # VAE: flux2 para modelos flux2/flux-2, ae.safetensors para LongCat y otros (16 canales)
        if any(x in flux_version for x in ["flux2", "flux-2"]):
            vae_name = "flux2_vae.safetensors"
        else:
            vae_name = "ae.safetensors"

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

    def _is_probable_gray_failure(self, image: Image.Image) -> bool:
        arr = np.array(image.convert("RGB"), dtype=np.float32)
        luma = arr.mean(axis=2)
        channel_spread = np.abs(arr[:, :, 0] - arr[:, :, 1]).mean() + np.abs(arr[:, :, 1] - arr[:, :, 2]).mean()
        return luma.std() < 8.0 and channel_spread < 8.0

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
        seed: int = None,
        denoise: float = 0.60,
        mask_image: Optional[Image.Image] = None,
        lora_name: Optional[str] = None,
        lora_strength: float = 1.0,
        progress_callback=None,
        reference_latents_method: str = "index_timestep_zero",
        grow_mask_by: int = 4,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        # Prompt negativo base (mínimo para calidad)
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, low resolution"

        is_longcat = "LongCat" in self._flux_version

        t0 = time.time()
        image = image.convert("RGB")
        w, h = image.size
        new_w, new_h = (w // 64) * 64, (h // 64) * 64
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
        # LongCat maneja hasta 1024px nativamente; otros modelos 768px
        max_dim = 1024 if is_longcat else 768
        if new_w > max_dim or new_h > max_dim:
            image.thumbnail((max_dim, max_dim), Image.LANCZOS)
            new_w, new_h = image.size
            new_w, new_h = (new_w // 64) * 64, (new_h // 64) * 64
            image = image.resize((new_w, new_h), Image.LANCZOS)

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
            "3": {"class_type": "DualCLIPLoaderGGUF" if self._dual_clip else ("CLIPLoaderGGUF" if self._is_gguf_clip else "CLIPLoader"), "inputs": {"clip_name1": self._clip_name, "clip_name2": self._clip_name2, "type": self._clip_type} if self._dual_clip else {"clip_name": self._clip_name, "type": self._clip_type, "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}},
        }

        # Manejo de LoRAs
        last_model = ["2", 0]
        last_clip = ["3", 0]
        if lora_name and lora_name != "None":
            wf["20"] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": last_model,
                    "clip": last_clip,
                    "lora_name": lora_name,
                    "strength_model": lora_strength,
                    "strength_clip": lora_strength
                }
            }
            last_model = ["20", 0]
            last_clip = ["20", 1]

        # LongCat Turbo es distilled (CFG=1.0 forzado, pasos nativos ~8).
        if is_longcat and self._is_longcat_turbo:
            actual_cfg = 1.0
            actual_denoise = 1.0
            num_inference_steps = min(num_inference_steps, 12)
        else:
            actual_cfg = guidance_scale
            actual_denoise = denoise

        if is_longcat:
            wf["16"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}}
            if self._is_longcat_turbo:
                wf["17"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": last_model, "shift": 3.1}}
                model_input = ["17", 0]
            else:
                model_input = last_model

            wf["6"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": last_clip, "prompt": prompt, "vae": ["4", 0], "image1": ["16", 0], "image2": None, "image3": None}
            }
            ref_method = reference_latents_method or "index_timestep_zero"
            if ref_method not in ("offset", "index", "uxo/uno", "index_timestep_zero"):
                ref_method = "index_timestep_zero"
            wf["11"] = {
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["6", 0], "reference_latents_method": ref_method},
            }
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            positive_input = ["11", 0]
            negative_input = ["7", 0]
        else:
            wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": last_clip}}
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            model_input = last_model
            positive_input = ["6", 0]
            negative_input = ["7", 0]

        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_input, "positive": positive_input, "negative": negative_input,
                "latent_image": ["5", 0], "seed": seed or int(t0) % 1000000,
                "steps": num_inference_steps, "cfg": actual_cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "simple",
                "denoise": actual_denoise
            }
        }
        # Cambiado a VAEDecode estándar para mayor estabilidad en 8GB. 
        # VAEDecodeTiled con parámetros temporales era erróneo para imágenes estáticas.
        wf["9"] = {
            "class_type": "VAEDecode",
            "inputs": {
                "samples": ["8", 0], 
                "vae": ["4", 0]
            }
        }
        wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "FastEdit"}}

        if mname:
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": mname, "upload": "image"}}
            if is_longcat:
                wf["19"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["14", 0]}}
            wf["15"] = {"class_type": "ImageToMask", "inputs": {"image": ["19", 0] if is_longcat else ["14", 0], "channel": "red"}}
            wf["5"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["16", 0] if is_longcat else ["1", 0],
                    "vae": ["4", 0],
                    "mask": ["15", 0],
                    "grow_mask_by": max(0, int(grow_mask_by)),
                },
            }
        else:
            wf["5"] = {"class_type": "VAEEncode", "inputs": {"pixels": ["16", 0] if is_longcat else ["1", 0], "vae": ["4", 0]}}

        if is_longcat:
            print(
                "[FluxClient] LongCat edit workflow: "
                f"cfg={wf['8']['inputs']['cfg']}, denoise={actual_denoise:.2f}, "
                f"ref={ref_method}, "
                f"mask={'inpaint' if mname else 'global'}, "
                f"nodes={sorted(wf.keys(), key=int)}",
                flush=True,
            )

        r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:200])
            return None, f"ComfyUI rechazó el prompt: {err}"

        pid = r.json().get("prompt_id")
        if not pid:
            return None, "ComfyUI no devolvió prompt_id"

        def _on_progress(prog):
            if progress_callback:
                progress_callback(prog)
            elif int(prog.get("elapsed") or 0) % 15 == 0 and prog.get("phase"):
                print(f"[FluxClient] {prog.get('phase')} ({int(prog.get('elapsed') or 0)}s)", flush=True)

        img_meta, wait_msg = wait_for_comfy_image(
            comfy_url,
            pid,
            timeout=3600,
            steps_hint=num_inference_steps,
            progress_callback=_on_progress,
            cancel_check=kwargs.get("cancel_check"),
        )
        if img_meta is None:
            return None, wait_msg

        res = requests.get(f"{comfy_url}/view?filename={img_meta['filename']}", timeout=30)
        if res.status_code != 200:
            return None, f"Error descargando imagen: {res.status_code}"

        elapsed = time.time() - t0
        pil_img = Image.open(io.BytesIO(res.content)).convert("RGB")
        arr = np.array(pil_img.convert("RGB"))
        std = arr.std()
        mean = arr.mean()
        print(f"[FluxClient] Generación completa. std={std:.2f}, mean={mean:.2f}")

        if self._is_probable_gray_failure(pil_img):
            print(f"[FluxClient] Imagen gris detectada (std={std:.1f}, mean={mean:.1f}).")
            return None, "ComfyUI devolvió una imagen casi gris/uniforme. Revisa VRAM, VAE y workflow del motor seleccionado."
        return GenResult(image=pil_img, time_taken=elapsed), f"OK ({elapsed:.0f}s)"

_client = None
def get_flux_edit_comfy_client() -> FluxEditComfyClient:
    global _client
    if _client is None:
        _client = FluxEditComfyClient()
    return _client
