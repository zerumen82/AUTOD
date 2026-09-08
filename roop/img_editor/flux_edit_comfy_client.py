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
        self._is_qwen_image_edit = False
        self._is_qwen_q5 = False  # v5.1040 fix: always defined to avoid AttributeError on klein/flux
        self._is_kontext_dev = False

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
            # GGUF models can be in unet/ instead of diffusion_models/
            if subdir == "diffusion_models" and name.endswith(".gguf"):
                alt_path = os.path.join(get_models_base(), "unet", name)
                if os.path.exists(alt_path):
                    return True
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
            "flux-2-klein-9b-Q4_K_S.gguf": ("Qwen_Qwen3-8B-Q4_0.gguf", "flux2"),  # 12288-dim text encoder para Klein 9B
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
        elif flux_version == "Qwen_Image_Edit-Q2_K.gguf":
            clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            clip_type = "qwen_image"
            self._dual_clip = False
            self._is_gguf_clip = False
            self._clip_name2 = None
            self._is_longcat_turbo = False
            self._is_qwen_image_edit = True
            self._is_qwen_q5 = False
        elif "kontext" in flux_version:
            # FLUX.1 Kontext dev GGUF — dual CLIP (clip_l + t5-xxl GGUF)
            # Use T5 GGUF (2.7GB) instead of T5 fp8 (3GB) to save VRAM
            clip_name = "clip_l.safetensors"
            clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            clip_type = "flux"
            self._dual_clip = True
            self._clip_name2 = clip_name2
            self._is_longcat_turbo = False
            self._is_qwen_image_edit = False
            self._is_kontext_dev = True
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
        
        # VAE: flux2 para modelos flux2/flux-2, Qwen_Image-VAE para Qwen-Image-Edit, ae.safetensors para LongCat y otros (16 canales)
        if "Qwen_Image_Edit" in flux_version:
            vae_name = "Qwen_Image-VAE.safetensors"
        elif any(x in flux_version for x in ["flux2", "flux-2"]):
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
        no_kontext: bool = False,
        drop_edit_image_ref: bool = False,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        # También aceptado vía kwargs por llamadas genéricas del manager
        drop_edit_image_ref = bool(drop_edit_image_ref or kwargs.get("drop_edit_image_ref", False))
        no_kontext = bool(no_kontext or kwargs.get("no_kontext", False))
        ref_method = reference_latents_method or "index_timestep_zero"
        if no_kontext:
            ref_method = "none"

        # Prompt negativo base (mínimo para calidad)
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, low resolution"

        is_longcat = "LongCat" in self._flux_version
        is_kontext = self._is_kontext_dev
        is_qwen = self._is_qwen_image_edit or self._is_qwen_q5
        is_klein_9b = "klein-9b" in self._flux_version
        is_klein = is_klein_9b or ("klein" in (self._flux_version or ""))
        is_klein_edit = is_klein
        # ImageEditor Klein = REAL EDIT (ReferenceLatent): latente vacío + la imagen
        # de referencia se inyecta como conditioning("reference_latents") → el modelo
        # regenera con composición/identidad de la ref aplicando la instrucción.

        klein_empty_latent = None

        t0 = time.time()
        image = image.convert("RGB")
        w, h = image.size
        new_w, new_h = (w // 64) * 64, (h // 64) * 64
        if new_w != w or new_h != h:
            image = image.resize((new_w, new_h), Image.LANCZOS)
        # LongCat/Kontext hasta 1024px; Klein 9B hasta 1024px; Qwen-Image-Edit 640px; otros 768px
        max_dim = 1024 if (is_longcat or is_kontext or is_klein_9b) else (640 if is_qwen else 768)
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
        if is_klein_edit:
            # El EDIT real de Klein usa la imagen COMPLETA como referencia (ReferenceLatent).
            # Una máscara rompería el conditioning — ignorarla siempre aquí.
            mask_image = None
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
        if is_klein_edit:
            # FLUX.2 Klein EDIT real: latente VACÍO + ReferenceLatent ( refs en
            # conditioning) → denoise 1.0 obligatorio, 4 pasos nativos, CFG 1.0.
            actual_cfg = 1.0
            actual_denoise = 1.0
            num_inference_steps = min(num_inference_steps, 4)
        elif is_longcat and self._is_longcat_turbo:
            actual_cfg = 1.0
            actual_denoise = 1.0
            num_inference_steps = min(num_inference_steps, 12)
        else:
            actual_cfg = guidance_scale
            actual_denoise = denoise

        if is_kontext:
            # FLUX.1 Kontext dev: standard FLUX img2img workflow
            # No FluxKontextImageScale or FluxKontextMultiReferenceLatentMethod
            model_input = last_model
            wf["6"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": last_clip},
            }
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            positive_input = ["6", 0]
            negative_input = ["7", 0]
        elif is_longcat:
            wf["16"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}}
            if self._is_longcat_turbo:
                wf["17"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": last_model, "shift": 3.1}}
                model_input = ["17", 0]
            else:
                model_input = last_model

            if no_kontext:
                # Bypass de latentes de referencia (sin FluxKontextMultiReferenceLatentMethod).
                # Con visión (drop=False): Qwen-VL VE la foto → género, pose y
                # proporciones correctos, pero sin ancla estructural a la ropa.
                # Sin visión (drop=True): solo-texto, más rápido pero alucina cuerpos.
                if drop_edit_image_ref:
                    print("[FluxClient] Kontext BYPASS + text-encode sin image-ref")
                    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": last_clip}}
                else:
                    print("[FluxClient] Kontext BYPASS con visión (ve la imagen, sin latentes ref)")
                    wf["6"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": prompt,
                            "vae": ["4", 0],
                            "image1": ["16", 0],
                            "image2": None,
                            "image3": None,
                        },
                    }
                wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
                positive_input = ["6", 0]
                negative_input = ["7", 0]
            else:
                wf["6"] = {
                    "class_type": "TextEncodeQwenImageEditPlus",
                    "inputs": {
                        "clip": last_clip,
                        "prompt": prompt,
                        "vae": ["4", 0],
                        "image1": ["16", 0],
                        "image2": None,
                        "image3": None,
                    },
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
        elif is_qwen:
            wf["16"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["1", 0]}}
            model_input = last_model
            wf["6"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {"clip": last_clip, "prompt": prompt, "vae": ["4", 0], "image1": ["16", 0], "image2": None, "image3": None}
            }
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            positive_input = ["6", 0]
            negative_input = ["7", 0]
            if lora_name and lora_name != "None":
                actual_cfg = 1.0
                num_inference_steps = min(num_inference_steps, 12)
            else:
                actual_cfg = guidance_scale if guidance_scale >= 1.0 else 2.5
            # Qwen-Image-Edit usa VAEEncode con imagen escalada (FluxKontextImageScale)
        else:
            wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": last_clip}}
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            if is_klein_edit:
                # ===== FLUX.2 Klein EDIT REAL (ReferenceLatent) =====
                # Latente VACÍO del tamaño de la imagen + la imagen de referencia se
                # inyecta COMO conditioning ("reference_latents") → el modelo regenera
                # manteniendo composición/identidad de la ref y aplica la instrucción.
                # (flujo oficial de FLUX.2 Klein edit / one-node)
                wf["5d"] = {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}}
                wf["5e"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["5d", 0], "height": ["5d", 1], "batch_size": 1}}
                wf["6r"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["5e", 0]}}
                wf["7r"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["5e", 0]}}
                wf["17"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"shift": 3.0, "model": last_model}}
                model_input = ["17", 0]
                positive_input = ["6r", 0]
                negative_input = ["7r", 0]
                klein_empty_latent = ["5e", 0]
            else:
                model_input = last_model
                positive_input = ["6", 0]
                negative_input = ["7", 0]
                klein_empty_latent = None

        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_input, "positive": positive_input, "negative": negative_input,
                "latent_image": klein_empty_latent or ["5", 0], "seed": seed or int(t0) % 1000000,
                "steps": num_inference_steps, "cfg": actual_cfg,
                "sampler_name": ("er_sde" if is_klein_edit else ("euler" if is_qwen else "euler_ancestral")),
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

        uses_kontext = is_longcat and not no_kontext  # Only LongCat uses FluxKontext nodes
        if mname:
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": mname, "upload": "image"}}
            if uses_kontext:
                wf["19"] = {"class_type": "FluxKontextImageScale", "inputs": {"image": ["14", 0]}}
            wf["15"] = {"class_type": "ImageToMask", "inputs": {"image": ["19", 0] if uses_kontext else ["14", 0], "channel": "red"}}
            wf["5"] = {
                "class_type": "VAEEncodeForInpaint",
                "inputs": {
                    "pixels": ["16", 0] if uses_kontext else ["1", 0],
                    "vae": ["4", 0],
                    "mask": ["15", 0],
                    "grow_mask_by": max(0, int(grow_mask_by)),
                },
            }
        else:
            wf["5"] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["16", 0] if uses_kontext else ["1", 0],
                    "vae": ["4", 0],
                },
            }

        def _wf_sort_key(k):
            try:
                return int(k)
            except ValueError:
                return int("".join(ch for ch in k if ch.isdigit()) or 0)

        if is_klein_edit:
            print(
                "[FluxClient] FLUX.2 Klein EDIT workflow (ReferenceLatent + EmptyFlux2Latent): "
                f"cfg=1.0, denoise=1.0, steps={num_inference_steps}, sampler=er_sde, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_kontext:
            print(
                "[FluxClient] FLUX.1 Kontext dev workflow (standard FLUX img2img): "
                f"cfg={wf['8']['inputs']['cfg']}, denoise={actual_denoise:.2f}, "
                f"mask={'inpaint' if mname else 'global'}, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_longcat:
            print(
                "[FluxClient] LongCat edit workflow: "
                f"cfg={wf['8']['inputs']['cfg']}, denoise={actual_denoise:.2f}, "
                f"ref={ref_method}, "
                f"drop_img_ref={drop_edit_image_ref}, "
                f"mask={'inpaint' if mname else 'global'}, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_qwen:
            print(
                "[FluxClient] Qwen-Image-Edit workflow: "
                f"cfg={wf['8']['inputs']['cfg']}, steps={num_inference_steps}, "
                f"mask={'inpaint' if mname else 'global'}, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
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
