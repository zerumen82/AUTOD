#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import time
import requests
import io
import json
from typing import Optional, Tuple, Dict
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
    final_prompt: str = ""
    time_taken: float = 0.0

class FluxGenComfyClient:
    """Cliente para generacion pura (txt2img) en ComfyUI usando FLUX/LongCat/SDXL"""
    def __init__(self):
        self._loaded = False
        self._flux_version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        self._alias = "longcat"
        self._clip_name = None
        self._clip_name2 = None
        self._clip_type = "flux"
        self._vae_name = "ae.safetensors"
        self._is_dual_clip = False
        self._is_longcat = False
        self._is_longcat_turbo = False
        self._is_sdxl = False
        self._model_configs = self._load_model_configs()

    def _load_model_configs(self):
        path = os.path.join(get_project_root(), "config", "model_configs.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    def is_available(self):
        try:
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _model_exists(self, subdir: str, name: str) -> bool:
        path = os.path.join(get_models_base(), subdir, name)
        return os.path.exists(path)

    def load(self, flux_version="longcat") -> Tuple[bool, str]:
        """Configura los modelos a usar"""
        alias_map = {
            "longcat": "LongCat-Image-Edit-Turbo-Q4_K_S.gguf",
            "longcat_full": "LongCat-Image-Edit-Q4_K_S.gguf",
            "klein_base": "flux-2-klein-base-4b-Q4_K_S.gguf",
            "flux_q2": "flux1-dev-Q2_K.gguf",
            "flux_dev_abliterated": "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf",
            "miamodel_nsfw": "miamodelSFWNSFWSDXL_v30.safetensors",
            "nova_nsfw": "novaillustrousNSFW_v20.safetensors",
            "lazy_nsfw": "realisticLazyMixNSFW_v10.safetensors"
        }
        
        self._alias = flux_version
        real_model = alias_map.get(flux_version, flux_version)
        self._flux_version = real_model
        
        self._is_sdxl = any(x in real_model.lower() for x in ["sdxl", "miamodel", "nova", "lazy"])
        self._is_longcat = "LongCat" in real_model
        self._is_longcat_turbo = "Turbo" in real_model
        self._is_dual_clip = False
        self._clip_name2 = None

        if self._is_sdxl:
            self._clip_type = "sdxl"
            self._vae_name = "baked"
            self._loaded = True
            return True, "Listo (SDXL)"

        if self._is_longcat:
            self._clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
            self._clip_type = "longcat_image"
            self._vae_name = "ae.safetensors"
        elif "schnell" in real_model.lower():
            self._clip_name = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            self._clip_type = "flux"
            self._vae_name = "ae.safetensors"
        elif "flux2" in real_model.lower() or "flux-2" in real_model.lower():
            self._clip_name = "qwen_3_4b_fp4_flux2.safetensors"
            self._clip_type = "flux2"
            self._vae_name = "flux2_vae.safetensors"
        else:
            self._clip_name = "clip_l.safetensors"
            self._clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            self._clip_type = "flux"
            self._is_dual_clip = True
            self._vae_name = "ae.safetensors"

        checks = [("diffusion_models", self._flux_version, "Modelo")]
        checks.append(("text_encoders", self._clip_name, "CLIP 1"))
        if self._is_dual_clip: checks.append(("text_encoders", self._clip_name2, "CLIP 2"))
        checks.append(("vae", self._vae_name, "VAE"))

        missing = []
        for d, f, l in checks:
            if not self._model_exists(d, f): missing.append(f)
        
        if missing:
            return False, f"Faltan archivos: {', '.join(missing)}"
        
        self._loaded = True
        return True, "Listo"

    def _prepare_prompt(self, prompt: str) -> str:
        """Preparación básica: Traducción y limpieza de prefijos."""
        try:
            from roop.img_editor.prompt_translator import translate_prompt
            final = translate_prompt(prompt).strip()
            if final.lower().startswith("instruction:"):
                final = final[12:].strip()
            if self._is_longcat:
                final = f"Instruction: {final}"
            return final
        except:
            return prompt

    def _prepare_prompt_intelligent(self, prompt: str) -> Tuple[str, Dict]:
        """Traduce el prompt y ajusta parámetros básicos sin añadir palabras ocultas."""
        try:
            from roop.img_editor.prompt_translator import translate_prompt
            translated = translate_prompt(prompt).strip()

            if translated.lower().startswith("instruction:"):
                translated = translated[12:].strip()

            if self._is_longcat:
                final_prompt = f"Instruction: {translated}"
            else:
                final_prompt = translated

            # Cargar config dinámica (solo para steps/cfg, no para el prompt)
            conf = self._model_configs.get(self._alias, self._model_configs.get("default", {}))
            
            p = {
                "steps": conf.get("steps", 25),
                "cfg": conf.get("cfg", 3.5),
                "neg": conf.get("negative_prompt", "")
            }
            
            # NO AÑADIMOS PREFIJOS NI TAGS HARDCODEADOS

            print(f"[GenFlux] Prompt Final: {final_prompt[:200]}...")
            return final_prompt, p
        except Exception as e:
            print(f"[GenFlux] Error: {e}")
            return prompt, {"steps": 25, "cfg": 7.5, "neg": ""}

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        width: int = 768,
        height: int = 1024,
        _skip_rewrite: bool = False,
        use_ai: bool = False,
        lora_name: str = None,
        lora_strength: float = 1.0,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        comfy_url = get_comfyui_url()
        if not self.is_available(): return None, "ComfyUI no disponible"
        if not self._loaded: self.load()

        start = time.time()
        w, h = (width // 64) * 64, (height // 64) * 64
        
        conf = self._model_configs.get(self._alias, self._model_configs.get("default", {}))

        # 1. Resolución de Prompt, Parámetros y Negativo
        current_neg = negative_prompt
        if use_ai and not _skip_rewrite:
            final_prompt, ai_params = self._prepare_prompt_intelligent(prompt)
            actual_steps = ai_params["steps"]
            actual_cfg = ai_params["cfg"]
            if ai_params.get("neg"):
                current_neg = f"{ai_params['neg']}, {negative_prompt}"
        else:
            final_prompt = self._prepare_prompt(prompt) if not _skip_rewrite else prompt
            actual_steps = steps if steps is not None else conf.get("steps", 25)
            actual_cfg = guidance_scale if guidance_scale is not None else conf.get("cfg", 3.5)

        sampler_name = conf.get("sampler", "euler_ancestral")
        scheduler = conf.get("scheduler", "simple")

        if self._is_sdxl:
            wf = {
                "1": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self._flux_version}},
            }
            
            last_model = ["2", 0]
            last_clip = ["2", 1]
            
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

            wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": final_prompt, "clip": last_clip}}
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": current_neg, "clip": last_clip}}
            wf["8"] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": last_model, "positive": ["6", 0], "negative": ["7", 0],
                    "latent_image": ["1", 0], "seed": seed or int(time.time()) % 1000000,
                    "steps": actual_steps, "cfg": actual_cfg,
                    "sampler_name": sampler_name, "scheduler": scheduler, "denoise": 1.0
                }
            }
            wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["2", 2]}}
            wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "GenSDXL"}}
        else:
            # Workflow FLUX / LongCat
            wf = {
                "1": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": self._flux_version}},
                "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}},
            }

            if self._is_dual_clip:
                wf["3"] = {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": self._clip_name, "clip_name2": self._clip_name2, "type": self._clip_type}}
            else:
                wf["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": self._clip_name, "type": self._clip_type}}

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

            if self._is_longcat:
                wf["13"] = {"class_type": "EmptyImage", "inputs": {"width": w, "height": h, "batch_size": 1, "color": 0}}
                wf["6"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": last_clip, "prompt": final_prompt, "vae": ["4", 0], "image1": ["13", 0]}}
                wf["11"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}}
                pos_cond = "11"
            else:
                wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": final_prompt, "clip": last_clip}}
                pos_cond = "6"

            if actual_cfg > 1.0:
                wf["14"] = {"class_type": "FluxGuidance", "inputs": {"conditioning": [pos_cond, 0], "guidance": actual_cfg}}
                positive_input = ["14", 0]
                ksampler_cfg = 1.0
            else:
                positive_input = [pos_cond, 0]
                ksampler_cfg = actual_cfg

            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": current_neg, "clip": last_clip}}
            
            model_node = last_model
            if self._is_longcat_turbo:
                wf["17"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": last_model, "shift": 3.1}}
                model_node = ["17", 0]

            wf["8"] = {
                "class_type": "KSampler",
                "inputs": {
                    "model": model_node, "positive": positive_input, "negative": ["7", 0],
                    "latent_image": ["1", 0], "seed": seed or int(time.time()) % 1000000,
                    "steps": actual_steps, "cfg": ksampler_cfg,
                    "sampler_name": sampler_name, "scheduler": scheduler, "denoise": 1.0
                }
            }
            wf["9"] = {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["4", 0]}}
            wf["10"] = {"class_type": "SaveImage", "inputs": {"images": ["9", 0], "filename_prefix": "GenFlux"}}


        try:
            r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
            pid = r.json().get("prompt_id")
            print(f"[GenFlux] Ejecutando: {self._flux_version} | Steps={actual_steps} | CFG={actual_cfg}")
            
            while True:
                r = requests.get(f"{comfy_url}/history/{pid}")
                if r.status_code == 200 and pid in r.json():
                    hist = r.json()[pid]
                    for node_out in hist.get("outputs", {}).values():
                        if "images" in node_out:
                            img_data = node_out["images"][0]
                            res = requests.get(f"{comfy_url}/view?filename={img_data['filename']}&subfolder={img_data.get('subfolder','')}&type={img_data.get('type','output')}")
                            return GenResult(image=Image.open(io.BytesIO(res.content)).convert("RGB"), final_prompt=final_prompt, time_taken=time.time()-start), "OK"
                time.sleep(1.5)
        except Exception as e:
            return None, str(e)

    def generate_ai(
        self,
        prompt: str,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:
        """
        Genera usando el sistema de Análisis Inteligente.
        Bypassea el rewriter de lenguaje si se solicita.
        """
        # Limpiar use_ai de kwargs si viene de la UI para evitar duplicados
        kwargs.pop("use_ai", None)
        
        # Extraer skip_rewrite si existe en kwargs, por defecto False para IA
        skip_rewrite = kwargs.pop("_skip_rewrite", False)
        
        return self.generate(
            prompt=prompt,
            use_ai=True,
            _skip_rewrite=skip_rewrite,
            **kwargs
        )

_gen_client = None
def get_flux_gen_client() -> FluxGenComfyClient:
    global _gen_client
    if _gen_client is None: _gen_client = FluxGenComfyClient()
    return _gen_client
