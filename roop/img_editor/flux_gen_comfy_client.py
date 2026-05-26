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
    """Cliente para generacion pura (txt2img) en ComfyUI usando FLUX/LongCat"""
    def __init__(self):
        self._loaded = False
        self._flux_version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
        self._clip_name = None
        self._clip_name2 = None
        self._clip_type = "flux"
        self._vae_name = "ae.safetensors"
        self._is_dual_clip = False
        self._is_longcat = False
        self._is_longcat_turbo = False

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
        """Configura los modelos a usar (con alias compatibles con Image Editor)"""
        
        # Mapeo de alias de Image Editor a archivos GGUF
        alias_map = {
            "longcat": "LongCat-Image-Edit-Turbo-Q4_K_S.gguf",
            "longcat_full": "LongCat-Image-Edit-Q4_K_S.gguf",
            "klein_base": "flux-2-klein-base-4b-Q4_K_S.gguf",
            "flux_q2": "flux1-dev-Q2_K.gguf",
            "flux_dev_abliterated": "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf",
        }
        
        # Si es un alias, convertirlo al nombre de archivo real
        real_model = alias_map.get(flux_version, flux_version)
        
        self._flux_version = real_model
        self._is_longcat = "LongCat" in real_model
        self._is_longcat_turbo = "Turbo" in real_model
        self._is_dual_clip = False
        self._clip_name2 = None

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
            # Dev / Abliterated
            self._clip_name = "clip_l.safetensors"
            self._clip_name2 = "t5-v1_1-xxl-encoder-Q4_K_S.gguf"
            self._clip_type = "flux"
            self._is_dual_clip = True
            self._vae_name = "ae.safetensors"

        # Verificar
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

    def _rewrite_prompt(self, prompt: str) -> Tuple[str, float]:
        """Traducción limpia y directa sin intervención creativa de LLM para máxima fidelidad"""
        try:
            # 1. Traducir de forma literal (español -> inglés)
            from roop.img_editor.prompt_translator import translate_prompt
            final_prompt = translate_prompt(prompt)
            
            # 2. Magnitud fija neutra (ya no afecta a los parámetros numéricos)
            magnitude = 0.5
            
            # 3. Formatear para LongCat si es necesario
            if self._is_longcat and not final_prompt.lower().startswith("instruction:"):
                final_prompt = f"Instruction: {final_prompt}"

            print(f"[GenFlux] Prompt Enviado: {final_prompt}")
            
            return final_prompt, magnitude
        except Exception as e:
            print(f"[GenFlux] Error en traducción: {e}")
            if self._is_longcat and not prompt.lower().startswith("instruction:"):
                prompt = f"Instruction: {prompt}"
            return prompt, 0.5

    def _rewrite_prompt_semantic(self, prompt: str) -> Tuple[str, float]:
        """Analiza semánticamente el prompt usando embeddings (como Image Editor)"""
        try:
            from roop.img_editor.prompt_translator import translate_prompt
            translated = translate_prompt(prompt)
            
            from roop.img_editor.nlp.semantic_analyzer import SemanticIntentAnalyzer
            nlp = SemanticIntentAnalyzer()
            magnitude = nlp.get_magnitude(prompt)
            
            # LongCat es un modelo instructivo
            if "LongCat" in self._flux_version and not translated.lower().startswith("instruction:"):
                translated = f"Instruction: {translated}"

            print(f"[GenFlux] Semantic embedding analysis: Mag={magnitude:.2f}")
            print(f"[GenFlux] Final Translated: {translated[:80]}...")
            
            return translated, magnitude
        except Exception as e:
            print(f"[GenFlux] Semantic embedding error, falling back: {e}")
            return self._rewrite_prompt(prompt)

    def _resolve_models(self) -> Optional[Tuple[str, str, str]]:
        """Resuelve los nombres de modelo, CLIP y VAE desde ComfyUI o modelo local."""
        comfy_url = get_comfyui_url()
        flux_version = self._flux_version

        # CLIP para LongCat
        clip_name = "qwen_2.5_vl_7b_fp8_scaled.safetensors"
        clip_type = "longcat_image"

        # VAE: flux2 para modelos flux2/flux-2, ae.safetensors para LongCat y otros (16 canales)
        if "flux2" in flux_version or "flux-2" in flux_version:
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

        if not self._loaded:
            self.load() # Cargar default si no se cargo nada

        start = time.time()

        # Dimensiones multiplo de 64
        width = (width // 64) * 64
        height = (height // 64) * 64

        flux_version = self._flux_version
        clip_name = self._clip_name
        clip_name2 = self._clip_name2
        clip_type = self._clip_type
        vae_name = self._vae_name

        # 1. Análisis Semántico y Traducción Dinámica
        final_prompt, magnitude = self._rewrite_prompt(prompt)

        # 2. Resolución Dinámica de Parámetros
        is_longcat = self._is_longcat
        is_longcat_turbo = self._is_longcat_turbo

        # 2. Resolución de Parámetros - RESPETO TOTAL A LA ENTRADA
        if steps is None:
            # Solo default si no se especifica. No escalamos nada.
            steps = 20 if not is_longcat_turbo else 8

        if guidance_scale is None:
            # Solo default si no se especifica.
            guidance_scale = 3.5 if not is_longcat_turbo else 1.0

        cfg = guidance_scale
        denoise = 1.0

        print(f"[GenFlux] Params Finales: Steps={steps}, CFG={cfg:.1f}")

        # Construir workflow
        wf = {
            "1": {
                "class_type": "EmptyLatentImage",
                "inputs": {"width": width, "height": height, "batch_size": 1}
            },
            "2": {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": flux_version}
            },
            "4": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name}
            },
        }

        # CLIP Loader
        if self._is_dual_clip:
            wf["3"] = {
                "class_type": "DualCLIPLoaderGGUF",
                "inputs": {"clip_name1": clip_name, "clip_name2": clip_name2, "type": clip_type}
            }
        else:
            wf["3"] = {
                "class_type": "CLIPLoader",
                "inputs": {"clip_name": clip_name, "type": clip_type}
            }

        # Prompt Encoding
        if is_longcat:
            wf["6"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": {
                    "clip": ["3", 0], "prompt": final_prompt, "vae": ["4", 0],
                    "image1": None, "image2": None, "image3": None
                }
            }
            wf["11"] = {
                "class_type": "FluxKontextMultiReferenceLatentMethod",
                "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}
            }
            positive_input = ["11", 0]
        else:
            wf["6"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": final_prompt, "clip": ["3", 0]}
            }
            positive_input = ["6", 0]

        wf["7"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": negative_prompt, "clip": ["3", 0]}
        }

        # Turbo Sampler Shift
        if is_longcat_turbo:
            wf["17"] = {
                "class_type": "ModelSamplingAuraFlow",
                "inputs": {"model": ["2", 0], "shift": 3.1}
            }
            model_input = ["17", 0]
        else:
            model_input = ["2", 0]

        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_input,
                "positive": positive_input,
                "negative": ["7", 0],
                "latent_image": ["1", 0],
                "seed": seed or int(time.time()) % 1000000,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler_ancestral",
                "scheduler": "simple",
                "denoise": denoise
            }
        }
        wf["9"] = {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["4", 0]}
        }
        wf["10"] = {
            "class_type": "SaveImage",
            "inputs": {"images": ["9", 0], "filename_prefix": "GenFlux"}
        }


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

    def generate_ai(
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
        """Genera con análisis semántico completo (como Image Editor)"""
        # 1. Traducir y Analizar con LLM (Qwen)
        final_prompt, magnitude = self._rewrite_prompt(prompt)
        
        # 2. Detectar parámetros recomendados
        from roop.img_editor.img_editor_manager import ImgEditorManager
        manager = ImgEditorManager()
        analysis = {"magnitude": magnitude, "prompt": final_prompt}
        
        # Determinar el engine semántico para auto_detect_params
        sem_engine = "longcat" if self._is_longcat else "flux_dev"
        if "schnell" in self._flux_version.lower(): sem_engine = "flux_schnell"
        
        params = manager.auto_detect_params(analysis, sem_engine)
        
        # Priorizar sliders de la UI si NO son None, sino usar los de la IA
        s = steps if steps is not None else params.get("num_inference_steps", 20)
        c = guidance_scale if guidance_scale is not None else params.get("guidance_scale", 3.5)
        
        print(f"[GenFlux] AI Generation: Mag={magnitude:.2f}, Final Steps={s}, Final CFG={c:.1f}")
        
        return self.generate(
            prompt=final_prompt,
            negative_prompt=negative_prompt,
            steps=s,
            guidance_scale=c,
            seed=seed,
            width=width,
            height=height
        )

_gen_client = None
def get_flux_gen_client() -> FluxGenComfyClient:
    global _gen_client
    if _gen_client is None:
        _gen_client = FluxGenComfyClient()
    return _gen_client
