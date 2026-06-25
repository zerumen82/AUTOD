#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import re
import time
import requests
import io
import json
from typing import Optional, Tuple, Dict, List
from PIL import Image
from dataclasses import dataclass

from roop.comfy_workflows import get_comfyui_url
from roop.img_editor.comfy_progress import wait_for_comfy_image

# Solo modelos SDXL NSFW instalados en este proyecto (checkpoints + diffusion_models)
GENERATION_MODEL_REGISTRY: List[Tuple[str, Tuple[str, ...], str, int, bool]] = [
    ("pony_realism", ("ponyrealism",), "PonyRealism (recomendado)", 0, True),
    ("juggernaut_xl", ("juggernaut",), "Juggernaut Ragnarok", 1, True),
    ("cyberrealistic_pony", ("cyberrealistic", "pony"), "CyberRealistic Pony v1.8", 2, False),
    ("talmendo_xl", ("talmendo",), "TalmendoXL", 3, True),
    ("helloworld_xl", ("helloworld",), "HelloWorld XL 7.0", 4, True),
    ("realism_engine", ("realismengine", "reprise"), "Realism Engine", 5, True),
    ("miamodel_nsfw", ("miamodel",), "Miamodel", 6, True),
    ("lazy_nsfw", ("lazymix", "realisticlazy"), "Lazy Mix", 7, True),
    ("nova_nsfw", ("novaillustrous",), "Nova", 8, True),
]

GENERATION_SKIP_CHECKPOINTS = ("framepack", "wan2", "qwenimage", "qwen_image")

GENERATION_ALIAS_FILES = {
    "pony_realism": "ponyRealism_V22.safetensors",
    "juggernaut_xl": "juggernautXL_ragnarok.safetensors",
    "cyberrealistic_pony": "cyberrealisticPony_v180Coreshift.safetensors",
    "talmendo_xl": "talmendoxlSDXL_v11Beta.safetensors",
    "helloworld_xl": "leosamsHelloworldXL_helloworldXL70.safetensors",
    "realism_engine": "realismEngineReprise4_50Fp8.safetensors",
    "miamodel_nsfw": "miamodelSFWNSFWSDXL_v30.safetensors",
    "lazy_nsfw": "realisticLazyMixNSFW_v10.safetensors",
    "nova_nsfw": "novaillustrousNSFW_v20.safetensors",
}


def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_models_base():
    return os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models")


def _checkpoint_search_dirs() -> List[str]:
    base = get_models_base()
    return [
        os.path.join(base, "checkpoints"),
        os.path.join(base, "diffusion_models"),
    ]


def _norm_model_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (name or "").lower())


def _list_generation_checkpoint_files() -> List[str]:
    found: Dict[str, str] = {}
    for folder in _checkpoint_search_dirs():
        if not os.path.isdir(folder):
            continue
        for entry in os.listdir(folder):
            if entry.lower().endswith(".safetensors"):
                key = entry.lower()
                found.setdefault(key, entry)
    return sorted(found.values(), key=str.lower)


def _match_checkpoint(filename: str, patterns: Tuple[str, ...], match_any: bool) -> bool:
    norm = _norm_model_name(os.path.splitext(filename)[0])
    hits = sum(1 for p in patterns if _norm_model_name(p) in norm)
    if match_any:
        return hits > 0
    return hits == len(patterns)


def _is_generation_checkpoint(filename: str) -> bool:
    norm = _norm_model_name(filename)
    return not any(skip in norm for skip in GENERATION_SKIP_CHECKPOINTS)


_CHECKPOINT_ARCH_CACHE: Dict[str, str] = {}


def _checkpoint_file_path(filename: str) -> Optional[str]:
    for folder in _checkpoint_search_dirs():
        path = os.path.join(folder, filename)
        if os.path.isfile(path):
            return path
    return None


def get_checkpoint_architecture(filename: str) -> str:
    """
    sdxl_full = checkpoint SDXL con CLIP integrado (válido para GENERAR).
    dit_unet  = solo UNet DiT/Flux (NO compatible con workflow SDXL actual).
    """
    key = (filename or "").lower()
    if key in _CHECKPOINT_ARCH_CACHE:
        return _CHECKPOINT_ARCH_CACHE[key]
    path = _checkpoint_file_path(filename)
    arch = "unknown"
    if path:
        try:
            from safetensors import safe_open
            with safe_open(path, framework="pt") as sf:
                keys = list(sf.keys())
            if any("conditioner.embedders" in k or "cond_stage_model" in k for k in keys):
                arch = "sdxl_full"
            elif any("cap_embedder" in k for k in keys):
                arch = "dit_unet"
            elif any("model.diffusion_model.input_blocks" in k for k in keys):
                arch = "unet_only"
        except Exception:
            arch = "unknown"
    _CHECKPOINT_ARCH_CACHE[key] = arch
    return arch


def _resolve_registry_alias(alias: str, files: List[str]) -> Optional[str]:
    for reg_alias, patterns, _label, _order, match_any in GENERATION_MODEL_REGISTRY:
        if reg_alias != alias:
            continue
        for filename in files:
            if _match_checkpoint(filename, patterns, match_any):
                return filename
        fallback = GENERATION_ALIAS_FILES.get(alias)
        if fallback:
            for filename in files:
                if filename.lower() == fallback.lower():
                    return filename
        return None
    return None


def resolve_generation_engine(engine_id: str) -> Tuple[str, Optional[str]]:
    """Devuelve (alias_config, nombre_archivo_checkpoint o None si no está instalado)."""
    engine_id = (engine_id or "pony_realism").strip()
    files = _list_generation_checkpoint_files()

    ckpt = _resolve_registry_alias(engine_id, files)
    if ckpt:
        return engine_id, ckpt

    if engine_id.lower().endswith(".safetensors"):
        for filename in files:
            if filename.lower() == engine_id.lower():
                return engine_id, filename
        return engine_id, None

    pony = _resolve_registry_alias("pony_realism", files)
    return "pony_realism", pony


def get_installed_generation_engines() -> List[Tuple[str, str]]:
    """Solo modelos del registro que existen en disco ahora mismo."""
    files = _list_generation_checkpoint_files()
    engines: List[Tuple[int, str, str]] = []

    for alias, patterns, label, order, match_any in GENERATION_MODEL_REGISTRY:
        ckpt = _resolve_registry_alias(alias, files)
        if ckpt and get_checkpoint_architecture(ckpt) == "sdxl_full":
            engines.append((order, label, alias))
        elif ckpt:
            print(f"[GenFlux] Omitido en UI (no SDXL completo): {ckpt}")

    engines.sort(key=lambda x: (x[0], x[1].lower()))
    return [(label, alias) for _order, label, alias in engines]


@dataclass
class GenResult:
    image: Image.Image
    final_prompt: str = ""
    user_translated: str = ""
    modifier_suffix: str = ""
    lora_applied: str = ""
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

    def _get_model_conf(self) -> Dict:
        conf = self._model_configs.get(self._alias)
        if conf:
            return conf
        if self._is_sdxl:
            return (
                self._model_configs.get("generic_nsfw_sdxl")
                or self._model_configs.get("pony_realism")
                or self._model_configs.get("default", {})
            )
        return self._model_configs.get("default", {})

    def is_available(self):
        try:
            r = requests.get(f"{get_comfyui_url()}/system_stats", timeout=5)
            return r.status_code == 200
        except:
            return False

    def _model_exists(self, subdir: str, name: str) -> bool:
        path = os.path.join(get_models_base(), subdir, name)
        return os.path.exists(path)

    def _lora_exists(self, name: str) -> bool:
        if not name or name == "None":
            return False
        return os.path.exists(os.path.join(get_models_base(), "loras", name))

    def _resolve_lora_stack(
        self,
        lora_name: str,
        lora_strength: float,
        conf: Dict,
        prompt_en: str = "",
        image_type: str = "auto",
    ) -> Tuple[List[Tuple[str, float]], str, str]:
        """Apila LoRAs automáticamente según escena del prompt (sin UI)."""
        from roop.img_editor.gen_lora_resolver import resolve_scene_loras, format_lora_log

        requested = (lora_name or "Auto").strip()
        auto_strength = float(conf.get("auto_lora_strength", 0.55))

        if requested not in ("", "None", "Auto"):
            strength = float(lora_strength) if lora_strength is not None else auto_strength
            log = f"{requested}@{strength:.2f}"
            return [(requested, strength)], "", log

        if requested == "None" or not (self._is_sdxl and conf.get("explicit", False)):
            return [], "", ""

        picks, boost = resolve_scene_loras(
            prompt_en, self._alias, base_strength=auto_strength, image_type=image_type,
        )
        log = format_lora_log(picks) if picks else ""
        if picks:
            print(f"[GenFlux] Auto LoRAs: {log}")
            if boost:
                print(f"[GenFlux] LoRA boost prompt: {boost}")
        else:
            print("[GenFlux] Auto LoRA: ninguna LoRA de escena encontrada")
        return [(f, s) for f, s, _lbl in picks], boost, log

    def _chain_lora_loaders(self, wf: Dict, last_model, last_clip, lora_stack: List[Tuple[str, float]], start_id: int = 20):
        from roop.img_editor.gen_lora_resolver import lora_has_clip_keys

        for i, (name, strength) in enumerate(lora_stack):
            nid = str(start_id + i)
            if lora_has_clip_keys(name):
                wf[nid] = {
                    "class_type": "LoraLoader",
                    "inputs": {
                        "model": last_model,
                        "clip": last_clip,
                        "lora_name": name,
                        "strength_model": strength,
                        "strength_clip": strength,
                    },
                }
                last_model = [nid, 0]
                last_clip = [nid, 1]
            else:
                wf[nid] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": last_model,
                        "lora_name": name,
                        "strength_model": strength,
                    },
                }
                last_model = [nid, 0]
        return last_model, last_clip

    def load(self, flux_version="pony_realism") -> Tuple[bool, str]:
        """Configura los modelos a usar"""
        flux_alias_map = {
            "longcat": "LongCat-Image-Edit-Turbo-Q4_K_S.gguf",
            "longcat_full": "LongCat-Image-Edit-Q4_K_S.gguf",
            "klein_base": "flux-2-klein-base-4b-Q4_K_S.gguf",
            "flux_q3": "flux1-dev-Q3_K_S.gguf",
            "flux_q2": "flux1-dev-Q2_K.gguf",
            "flux_dev_abliterated": "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf",
            "flux_dev_q4": "flux1-dev-Q4_K.gguf",
            "flux_dev": "flux1-dev-Q4_K.gguf",
            "schnell": "flux1-schnell-Q4_K_S.gguf",
            "hart": "hart",
        }

        if flux_version in flux_alias_map:
            self._alias = flux_version
            real_model = flux_alias_map[flux_version]
            self._is_sdxl = False
        else:
            config_alias, real_model = resolve_generation_engine(flux_version)
            self._alias = config_alias
            if not real_model:
                return False, (
                    f"Modelo '{config_alias}' no instalado. "
                    f"Copia el .safetensors a ui/tob/ComfyUI/models/checkpoints/ y reinicia."
                )
            arch = get_checkpoint_architecture(real_model)
            if arch == "dit_unet":
                return False, (
                    f"'{real_model}' es solo UNet DiT (sin CLIP). "
                    f"No funciona en GENERAR. Usa PonyRealism o Juggernaut."
                )
            if arch not in ("sdxl_full",) and real_model.lower().endswith(".safetensors"):
                return False, (
                    f"'{real_model}' no es un checkpoint SDXL completo. "
                    f"Elige otro modelo en el desplegable."
                )
            self._is_sdxl = arch == "sdxl_full"

        self._flux_version = real_model
        self._is_longcat = "LongCat" in real_model
        self._is_longcat_turbo = "Turbo" in real_model
        self._is_hart = "hart" in real_model.lower() or flux_version == "hart"
        self._is_dual_clip = False
        self._clip_name2 = None

        if self._is_hart:
            self._loaded = True
            return True, "Listo (HART autoregresivo - puro txt2img ligero)"

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

        if self._is_hart:
            # HART usa sus propios modelos (0.7B), no los checks estándar Flux
            self._loaded = True
            return True, "Listo (HART)"

        checks = [("diffusion_models", self._flux_version, "Modelo")]
        # Always require the standard clips for reliable pure T2I in Generate
        checks.append(("text_encoders", "clip_l.safetensors", "CLIP L"))
        checks.append(("text_encoders", "t5-v1_1-xxl-encoder-Q4_K_S.gguf", "T5"))
        # For longcat we also prefer its special clip if present (used in Image Editor)
        if self._clip_name and self._model_exists("text_encoders", self._clip_name):
            checks.append(("text_encoders", self._clip_name, "CLIP LongCat"))
        if self._is_dual_clip and self._clip_name2:
            checks.append(("text_encoders", self._clip_name2, "CLIP 2"))
        checks.append(("vae", self._vae_name, "VAE"))

        missing = []
        for d, f, l in checks:
            if not self._model_exists(d, f): missing.append(f)
        
        if missing:
            return False, f"Faltan archivos: {', '.join(missing)}"
        
        self._loaded = True
        return True, "Listo"

    def _prepare_prompt(self, prompt: str) -> str:
        """Básico para generate txt2img: translate (estilo Grok Imagine)."""
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

    def _prepare_prompt_intelligent(
        self,
        prompt: str,
        prompt_modifiers: Dict = None,
        image_type: str = "auto",
    ) -> Tuple[str, Dict]:
        """Prepara prompt para GENERAR (puro txt2img).
        1) Traduce solo el texto del usuario.
        2) Añade modificadores EN al final (sin traducir).
        3) Prefijo del modelo (amateur-aware si aplica).
        """
        try:
            from roop.img_editor.prompt_translator import translate_prompt
            from roop.img_editor.gen_prompt_modifiers import (
                assemble_generation_prompt,
                get_effective_negative,
            )

            user_text = (prompt or "").strip()
            base_translated = translate_prompt(user_text).strip()
            if base_translated.lower().startswith("instruction:"):
                base_translated = base_translated[12:].strip()
            translated = base_translated
            adult_intent = False

            try:
                from .prompt_rewriter import get_prompt_rewriter
                from roop.img_editor.gen_prompt_modifiers import rewriter_caption_trustworthy
                rewriter = get_prompt_rewriter()
                rw = rewriter.rewrite(
                    user_text,
                    image_context=f"Draft EN: {base_translated}",
                    mode="txt2img",
                )
                adult_intent = bool(rw.get("adult", False)) if rw else False
                rw_prompt = (rw.get("prompt") or "").strip() if rw else ""
                if rw_prompt and rewriter_caption_trustworthy(base_translated, rw_prompt):
                    translated = rw_prompt
                    print("[GenFlux] Rewriter caption aceptado")
                elif rw_prompt:
                    print("[GenFlux] Rewriter caption rechazado (alucinación/contradicción) → traducción base")
            except Exception as e:
                print(f"[GenFlux] Rewriter txt2img skipped: {e}")

            modifiers = dict(prompt_modifiers or {})
            if not modifiers.get("image_type"):
                modifiers["image_type"] = image_type or "auto"
            from roop.img_editor.gen_prompt_modifiers import resolve_effective_image_type
            effective_type = resolve_effective_image_type(
                modifiers.get("image_type", "auto") or "auto",
                self._alias,
                self._model_configs,
            )
            final_prompt, suffix, image_type = assemble_generation_prompt(
                self._alias,
                translated,
                modifiers,
                self._model_configs,
                is_sdxl=self._is_sdxl,
            )
            image_type = effective_type

            if self._is_longcat:
                final_prompt = f"Instruction: {final_prompt}"

            conf = self._get_model_conf()
            steps = conf.get("steps", 20)
            cfg = conf.get("cfg", 3.5)
            sampler_name = conf.get("sampler", "euler_ancestral")
            scheduler = conf.get("scheduler", "simple")
            neg = conf.get("negative_prompt",
                "lowres, blurry, deformed, bad anatomy, extra limbs, watermark, text, cartoon, 3d render, oversaturated, ugly, poorly drawn face")
            neg = get_effective_negative(
                self._alias, neg, modifiers, self._model_configs,
                prompt_en=translated,
            )
            from roop.img_editor.gen_prompt_modifiers import _model_explicit
            is_explicit = _model_explicit(self._alias, self._model_configs) or bool(conf.get("explicit"))

            p = {
                "steps": steps, "cfg": cfg, "neg": neg,
                "translated": translated, "suffix": suffix, "adult": adult_intent,
                "effective_image_type": image_type,
            }

            print(f"[GenFlux] [Smart] modelo={self._alias} steps={steps} cfg={cfg} sampler={sampler_name} scheduler={scheduler} tipo={image_type} explicit={is_explicit}")
            print(f"[GenFlux] [1/4] Usuario: {user_text}")
            print(f"[GenFlux] [2/4] Traducción base: {base_translated}")
            if translated != base_translated:
                print(f"[GenFlux] [2b/4] Caption final: {translated}")
            print(f"[GenFlux] [3/4] Añadidos UI: {suffix or '(ninguno)'}")
            from roop.img_editor.gen_prompt_modifiers import _estimate_clip_tokens
            print(f"[GenFlux] [4/4] Final → ComfyUI: {final_prompt}")
            print(f"[GenFlux] Tokens~{_estimate_clip_tokens(final_prompt)} | Neg~{_estimate_clip_tokens(neg)}")
            print(f"[GenFlux] Negativo: {neg}")
            return final_prompt, p

        except Exception as e:
            print(f"[GenFlux] Smart prepare fallback: {e}")
            try:
                from roop.img_editor.prompt_translator import translate_prompt
                translated = translate_prompt(prompt).strip()
                if self._is_longcat:
                    final_prompt = f"Instruction: {translated}"
                else:
                    final_prompt = translated
                    if final_prompt.lower().startswith("instruction:"):
                        final_prompt = final_prompt[12:].strip()
            except:
                final_prompt = prompt

            conf = self._get_model_conf()
            steps = conf.get("steps", 20)
            cfg = conf.get("cfg", 3.5)
            neg = conf.get("negative_prompt",
                "lowres, blurry, deformed, bad anatomy, extra limbs, watermark, text, cartoon, 3d render, oversaturated, ugly, poorly drawn face")
            return final_prompt, {"steps": steps, "cfg": cfg, "neg": neg}

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
        lora_name: str = "Auto",
        lora_strength: float = None,
        modifier_suffix: str = "",
        image_type: str = "auto",
        prompt_modifiers: Dict = None,
        progress_callback=None,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        comfy_url = get_comfyui_url()
        if not self.is_available(): return None, "ComfyUI no disponible"
        if not self._loaded: self.load()

        path_type = "PURE T2I (recomendado para GENERAR sin foto)" if not self._is_longcat else "EDIT (LongCat - mejor con foto de referencia)"
        print(f"[GenFlux] Usando path: {path_type} para {self._flux_version} (ultra realista SIN CENSURA, ligero para 8GB - usa 'schnell' para velocidad)")

        start = time.time()
        w, h = (width // 64) * 64, (height // 64) * 64

        def _report(phase: str, detail: str = "", pct: float = 0):
            if progress_callback:
                progress_callback({
                    "phase": phase,
                    "detail": detail,
                    "progress": pct / 100.0,
                    "step": 0,
                    "total": 0,
                    "eta": 0,
                    "elapsed": time.time() - start,
                })

        conf = self._get_model_conf()

        # 1. Resolución de Prompt, Parámetros y Negativo
        _report("Preparando prompt", f"Modelo: {self._alias}")
        current_neg = negative_prompt
        user_translated = ""
        mod_suffix_used = (modifier_suffix or "").strip()
        adult_intent = False
        ai_params: Dict = {}
        if use_ai and not _skip_rewrite:
            mods = prompt_modifiers or {
                "image_type": image_type or "auto",
            }
            final_prompt, ai_params = self._prepare_prompt_intelligent(
                prompt,
                prompt_modifiers=mods,
                image_type=image_type,
            )
            actual_steps = ai_params["steps"]
            actual_cfg = ai_params["cfg"]
            user_translated = ai_params.get("translated", "")
            mod_suffix_used = ai_params.get("suffix", mod_suffix_used)
            adult_intent = bool(ai_params.get("adult", False))
            if ai_params.get("neg"):
                current_neg = f"{ai_params['neg']}, {negative_prompt}"
        else:
            final_prompt = self._prepare_prompt(prompt) if not _skip_rewrite else prompt
            user_translated = final_prompt
            actual_steps = steps if steps is not None else conf.get("steps", 20)
            actual_cfg = guidance_scale if guidance_scale is not None else conf.get("cfg", 3.5)
            neg_conf = conf.get("negative_prompt")
            if neg_conf:
                current_neg = f"{neg_conf}, {negative_prompt}" if negative_prompt else neg_conf

        prompt_for_loras = user_translated or final_prompt or prompt
        effective_lora_type = "auto"
        if use_ai and not _skip_rewrite:
            effective_lora_type = ai_params.get("effective_image_type", "auto")
        elif prompt_modifiers:
            from roop.img_editor.gen_prompt_modifiers import resolve_effective_image_type
            effective_lora_type = resolve_effective_image_type(
                (prompt_modifiers or {}).get("image_type", "auto"),
                self._alias,
                self._model_configs,
            )
        lora_stack, lora_boost, lora_log = self._resolve_lora_stack(
            lora_name, lora_strength, conf, prompt_en=prompt_for_loras,
            image_type=effective_lora_type,
        )
        if lora_boost and lora_boost.lower() not in (final_prompt or "").lower():
            final_prompt = f"{final_prompt}, {lora_boost}"

        sampler_name = conf.get("sampler", "euler_ancestral")
        scheduler = conf.get("scheduler", "simple")

        if self._is_hart:
            try:
                from roop.img_editor.hart_edit_comfy_client import get_hart_edit_comfy_client
                hart_client = get_hart_edit_comfy_client()
                # HART carga sus modelos internamente en subprocess
                res, msg = hart_client.generate(
                    prompt=final_prompt,
                    num_inference_steps=actual_steps,
                    guidance_scale=actual_cfg,
                    seed=seed,
                    width=w,
                    height=h
                )
                if res:
                    return res, msg
                return None, msg or "HART generation failed"
            except Exception as e:
                return None, f"HART error: {str(e)}"

        if self._is_sdxl:
            wf = {
                "1": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                "2": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": self._flux_version}},
            }
            
            last_model = ["2", 0]
            last_clip = ["2", 1]
            if lora_stack:
                last_model, last_clip = self._chain_lora_loaders(wf, last_model, last_clip, lora_stack)

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
            # Workflow FLUX / LongCat  -- PURE T2I for Generate tab
            unet_loader_class = "UNETLoader" if self._flux_version.endswith(".safetensors") else "UnetLoaderGGUF"
            unet_inputs = {"unet_name": self._flux_version}
            if unet_loader_class == "UNETLoader":
                unet_inputs["weight_dtype"] = "default"

            wf = {
                "1": {"class_type": "EmptyLatentImage", "inputs": {"width": w, "height": h, "batch_size": 1}},
                "2": {"class_type": unet_loader_class, "inputs": unet_inputs},
                "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}},
            }

            # Choose correct text encoding depending on model
            if self._is_longcat:
                # LongCat-Image-Edit GGUF requires its own Qwen clip + edit encoder + a reference image
                # (even for pure gen we use a neutral reference image)
                wf["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": self._clip_name, "type": self._clip_type}}

                last_model = ["2", 0]
                last_clip = ["3", 0]

                if lora_stack:
                    last_model, last_clip = self._chain_lora_loaders(wf, last_model, last_clip, lora_stack)

                # Use a neutral gray image instead of pure black for slightly better behavior in pure gen
                wf["13"] = {"class_type": "EmptyImage", "inputs": {"width": w, "height": h, "batch_size": 1, "color": 128}}
                # Note: for LongCat in Generate we still have to use the edit encoder path because of embedding dimensions.
                # For maximum instruction following use the recommended standard FLUX model.
                wf["6"] = {"class_type": "TextEncodeQwenImageEditPlus", "inputs": {"clip": last_clip, "prompt": final_prompt, "vae": ["4", 0], "image1": ["13", 0]}}
                wf["11"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"}}
                pos_cond = "11"
            else:
                if self._clip_type == "flux2":
                    # FLUX.2 Klein: usa Qwen3 clip, sin FluxGuidance
                    wf["3"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": self._clip_name, "type": "flux2"}}
                else:
                    # Standard FLUX GGUF models
                    wf["3"] = {"class_type": "DualCLIPLoaderGGUF", "inputs": {"clip_name1": "clip_l.safetensors", "clip_name2": "t5-v1_1-xxl-encoder-Q4_K_S.gguf", "type": "flux"}}

                last_model = ["2", 0]
                last_clip = ["3", 0]

                if lora_stack:
                    last_model, last_clip = self._chain_lora_loaders(wf, last_model, last_clip, lora_stack)

                wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": final_prompt, "clip": last_clip}}
                pos_cond = "6"

            # FluxGuidance solo para FLUX.1 (no FLUX.2)
            if self._clip_type != "flux2" and actual_cfg > 1.0:
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
            _report("Enviando a ComfyUI", f"{w}×{h} · {actual_steps} pasos", pct=2)
            r = requests.post(f"{comfy_url}/prompt", json={"prompt": wf})
            pid = r.json().get("prompt_id")
            if not pid:
                return None, f"ComfyUI sin prompt_id: {r.text[:200]}"
            print(f"[GenFlux] Ejecutando: {self._flux_version} | Steps={actual_steps} | CFG={actual_cfg} | Sampler={sampler_name} | Scheduler={scheduler}")

            def _comfy_cb(prog):
                if progress_callback:
                    prog = dict(prog)
                    prog.setdefault("detail", lora_log or "")
                    progress_callback(prog)

            img_meta, wait_msg = wait_for_comfy_image(
                comfy_url,
                pid,
                steps_hint=actual_steps,
                progress_callback=_comfy_cb,
                cancel_check=kwargs.get("cancel_check"),
            )
            if not img_meta:
                return None, wait_msg

            res = requests.get(
                f"{comfy_url}/view?filename={img_meta['filename']}"
                f"&subfolder={img_meta.get('subfolder', '')}"
                f"&type={img_meta.get('type', 'output')}",
                timeout=60,
            )
            out_img = Image.open(io.BytesIO(res.content)).convert("RGB")
            try:
                from roop.img_editor.hyperreal_polish import polish_result_image
                out_img, polish_note, _ = polish_result_image(out_img, tier="hd")
                if polish_note and "omitido" not in polish_note:
                    print(f"[GenFlux] Post-acabado: {polish_note}")
            except Exception as e:
                print(f"[GenFlux] Post-acabado omitido: {e}")
            return GenResult(
                image=out_img,
                final_prompt=final_prompt,
                user_translated=user_translated,
                modifier_suffix=mod_suffix_used,
                lora_applied=lora_log,
                time_taken=time.time() - start,
            ), "OK"
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
