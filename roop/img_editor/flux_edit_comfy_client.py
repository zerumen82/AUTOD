#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import json
import time
import requests
import io
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


# A measured hole at strength 1 is empty noise. On the hair mask that noise
# became a second head (FastEdit_00841_). Refining an inpaint of the hole at
# this strength left the previous surface as a shadow (edit_1790869075.png).
# Region edits that insert a new face can still use it.
SURFACE_CONTINUATION_DENOISE = 0.60


def _appearance_sampling_config(recipe_key="appearance_edit"):
    path = os.path.join(get_project_root(), "config", "model_configs.json")
    with open(path, encoding="utf-8") as stream:
        model = json.load(stream)["longcat"]
    appearance = model[recipe_key]
    steps = int(model["steps"])
    denoise = float(appearance["denoise"])
    spatial_reference = appearance["spatial_reference"]
    if steps < 1 or not 0.0 < denoise <= 1.0 or not isinstance(spatial_reference, bool):
        raise ValueError("Invalid LongCat appearance sampling configuration")
    return steps, denoise, spatial_reference


APPEARANCE_SAMPLE_STEPS, APPEARANCE_SAMPLE_DENOISE, APPEARANCE_SPATIAL_REFERENCE = _appearance_sampling_config()
REGION_SAMPLE_STEPS, REGION_SAMPLE_DENOISE, REGION_SPATIAL_REFERENCE = _appearance_sampling_config("region_edit")


def measured_appearance_sample(has_mask: bool) -> dict:
    """One LongCat pass for a measured appearance or surface region.

    A measured region uses the configured native schedule and the original
    spatial reference. Full-strength noise without that reference added a
    head or hand in real edits. The measured mask still limits the change.
    """
    if has_mask:
        return {
            "num_inference_steps": APPEARANCE_SAMPLE_STEPS,
            "denoise": APPEARANCE_SAMPLE_DENOISE,
            "step_cap": APPEARANCE_SAMPLE_STEPS,
            "no_kontext": not APPEARANCE_SPATIAL_REFERENCE,
        }
    return {"num_inference_steps": 12, "denoise": 1.0, "no_kontext": True}


def measured_region_sample(has_mask: bool) -> dict:
    """Use the configured native recipe only for an actually measured region."""
    if has_mask:
        return {
            "num_inference_steps": REGION_SAMPLE_STEPS,
            "denoise": REGION_SAMPLE_DENOISE,
            "step_cap": REGION_SAMPLE_STEPS,
            "no_kontext": not REGION_SPATIAL_REFERENCE,
        }
    return measured_appearance_sample(False)


def longcat_turbo_denoise(preserve_frame, has_mask, keep_source_pixels, requested=1.0, prompt_authority=False):
    """Strength for LongCat Turbo.

    A measured region that keeps the photo pixels uses the requested strength.
    Region edits request 1.0. A measured appearance edit requests
    APPEARANCE_SAMPLE_DENOISE. A continuation of the surrounding surface
    passes SURFACE_CONTINUATION_DENOISE: strength 1 painted a new person into
    the hole, and 0.85 left the previous pixels (edit_1790790024.png). Body
    edits with prompt authority run without a mask, or with the measured
    subject mask keeping the photo pixels, at the requested strength; other
    unmasked preservation uses 0.55.
    """
    if prompt_authority and not has_mask:
        return requested
    if preserve_frame and not has_mask:
        return 0.55
    if preserve_frame and has_mask and not keep_source_pixels:
        return 0.85
    return requested


def _server_vram_free_mb(comfy_url):
    """v5.1133b — VRAM libre del servidor ComfyUI (free_mb, total_mb) o (None, None)."""
    try:
        _ss = requests.get(f"{comfy_url}/system_stats", timeout=10).json()
        _d = _ss.get("devices", [{}])[0]
        return int(_d.get("vram_free", 0)) / (1024 * 1024), int(_d.get("vram_total", 0)) / (1024 * 1024)
    except Exception:
        return None, None


def _sync_unload(comfy_url, reason: str) -> bool:
    """v5.1133b — unload SINCRONO con resultado VERIFICADO y LOG (antes silencioso)."""
    try:
        _r = requests.post(f"{comfy_url}/autoauto/unload", json={}, timeout=60)
        if _r.status_code == 200 and _r.json().get("ok"):
            _vf = _r.json().get("vram_free_mb", 0)
            _vt = _r.json().get("vram_total_mb", 0)
            print(f"[FluxClient] Unload sincrono ({reason}): OK — {_vf:.0f}MB libres de {_vt:.0f}MB", flush=True)
            return True
        print(f"[FluxClient] Unload sincrono ({reason}): endpoint respondió {_r.status_code} — fallback /free lazy", flush=True)
    except Exception as _e:
        print(f"[FluxClient] Unload sincrono ({reason}) FALLÓ: {_e} — fallback /free lazy", flush=True)
    try:
        requests.post(
            f"{comfy_url}/free",
            json={"unload_models": True, "free_memory": True},
            timeout=15,
        )
        return False
    except Exception:
        return False


def _queue_is_idle(comfy_url) -> bool:
    """No descargar pesos mientras ComfyUI atiende otra petición."""
    try:
        queue = requests.get(f"{comfy_url}/queue", timeout=5).json()
        return not queue.get("queue_running") and not queue.get("queue_pending")
    except Exception:
        return False

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
        except Exception:
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
        # v5.1094: verificar que esta instancia tenga nodos GGUF. Si el servidor
        # enganchó un ComfyUI ajeno (p.ej. puerto 8189 con Python del sistema),
        # el workflow falla con "UnetLoaderGGUF no encontrado". Error claro en vez
        # de un fallo críptico después.
        try:
            _oi = requests.get(f"{comfy_url}/object_info/UnetLoaderGGUF", timeout=5)
            if _oi.status_code != 200 or "UnetLoaderGGUF" not in _oi.json():
                raise ValueError("sin UnetLoaderGGUF")
        except Exception:
            return False, (
                f"El ComfyUI de {comfy_url} NO tiene nodos GGUF (falta UnetLoaderGGUF). "
                "Seguramente es otra instalación (p.ej. puerto 8189). Cerrá ese ComfyUI "
                "y reiniciá el servidor para que use el propio (puerto 8188)."
            )

        # v5.1133b — chequeo de VRAM ANTES del early-return de reutilización:
        # incluso con la MISMA versión, si el servidor quedó con la VRAM sucia
        # (stack residente + ESRGAN/CodeFormer del proceso principal), la run
        # siguiente hace thrash de preparación (450s+ observado en la 2ª petición
        # con el MISMO LongCat Full). Mejor pagar ~45-60s de recarga que 8min de
        # ping-pong VAE↔TE↔UNet.
        _pre_vram_free, _pre_vram_total = _server_vram_free_mb(comfy_url)
        # LongCat mantiene el UNet en VRAM tras la primera edición. En una
        # RTX 3060 Ti, el encoder de 7.9 GB de la siguiente petición compite
        # con ese UNet: 5198 MB libres parecían suficientes, pero el segundo
        # muestreo pasó de 7 s/paso a decenas de segundos/paso. Una recarga
        # limpia cuesta menos que ese intercambio continuo de pesos.
        _reuse_floor_mb = (
            5600 if flux_version.startswith("LongCat-") and _pre_vram_total is not None
            and _pre_vram_total <= 10000 else 2200
        )
        _low_vram = (
            _pre_vram_free is not None
            and _pre_vram_free < _reuse_floor_mb
        )
        _version_change = bool(getattr(self, '_flux_version', None) and self._flux_version != flux_version)

        if self._loaded and getattr(self, '_flux_version', None) == flux_version:
            if _low_vram and _queue_is_idle(comfy_url):
                _sync_unload(comfy_url, f"VRAM baja ({_pre_vram_free:.0f}MB libres) — misma versión, recarga limpia")
            else:
                if _pre_vram_free is not None:
                    _state = "cola ocupada; sin descarga" if _low_vram else "reutilizando modelo"
                    print(f"[FluxClient] VRAM servidor: {_pre_vram_free:.0f}MB libres de {_pre_vram_total:.0f}MB — {_state}", flush=True)
                return True, f"FLUX {flux_version} ya cargado (reutilizado)"
        elif self._loaded:
            # El cliente YA tiene un modelo, pero se pide otro. Dejar ambos
            # residentes en una RTX 3060 Ti de 8 GB provoca offload continuo:
            # el log parece avanzar, pero el primer paso puede tardar minutos.
            # `/free` no sirve aquí porque ComfyUI lo consume demasiado tarde;
            # el custom endpoint descarga SINCRÓNICAMENTE antes de encolar el
            # workflow nuevo.
            _vf, _vt = _server_vram_free_mb(comfy_url)
            if _vf is not None:
                print(f"[FluxClient] Cambio de versión {self._flux_version} -> {flux_version} (VRAM servidor antes: {_vf:.0f}MB de {_vt:.0f}MB)", flush=True)
            _sync_unload(
                comfy_url,
                f"cambio de versión {self._flux_version} -> {flux_version}",
            )
            self._loaded = False
        elif _version_change or _low_vram:
            # (a) cambio de modelo (v5.1132) o (b) primera carga del cliente con
            # VRAM sucia de una sesión anterior: descargar antes de cargar.
            _reason = "cambio de modelo" if _version_change else f"VRAM baja ({_pre_vram_free:.0f}MB libres)"
            _sync_unload(comfy_url, _reason)

        # v5.1131c — CAMBIO DE MODELO: diagnóstico de VRAM del servidor. NOTA
        # verificada en el fuente de este ComfyUI 0.19.0: /free es LAZY (server.py
        # l.1016 solo pone flags en la cola; main.py l.341 los consume tras
        # ejecutar el próximo prompt). El unload REAL sucede si la run ANTERIOR
        # envió /free al completar (_free_server_vram en generate). Aquí solo
        # medimos y avisamos: si la run anterior no lo envió (p.ej. crash o 1ª
        # run tras restart con cola), la nueva carga convive con el modelo viejo.

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
        self._is_klein_9b = False
        self._is_klein_4b = False
        # Reset total de flags de motor: el singleton se reutiliza entre runs
        # con versiones distintas (Turbo/Kontext/Klein). Sin esto, el flag del
        # run anterior sobrevivía y generate() entraba en la rama de workflow
        # equivocada (1º bien, siguientes deformados).
        self._is_longcat_turbo = False
        self._is_qwen_image_edit = False
        self._is_qwen_q5 = False
        self._is_kontext_dev = False

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

        if "klein-9b" in flux_version:
            self._is_klein_9b = True
        elif "klein" in flux_version:
            self._is_klein_4b = True

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
        # v5.1091: el ruido puro del bug ReferenceLatent daba std global ~5 con
        # spread de color alto (estática RGB) → el gate viejo no lo cazaba.
        # Una foto natural tiene std global ~40-70. std < 12 = gris/ruido seguro.
        if float(arr.std()) < 12.0:
            return True
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
        prompt_authority: bool = False,
        **kwargs
    ) -> Tuple[Optional[GenResult], str]:

        if not self._loaded:
            return None, "No cargado - llama a load() primero"

        # También aceptado vía kwargs por llamadas genéricas del manager
        drop_edit_image_ref = bool(drop_edit_image_ref or kwargs.get("drop_edit_image_ref", False))
        no_kontext = bool(no_kontext or kwargs.get("no_kontext", False))
        prompt_authority = bool(prompt_authority or kwargs.get("prompt_authority", False))
        preserve_frame = bool(kwargs.get("preserve_frame", False))
        # The inpaint encoder paints the hole flat gray. On a measured head
        # that gray is what comes back (edit_1790789518.png). Keeping the
        # photo pixels and noising only the mask leaves the face for the
        # sampler to edit.
        keep_source_pixels = bool(kwargs.get("keep_source_pixels", False))
        # v5.1104 — img2img PURO para uncover (solo lo pasa el manager en
        # kontext+uncover): KSampler sobre el latente de la foto con denoise
        # alto, SIN ReferenceLatent. None = modo EDIT (ReferenceLatent).
        _img2img_denoise = kwargs.get("kontext_img2img_denoise", None)
        try:
            _img2img_denoise = float(_img2img_denoise) if _img2img_denoise else None
        except (TypeError, ValueError):
            _img2img_denoise = None
        if _img2img_denoise:
            _img2img_denoise = max(0.80, min(0.90, _img2img_denoise))
        ref_method = reference_latents_method or "index_timestep_zero"
        if no_kontext:
            ref_method = "none"
        # ControlNet Pose (garantía de postura)
        cn_pose_image = kwargs.get("controlnet_pose_image", None)
        cn_strength = float(kwargs.get("controlnet_strength", 0.75))

        # Prompt negativo base (mínimo para calidad)
        if not negative_prompt:
            negative_prompt = "low quality, blurry, distorted, low resolution"

        is_longcat = "LongCat" in self._flux_version
        is_kontext = self._is_kontext_dev
        is_qwen = self._is_qwen_image_edit or self._is_qwen_q5
        is_klein_9b = getattr(self, '_is_klein_9b', False)
        is_klein_4b = getattr(self, '_is_klein_4b', False)
        is_klein_edit = is_klein_9b or is_klein_4b
        # ImageEditor Klein = REAL EDIT (ReferenceLatent): latente vacío + la imagen
        # de referencia se inyecta como conditioning("reference_latents") → el modelo
        # regenera con composición/identidad de la ref aplicando la instrucción.

        klein_empty_latent = None

        t0 = time.time()
        image = image.convert("RGB")
        w, h = image.size

        if is_longcat:
            # v5.1125 — LongCat (Qwen-VL) opera a 1 Megapíxel (1024*1024 píxeles totales).
            # TextEncodeQwenImageEditPlus en nodes_qwen.py reescala internamente los ref_latents
            # usando: scale_by = sqrt((1024*1024) / (w * h)).
            # Con thumbnail((1024, 1024)) una imagen vertical (p.ej. 640x1024) tenía solo 655k px,
            # forzando a Qwen a escalar ref_latents a 808x1296 (1.265x) mientras EmptySD3LatentImage
            # permanecía a 640x1024 → recorte forzado del 26.5% y zoom en formatos no cuadrados.
            # Al normalizar la imagen a 1MP preservando su relación de aspecto nativa (múltiplos de 16):
            # scale_by en Qwen es exactamente 1.0 y ref_latents coincide 1:1 con el canvas.
            total_px = 1024 * 1024
            scale = (total_px / float(w * h)) ** 0.5
            new_w = round(w * scale / 16.0) * 16
            new_h = round(h * scale / 16.0) * 16
            new_w = max(512, min(1536, new_w))
            new_h = max(512, min(1536, new_h))
            print(
                f"[FluxClient] Resolución efectiva LongCat: {w}x{h} -> {new_w}x{new_h} "
                f"({new_w * new_h / 1_000_000:.2f} MP)",
                flush=True,
            )
            image = image.resize((new_w, new_h), Image.LANCZOS)
        else:
            new_w, new_h = (w // 64) * 64, (h // 64) * 64
            if new_w != w or new_h != h:
                image = image.resize((new_w, new_h), Image.LANCZOS)
            # Kontext hasta 1024px; Klein 4B/9B hasta 1024px; Qwen-Image-Edit 640px; otros 768px
            # v5.1144 — Kontext 1024→768px: el UNet Q4_K_M (6.6GB) NO cabe en 8GB
            # (5642MB loaded + 1004MB offloaded) y el coste de sampling escala ~cuadrático
            # con los píxeles. Log real (2026-09-15): 9.1→48 s/it (×5.2 de degradación
            # intra-run, +2.4 s/it por paso) y run completo de 10+ min. A 768px (~56% de
            # los píxeles) el run baja a ~la mitad manteniendo la adherencia del Q4.
            max_dim = 768 if is_kontext else (1024 if is_klein_9b or is_klein_4b else (640 if is_qwen else 768))
            if new_w > max_dim or new_h > max_dim:
                image.thumbnail((max_dim, max_dim), Image.LANCZOS)
                new_w, new_h = image.size
                new_w, new_h = (new_w // 64) * 64, (new_h // 64) * 64
                image = image.resize((new_w, new_h), Image.LANCZOS)

        iname = f"fast_{int(t0)}.png"
        buf = io.BytesIO()
        image.save(buf, "PNG")
        buf.seek(0)
        comfy_url = get_comfyui_url()
        try:
            r = requests.post(
                f"{comfy_url}/upload/image",
                files={"image": (iname, buf, "image/png")},
                timeout=60,
            )
        except requests.RequestException as exc:
            return None, f"Error subiendo imagen a ComfyUI: {exc}; no se reintenta automáticamente"
        if r.status_code != 200:
            detail = (r.text or "").strip().replace("\n", " ")[:160]
            return None, (
                f"Error subiendo imagen a ComfyUI ({r.status_code}): {detail}; "
                "no se reintenta automáticamente"
            )

        mname = None
        if is_klein_edit or is_kontext:
            # El EDIT real (ReferenceLatent) usa la imagen COMPLETA como referencia
            # y arranca de latente vacío — una máscara de inpaint no puede aplicar
            # aquí (rompería el conditioning). Ignorada siempre en klein/kontext.
            if mask_image is not None:
                print(
                    "[FluxClient] Máscara recibida pero no compatible con el workflow "
                    f"{'Kontext' if is_kontext else 'Klein'}; se omite",
                    flush=True,
                )
            mask_image = None
        if mask_image:
            mname = f"mask_{int(t0)}.png"
            mbuf = io.BytesIO()
            mask_image.convert("L").resize((new_w, new_h)).save(mbuf, "PNG")
            mbuf.seek(0)
            requests.post(f"{comfy_url}/upload/image", files={"image": (mname, mbuf, "image/png")})

        # Appearance only. Qwen copies a large measured surface from image1
        # (edit_1790892396). The extra view covers that surface. Node 5
        # keeps encoding the photo.
        surface_reference = kwargs.get("reference_image")
        use_surface_reference = (
            is_longcat
            and no_kontext
            and surface_reference is not None
            and prompt_authority
            and not drop_edit_image_ref
            and mask_image is not None
            and keep_source_pixels
        )
        reference_name = None
        if use_surface_reference:
            reference_name = f"ref_{int(t0)}.png"
            rbuf = io.BytesIO()
            surface_reference.convert("RGB").resize((new_w, new_h), Image.LANCZOS).save(rbuf, "PNG")
            rbuf.seek(0)
            uploaded = requests.post(
                f"{comfy_url}/upload/image",
                files={"image": (reference_name, rbuf, "image/png")},
                timeout=60,
            )
            if uploaded.status_code != 200:
                reference_name = None

        wf = {
            "1": {"class_type": "LoadImage", "inputs": {"image": iname, "upload": "image"}},
            "2": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": self._flux_version, "device": "default"}},
            "3": {"class_type": "DualCLIPLoaderGGUF" if self._dual_clip else ("CLIPLoaderGGUF" if self._is_gguf_clip else "CLIPLoader"), "inputs": {"clip_name1": self._clip_name, "clip_name2": self._clip_name2, "type": self._clip_type} if self._dual_clip else {"clip_name": self._clip_name, "type": self._clip_type, "device": "default"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": self._vae_name}},
        }
        if reference_name:
            wf["1r"] = {
                "class_type": "LoadImage",
                "inputs": {"image": reference_name, "upload": "image"},
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
        if is_kontext:
            # v5.1099 — FLUX.1 Kontext dev es un MODELO DE EDICIÓN, no img2img.
            # Workflow nativo: VAEEncode de la foto SOLO como ReferenceLatent en el
            # conditioning + KSampler desde latente VACÍO + FluxGuidance (2.0-5.0).
            # Antes (img2img estándar): denoise alto destruía la foto (no hacía caso
            # al prompt) y denoise bajo la congelaba (ignoraba el prompt).
            # Con ReferenceLatent la imagen SIEMPRE se preserva como ancla y el
            # guidance es el único control de adherencia — como el dev original.
            actual_denoise = 1.0  # irrelevante: arranca de latente vacío
            actual_cfg = 1.0      # negative vía FluxGuidance, no CFG
            num_inference_steps = max(20, min(num_inference_steps, 28))
            # guidance_scale del manager (2.5-4.5) mapea directo al rango nativo
            # de Kontext dev (FluxGuidance 2.0-5.0).
            _kg = max(2.0, min(5.0, float(guidance_scale)))
            # Referencia: VAEEncode de la foto real (el foto real, no el vacío).
            # ref_method "none" (no_kontext) significa también sin latente-ref.
            _use_img_ref = ref_method != "none"
        elif is_klein_edit:
            # FLUX.2 Klein EDIT real: latente VACÍO + ReferenceLatent ( refs en
            # conditioning) → denoise 1.0 obligatorio, 4 pasos nativos, CFG 1.0.
            actual_cfg = 1.0
            actual_denoise = 1.0
            num_inference_steps = min(num_inference_steps, 4)
        elif is_longcat and self._is_longcat_turbo:
            actual_cfg = 1.0
            try:
                requested_denoise = float(denoise)
            except (TypeError, ValueError):
                requested_denoise = 1.0
            if not 0.0 < requested_denoise <= 1.0:
                requested_denoise = 1.0
            actual_denoise = longcat_turbo_denoise(
                preserve_frame, mask_image is not None, keep_source_pixels, requested_denoise,
                prompt_authority=prompt_authority,
            )
            # 12 steps makes denoise 0.85 almost a full redraw and 0.65 leave
            # the previous surface. A longer schedule lets a mid strength
            # change that surface while the photo around the mask stays.
            try:
                step_cap = int(kwargs.get("step_cap", 12) or 12)
            except (TypeError, ValueError):
                step_cap = 12
            num_inference_steps = min(num_inference_steps, max(4, min(step_cap, 28)))
        elif is_longcat and no_kontext:
            # Edición fuerte de LongCat. La foto sigue siendo el latente inicial
            # y la referencia visual: un latente vacío convierte el edit en
            # text-to-image y degrada composición, color y textura.
            # v5.1126 — CFG clamp 1.5-2.0 era DEMASIADO BAJO: en bypass el TEXTO es
            # la única autoridad y el CFG bajo diluyó la instrucción (log 11:37:
            # 'the boy must be bald' con cfg=2.0 → no ejecutó). La receta oficial
            # Full es CFG 2.0-3.5; CFGNorm (nodo 17) estabiliza la norma del latente,
            # así que el rango alto NO quema colores. Clamp 2.0-3.5.
            # v5.1149 — REVERSIÓN parcial de v5.1148: el CFG 3.0 da piel de
            # cera/muñeco (el CFG alto aplana la micro-textura de piel; el
            # usuario reporta "parecen muñecos y no personas"). La textura se
            # gana con steps (suelo 36 intacto) + el acabado con grano, no con
            # CFG. Vuelta a 2.0-2.5, receta que no plastifica. CFGNorm sigue.
            actual_cfg = max(2.0, min(2.5, float(guidance_scale))) if guidance_scale else 2.5
            actual_denoise = 1.0
            # v5.1126 — 24-30 steps eran cortos para following complejo (log 11:37:
            # 'EL CHICO ES CALVO' no ejecutado a 29 steps). 36-40 pasos Full.
            num_inference_steps = max(36, min(num_inference_steps, 40))
        else:
            actual_cfg = guidance_scale
            actual_denoise = denoise

        if is_kontext:
            # v5.1099 — Workflow EDIT nativo de Kontext dev (guía oficial ComfyUI):
            #   photo → VAEEncode → ReferenceLatent(conditioning) + FluxGuidance
            #   KSampler desde latente VACÍO del tamaño de la foto.
            # Antes era img2img estándar (denoise<1.0 anclando el latente original):
            #   denoise alto = foto destruida, denoise bajo = prompt ignorado.
            wf["5d"] = {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}}
            wf["5e"] = {"class_type": "EmptySD3LatentImage", "inputs": {"width": ["5d", 0], "height": ["5d", 1], "batch_size": 1}}
            model_input = last_model
            wf["6"] = {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": prompt, "clip": last_clip},
            }
            wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
            wf["6g"] = {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["6", 0], "guidance": _kg},
            }
            if _img2img_denoise and _use_img_ref:
                # v5.1104 — img2img PURO (solo uncover): la evidencia (Q2+offset
                # =copia, Q4+offset=copia, Q4+uxo=débil sin cambio) demuestra que
                # CUALQUIER ReferenceLatent con tokens limpios ancla demasiado
                # para quitar ropa: el modelo siempre resuelve "foto + desnuda"
                # a favor de la foto. Sin tokens limpios, el latente con ruido
                # (denoise 0.80-0.90) deja regenerar la ropa manteniendo layout
                # aproximado. La cara puede derivar (preferencia del usuario:
                # adherencia > preservación facial).
                actual_denoise = _img2img_denoise
                positive_input = ["6g", 0]
                klein_empty_latent = None  # KSampler usa ["5", 0] (foto real)
            elif _use_img_ref:
                wf["6r"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6g", 0], "latent": ["5", 0]}}
                # v5.1101 — método de referencia EXPLÍCITO (nodo 6m tras 6r).
                # Sin este nodo el DiT usa default_ref_method="offset" implícito.
                # v5.1102: el método lo decide el manager (uxo/uno u offset);
                # el valor por defecto aquí es "offset" = default oficial.
                _ctx_method = ref_method if ref_method in ("offset", "index", "uxo/uno", "index_timestep_zero") else "offset"
                wf["6m"] = {"class_type": "FluxKontextMultiReferenceLatentMethod", "inputs": {"conditioning": ["6r", 0], "reference_latents_method": _ctx_method}}
                positive_input = ["6m", 0]
                klein_empty_latent = ["5e", 0]
            else:
                # no_kontext: sin ancla de imagen (el texto manda en todo)
                positive_input = ["6g", 0]
                klein_empty_latent = ["5e", 0]
            negative_input = ["7", 0]
        elif is_longcat:
            # LongCat no debe usar FluxKontextImageScale porque recorta al centro ("center" crop)
            # y fuerza buckets fijos (672..1568) provocando zoom y recorte de los bordes.
            # Qwen-VL / LongCat procesa ["1", 0] directamente a la resolución nativa.
            if self._is_longcat_turbo:
                wf["17"] = {"class_type": "ModelSamplingAuraFlow", "inputs": {"model": last_model, "shift": 3.1}}
                model_input = ["17", 0]
            elif actual_cfg > 1.0:
                # CFGNorm estabiliza la norma del latente en CFG > 1.0 evitando colores quemados
                wf["17"] = {"class_type": "CFGNorm", "inputs": {"model": last_model, "strength": 1.0}}
                model_input = ["17", 0]
            else:
                model_input = last_model

            if no_kontext:
                # Edición fuerte guiada por referencia. La foto sigue entrando en
                # el conditioning y en el sampler: sin ello LongCat genera una
                # escena nueva de baja fidelidad en lugar de editar la existente.
                # prompt_authority es la excepción: Qwen-VL sigue viendo la foto,
                # pero el latente de referencia (método index) copiaba la pose y
                # el cuerpo y la instrucción no se aplicaba.
                if prompt_authority and not drop_edit_image_ref and mask_image is not None:
                    # The measured subject is repainted. The sampler must use
                    # the photo latent with that mask, not an empty frame.
                    qwen_image = ["1", 0]
                    if reference_name:
                        qwen_image = ["1r", 0]
                        print(
                            "[FluxClient] Qwen y el latente ven la superficie pedida cubierta "
                            "con el color de alrededor; fuera de la zona sigue la foto",
                            flush=True,
                        )
                    print(
                        "[FluxClient] LongCat edición fuerte: sujeto medido, el resto de la foto se conserva",
                        flush=True,
                    )
                    wf["6"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": prompt,
                            "image1": qwen_image,
                        },
                    }
                    if self._is_longcat_turbo:
                        positive_input = ["6", 0]
                        negative_input = positive_input
                    else:
                        wf["7"] = {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": negative_prompt, "clip": last_clip},
                        }
                        positive_input = ["6", 0]
                        negative_input = ["7", 0]
                    klein_empty_latent = None
                elif prompt_authority and preserve_frame and not drop_edit_image_ref:
                    print(
                        "[FluxClient] LongCat edición fuerte: foto conservada, sujeto no medido",
                        flush=True,
                    )
                    wf["6"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": prompt,
                            "image1": ["1", 0],
                        },
                    }
                    if self._is_longcat_turbo:
                        positive_input = ["6", 0]
                        negative_input = positive_input
                    else:
                        wf["7"] = {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": negative_prompt, "clip": last_clip},
                        }
                        positive_input = ["6", 0]
                        negative_input = ["7", 0]
                    klein_empty_latent = None
                elif prompt_authority and not drop_edit_image_ref:
                    print("[FluxClient] LongCat edición fuerte: Qwen ve la foto, sin ancla de latente", flush=True)
                    wf["5d"] = {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}}
                    wf["5e"] = {
                        "class_type": "EmptySD3LatentImage",
                        "inputs": {"width": ["5d", 0], "height": ["5d", 1], "batch_size": 1},
                    }
                    wf["6"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": prompt,
                            "image1": ["1", 0],
                        },
                    }
                    if self._is_longcat_turbo:
                        positive_input = ["6", 0]
                        negative_input = positive_input
                    else:
                        wf["7"] = {
                            "class_type": "CLIPTextEncode",
                            "inputs": {"text": negative_prompt, "clip": last_clip},
                        }
                        positive_input = ["6", 0]
                        negative_input = ["7", 0]
                    klein_empty_latent = ["5e", 0]
                elif drop_edit_image_ref:
                    print("[FluxClient] LongCat edición fuerte sin imagen de referencia")
                    wf["6"] = {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": last_clip}}
                    wf["7"] = {"class_type": "CLIPTextEncode", "inputs": {"text": negative_prompt, "clip": last_clip}}
                    positive_input = ["6", 0]
                    negative_input = ["7", 0]
                else:
                    print("[FluxClient] LongCat edición fuerte: imagen inicial + referencia index")
                    wf["6"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": prompt,
                            "vae": ["4", 0],
                            "image1": ["1", 0],
                            "image2": None,
                            "image3": None,
                        },
                    }
                    wf["7"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": negative_prompt,
                        },
                    }
                    wf["11"] = {
                        "class_type": "FluxKontextMultiReferenceLatentMethod",
                        "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index"},
                    }
                    wf["12"] = {
                        "class_type": "FluxKontextMultiReferenceLatentMethod",
                        "inputs": {"conditioning": ["7", 0], "reference_latents_method": "index"},
                    }
                    positive_input = ["11", 0]
                    negative_input = ["12", 0]
                    # La foto ya entra por el conditioning positivo y por el
                    # latente inicial del sampler. Repetir VAE+Qwen con la misma
                    # imagen en el negativo duplica el trabajo de 1 MP y, en
                    # 8 GB, puede provocar intercambio continuo TE/UNet en la
                    # segunda edición fuerte. La rama negativa solo necesita
                    # el texto, igual que la ruta LongCat Full normal.
                # El nodo 5 (VAEEncode de la foto) se crea al final del workflow.
                # El sampler debe usarlo y no arrancar de EmptySD3LatentImage.
                klein_empty_latent = None
            else:
                wf["6"] = {
                    "class_type": "TextEncodeQwenImageEditPlus",
                    "inputs": {
                        "clip": last_clip,
                        "prompt": prompt,
                        "vae": ["4", 0],
                        "image1": ["1", 0],
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
                positive_input = ["11", 0]
                if self._is_longcat_turbo:
                    # Turbo usa CFG=1.0: en classifier-free guidance la rama
                    # incondicional tiene peso cero. Codificar de nuevo foto +
                    # Qwen para el negativo no cambia el resultado, pero en 8GB
                    # obliga a alternar VAE/Qwen y añade varios minutos de carga.
                    # Reusar el conditioning positivo conserva el workflow de
                    # edición con referencia sin convertirlo en img2img.
                    negative_input = positive_input
                    print("[FluxClient] LongCat Turbo CFG=1: omitida codificación negativa redundante", flush=True)
                else:
                    # Full sí usa CFG > 1, así que conserva una rama negativa
                    # real. La referencia visual ya viaja íntegra en positivo;
                    # repetir foto+VAE en negativo no añade información y, con
                    # 8GB, fuerza el ping-pong VAE↔Qwen que bloqueaba minutos.
                    # El encoder Qwen recibe el texto negativo en su formato
                    # nativo, pero no una segunda imagen de referencia.
                    wf["7"] = {
                        "class_type": "TextEncodeQwenImageEditPlus",
                        "inputs": {
                            "clip": last_clip,
                            "prompt": negative_prompt,
                        },
                    }
                    wf["12"] = {
                        "class_type": "FluxKontextMultiReferenceLatentMethod",
                        "inputs": {"conditioning": ["7", 0], "reference_latents_method": ref_method},
                    }
                    negative_input = ["12", 0]
                    print("[FluxClient] LongCat Full: negativo textual; referencia visual codificada una vez", flush=True)
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
                # KSampler arranca de latente VACÍO (ruido) del tamaño de la imagen,
                # y la FOTO REAL se inyecta como conditioning vía ReferenceLatent
                # (tokens de referencia = VAEEncode de la imagen ["5", 0]).
                # v5.1091 FIX: antes se pasaba ["5e", 0] (vacío) al ReferenceLatent
                # → el modelo nunca veía la foto → ruido puro (std~5). Ahora ["5",0].
                # (flujo oficial de FLUX.2 Klein edit / one-node)
                wf["5d"] = {"class_type": "GetImageSize", "inputs": {"image": ["1", 0]}}
                wf["5e"] = {"class_type": "EmptyFlux2LatentImage", "inputs": {"width": ["5d", 0], "height": ["5d", 1], "batch_size": 1}}
                wf["6r"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["6", 0], "latent": ["5", 0]}}
                wf["7r"] = {"class_type": "ReferenceLatent", "inputs": {"conditioning": ["7", 0], "latent": ["5", 0]}}
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

        # ControlNet Pose: aplicar condicionamiento de postura si se pasa pose map
        if cn_pose_image is not None:
            try:
                cn_name = f"pose_{int(t0)}.png"
                cbuf = io.BytesIO()
                if hasattr(cn_pose_image, 'convert'):
                    cn_pose_image.convert("RGB").resize((new_w, new_h)).save(cbuf, "PNG")
                else:
                    Image.fromarray(cn_pose_image).convert("RGB").resize((new_w, new_h)).save(cbuf, "PNG")
                cbuf.seek(0)
                requests.post(f"{comfy_url}/upload/image", files={"image": (cn_name, cbuf, "image/png")})
                wf["cn_img"] = {"class_type": "LoadImage", "inputs": {"image": cn_name, "upload": "image"}}
                wf["cn_load"] = {"class_type": "ControlNetLoader", "inputs": {"control_net_name": "flux-union-pro2-fp8.safetensors"}}
                wf["cn_apply"] = {
                    "class_type": "ControlNetApplyAdvanced",
                    "inputs": {
                        "positive": positive_input, "negative": negative_input,
                        "control_net": ["cn_load", 0], "image": ["cn_img", 0],
                        "vae": ["4", 0],
                        "strength": max(0.1, min(1.0, cn_strength)),
                        "start_percent": 0.0, "end_percent": 1.0,
                    },
                }
                positive_input = ["cn_apply", 0]
                negative_input = ["cn_apply", 1]
                print(f"[FluxClient] ControlNet Pose activo (strength={cn_strength:.2f})", flush=True)
            except Exception as e:
                print(f"[FluxClient] ControlNet Pose falló, sin ControlNet: {e}", flush=True)

        wf["8"] = {
            "class_type": "KSampler",
            "inputs": {
                "model": model_input, "positive": positive_input, "negative": negative_input,
                "latent_image": klein_empty_latent or ["5", 0], "seed": seed or int(t0) % 1000000,
                "steps": num_inference_steps, "cfg": actual_cfg,
                "sampler_name": ("er_sde" if is_klein_edit else ("euler" if (is_qwen or is_kontext or is_longcat) else "euler_ancestral")),
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

        mask_mode = "global"
        if mname:
            wf["14"] = {"class_type": "LoadImage", "inputs": {"image": mname, "upload": "image"}}
            wf["15"] = {"class_type": "ImageToMask", "inputs": {"image": ["14", 0], "channel": "red"}}
            if keep_source_pixels and klein_empty_latent is None:
                # The covered view is the latent only inside the measured
                # zone: outside that mask the cover is the original photo.
                # Starting from the photo kept a large hair mass
                # (edit_1790893724, luminance 42 to 48).
                source_pixels = ["1r", 0] if reference_name else ["1", 0]
                wf["5"] = {
                    "class_type": "VAEEncode",
                    "inputs": {
                        "pixels": source_pixels,
                        "vae": ["4", 0],
                    },
                }
                wf["16"] = {
                    "class_type": "SetLatentNoiseMask",
                    "inputs": {
                        "samples": ["5", 0],
                        "mask": ["15", 0],
                    },
                }
                wf["8"]["inputs"]["latent_image"] = ["16", 0]
                mask_mode = "foto"
            else:
                wf["5"] = {
                    "class_type": "VAEEncodeForInpaint",
                    "inputs": {
                        "pixels": ["1", 0],
                        "vae": ["4", 0],
                        "mask": ["15", 0],
                        "grow_mask_by": max(0, int(grow_mask_by)),
                    },
                }
                mask_mode = "inpaint"
        else:
            wf["5"] = {
                "class_type": "VAEEncode",
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 0],
                },
            }

        def _wf_sort_key(k):
            try:
                return int(k)
            except ValueError:
                return int("".join(ch for ch in k if ch.isdigit()) or 0)

        if is_klein_9b:
            print(
                "[FluxClient] FLUX.2 Klein 9B EDIT workflow (ReferenceLatent + EmptyFlux2Latent): "
                f"cfg=1.0, denoise=1.0, steps={num_inference_steps}, sampler=er_sde, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_klein_4b:
            print(
                "[FluxClient] FLUX.2 Klein 4B EDIT workflow (ReferenceLatent + EmptyFlux2Latent): "
                f"cfg=1.0, denoise=1.0, steps={num_inference_steps}, sampler=er_sde, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_kontext:
            _mode = f"img2img denoise={actual_denoise:.2f} (SIN ReferenceLatent)" if (_img2img_denoise and _use_img_ref) else f"EDIT ref={wf.get('6m', {}).get('inputs', {}).get('reference_latents_method', 'offset(default)') if _use_img_ref else 'no-ref'}"
            print(
                "[FluxClient] FLUX.1 Kontext dev workflow (FluxGuidance + euler): "
                f"guidance={_kg:.1f}, steps={num_inference_steps}, {_mode}, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_longcat:
            _workflow_ref = wf.get("11", {}).get("inputs", {}).get("reference_latents_method", "none")
            print(
                "[FluxClient] LongCat edit workflow: "
                f"cfg={wf['8']['inputs']['cfg']}, denoise={actual_denoise:.2f}, "
                f"ref={_workflow_ref}, "
                f"drop_img_ref={drop_edit_image_ref}, "
                f"mask={mask_mode}, "
                f"nodes={sorted(wf.keys(), key=_wf_sort_key)}",
                flush=True,
            )
        elif is_qwen:
            print(
                "[FluxClient] Qwen-Image-Edit workflow: "
                f"cfg={wf['8']['inputs']['cfg']}, steps={num_inference_steps}, "
                f"mask={mask_mode}, "
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

        def _free_server_vram():
            # v5.1133c — NO descargar post-run. Evidencia de logs:
            #   - run 1: unload post-run OK (7078MB libres) → run 2 SAME-model:
            #     "reutilizando modelo" pero ComfyUI recargó TE+UNet DESDE CERO
            #     (no había nada residente que reutilizar) → 300s+ de preparación.
            #   - el modelo residente ES el cache: la reutilización same-model
            #     samplea a 5.4s/it directo. La protección correcta es el guard
            #     PRE-RUN de load() (cambio de versión o VRAM libre <1500MB →
            #     unload sincrono + recarga limpia), que ya cubre ambos casos.
            _free_now, _tot = _server_vram_free_mb(comfy_url)
            if _free_now is not None:
                print(f"[FluxClient] Modelo residente para reutilización (VRAM libre: {_free_now:.0f}MB de {_tot:.0f}MB)", flush=True)

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
        _free_server_vram()
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
