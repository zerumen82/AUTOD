#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import roop.globals

class ImgEditorManager:
    def __init__(self):
        self.flux_klein_client = None
        self.flux_schnell_client = None
        self.flux_dev_client = None
        self.flux_dev_abl_client = None
        self.omnigen2_client = None
        self.face_preserver = None
        self.clip_masker = None
        self.prompt_analyzer = None
        self.prompt_rewriter = None
        self._last_context = "normal"

    def _get_prompt_analyzer(self):
        if self.prompt_analyzer is None:
            from roop.img_editor.prompt_analyzer import PromptAnalyzer, EditingMode
            self.prompt_analyzer = PromptAnalyzer()
            self.EditingMode = EditingMode
        return self.prompt_analyzer

    def _get_face_preserver(self):
        if self.face_preserver is None:
            from roop.img_editor.face_preserver import FacePreserver
            fp = FacePreserver()
            ok, _ = fp.initialize()
            if ok:
                self.face_preserver = fp
        return self.face_preserver

    def _get_rewriter(self):
        if self.prompt_rewriter is None:
            from roop.img_editor.prompt_rewriter import get_prompt_rewriter
            self.prompt_rewriter = get_prompt_rewriter()
        return self.prompt_rewriter

    def _get_clipseg_masker(self):
        if self.clip_masker is None:
            from roop.img_editor.clipseg_masker import get_clipseg_masker
            self.clip_masker = get_clipseg_masker()
        return self.clip_masker

    def auto_detect_params(self, analysis: Dict, engine: str) -> Dict:
        """
        Calcula parámetros dinámicos basados puramente en el análisis semántico del LLM.
        """
        magnitude = analysis.get("magnitude", 0.5)
        
        # Escalar denoise para img2img: proteger imagen original
        # Cambios sutiles (0-0.3): 0.15-0.25
        # Cambios medios (0.3-0.6): 0.25-0.35  
        # Cambios radicales (0.6-1): 0.35-0.45
        if magnitude < 0.3:
            denoise = 0.15 + (magnitude * 0.333)
        elif magnitude < 0.6:
            denoise = 0.25 + ((magnitude - 0.3) * 0.333)
        else:
            denoise = 0.35 + ((magnitude - 0.6) * 0.25)
            
        denoise = min(denoise, 0.45)
        
        # Escalar pasos - reducir para mayor velocidad
        steps = int(8 + (magnitude * 8))  # 8-16 steps
        
        # Escalar guidance - aumentar un poco para mejor seguimiento del prompt
        guidance = 4.0 + (magnitude * 2.5)

        # Ajustes según motor
        if engine == "flux_schnell":
            steps = min(steps, 6)
        elif "klein" in engine:
            steps = min(steps, 15)

        # Determinar modo
        mode = "img2img"
        if magnitude < 0.35:
            # Para cambios muy pequeños, el sistema intentará inpaint quirúrgico
            pass

        print(f"[ImgEditor] LLM Analysis Applied: Denoise={denoise:.2f}, Steps={steps}, Guidance={guidance:.1f}")
        
        return {
            "denoise": denoise,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "mode": mode
        }

    def _compose_generation_prompt(self, user_prompt: str, analysis: Dict) -> str:
        """
        Mantiene la instrucción del usuario como señal principal y añade la
        traducción técnica solo si el LLM devolvió algo útil.
        """
        user_prompt = (user_prompt or "").strip()
        rewritten = str(analysis.get("prompt", "") or "").strip()
        if not rewritten or rewritten.lower() == user_prompt.lower():
            return user_prompt

        bad_fragments = (
            "traduccion detallada",
            "traducción detallada",
            "technical translation",
            "prompt real",
            "json",
        )
        rewritten_lw = rewritten.lower()
        if any(fragment in rewritten_lw for fragment in bad_fragments):
            return user_prompt

        # Crear prompt más claro para FLUX img2img
        # Incluir referencia a la imagen original y descripción clara
        return f"Photo of {rewritten}"

    def _is_usable_mask_target(self, mask_target: str) -> bool:
        target = (mask_target or "").strip().lower()
        if not target:
            return False
        broad_targets = {
            "subject", "person", "human", "body", "whole body", "full body",
            "image", "photo", "scene", "everything", "all"
        }
        return target not in broad_targets

    def _should_preserve_faces(self, face_preserve: bool, analysis: Dict, prompt: str) -> bool:
        if not face_preserve:
            return False

        text = f"{prompt} {analysis.get('mask_target', '')}".lower()
        facial_terms = (
            "face", "rostro", "cara", "eyes", "ojos", "mouth", "boca",
            "lips", "labios", "smile", "sonrisa", "expression", "expresion",
            "expresión", "blink", "wink", "guiño", "hair", "pelo", "cabello"
        )
        return not any(term in text for term in facial_terms)

    def generate_intelligent(
        self,
        image,
        prompt: str,
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        face_preserve: bool = True,
        use_rewriter: bool = True,
        ref_metadata: dict = None,
        engine: str = "flux_klein",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = ""
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        prompt = (prompt or "").strip()

        # 1. Análisis de la imagen original (describe qué hay para preservar contexto)
        img_description = ""
        try:
            from scripts.moondream_analyzer import analyze_image_with_moondream
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                res = analyze_image_with_moondream(tmp.name)
                img_description = res.get('positive', '')
            if img_description:
                print(f"[ImgEditor] Imagen analizada: {img_description[:100]}...")
        except Exception as e:
            print(f"[ImgEditor] Warning: No se pudo analizar imagen: {e}")

        # 2. Análisis Semántico del prompt del usuario (primero, para obtener magnitud/mask correctos)
        analysis = {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject"}
        if use_rewriter:
            try:
                # Primero analizar solo el prompt del usuario
                print(f"[ImgEditor] Analizando prompt: '{prompt}'")
                rewriter = self._get_rewriter()
                analysis = rewriter.rewrite(prompt)
                print(f"[ImgEditor] LLM análisis - Mag: {analysis.get('magnitude')}, Mask: {analysis.get('mask_target')}")
                
                # Luego agregar la descripción de imagen al prompt final
                if img_description:
                    analysis["prompt"] = f"{img_description[:150]}... {prompt}"
                    print(f"[ImgEditor] Prompt final: {analysis['prompt'][:100]}...")
            except Exception as e:
                print(f"[ImgEditor] Falló análisis semántico: {e}")

        # 2. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
        
        # Crear prompt final: primero el pedido del usuario, luego contexto
        rewritten = analysis.get("prompt", "").strip()
        if not rewritten or "traduccion" in rewritten.lower() or len(rewritten) < 10:
            rewritten = prompt
        else:
            # Si el LLM generó algo útil, combinar con el prompt del usuario
            rewritten = f"{prompt} {rewritten}"
        
        # Crear prompt más claro y directo para FLUX
        prompt_lower = prompt.lower()
        
        # Traducir palabras clave al inglés para mejor comprensión de FLUX
        prompt_english = prompt
        replacements = [
            ("desnudo", "naked"), ("desnuda", "naked"), ("desnudos", "naked"), ("desnudas", "naked"),
            ("cuerpo", "body"), ("ropa", "clothes"), ("traje", "suit"), ("camisa", "shirt"),
            ("pantalón", "pants"), ("sonreír", "smiling"), ("ojos", "eyes")
        ]
        for es, en in replacements:
            prompt_english = prompt_english.replace(es, en)
        
        # Crear prompt final más directo
        if any(w in prompt_lower for w in ["desnudo", "desnuda", "desnudos", "desnudas", "nude", "naked"]):
            # Para desnudos, crear prompt más explícito
            prompt_enhanced = f"Photo of people, {prompt_english}, wearing less clothing, partially undressed"
        else:
            # Para otros cambios
            if img_description:
                prompt_enhanced = f"{prompt_english}. {img_description[:80]}..."
            else:
                prompt_enhanced = prompt_english
            
        print(f"[ImgEditor] Prompt enviado: {prompt_enhanced[:180]}", flush=True)
        
        # Post-procesamiento: corregir mask_target y parámetros
        mask_target = analysis.get("mask_target", "subject").lower()
        
        # Sobrescribir mask_target y aumentar parámetros si hay palabras clave específicas
        if any(w in prompt_lower for w in ["desnudo", "desnuda", "desnudos", "desnudas", "nude", "naked", "cuerpo", "body"]):
            mask_target = "body"
            analysis["magnitude"] = max(analysis.get("magnitude", 0.5), 0.7)
            # Aumentar guidance para este tipo de cambios
            params["guidance_scale"] = max(params.get("guidance_scale", 4.5), 6.0)
            print(f"[ImgEditor] Corregido -> body, guidance aumentado a 6.0")
        elif any(w in prompt_lower for w in ["ropa", "clothes", "shirt", "camisa", "traje", "pantalón"]):
            mask_target = "clothes"
        
        print(f"[ImgEditor] Mask target final: {mask_target}", flush=True)
        analysis["mask_target"] = mask_target
        
        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
        if guidance_scale: params["guidance_scale"] = guidance_scale

        # 3. Máscara Automática - SOLO para cambios sutiles/medios, NO para nudidad
        final_mask = mask_image
        mask_target = analysis.get("mask_target", "subject")
        
        # NO usar máscara para body/nude edits - usar edición global
        if mask_target in ["body", "subject"]:
            print(f"[ImgEditor] Edición global (sin máscara) para cambio de cuerpo")
        elif final_mask is None and analysis.get("magnitude", 0.5) < 0.5:
            try:
                mask_target = analysis.get("mask_target", "subject")
                if self._is_usable_mask_target(mask_target):
                    print(f"[ImgEditor] Intentando auto-máscara para: {mask_target}")
                    masker = self._get_clipseg_masker()
                    auto_mask = masker.generate_mask(img, mask_target)
                    if auto_mask:
                        final_mask = auto_mask
                        print(f"[ImgEditor] Máscara quirúrgica (LLM driven) GENERADA")
                else:
                    print(f"[ImgEditor] Auto-máscara omitida por target genérico: {mask_target}")
            except Exception as e:
                print(f"[ImgEditor] Error en máscara: {e}")

        try:
            result = None
            msg = ""
            print(f"[ImgEditor] Iniciando generación con motor: {engine}")
            
            # ... (dentro del bloque try anterior) ...
            if engine == "flux_schnell":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_schnell_client is None:
                    self.flux_schnell_client = get_flux_edit_comfy_client()
                client = self.flux_schnell_client
                success, msg = client.load(flux_version="flux1-schnell-Q4_K_S.gguf")
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (Flux Schnell)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=final_mask if mask_mode in ["manual", "global"] else None
                )
                if result_obj: result = result_obj.image

            elif engine == "flux_dev":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_dev_client is None:
                    self.flux_dev_client = get_flux_edit_comfy_client()
                client = self.flux_dev_client
                success, msg = client.load(flux_version="flux1-dev-Q4_K.gguf")
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (Flux Dev)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=final_mask if mask_mode in ["manual", "global"] else None
                )
                if result_obj: result = result_obj.image

            elif engine == "klein_base":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None:
                    self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                success, msg = client.load(flux_version="flux-2-klein-base-4b-Q4_K_S.gguf")
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (Flux 2 Klein)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=final_mask if mask_mode in ["manual", "global"] else None
                )
                if result_obj: result = result_obj.image

            elif engine == "flux_dev_abliterated":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_dev_abl_client is None:
                    self.flux_dev_abl_client = get_flux_edit_comfy_client()
                client = self.flux_dev_abl_client
                success, msg = client.load(flux_version="T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf")
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (Flux Dev Abliterated)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=final_mask if mask_mode in ["manual", "global"] else None
                )
                if result_obj: result = result_obj.image

            elif engine == "omnigen2":
                from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                if self.omnigen2_client is None:
                    self.omnigen2_client = get_omnigen2_comfy_client()
                client = self.omnigen2_client
                success, msg = client.load()
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (OmniGen 2)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"]
                )
                if result_obj: result = result_obj.image

            else:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None:
                    self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                success, msg = client.load(flux_version="flux-2-klein-base-4b-Q4_K_S.gguf")
                if not success:
                    return None, msg, final_mask
                print(f"[ImgEditor] Enviando a ComfyUI (Default/Flux 2)...")
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=final_mask if mask_mode in ["manual", "global"] else None
                )
                if result_obj: result = result_obj.image

            if result is not None and self._should_preserve_faces(face_preserve, analysis, prompt):
                try:
                    fp = self._get_face_preserver()
                    if fp:
                        result = fp.preserve_faces(img, result, method="blend")
                        print(f"[ImgEditor] Face preservation aplicada")
                except Exception as e:
                    print(f"[ImgEditor] Face preservation falló: {e}")
            elif result is not None and face_preserve:
                print("[ImgEditor] Face preservation omitida porque el prompt edita rasgos faciales")

            if result is not None:
                return result, msg, final_mask
            return None, msg if msg else "Error: no se generó resultado", final_mask

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}", final_mask

_manager = None
def get_img_editor_manager() -> ImgEditorManager:
    global _manager
    if _manager is None:
        _manager = ImgEditorManager()
    return _manager
