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
        self.omnigen2_client = None
        self.face_preserver = None
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

    def auto_detect_params(self, prompt: str, engine: str) -> Dict:
        analyzer = self._get_prompt_analyzer()
        mode, confidence = analyzer.analyze(prompt)

        prompt_lower = prompt.lower()
        has_scene_change = any(kw in prompt_lower for kw in [
            "fondo", "background", "playa", "beach", "ciudad", "city",
            "bosque", "forest", "montaña", "mountain", "espacio", "space",
            "calle", "street", "interior", "habitación", "room"
        ])
        has_clothing_change = any(kw in prompt_lower for kw in [
            "ropa", "clothing", "outfit", "vestido", "dress", "camisa",
            "shirt", "pantalones", "pants", "zapatos", "shoes"
        ])
        has_expression_change = any(kw in prompt_lower for kw in [
            "sonrisa", "smile", "feliz", "happy", "expresion", "expression",
            "serio", "serious", "triste", "sad"
        ])

        if mode == self.EditingMode.INPAINT:
            denoise = 0.65
            steps = 8
            guidance = 4.0
        elif mode == self.EditingMode.OUTPAINT:
            denoise = 0.75
            steps = 10
            guidance = 4.5
        else:
            if has_scene_change and has_clothing_change:
                denoise = 0.60
            elif has_scene_change:
                denoise = 0.55
            elif has_clothing_change:
                denoise = 0.50
            elif has_expression_change:
                denoise = 0.40
            else:
                denoise = 0.45
            steps = 8
            guidance = 3.5

        if engine == "flux_schnell":
            steps = min(steps, 4)
        elif "klein" in engine:
            steps = min(steps, 8)

        print(f"[ImgEditor] Auto-params: mode={mode.value}, denoise={denoise}, steps={steps}, guidance={guidance}")
        return {
            "denoise": denoise,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "mode": mode.value
        }

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
    ) -> Tuple[Optional[Image.Image], str]:

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        params = self.auto_detect_params(prompt, engine)
        if num_inference_steps:
            params["num_inference_steps"] = num_inference_steps
        if guidance_scale:
            params["guidance_scale"] = guidance_scale

        prompt_enhanced = prompt
        if use_rewriter:
            try:
                rewriter = self._get_rewriter()
                rewritten, intensity = rewriter.rewrite(prompt)
                if len(rewritten) > len(prompt):
                    prompt_enhanced = rewritten
                    print(f"[ImgEditor] Prompt mejorado: {prompt_enhanced[:60]}...")
            except Exception as e:
                print(f"[ImgEditor] Rewriter falló: {e}")

        print(f"[ImgEditor] Engine={engine} | Steps={params['num_inference_steps']} | Denoise={params['denoise']} | Mode={params['mode']}")

        try:
            result = None
            msg = ""

            if engine == "flux_schnell":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_schnell_client is None:
                    self.flux_schnell_client = get_flux_edit_comfy_client()
                client = self.flux_schnell_client
                success, msg = client.load(flux_version="flux1-schnell-Q4_K_S.gguf")
                if not success:
                    return None, msg
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=mask_image if mask_mode == "manual" else None
                )
                if result_obj: result = result_obj.image

            elif engine == "omnigen2":
                from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_client
                if self.omnigen2_client is None:
                    self.omnigen2_client = get_omnigen2_client()
                client = self.omnigen2_client
                success, msg = client.load()
                if not success:
                    return None, msg
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
                success, msg = client.load(flux_version="flux2-klein-4b-Q4_K_S.gguf")
                if not success:
                    return None, msg
                result_obj, msg = client.generate(
                    image=img, prompt=prompt_enhanced,
                    num_inference_steps=params["num_inference_steps"],
                    guidance_scale=params["guidance_scale"],
                    seed=seed, denoise=params["denoise"],
                    mask_image=mask_image if mask_mode == "manual" else None
                )
                if result_obj: result = result_obj.image

            if result is not None and face_preserve:
                try:
                    fp = self._get_face_preserver()
                    if fp:
                        result = fp.preserve_faces(img, result, method="blend")
                        print(f"[ImgEditor] Face preservation aplicada")
                except Exception as e:
                    print(f"[ImgEditor] Face preservation falló: {e}")

            if result is not None:
                return result, msg
            return None, msg if msg else "Error: no se generó resultado"

        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"

_manager = None
def get_img_editor_manager() -> ImgEditorManager:
    global _manager
    if _manager is None:
        _manager = ImgEditorManager()
    return _manager