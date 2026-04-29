#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImgEditor Manager - Edición de imágenes con ComfyUI y Motores Modernos (FLUX, HART, OmniGen2)
Versión Refactorizada y Alineada (2026)
"""

import os
import sys
import tempfile
import time
import json
import torch
import gc
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import cv2
import roop.globals

from roop.comfy_client import get_comfyui_url
from roop.img_editor.clothing_segmenter import get_clothing_segmenter, is_clipseg_available
from roop.img_editor.controlnet_utils import get_controlnet_utils

class ImgEditorManager:
    """Gestiona la edición de imágenes con ComfyUI y otros motores modernos"""

    def __init__(self):
        self.client = None
        self.face_swapper = None
        self.face_analyzer = None
        self.flux_client = None
        self.hart_edit_client = None
        self.omnigen2_client = None
        self.controlnet_utils = None
        self._last_context = "normal"

    def resolve_engine_params(self, engine: str, creativity: float, preserve: float, steps: int, resolution_label: str = "1024p") -> Dict:
        """
        Centraliza la resolución de parámetros según el motor y la intención.
        Sigue la 'Alignment Guide' para evitar overrides opacos.
        """
        # 1. Mapeo de denoise (basado en 'preserve')
        base_denoise = 1.0 - preserve
        
        # 2. Mapeo de guidance (basado en 'creativity')
        base_guidance = 1.5 + (creativity * 5.0) 

        params = {
            "engine": engine,
            "denoise": base_denoise,
            "guidance_scale": base_guidance,
            "num_inference_steps": steps,
            "target_width": 1024,
            "target_height": 1024,
            "supports_mask": True,
            "mode": "edit"
        }

        # 3. Ajustes específicos por motor
        if engine in ["flux_klein", "flux"]:
            params["guidance_scale"] = 3.5 + (creativity * 1.5)
            params["num_inference_steps"] = max(8, min(30, steps))
            params["denoise"] = max(0.15, min(0.95, base_denoise))
        
        elif engine == "omnigen2":
            params["guidance_scale"] = 1.0 + (creativity * 4.0)
            params["num_inference_steps"] = max(10, min(30, steps))
            params["denoise"] = max(0.2, min(0.9, base_denoise))

        # 4. Resolución de dimensiones
        try:
            max_side = int(resolution_label.replace('p', ''))
            params["target_width"] = max_side
            params["target_height"] = max_side 
        except:
            pass

        return params

    def rewrite_prompt(self, prompt: str) -> str:
        """
        Analizador semántico estilo Grok: realismo extremo y protección de identidad.
        """
        prompt_lower = prompt.lower()
        enhanced_prompt = prompt
        
        # Detección de acción/pose
        pose_keywords = ["salte", "jump", "rodillas", "kneel", "corra", "run", "parado", "stand", "sentado", "sit", "dance", "baila", "pose", "postura", "caminando", "walk", "movimiento", "move"]
        is_action = any(kw in prompt_lower for kw in pose_keywords)
        
        # Estilo de fotografía de ALTA CALIDAD - FOCO EN COLOR Y PIEL
        style = "professional high-definition photography, vivid colors, natural skin tones, cinematic sharp focus, 8k resolution"
        protection = "exact same characters, keep original clothes and background colors"
        
        if is_action:
            print("[ImgEditor] 🏃 Acción detectada - Optimizando para movimiento realista")
            enhanced_prompt = f"{style}, {prompt}, {protection}, in motion"
            self._last_context = "pose_change"
        else:
            enhanced_prompt = f"{style}, {prompt}, {protection}"
            self._last_context = "normal"
            
        return enhanced_prompt

    def generate_intelligent(
        self,
        image,
        prompt: str,
        num_inference_steps: int = 25,
        guidance_scale: float = 7.5,
        seed: int = None,
        face_preserve: bool = True,
        use_rewriter: bool = True,
        ref_metadata: dict = None,
        engine: str = "flux_klein",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = ""
    ) -> Tuple[Optional[Image.Image], str]:
        """Generación modular enfocada en ELIMINAR EL GRIS Y MANTENER IDENTIDAD."""
        
        # Preparar imagen original
        if isinstance(image, Image.Image):
            original_image = image.copy().convert("RGB")
        elif hasattr(image, 'name'):
            original_image = Image.open(image.name).copy().convert("RGB")
        else:
            return None, "Error: Imagen inválida"

        # 1. RESOLVER PARÁMETROS CENTRALIZADOS
        creativity = ref_metadata.get('creativity', 0.78) if ref_metadata else 0.78
        preserve = ref_metadata.get('preserve', 0.22) if ref_metadata else 0.22
        res_label = ref_metadata.get('resolution_label', '1024p') if ref_metadata else '1024p'
        
        p = self.resolve_engine_params(engine, creativity, preserve, num_inference_steps, res_label)
        
        # 2. REESCRITURA DE PROMPT
        final_prompt = self.rewrite_prompt(prompt) if use_rewriter else prompt

        # AJUSTE PARA ACCIÓN (Fuerza Bruta contra el gris)
        if self._last_context == "pose_change":
            p["denoise"] = 0.78
            p["num_inference_steps"] = 30 # Forzamos 30 pasos para detalle
            p["guidance_scale"] = 5.0
            if mask_mode == "global":
                mask_mode, mask_prompt = "smart", "person"

        print(f"[ImgEditor] === LÓGICA DE MOTOR: {engine} ===")
        print(f" > Prompt: {final_prompt[:60]}...")
        print(f" > Denoise: {p['denoise']:.3f} | CFG: {p['guidance_scale']:.1f} | Steps: {p['num_inference_steps']}")

        # 3. LÓGICA DE MÁSCARA
        working_mask = None
        if p["supports_mask"]:
            if mask_mode == "manual" and mask_image:
                working_mask = mask_image
            elif mask_mode == "smart" and (mask_prompt or self._last_context == "pose_change"):
                m_prompt = mask_prompt if mask_prompt else "person"
                try:
                    segmenter = get_clothing_segmenter()
                    mask_pil, _ = segmenter.segment_with_prompt(original_image, [m_prompt], threshold=0.3, dilation=50)
                    working_mask = mask_pil
                except Exception as e:
                    print(f"[ImgEditor] ❌ CLIPSeg falló: {e}")

        # 4. EJECUCIÓN POR MOTOR
        try:
            if engine in ["flux", "flux_klein", "flux_dev", "flux_schnell"]:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_client is None: self.flux_client = get_flux_edit_comfy_client()
                
                flux_v = "flux2-klein-4b-Q4_K_S.gguf" if "klein" in engine or engine=="flux" else \
                         ("flux1-dev-Q4_K.gguf" if "dev" in engine else "flux1-schnell-Q4_K_S.gguf")
                
                success, msg = self.flux_client.load(flux_version=flux_v)
                if not success: return None, f"Error FLUX: {msg}"

                result_obj, msg = self.flux_client.generate(
                    image=original_image, prompt=final_prompt,
                    num_inference_steps=p["num_inference_steps"],
                    guidance_scale=p["guidance_scale"], seed=seed,
                    denoise=p["denoise"], mask_image=working_mask,
                    target_width=p["target_width"], target_height=p["target_height"]
                )

                if result_obj and result_obj.image:
                    final_image = result_obj.image
                    if face_preserve:
                        final_image = self._restore_face(original_image, final_image)
                    return final_image, f"OK ({result_obj.time_taken:.1f}s)"
                return None, f"Fallo FLUX: {msg}"

            return None, "Motor no implementado"

        except Exception as e:
            return None, f"Error Motor {engine}: {str(e)}"

    def _init_face_swap(self):
        if self.face_analyzer is not None and self.face_swapper is not None: return True
        try:
            import insightface
            from insightface.app import FaceAnalysis
            self.face_analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            from roop.processors.FaceSwap import FaceSwap
            self.face_swapper = FaceSwap()
            self.face_swapper.Initialize({'devicename': 'cuda' if torch.cuda.is_available() else 'cpu', 'model': 'inswapper_128.onnx'})
            return True
        except Exception as e:
            print(f"[ImgEditor] Error face swap init: {e}")
            return False

    def _restore_face(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """Restaura caras con lógica de 'fallback' forzada."""
        if not self._init_face_swap(): return generated
        try:
            import roop.globals
            orig_cv = cv2.cvtColor(np.array(original), cv2.COLOR_RGB2BGR)
            gen_cv = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            
            faces_orig = self.face_analyzer.get(orig_cv)
            faces_gen = self.face_analyzer.get(gen_cv)
            
            if not faces_orig: return generated

            w_orig, h_orig = original.size
            w_gen, h_gen = generated.size
            
            def get_norm_center(face, w, h):
                x1, y1, x2, y2 = face.bbox
                return ((x1+x2)/2/w, (y1+y2)/2/h)

            orig_data = [(get_norm_center(f, w_orig, h_orig), f) for f in faces_orig]
            
            result = gen_cv.copy()
            assigned = [False] * len(faces_gen) if faces_gen else []
            
            for (ox, oy), f_o in orig_data:
                best_i, min_d = -1, 1.0
                if faces_gen:
                    gen_data = [(get_norm_center(f, w_gen, h_gen), f) for f in faces_gen]
                    for i, ((gx, gy), f_g) in enumerate(gen_data):
                        if assigned[i]: continue
                        d = ((ox-gx)**2 + (oy-gy)**2)**0.5
                        if d < min_d: min_d, best_i = d, i
                
                # Forzar swap: si hay match por proximidad o simplemente por coordenadas originales
                target_f = faces_gen[best_i] if best_i != -1 and min_d < 0.4 else f_o
                
                old_blend = roop.globals.blend_ratio
                roop.globals.blend_ratio = 1.0
                res = self.face_swapper.Run(f_o, target_f, result, paste_back=True)
                roop.globals.blend_ratio = old_blend
                if res is not None: result = res
            
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"[ImgEditor] Face Restore error: {e}")
            return generated

    def preview_smart_mask(self, image: Image.Image, mask_prompt: str = "", mask_image: Optional[Image.Image] = None, mask_mode: str = "smart") -> Tuple[Optional[Image.Image], str]:
        try:
            segmenter = get_clothing_segmenter()
            if mask_mode == "manual": m = mask_image
            else:
                ok, _ = segmenter.load()
                if not ok: return None, "Error CLIPSeg"
                m, _ = segmenter.segment_with_prompt(image, [mask_prompt])
            if not m: return None, "Sin máscara"
            return segmenter.visualize_mask(image, m), "Máscara OK"
        except Exception as e: return None, str(e)

_manager = None
def get_img_editor_manager() -> ImgEditorManager:
    global _manager
    if _manager is None: _manager = ImgEditorManager()
    return _manager
