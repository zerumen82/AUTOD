#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, threading, cv2
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import roop.globals
from roop.utils import get_vram_gb
from concurrent.futures import ThreadPoolExecutor
from roop.utilities import resolve_relative_path

class ImgEditorManager:
    def __init__(self):
        self.flux_klein_client = None
        self.flux_schnell_client = None
        self.flux_dev_client = None
        self.flux_dev_abl_client = None
        self.qwen_edit_client = None
        self.omnigen2_client = None
        self.face_preserver = None
        self.clip_masker = None
        self.prompt_analyzer = None
        self.semantic_analyzer = None
        self._last_context = "normal"
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._codeformer = None

    def _get_prompt_analyzer(self):
        if self.prompt_analyzer is None:
            from roop.img_editor.prompt_analyzer import PromptAnalyzer, EditingMode
            self.prompt_analyzer = PromptAnalyzer()
            self.EditingMode = EditingMode
        return self.prompt_analyzer

    def _get_semantic_analyzer(self):
        if self.semantic_analyzer is None:
            from roop.img_editor.nlp.semantic_analyzer import SemanticIntentAnalyzer
            self.semantic_analyzer = SemanticIntentAnalyzer()
        return self.semantic_analyzer

    def _get_face_preserver(self):
        if self.face_preserver is None:
            from roop.img_editor.face_preserver import FacePreserver
            fp = FacePreserver()
            ok, _ = fp.initialize()
            if ok:
                self.face_preserver = fp
        return self.face_preserver

    def _get_clipseg_masker(self):
        if self.clip_masker is None:
            from roop.img_editor.clipseg_masker import get_clipseg_masker
            self.clip_masker = get_clipseg_masker()
        return self.clip_masker

    def auto_detect_params(self, analysis: Dict, engine: str) -> Dict:
        """
        Calcula parámetros dinámicos basados puramente en el análisis semántico.
        Optimizado para velocidad en 8GB VRAM.
        """
        magnitude = analysis.get("magnitude", 0.5)
        
        # Escalamiento lineal dinámico basado en magnitud (0.0 a 1.0)
        # Denoise: 0.20 (sutil) a 0.90 (radical)
        denoise = 0.20 + (magnitude * 0.70)
        denoise = min(denoise, 0.95)

        # Pasos: 12 (rápido) a 28 (calidad) para modelos estándar
        min_steps = 12
        max_steps = 28
        steps = int(min_steps + (magnitude * (max_steps - min_steps)))

        # Guidance: 3.0 a 7.0
        guidance = 3.0 + (magnitude * 4.0)

        # Ajustes técnicos mínimos por motor (sin hardcoding semántico)
        if engine == "longcat":
            # LongCat es instructivo, requiere denoise alto para cambios notables
            if magnitude > 0.5:
                denoise = max(denoise, 0.75)
            guidance = 3.5 # Valor óptimo para el modelo LongCat
        elif engine == "longcat_full":
            steps = int(20 + (magnitude * 10))
            guidance = 4.5
        elif engine == "flux_schnell":
            steps = 4 # Limitación técnica del modelo Schnell
        elif "qwen" in engine:
            steps = min(steps, 8) # Qwen es muy lento, limitamos por UX
            guidance = min(guidance, 4.0)
        elif "klein" in engine or "abliterated" in engine:
            steps = min(steps, 20) # Balance velocidad/calidad

        print(f"[ImgEditor] Dynamic Params: Mag={magnitude:.2f}, Denoise={denoise:.2f}, Steps={steps}, CFG={guidance:.1f}")
        
        return {
            "denoise": denoise,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "mode": "img2img"
        }

    def _compose_generation_prompt(self, user_prompt: str, img_context: str = "", engine: str = "") -> str:
        user_prompt = (user_prompt or "").strip()
        img_context = (img_context or "").strip()
        
        is_longcat = "longcat" in engine.lower()

        if is_longcat:
            # PARA LONGCAT: Si hay una instrucción del usuario, usarla TAL CUAL (LongCat es instructivo)
            # Si no hay prompt de usuario, usar el contexto de la imagen como fallback
            base = user_prompt if user_prompt else img_context
            # Asegurar prefijo Instruction: si no lo tiene
            if base and not base.lower().startswith("instruction:"):
                base = f"Instruction: {base}"
            print(f"[ImgEditor] LongCat Prompt: {base}")
        elif img_context:
            base = f"{img_context}. {user_prompt}"
        else:
            base = user_prompt


        if not is_longcat and base and not base.lower().startswith(("photo", "a photo", "a picture")):
            # Eliminado prefijo hardcoded 'Photo of' para mayor fidelidad al prompt del LLM
            pass

        return base

    def _should_preserve_faces(self, face_preserve: bool, analysis: Dict, prompt: str) -> bool:
        if not face_preserve:
            return False

        # PRIORIDAD: Usar la decisión semántica si está disponible en el análisis
        if "preserve_face" in analysis:
            return bool(analysis["preserve_face"])

        return True

    def _apply_skin_tone_match(self, original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
        """Ajusta el tono de la piel generada para que coincida con la original"""
        orig_arr = np.array(original.convert("RGB"))
        gen_arr = np.array(generated.convert("RGB"))
        mask_arr = np.array(mask.convert("L")) / 255.0
        
        if mask_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            mask_arr = cv2.resize(mask_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        orig_arr_comp = orig_arr
        if orig_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            orig_arr_comp = cv2.resize(orig_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

        mask_center = mask_arr > 0.8
        if not np.any(mask_center): mask_center = mask_arr > 0.5
        
        if np.any(mask_center):
            avg_orig = orig_arr_comp[mask_center].mean(axis=0)
            avg_gen = gen_arr[mask_center].mean(axis=0)
            
            correction = avg_orig / (avg_gen + 1e-6)
            correction = np.clip(correction, 0.8, 1.2)
            
            for i in range(3):
                gen_arr[:,:,i] = np.clip(gen_arr[:,:,i] * correction[i], 0, 255)
                
            return Image.fromarray(gen_arr.astype(np.uint8))
        return generated

    def _feather_mask(self, mask: Image.Image, amount: int = 15) -> Image.Image:
        """Aplica un borde suave a la máscara"""
        from PIL import ImageFilter
        return mask.filter(ImageFilter.GaussianBlur(radius=amount))

    def _get_codeformer(self):
        if self._codeformer is None:
            from roop.processors.Enhance_CodeFormer import Enhance_CodeFormer
            model_path = resolve_relative_path('../models/CodeFormer/CodeFormerv0.1.onnx')
            if os.path.exists(model_path):
                try:
                    cf = Enhance_CodeFormer()
                    opts = {"devicename": "cuda"}
                    cf.Initialize(opts)
                    self._codeformer = cf
                    print("[ImgEditor] CodeFormer listo para enhancement facial")
                except Exception as e:
                    print(f"[ImgEditor] CodeFormer no disponible: {e}")
            else:
                print(f"[ImgEditor] CodeFormer model not found at {model_path}")
        return self._codeformer

    def _enhance_faces(self, image: Image.Image) -> Image.Image:
        """Mejora rostros en la imagen usando CodeFormer. Sin hardcoding."""
        cf = self._get_codeformer()
        if cf is None:
            return image

        fp = self._get_face_preserver()
        if fp is None:
            return image

        try:
            faces = fp.detect_faces(image)
            if not faces:
                return image

            img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            h, w = img_np.shape[:2]
            result = img_np.copy()

            for face in faces:
                bbox = face["bbox"]
                x1, y1, x2, y2 = [int(v) for v in bbox]
                margin = int(max(x2 - x1, y2 - y1) * 0.3)
                x1 = max(0, x1 - margin)
                y1 = max(0, y1 - margin)
                x2 = min(w, x2 + margin)
                y2 = min(h, y2 + margin)

                face_crop = img_np[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                enhanced_bgr, _ = cf.Run(None, None, face_crop)
                if enhanced_bgr.shape[:2] != face_crop.shape[:2]:
                    enhanced_bgr = cv2.resize(
                        enhanced_bgr, (face_crop.shape[1], face_crop.shape[0]),
                        cv2.INTER_LANCZOS4
                    )

                # Low blend: just subtle texture/quality improvement, keeps edit intact
                det_score = face.get("det_score", 0.9)
                blend = float(np.clip(det_score * 0.15, 0.05, 0.3))

                roi = result[y1:y2, x1:x2]
                result[y1:y2, x1:x2] = (
                    roi.astype(np.float32) * (1 - blend)
                    + enhanced_bgr.astype(np.float32) * blend
                ).astype(np.uint8)

            print(f"[ImgEditor] Enhancement aplicado a {len(faces)} rostro(s)")
            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        except Exception as e:
            print(f"[ImgEditor] Face enhancement error: {e}")
            return image

    def generate_intelligent(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = None,
        guidance_scale: float = None,
        seed: int = None,
        face_preserve: bool = True,
        use_rewriter: bool = True,
        ref_metadata: dict = None,
        engine: str = "flux_klein",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = "",
        enhance_faces: bool = False,
        lora_name: str = None,
        lora_strength: float = 1.0,
        denoise: float = None
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        from roop.img_editor.prompt_translator import translate_prompt
        prompt = translate_prompt((prompt or "").strip())
        negative_prompt = (negative_prompt or "").strip()

        if engine == "flux_dev_abliterated":
            vram = get_vram_gb()
            if 0 < vram <= 8:
                print(f"[ImgEditor] VRAM={vram}GB detectada. Flux Dev puede ser lento o fallar.")
        elif engine == "qwen_edit":
            pass

        img_description = ""

        # 2. Análisis semántico por EMBEDDINGS (IA Pura, Cero Hardcoding)
        try:
            nlp = self._get_semantic_analyzer()
            mag_suggested = nlp.get_magnitude(prompt)
            mask_target = nlp.detect_target(prompt)
            
            # Para detectar si es global, miramos si no hay un target claro o si el prompt es muy corto/genérico
            is_global = mask_target == "subject" and mag_suggested < 0.45
            
            print(f"[ImgEditor] Semantic Analysis: Magnitude={mag_suggested:.2f}, Target={mask_target}")
        except Exception as e:
            print(f"[NLP] Error en análisis semántico: {e}. Usando fallback.")
            mag_suggested = 0.5
            mask_target = "subject"
            is_global = True

        analysis = {
            "prompt": prompt, 
            "magnitude": mag_suggested, 
            "mask_target": mask_target,
            "is_global": is_global
        }

        # 3. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
        
        prompt_enhanced = self._compose_generation_prompt(prompt, img_context=img_description, engine=engine)
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        is_global = bool(analysis.get("is_global", False))

        print(f"[ImgEditor] Target: {mask_target} (Global: {is_global})", flush=True)
        print(f"[ImgEditor] Prompt final: {prompt_enhanced}", flush=True)
        analysis["mask_target"] = mask_target
        
        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
        if guidance_scale: params["guidance_scale"] = guidance_scale
        if denoise: params["denoise"] = denoise

        final_mask = mask_image
        
        if final_mask is None and not is_global:
            try:
                mask_query = mask_target
                print(f"[ImgEditor] Intentando auto-máscara para: {mask_query}")
                masker = self._get_clipseg_masker()
                auto_mask = masker.generate_mask(img, mask_query)
                if auto_mask:
                    final_mask = auto_mask
                    print(f"[ImgEditor] Máscara automática GENERADA para '{mask_query}'")
            except Exception as e:
                print(f"[ImgEditor] Error en máscara: {e}")

        if final_mask is None and is_global:
             print(f"[ImgEditor] Edición global detectada (sin máscara)")

        try:
            result = None
            msg = ""
            print(f"[ImgEditor] Iniciando generación con motor: {engine}")
            
            client = None
            version = None
            
            if engine == "longcat":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
            elif engine == "longcat_full":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "LongCat-Image-Edit-Q4_K_S.gguf"
            elif engine == "flux_dev_abliterated":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_dev_abl_client is None: self.flux_dev_abl_client = get_flux_edit_comfy_client()
                client = self.flux_dev_abl_client
                version = "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf"
            elif engine == "qwen_edit":
                from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client
                if self.qwen_edit_client is None: self.qwen_edit_client = get_qwen_edit_comfy_client()
                client = self.qwen_edit_client
                version = "q2"
            elif engine == "omnigen2":
                from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                if self.omnigen2_client is None: self.omnigen2_client = get_omnigen2_comfy_client()
                client = self.omnigen2_client
                version = None
            elif engine == "klein_base":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "flux-2-klein-base-4b-Q4_K_S.gguf"
            elif engine == "flux_q2":
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_dev_abl_client is None: self.flux_dev_abl_client = get_flux_edit_comfy_client()
                client = self.flux_dev_abl_client
                version = "flux1-dev-Q2_K.gguf"
            else:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

            if client:
                load_params = {"flux_version": version} if "qwen" not in engine else {"qwen_version": version}
                success, msg = client.load(**load_params)
                if not success:
                    return None, msg, final_mask
                
                print(f"[ImgEditor] Enviando a ComfyUI ({engine}, version={version})...")
                gen_params = {
                    "image": img, "prompt": prompt_enhanced,
                    "negative_prompt": negative_prompt,
                    "num_inference_steps": params["num_inference_steps"],
                    "guidance_scale": params["guidance_scale"],
                    "seed": seed, "denoise": params["denoise"],
                    "lora_name": lora_name,
                    "lora_strength": lora_strength
                }
                if "qwen" not in engine:
                    gen_params["mask_image"] = final_mask if mask_mode in ["manual", "global"] else None
                
                result_obj, msg = client.generate(**gen_params)
                if result_obj: 
                    result = result_obj.image
                    if result.size != img.size:
                        result = result.resize(img.size, Image.LANCZOS)
                    
                    if final_mask:
                        result = self._apply_skin_tone_match(img, result, final_mask)
                        soft_mask = self._feather_mask(final_mask, amount=15)
                        orig_bg = img.copy()
                        result = Image.composite(result, orig_bg, soft_mask)

            if result is not None and self._should_preserve_faces(face_preserve, analysis, prompt):
                mask_target = analysis.get("mask_target", "subject")
                preserve_override = analysis.get("preserve_face", True)
                try:
                    fp = self._get_face_preserver()
                    if fp:
                        result = fp.preserve_faces(img, result, method="swap")
                except Exception as e:
                    print(f"[ImgEditor] Face preservation falló: {e}")

            if result is not None and enhance_faces:
                result = self._enhance_faces(result)

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
