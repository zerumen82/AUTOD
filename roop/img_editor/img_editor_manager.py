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
        self.clip_masker = None
        self.clothing_segmenter = None
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

    def _get_semantic_analyzer(self, full_ai: bool = False):
        if self.semantic_analyzer is None:
            from roop.img_editor.nlp.semantic_analyzer import get_semantic_analyzer
            self.semantic_analyzer = get_semantic_analyzer(full_ai=full_ai)
        return self.semantic_analyzer

    def _get_clothing_segmenter(self):
        if self.clothing_segmenter is None:
            from roop.img_editor.clothing_segmenter import get_clothing_segmenter
            seg = get_clothing_segmenter()
            if seg.is_available():
                self.clothing_segmenter = seg
        return self.clothing_segmenter

    def _get_clipseg_masker(self):
        if self.clip_masker is None:
            from roop.img_editor.clipseg_masker import get_clipseg_masker
            self.clip_masker = get_clipseg_masker()
        return self.clip_masker

    def auto_detect_params(self, analysis: Dict, engine: str) -> Dict:
        """
        Calcula parámetros dinámicos basados puramente en el análisis semántico (embeddings).
        Cero hardcode de palabras clave de usuario.
        Optimizado para 8GB VRAM (LongCat Turbo etc).
        """
        magnitude = analysis.get("magnitude", 0.5)
        quality_only = bool(analysis.get("quality_only", False))

        if quality_only:
            denoise = 0.30
            steps = 32
            guidance = 3.8
            print(f"[ImgEditor] Modo CALIDAD (denoise bajo, preserva foto): denoise={denoise:.2f}")
            return {
                "denoise": denoise,
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "mode": "quality_enhance",
            }
        
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
        if engine in ("longcat", "imagine"):
            # Grok Imagine style (edit engine).
            # Use dynamic magnitude from semantic analyzer.
            # Higher base + stronger scaling to allow full body/clothing changes (undress etc) while keeping photo.
            # Semantic + rewriter decide how much. Turbo forces full denoise anyway.
            denoise = 0.58 + (magnitude * 0.42)
            denoise = min(denoise, 0.98)
            guidance = 2.8 if magnitude > 0.65 else 3.0
            # More steps for high mag to allow better following
            if magnitude > 0.6:
                steps = max(steps, 26)
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
        elif engine == "hart":
            # HART es autoregresivo puro. Usa guidance alto y pasos limitados.
            steps = min(8, max(4, steps))
            guidance = 4.5
            denoise = 1.0  # No aplica denoise real (generación)

        print(f"[ImgEditor] Dynamic Params: Mag={magnitude:.2f}, Denoise={denoise:.2f}, Steps={steps}, CFG={guidance:.1f}")
        
        return {
            "denoise": denoise,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "mode": "img2img"
        }

    def _compose_generation_prompt(
        self,
        user_prompt: str,
        img_context: str = "",
        engine: str = "",
        magnitude: float = 0.5,
        quality_only: bool = False,
    ) -> str:
        user_prompt = (user_prompt or "").strip()
        img_context = (img_context or "").strip()
        
        is_longcat = engine in ("longcat", "longcat_full", "imagine") or "longcat" in engine.lower()
        is_hart = engine == "hart"

        if is_longcat:
            raw = (user_prompt or img_context or "").strip()
            if quality_only:
                base = (
                    "Instruction: Edit this exact photo. "
                    "Keep identical composition, people, poses, clothing, background and scene. "
                    "Only enhance photographic quality: ultra realistic RAW photograph, sharp focus, "
                    "remove blur and noise, detailed skin texture and pores, natural colors, "
                    "cinematic clarity, hyperrealistic DSLR detail."
                )
                print(f"[ImgEditor] Imagine/LongCat Prompt (UI quality mode): {base}")
                return base

            likeness = (
                "Keep the same person, facial likeness, body proportions, camera angle, lighting and scene as the reference photo. "
            )
            if magnitude > 0.75:
                preservation = (
                    f"Edit this exact photo. {likeness}"
                    "Apply the instruction exactly as described, as strongly and completely as possible."
                )
                base = f"Instruction: {raw}. {preservation}"
            elif magnitude > 0.6:
                preservation = (
                    f"Edit this exact photo. {likeness}"
                    "Apply the requested change (including pose, clothing or body) exactly as described and as strongly as possible."
                )
                base = f"Instruction: {raw}. {preservation}"
            else:
                preservation = (
                    f"Edit this exact photo. {likeness}"
                    "Apply only the requested change to the subject."
                )
                base = f"Instruction: {preservation} {raw}"
            print(f"[ImgEditor] Imagine/LongCat Prompt: {base}")
            return base
        elif is_hart:
            # HART autoregressive - prompt as provided (user or context)
            base = user_prompt if user_prompt else img_context
            print(f"[ImgEditor] HART Autoregressive Prompt: {base}")
            return base
        elif img_context:
            base = f"{img_context}. {user_prompt}"
        else:
            base = user_prompt

        if base and not base.lower().startswith(("photo", "a photo", "a picture")):
            # Eliminado prefijo hardcoded 'Photo of' para mayor fidelidad al prompt del LLM
            pass

        return base

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

        try:
            import roop.face_util as face_util
            img_np = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            faces_data = face_util.extract_face_images(img_np, target_face_detection=True)
            if not faces_data:
                return image

            h, w = img_np.shape[:2]
            result = img_np.copy()

            for face_obj, _crop in faces_data:
                bbox = face_obj.bbox
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

                det_score = float(getattr(face_obj, "det_score", 0.9) or 0.9)
                blend = float(np.clip(det_score * 0.15, 0.05, 0.3))

                roi = result[y1:y2, x1:x2]
                result[y1:y2, x1:x2] = (
                    roi.astype(np.float32) * (1 - blend)
                    + enhanced_bgr.astype(np.float32) * blend
                ).astype(np.uint8)

            print(f"[ImgEditor] Enhancement aplicado a {len(faces_data)} rostro(s)")
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
        use_rewriter: bool = True,
        ref_metadata: dict = None,
        engine: str = "imagine",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = "",
        enhance_faces: bool = False,
        lora_name: str = None,
        lora_strength: float = 1.0,
        denoise: float = None,
        progress_callback=None,
        auto_upscale: bool = True,
        quality_mode: bool = False,
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        from roop.img_editor.prompt_translator import translate_prompt
        quality_only = bool(quality_mode)
        if quality_only:
            prompt = ""
            use_rewriter = False
            print("[ImgEditor] Modo mejora de calidad (UI) — prompt ignorado")
        else:
            prompt = translate_prompt((prompt or "").strip())
        negative_prompt = (negative_prompt or "").strip()

        if engine == "flux_dev_abliterated":
            vram = get_vram_gb()
            if 0 < vram <= 8:
                print(f"[ImgEditor] VRAM={vram}GB detectada. Flux Dev puede ser lento o fallar.")
        elif engine == "qwen_edit":
            pass

        img_description = ""  # DELIBERADAMENTE vacío: no usamos descripciones generativas de imagen (evita alucinaciones). El análisis es texto-del-prompt + preservación fuerte.

        if quality_only:
            mag_suggested = 0.35
            mask_target = "subject"
            is_global = True
        else:
            try:
                nlp = self._get_semantic_analyzer(full_ai=False)
                mag_suggested = nlp.get_magnitude(prompt)
                mask_target = nlp.detect_target(prompt)
                is_global = mask_target == "subject" and mag_suggested < 0.45
                print(f"[ImgEditor] Semantic: Magnitude={mag_suggested:.2f}, Target={mask_target}")
            except Exception as e:
                print(f"[NLP] Error en análisis semántico: {e}. Usando fallback.")
                mag_suggested = 0.58
                mask_target = "subject"
                is_global = True

        analysis = {
            "prompt": prompt,
            "magnitude": mag_suggested,
            "mask_target": mask_target,
            "is_global": is_global,
            "quality_only": quality_only,
        }

        # 3. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
        
        prompt_enhanced = self._compose_generation_prompt(
            prompt, img_context=img_description, engine=engine, magnitude=mag_suggested,
            quality_only=quality_only,
        )
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        is_global = bool(analysis.get("is_global", False))
        
        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
        if guidance_scale: params["guidance_scale"] = guidance_scale
        if denoise: params["denoise"] = denoise

        if quality_only and not negative_prompt:
            negative_prompt = (
                "blurry, low quality, noise, jpeg artifacts, soft focus, out of focus, "
                "pixelated, washed out, oversmoothed, plastic skin"
            )

        # NOTE: Negative clothing logic moved after rewriter so it can adapt to whether the user is asking
        # to ADD clothes (e.g. "ropa raída con mangas") or REMOVE them. No hardcoding of intents.

        # For high mag, use rewriter to get a better, more detailed instruction (LLM understands intent better, no hardcode)
        rewrote = False
        if use_rewriter and mag_suggested > 0.6 and not quality_only:
            try:
                from .prompt_rewriter import get_prompt_rewriter
                re = get_prompt_rewriter()
                r = re.rewrite(prompt, image_context="", mode="img2img")
                if 'prompt' in r and r['prompt']:
                    prompt = r['prompt']
                    rewrote = True
                    print(f"[ImgEditor] Used rewriter for high mag prompt: {prompt[:80]}...")
                if 'mask_target' in r and r['mask_target']:
                    mask_target = r['mask_target'].lower()
                    analysis['mask_target'] = mask_target
                    print(f"[ImgEditor] Rewriter set target: {mask_target}")
                if 'magnitude' in r:
                    mag_suggested = float(r['magnitude'])
                    analysis['magnitude'] = mag_suggested
            except Exception as e:
                print(f"[ImgEditor] Rewriter failed for high mag, using original: {e}")

        # Re-compose (and refresh params) using the (possibly rewriter-improved) prompt + updated mag.
        # This ensures the final sent prompt contains the clean rewritten instruction (e.g. "completely naked...").
        if rewrote or mag_suggested != analysis.get("magnitude", mag_suggested):
            params = self.auto_detect_params(analysis, engine)
        prompt_enhanced = self._compose_generation_prompt(
            prompt, img_context=img_description, engine=engine, magnitude=mag_suggested,
            quality_only=quality_only,
        )

        # Refresh local views after possible rewriter + recompose
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        is_global = bool(analysis.get("is_global", False))
        mag_suggested = float(analysis.get("magnitude", mag_suggested))

        print(f"[ImgEditor] Target: {mask_target} (Global: {is_global})", flush=True)
        print(f"[ImgEditor] Prompt final: {prompt_enhanced}", flush=True)
        if quality_only:
            print("[ImgEditor] Pipeline: LongCat Full (denoise bajo) + post upscale/nitidez")

        # No automatic anti-clothing negative anymore.
        # It was fighting requests like "ropa raída con mangas".
        # Now we rely purely on the user's prompt + rewriter + "apply the change as strongly as possible".
        # This way it adapts to ANY clothing request (add, modify, remove, specific style) without lists or hardcodes.
        # The semantic + preservation text is enough to guide the model.

        final_mask = mask_image

        # Alta magnitud: edición global (LongCat Kontext mantiene parecido a la foto de referencia)
        force_global = (mag_suggested > 0.6 and mask_target in ("subject", "clothes", "face"))
        use_clothing_mask = (
            not force_global
            and final_mask is None
            and mask_target in ("clothes", "subject")
            and mag_suggested >= 0.45
        )
        if use_clothing_mask:
            try:
                seg = self._get_clothing_segmenter()
                if seg:
                    cloth_mask, _ = seg.segment_clothing(img, threshold=0.45)
                    if cloth_mask is not None:
                        final_mask = cloth_mask
                        print("[ImgEditor] Máscara de ropa (ClothingSegmenter) para edit focalizado")
            except Exception as e:
                print(f"[ImgEditor] ClothingSegmenter no disponible: {e}")

        if force_global:
            print("[ImgEditor] Alta magnitud: edición global (instrucción fuerte + ref foto)")
            final_mask = None
        elif final_mask is None and not is_global:
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
            
            if engine in ("longcat", "imagine"):
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                # Full: mejor seguimiento de instrucción (CFG/denoise reales). Turbo: más rápido en 8GB.
                if quality_only or (engine == "imagine" and mag_suggested >= 0.68):
                    version = "LongCat-Image-Edit-Q4_K_S.gguf"
                    reason = "modo calidad (UI)" if quality_only else f"mag={mag_suggested:.2f}"
                    print(f"[ImgEditor] {reason} → LongCat Full (denoise real, no Turbo)")
                else:
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
            elif engine == "hart":
                from roop.img_editor.hart_edit_comfy_client import get_hart_edit_comfy_client
                if not hasattr(self, 'hart_client') or self.hart_client is None:
                    self.hart_client = get_hart_edit_comfy_client()
                client = self.hart_client
                version = None  # HART usa su propio loader en subprocess
            else:
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

            if client:
                if engine == "hart":
                    # === HART Autoregressive (no img2img, no máscara, no denoise) ===
                    print(f"[ImgEditor] Generando con HART (autoregresivo) - prompt puro...")
                    # HART ignora imagen de entrada; usamos el prompt enriquecido
                    hart_prompt = prompt_enhanced or prompt
                    # Intentar load (aunque HART lo hace en generate via subprocess)
                    try:
                        ok, load_msg = client.load()
                        if not ok:
                            return None, load_msg or "HART no disponible", final_mask
                    except Exception:
                        pass  # generate() maneja el spawn

                    result_obj, msg = client.generate(
                        prompt=hart_prompt,
                        num_inference_steps=params.get("num_inference_steps", 8),
                        guidance_scale=params.get("guidance_scale", 4.5),
                    )
                else:
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
                        "lora_strength": lora_strength,
                        "progress_callback": progress_callback,
                    }
                    if "qwen" not in engine:
                        gen_params["mask_image"] = final_mask if mask_mode in ["manual", "global"] else None
                    
                    result_obj, msg = client.generate(**gen_params)
                if result_obj: 
                    result = result_obj.image
                    if result.size != img.size and engine != "hart":
                        result = result.resize(img.size, Image.LANCZOS)
                    
                    if final_mask and engine != "hart":
                        result = self._apply_skin_tone_match(img, result, final_mask)
                        soft_mask = self._feather_mask(final_mask, amount=15)
                        orig_bg = img.copy()
                        result = Image.composite(result, orig_bg, soft_mask)

            if result is not None and (enhance_faces or quality_only):
                result = self._enhance_faces(result)

            if result is not None and (quality_only or auto_upscale):
                try:
                    if progress_callback:
                        progress_callback({"phase": "Mejorando nitidez", "progress": 0.92, "elapsed": 0})
                    from roop.img_editor.image_quality_pipeline import get_quality_finisher
                    finisher = get_quality_finisher()
                    result, fin_note = finisher.finish(
                        result,
                        upscale=True if quality_only else auto_upscale,
                        sharpen_image=True,
                        denoise=quality_only,
                        ultra=quality_only,
                    )
                    msg = f"{msg} | {fin_note}" if fin_note else msg
                    print(f"[ImgEditor] Post-calidad: {fin_note}")
                except Exception as e:
                    print(f"[ImgEditor] Post-calidad omitido: {e}")

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
