#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, threading, cv2
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import roop.globals
from roop.utils import get_vram_gb

from roop.utilities import resolve_relative_path
class ImgEditorManager:
    def __init__(self):
        self.flux_klein_client = None
        self.flux_schnell_client = None
        self.flux_dev_client = None
        self.flux_dev_abl_client = None
        self.qwen_edit_client = None
        self.omnigen2_client = None
        self.icedit_client = None
        self.clip_masker = None
        self.clothing_segmenter = None
        self.prompt_analyzer = None
        self.semantic_analyzer = None
        self._last_context = "normal"
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
        has_quality = bool(analysis.get("has_quality_request", False))
        body_transform = bool(analysis.get("body_transform", False))
        body_intensity = float(analysis.get("body_transform_intensity", 0.0))

        if quality_only:
            tier = str(analysis.get("enhance_tier", "hd")).lower()
            print(
                f"[ImgEditor] Modo MEJORA UI tier={tier}: "
                "pipeline TOP (LongCat realismo + Lanczos, sin rejilla)"
            )
            return {
                "denoise": 0.0,
                "num_inference_steps": 0,
                "guidance_scale": 0.0,
                "mode": "quality_classical",
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

        high_structural = bool(analysis.get("high_structural", False))
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        background_focus = bool(analysis.get("background_focus", False))
        quality_hybrid = bool(analysis.get("quality_hybrid", False))
        quality_primary = bool(analysis.get("quality_primary", False))

        # Ajustes técnicos mínimos por motor (sin hardcoding semántico)
        if engine in ("longcat", "imagine"):
            # Grok Imagine style (edit engine).
            # Use dynamic magnitude from semantic analyzer.
            # Higher base + stronger scaling to allow full body/clothing changes (undress etc) while keeping photo.
            # Semantic + rewriter decide how much. Turbo forces full denoise anyway.
            denoise = 0.58 + (magnitude * 0.42)
            if body_transform:
                denoise = max(denoise, 0.86 + body_intensity * 0.10)
                denoise = min(denoise, 0.98)
                guidance = 2.3
                steps = max(steps, 28)
            elif quality_primary:
                denoise = min(denoise, 0.42)
            elif quality_hybrid and has_quality and not body_transform:
                denoise = min(denoise, 0.55)
            elif has_quality and magnitude > 0.6:
                denoise = min(denoise, 0.82)
            denoise = min(denoise, 0.98)
            if not body_transform:
                guidance = 2.8 if magnitude > 0.65 else 3.0
            # More steps for high mag to allow better following
            if magnitude > 0.6:
                steps = max(steps, 26)
            if background_focus:
                denoise = 0.40 + (magnitude * 0.22)
                if quality_hybrid or has_quality:
                    denoise = min(denoise, 0.58)
                else:
                    denoise = min(denoise, 0.65)
                guidance = 3.0
                steps = max(steps, 22)
                print(
                    f"[ImgEditor] Fondo focal (máscara): denoise={denoise:.2f}, steps={steps}"
                )
            elif high_structural:
                if structural_bias == "remove":
                    denoise = 1.0
                    steps = max(steps, 26)
                    guidance = 1.0
                    ref_note = "Flux Fill"
                elif structural_bias == "add":
                    denoise = max(denoise, 0.94)
                    denoise = min(denoise, 0.98)
                    steps = max(steps, 30)
                    guidance = 2.5
                    ref_note = "inpaint add"
                else:
                    denoise = max(denoise, 0.92)
                    denoise = min(denoise, 0.98)
                    steps = max(steps, 28)
                    guidance = 2.5
                    ref_note = "offset"
                print(
                    f"[ImgEditor] Structural bias={structural_bias}: "
                    f"denoise={denoise:.2f}, steps={steps}, model={ref_note}"
                )
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

    _STOPWORDS = frozenset({"the", "and", "for", "are", "but", "not", "you", "all", "can",
                            "had", "her", "was", "one", "our", "out", "has", "how", "its",
                            "may", "now", "old", "see", "way", "who", "did", "got", "let",
                            "say", "she", "too", "use", "any", "per", "que", "del", "las",
                            "los", "con", "por", "una"})

    def _light_score(self, prompt_l: str, texts: list) -> float:
        p_words = set(w for w in prompt_l.split() if len(w) > 2 and w not in self._STOPWORDS)
        if not p_words:
            return 0.0
        best = 0.0
        for t in texts:
            t_words = set(w for w in t.lower().split() if len(w) > 2 and w not in self._STOPWORDS)
            if t_words:
                overlap = len(p_words & t_words) / max(len(t_words), 1)
                for pw in p_words:
                    for tw in t_words:
                        if pw in tw or tw in pw:
                            overlap += 0.08
                            break
                best = max(best, overlap)
        return min(1.0, best)

    def _compose_generation_prompt(
        self,
        user_prompt: str,
        img_context: str = "",
        engine: str = "",
        magnitude: float = 0.5,
        quality_only: bool = False,
        has_quality_request: bool = False,
        enhance_tier: str = "hd",
        high_structural: bool = False,
        structural_bias: str = "neutral",
        body_transform: bool = False,
        mask_target: str = "subject",
        quality_hybrid: bool = False,
    ) -> str:
        user_prompt = (user_prompt or "").strip()
        img_context = (img_context or "").strip()
        
        is_longcat = engine in ("longcat", "longcat_full", "imagine") or "longcat" in engine.lower()
        is_hart = engine == "hart"
        mask_target = (mask_target or "subject").lower()
        background_focus = mask_target == "background" and not body_transform
        scene_change = background_focus or (
            high_structural and structural_bias not in ("remove", "add")
        )

        if is_longcat:
            raw = (user_prompt or img_context or "").strip()
            if quality_only:
                tier = (enhance_tier or "hd").lower()
                base = (
                    f"Quality enhance tier={tier} (classical pipeline — no generative edit)"
                )
                print(f"[ImgEditor] Modo mejora tier={tier}: sin prompt generativo (preserva foto)")
                return base

            if high_structural and not background_focus:
                if structural_bias == "remove":
                    preservation = (
                        "Edit this exact photo. Keep the same environment, lighting, camera angle and scene style. "
                        "Completely remove the specified people from the image with no visible trace, ghosting, "
                        "silhouette or leftover body parts. Fill the area naturally with coherent background."
                    )
                elif structural_bias == "add":
                    preservation = (
                        "Edit this exact photo. Keep existing main subjects, lighting and camera perspective. "
                        "Seamlessly add the requested people into the scene with correct scale, shadows, "
                        "perspective and photographic integration."
                    )
                else:
                    preservation = (
                        "Edit this exact photo. Keep the face and identity of existing main subjects. "
                        "Apply the requested scene change (adding or removing people, objects, or elements) "
                        "as completely, strongly and obviously as possible."
                    )
                if has_quality_request:
                    preservation += " Also improve colors, sharpness and photographic quality naturally."
                base = f"Instruction: {raw}. {preservation}"
            elif background_focus or (scene_change and not body_transform):
                preservation = (
                    "Edit this exact photo. Keep the face, identity and main subjects completely unchanged. "
                    "Change and enhance only the background, environment and scenery as described, "
                    "with natural photographic integration and realistic detail."
                )
                if has_quality_request:
                    preservation += (
                        " Also improve colors, sharpness, microcontrast and photographic quality naturally."
                    )
                base = f"Instruction: {raw}. {preservation}"
            elif (
                quality_hybrid
                and has_quality_request
                and not body_transform
                and not background_focus
            ):
                preservation = (
                    "Edit this exact photo. Keep the face and identity. "
                    "Enhance photographic realism, colors, lighting, sharpness, fine detail "
                    "and skin texture across the entire image as described, as strongly as possible."
                )
                base = f"Instruction: {raw}. {preservation}"
            elif body_transform or magnitude > 0.75:
                preservation = (
                    "Edit this exact photo. Keep only the face and identity. "
                    "Change the body, clothing and skin exposure exactly as the instruction says, "
                    "as completely, strongly and obviously as possible."
                )
                if scene_change:
                    preservation += (
                        " Change the background, environment and scenery exactly as requested "
                        "while keeping subjects integrated with the new scene."
                    )
                if has_quality_request:
                    preservation += (
                        " Also remove posterization and AI artifacts; improve colors, vibrance, "
                        "sharpness, skin texture and overall photographic quality naturally."
                    )
                base = f"Instruction: {raw}. {preservation}"
            elif magnitude > 0.6:
                if scene_change:
                    preservation = (
                        "Edit this exact photo. Keep the face, identity and main subjects. "
                        "Change the background, environment and scenery exactly as described. "
                        "Apply all other requested changes as strongly as possible."
                    )
                    base = f"Instruction: {raw}. {preservation}"
                else:
                    base = f"Instruction: {raw}"
                if has_quality_request:
                    base += (
                        " Also improve quality, sharpness, color and detail."
                    )
            else:
                if scene_change:
                    preservation = (
                        "Edit this exact photo. Keep the face, identity and main subjects. "
                        "Change only the background, environment and scenery as requested."
                    )
                else:
                    preservation = (
                        "Edit this exact photo. Keep the face, identity, lighting, background and overall scene exactly the same. "
                        "Apply the requested change to the subject (clothing, body, pose etc as needed)."
                    )
                if has_quality_request:
                    preservation += " Also improve sharpness, color and photographic quality; reduce posterization."
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

    def _mask_coverage(self, mask: Image.Image) -> float:
        arr = np.array(mask.convert("L"))
        return float((arr > 128).sum()) / float(arr.size)

    def _mask_from_face_bboxes(
        self,
        image: Image.Image,
        faces,
        width_mult: float = 4.0,
        height_mult: float = 6.0,
        head_pad: float = 0.5,
    ) -> Optional[Image.Image]:
        w, h = image.size
        union = np.zeros((h, w), dtype=np.uint8)
        for face_obj in faces:
            x1, y1, x2, y2 = [int(v) for v in face_obj.bbox]
            fw, fh = max(1, x2 - x1), max(1, y2 - y1)
            cx = (x1 + x2) // 2
            bw = int(fw * width_mult)
            top = max(0, y1 - int(fh * head_pad))
            bottom = min(h, top + int(fh * height_mult))
            left = max(0, cx - bw // 2)
            right = min(w, cx + bw // 2)
            union[top:bottom, left:right] = 255
        if union.max() == 0:
            return None
        union = cv2.GaussianBlur(union, (31, 31), 0)
        return Image.fromarray(union).convert("L")

    def _release_local_gpu_before_comfy(self):
        """Libera VRAM local (InsightFace/CLIPSeg) antes de que ComfyUI use la GPU."""
        try:
            seg = self.clothing_segmenter
            if seg is not None and getattr(seg, "is_loaded", lambda: False)():
                seg.unload()
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass
        import gc
        gc.collect()
        print("[ImgEditor] VRAM local liberada — ComfyUI puede usar la GPU")

    def _union_grayscale_masks(self, *masks: Image.Image) -> Optional[Image.Image]:
        arrays = []
        w, h = None, None
        for mask in masks:
            if mask is None:
                continue
            arr = np.array(mask.convert("L"))
            if w is None:
                w, h = mask.size
            elif arr.shape[:2] != (h, w):
                arr = cv2.resize(arr, (w, h), interpolation=cv2.INTER_LINEAR)
            arrays.append(arr)
        if not arrays:
            return None
        union = arrays[0].copy()
        for arr in arrays[1:]:
            union = np.maximum(union, arr)
        return Image.fromarray(union).convert("L") if union.max() > 0 else None

    def _build_clipseg_person_mask(
        self,
        image: Image.Image,
        dilation: int = 20,
    ) -> Optional[Image.Image]:
        """CLIPSeg segmentación de cuerpo completo. Múltiples thresholds, sin filtro restrictivo."""
        try:
            import torch
            seg = self._get_clothing_segmenter()
            if not seg or not seg.is_available():
                return None
            prev_dev = getattr(seg, "device", None)
            try:
                seg.device = torch.device("cpu")
                if seg.model is not None:
                    seg.model.to(seg.device)
            except Exception:
                pass

            prompts = [
                "person", "human body", "full body person", "people",
                "man", "woman", "body", "torso",
            ]

            best_mask = None
            best_cov = 0.0
            for th in (0.35, 0.40, 0.50, 0.30, 0.60):
                mask_img, _ = seg.segment_with_prompt(
                    image, prompts,
                    threshold=th, dilation=dilation,
                )
                if mask_img is None or np.array(mask_img).max() == 0:
                    continue
                cov = self._mask_coverage(mask_img)
                if cov > 0.05 and cov > best_cov:
                    best_mask = mask_img
                    best_cov = cov

            if prev_dev is not None:
                try:
                    seg.device = prev_dev
                    if seg.model is not None:
                        seg.model.to(prev_dev)
                except Exception:
                    pass

            if best_mask is not None:
                print(f"[ImgEditor] CLIPSeg persona: cov={best_cov*100:.1f}%")
                return best_mask
        except Exception as e:
            print(f"[ImgEditor] CLIPSeg persona falló: {e}")
        return None

    def _build_person_removal_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """Máscara cuerpo/persona para inpaint remove. CLIPSeg + caras + dilatación agresiva + fallback full-frame."""
        faces = []
        face_count = 0
        try:
            from roop.face_util import get_face_analyser_cpu, preprocess_image_for_detection
            img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            analyser = get_face_analyser_cpu()
            faces = analyser.get(img_bgr) or []
            if len(faces) < 2:
                enhanced = preprocess_image_for_detection(img_bgr)
                extra = analyser.get(enhanced) or []
                seen = {tuple(int(v) for v in f.bbox) for f in faces}
                for f in extra:
                    key = tuple(int(v) for v in f.bbox)
                    if key not in seen:
                        faces.append(f)
                        seen.add(key)
            face_count = len(faces)
        except Exception as e:
            print(f"[ImgEditor] InsightFace persona falló: {e}")

        clipseg_mask = self._build_clipseg_person_mask(image)
        face_mask = None
        if faces:
            face_mask = self._mask_from_face_bboxes(
                image, faces, width_mult=4.0, height_mult=6.0, head_pad=0.5,
            )

        merged = None
        parts = []
        if clipseg_mask is not None:
            merged = clipseg_mask
            parts.append("CLIPSeg")
            if face_mask is not None:
                clip_arr = np.array(clipseg_mask.convert("L"))
                face_arr = np.array(face_mask.convert("L"))
                supplement = np.where((face_arr > 128) & (clip_arr <= 128), 255, 0).astype(np.uint8)
                if supplement.max() > 0:
                    merged = Image.fromarray(np.maximum(clip_arr, supplement)).convert("L")
                    parts.append(f"InsightFace({face_count})")
        elif face_mask is not None:
            merged = face_mask
            parts.append(f"InsightFace({face_count})")

        if merged is not None:
            arr = np.array(merged.convert("L"))
            h, w = arr.shape

            close_k = max(7, min(h, w) // 40)
            arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, np.ones((close_k, close_k), np.uint8))

            dilate_k = max(15, min(h, w) // 20)
            arr = cv2.dilate(arr, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

            cov = (arr > 128).sum() / float(arr.size)
            if cov < 0.30:
                arr = cv2.dilate(arr, np.ones((31, 31), np.uint8), iterations=1)
                cov = (arr > 128).sum() / float(arr.size)

            merged = Image.fromarray(arr).convert("L")
            print(
                f"[ImgEditor] Máscara persona vía {' + '.join(parts)} "
                f"(cov={cov*100:.1f}%)"
            )
            return merged

        if face_count > 0:
            h, w = np.array(image.convert("L")).shape
            full = np.ones((h, w), dtype=np.uint8) * 255
            margin_x, margin_y = w // 20, h // 20
            full[:margin_y, :] = 0
            full[-margin_y:, :] = 0
            full[:, :margin_x] = 0
            full[:, -margin_x:] = 0
            print(f"[ImgEditor] Fallback full-frame mask ({face_count} caras detectadas)")
            return Image.fromarray(full).convert("L")

        return None

    def _build_person_addition_mask(self, image: Image.Image) -> Optional[Image.Image]:
        """
        Máscara INVERSA para ADD person.
        Protege (negro) las áreas con personas existentes (caras + CLIPSeg cuerpo).
        Deja blanco (editables) las áreas libres donde se puede añadir gente nueva.
        Usa CLIPSeg persona + InsightFace + dilatación generosa + feathering suave.
        """
        faces = []
        face_count = 0
        try:
            from roop.face_util import get_face_analyser_cpu, preprocess_image_for_detection
            img_bgr = cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)
            analyser = get_face_analyser_cpu()
            faces = analyser.get(img_bgr) or []
            if len(faces) < 2:
                enhanced = preprocess_image_for_detection(img_bgr)
                extra = analyser.get(enhanced) or []
                seen = {tuple(int(v) for v in f.bbox) for f in faces}
                for f in extra:
                    key = tuple(int(v) for v in f.bbox)
                    if key not in seen:
                        faces.append(f)
                        seen.add(key)
            face_count = len(faces)
        except Exception as e:
            print(f"[ImgEditor] InsightFace para add falló: {e}")

        clipseg_mask = self._build_clipseg_person_mask(image, dilation=15)
        face_mask = None
        if faces:
            face_mask = self._mask_from_face_bboxes(
                image, faces, width_mult=4.0, height_mult=5.0, head_pad=0.4,
            )

        # Unir CLIPSeg + caras en una sola máscara de protección (negro = proteger)
        protection = None
        parts = []

        if clipseg_mask is not None and face_mask is not None:
            clip_arr = np.array(clipseg_mask.convert("L"))
            face_arr = np.array(face_mask.convert("L"))
            # Unión de protección
            protection = np.maximum(clip_arr, face_arr)
            parts.append("CLIPSeg+InsightFace")
        elif clipseg_mask is not None:
            protection = np.array(clipseg_mask.convert("L"))
            parts.append("CLIPSeg")
        elif face_mask is not None:
            protection = np.array(face_mask.convert("L"))
            parts.append(f"InsightFace({face_count})")

        if protection is not None:
            h, w = protection.shape[:2]

            # Dilatar protección para dejar espacio alrededor
            dilate_k = max(25, min(h, w) // 15)
            protection = cv2.dilate(protection, np.ones((dilate_k, dilate_k), np.uint8), iterations=1)

            # Feathering en bordes para transición suave
            blur_k = max(15, min(h, w) // 30)
            if blur_k % 2 == 0:
                blur_k += 1
            protection = cv2.GaussianBlur(protection, (blur_k, blur_k), 0)

            # INVERTIR: lo que estaba blanco (persona) ahora es negro (protegido)
            # y lo que estaba negro (fondo) ahora es blanco (área editable)
            invert = 255 - protection

            # Margen mínimo editable alrededor de bordes de imagen
            invert[:max(3, h//40), :] = 0
            invert[-max(3, h//40):, :] = 0
            invert[:, :max(3, w//40)] = 0
            invert[:, -max(3, w//40):] = 0

            cov = (invert > 128).sum() / float(invert.size)
            if cov < 0.15:
                print(f"[ImgEditor] ADD mask cov={cov*100:.1f}% — muy poca área libre, reduciendo protección")
                # Reducir dilatación para dejar más área
                protection = protection.astype(np.float32)
                protection = cv2.GaussianBlur(protection, (blur_k, blur_k), sigmaX=blur_k*0.3)
                invert = (255 - protection).astype(np.uint8)
                cov = (invert > 128).sum() / float(invert.size)

            if face_count > 0:
                print(
                    f"[ImgEditor] Máscara ADD vía {' + '.join(parts)} "
                    f"(area_libre={cov*100:.1f}%, caras={face_count})"
                )
            else:
                print(
                    f"[ImgEditor] Máscara ADD vía {' + '.join(parts)} "
                    f"(area_libre={cov*100:.1f}%)"
                )
            return Image.fromarray(invert).convert("L")

        # Sin detección: máscara global (todo editable)
        print("[ImgEditor] ADD mask: sin detección → todo editable (global)")
        return None

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

    def _enhance_faces(
        self,
        image: Image.Image,
        progress_callback=None,
        quality_mode: bool = False,
        face_enhance_cap: float = None,
    ) -> Image.Image:
        """Mejora rostros en la imagen usando CodeFormer. Sin hardcoding."""
        cf = self._get_codeformer()
        if cf is None:
            return image

        try:
            if progress_callback:
                progress_callback({"phase": "Mejorando caras", "progress": 0.94, "detail": "CodeFormer"})
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
                cap = float(face_enhance_cap) if face_enhance_cap is not None else (0.32 if quality_mode else 0.30)
                if quality_mode:
                    blend = float(np.clip(det_score * 0.20, 0.08, cap))
                else:
                    blend = float(np.clip(det_score * 0.15, 0.05, min(cap, 0.30)))

                fh, fw = face_crop.shape[:2]
                cx, cy = fw // 2, fh // 2
                axes = (max(8, int(fw * 0.46)), max(8, int(fh * 0.52)))
                mask = np.zeros((fh, fw), dtype=np.float32)
                cv2.ellipse(mask, (cx, cy), axes, 0, 0, 360, 1.0, -1)
                mask = cv2.GaussianBlur(mask, (0, 0), max(6.0, min(fh, fw) * 0.08))
                mask *= float(blend)
                mask3 = mask[..., None]
                roi = result[y1:y2, x1:x2].astype(np.float32)
                result[y1:y2, x1:x2] = (
                    roi * (1.0 - mask3) + enhanced_bgr.astype(np.float32) * mask3
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
        use_rewriter: bool = False,
        use_semantic: bool = True,
        ref_metadata: dict = None,
        engine: str = "imagine",
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = "",
        enhance_faces: bool = True,
        lora_name: str = None,
        lora_strength: float = 1.0,
        denoise: float = None,
        progress_callback=None,
        auto_upscale: bool = False,
        quality_mode: bool = False,
        enhance_tier: str = "hd",
        quality_enhance_style: str = "auto",
        quality_use_generative: bool = True,
        quality_preserve_faces: bool = True,
        face_preserve: bool = True,
        cancel_check=None,
    ) -> Tuple[Optional[Image.Image], str, Optional[Image.Image]]:

        if cancel_check and cancel_check():
            return None, "Cancelado", None

        if isinstance(image, Image.Image):
            img = image.copy().convert("RGB")
        else:
            img = Image.open(image.name).copy().convert("RGB")

        from roop.img_editor.prompt_translator import translate_prompt
        prompt_raw = (prompt or "").strip()
        quality_enhance = bool(quality_mode)
        quality_only = quality_enhance and not prompt_raw
        quality_hybrid = quality_enhance and bool(prompt_raw)
        quality_style = (quality_enhance_style or "auto").lower()
        if quality_style not in ("auto", "solo_mejora", "hibrido"):
            quality_style = "auto"
        enhance_tier = (enhance_tier or "hd").lower()
        if enhance_tier not in ("hd", "4k", "8k"):
            enhance_tier = "hd"
        if quality_only:
            prompt = ""
            use_rewriter = False
            print(f"[ImgEditor] Modo mejora imagen (UI tier={enhance_tier}) — sin instrucción de prompt")
        elif quality_hybrid:
            prompt = translate_prompt(prompt_raw)
            use_rewriter = False
            short = prompt_raw if len(prompt_raw) <= 64 else prompt_raw[:64] + "…"
            print(
                f"[ImgEditor] Modo híbrido tier={enhance_tier} — "
                f"edit: «{short}» + mejora TOP después"
            )
        else:
            prompt = translate_prompt(prompt_raw)
        negative_prompt = (negative_prompt or "").strip()

        if engine == "flux_dev_abliterated":
            vram = get_vram_gb()
            if 0 < vram <= 8:
                print(f"[ImgEditor] VRAM={vram}GB detectada. Flux Dev puede ser lento o fallar.")
        elif engine == "qwen_edit":
            pass

        img_description = ""  # DELIBERADAMENTE vacío: no usamos descripciones generativas de imagen (evita alucinaciones). El análisis es texto-del-prompt + preservación fuerte.

        structural_score = 0.0
        high_structural = False
        structural_bias = "neutral"
        body_transform_intensity = 0.0
        background_focus = False
        if quality_only:
            mag_suggested = 0.35
            mask_target = "subject"
            axis_scores = {}
            is_global = True
        elif use_semantic:
            try:
                nlp = self._get_semantic_analyzer(full_ai=False)
                mag_suggested = nlp.get_magnitude(prompt)
                mask_target = nlp.detect_target(prompt)
                axis_scores = nlp.get_axis_scores(prompt) if hasattr(nlp, "get_axis_scores") else {}
                structural_score = float(axis_scores.get("structural", 0.0))
                if hasattr(nlp, "get_body_transform_intensity"):
                    body_transform_intensity = float(nlp.get_body_transform_intensity(prompt))
                if hasattr(nlp, "is_structural_dominant"):
                    high_structural = nlp.is_structural_dominant(prompt)
                else:
                    struct_thresh = getattr(nlp, "STRUCTURAL_SIGNAL_THRESHOLD", 0.06)
                    high_structural = structural_score > struct_thresh
                if hasattr(nlp, "get_structural_bias"):
                    structural_bias = nlp.get_structural_bias(prompt)
                background_focus = (
                    mask_target == "background" and structural_bias == "neutral"
                )
                is_global = (
                    not background_focus
                    and (
                        high_structural
                        or (mask_target == "subject" and mag_suggested < 0.45)
                    )
                )
                print(
                    f"[ImgEditor] Semantic: Magnitude={mag_suggested:.2f}, Target={mask_target}, "
                    f"Structural={structural_score:.2f}, Bias={structural_bias}, "
                    f"BodyTransform={body_transform_intensity:.2f}"
                    + (", BackgroundFocus=True" if background_focus else "")
                )
            except Exception as e:
                print(f"[NLP] Error en análisis semántico: {e}. Usando fallback.")
                mag_suggested = 0.58
                mask_target = "subject"
                structural_score = 0.0
                high_structural = False
                structural_bias = "neutral"
                axis_scores = {}
                background_focus = False
                is_global = True
        else:
            mag_suggested = 0.58
            mask_target = "subject"
            structural_score = 0.0
            high_structural = False
            structural_bias = "neutral"
            axis_scores = {}
            background_focus = False
            is_global = True
            print("[ImgEditor] Análisis semántico desactivado — params por defecto")

        body_transform = (
            not quality_only
            and body_transform_intensity >= 0.08
            and mag_suggested >= 0.52
        )
        if body_transform:
            mag_suggested = max(mag_suggested, 0.68 + body_transform_intensity * 0.12)
            is_global = True
            background_focus = False
        elif not quality_only:
            background_focus = (
                mask_target == "background"
                and structural_bias == "neutral"
                and not body_transform
            )
            if background_focus:
                is_global = False

        quality_anchors = [
            "improve quality sharpness detail resolution clarity enhance deblur upscale",
            "mejorar calidad nitidez detalle sharper mejorar color realzar ultra realista hiperrealista",
            "better colors vibrant lighting higher detail photographic quality",
            "desposterizar posterización posterization natural skin texture remove artifacts photographic realism",
        ]
        prompt_lower = prompt.lower()
        has_quality_request = self._light_score(prompt_lower, quality_anchors) > 0.03

        env_anchors = [
            "entorno", "environment", "fondo", "background", "escena", "scenery",
            "paisaje", "ambiente", "surroundings", "landscape",
        ]
        bg_in_prompt = self._light_score(prompt_lower, env_anchors) > 0.04
        subject_scene_anchors = [
            "personajes", "personas", "characters", "people", "sujetos", "subjects",
            "personaje", "persona", "cuerpos", "bodies", "caras", "faces", "figuras",
        ]
        subject_in_prompt = self._light_score(prompt_lower, subject_scene_anchors) > 0.03
        global_scene_realism = (
            bg_in_prompt and subject_in_prompt and has_quality_request and quality_hybrid
        )
        if global_scene_realism and not quality_only:
            background_focus = False
            is_global = True
            mask_target = "subject"
            high_structural = False
            structural_bias = "neutral"
            mag_suggested = min(mag_suggested, 0.52)
            print(
                "[ImgEditor] Fondo + personajes realismo → edición global de escena + TOP "
                "(no solo fondo, no structural add)"
            )
        elif bg_in_prompt and quality_hybrid and not quality_only:
            mask_target = "background"
            background_focus = True
            is_global = False

        quality_primary = False
        if quality_style == "solo_mejora" and quality_hybrid:
            if bg_in_prompt:
                mask_target = "background"
                background_focus = True
                is_global = False
                print(
                    "[ImgEditor] Prompt pide fondo/entorno → edit de fondo + TOP "
                    "(«Solo mejora» no aplica a cambios de escena)"
                )
            else:
                quality_primary = True
                mag_suggested = min(mag_suggested, 0.42)
                is_global = True
                print(
                    "[ImgEditor] Modo solo mejora (UI) → pipeline TOP clásico "
                    "(sin edit LongCat fuerte)"
                )
        elif quality_style == "hibrido":
            quality_primary = False
            if quality_hybrid:
                print("[ImgEditor] Modo híbrido forzado (UI) → edit + mejora TOP")
        elif (
            quality_hybrid
            and has_quality_request
            and not body_transform
            and not high_structural
            and not background_focus
            and not bg_in_prompt
        ):
            nlp = self._get_semantic_analyzer(full_ai=False)
            if hasattr(nlp, "is_quality_dominant") and nlp.is_quality_dominant(prompt):
                quality_primary = True
                mag_suggested = min(mag_suggested, 0.42)
                is_global = True
                print(
                    "[ImgEditor] Híbrido calidad-primaria (auto) → pipeline TOP clásico "
                    "(sin edit LongCat fuerte; evita sobre-procesar)"
                )

        if quality_hybrid and not quality_only and bg_in_prompt and not global_scene_realism:
            print(
                "[ImgEditor] Híbrido fondo/entorno → edit focal de fondo + TOP "
                "(personas preservadas)"
            )
        elif (
            quality_hybrid
            and not quality_only
            and not quality_primary
            and has_quality_request
            and not body_transform
            and not background_focus
        ):
            is_global = True
            print(
                "[ImgEditor] Híbrido calidad → edición global "
                "(sin máscara, evita costuras)"
            )

        analysis = {
            "prompt": prompt,
            "magnitude": mag_suggested,
            "mask_target": mask_target,
            "is_global": is_global,
            "quality_only": quality_only,
            "quality_enhance": quality_enhance,
            "quality_hybrid": quality_hybrid,
            "quality_primary": quality_primary,
            "quality_enhance_style": quality_style,
            "quality_use_generative": bool(quality_use_generative),
            "quality_preserve_faces": bool(quality_preserve_faces),
            "has_quality_request": has_quality_request or quality_enhance,
            "enhance_tier": enhance_tier,
            "structural_score": structural_score,
            "high_structural": high_structural,
            "structural_bias": structural_bias,
            "axis_scores": axis_scores,
            "body_transform_intensity": body_transform_intensity,
            "body_transform": body_transform,
            "background_focus": background_focus,
            "global_scene_realism": global_scene_realism,
        }

        # 3. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
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
        elif high_structural and not negative_prompt:
            if structural_bias == "remove":
                negative_prompt = (
                    "people visible, person in frame, leftover figure, ghost silhouette, "
                    "incomplete removal, duplicate person, partial body, blurry patch"
                )
            elif structural_bias == "add":
                negative_prompt = (
                    "floating figure, cutout collage, mismatched lighting, pasted person, "
                    "wrong scale, extra limbs, blurry face, bad integration"
                )
        elif body_transform and not negative_prompt:
            negative_prompt = (
                "wearing clothes, clothed, dressed, fabric covering body, underwear, bra, panties, "
                "partial clothing, incomplete exposure, censored, covered skin, leftover garment"
            )

        has_quality_request = bool(analysis.get("has_quality_request", False))
        high_structural = bool(analysis.get("high_structural", False))
        structural_bias = str(analysis.get("structural_bias", "neutral"))
        if high_structural and structural_bias == "remove":
            has_quality_request = False
        analysis["has_quality_request"] = has_quality_request
        mask_target = str(analysis.get("mask_target", "subject")).lower()
        prompt_enhanced = self._compose_generation_prompt(
            prompt, img_context=img_description, engine=engine, magnitude=mag_suggested,
            quality_only=quality_only, has_quality_request=has_quality_request,
            enhance_tier=enhance_tier, high_structural=high_structural,
            structural_bias=structural_bias, body_transform=body_transform,
            mask_target=mask_target, quality_hybrid=quality_hybrid,
        )
        is_global = bool(analysis.get("is_global", False))
        mag_suggested = float(analysis.get("magnitude", mag_suggested))

        print(f"[ImgEditor] Target: {mask_target} (Global: {is_global})", flush=True)
        print(f"[ImgEditor] Prompt final: {prompt_enhanced}", flush=True)
        if quality_only:
            print(
                f"[ImgEditor] Pipeline TOP: desposterizar + LongCat realismo + "
                f"Lanczos tier={enhance_tier} (sin rejilla ONNX)"
            )
        elif has_quality_request and mag_suggested > 0.6:
            print("[ImgEditor] Compound prompt detected (main change + quality/color polish)")
        elif body_transform:
            print(
                f"[ImgEditor] Cambio corporal/ropa (semántico, intensity={body_transform_intensity:.2f}) "
                "→ global + LongCat Full + ref offset"
            )
        elif background_focus:
            print(
                "[ImgEditor] Edición de fondo focalizada — máscara clipseg, denoise moderado, "
                "sujetos preservados"
            )

        # No automatic anti-clothing negative anymore.
        # It was fighting requests like "ropa raída con mangas".
        # Prompt del usuario + análisis semántico local + preservación mag-dependent.
        # This way it adapts to ANY clothing request (add, modify, remove, specific style) without lists or hardcodes.
        # The semantic + preservation text is enough to guide the model.

        final_mask = mask_image

        if high_structural and structural_bias == "remove" and final_mask is None:
            final_mask = self._build_person_removal_mask(img)
            if final_mask is None:
                print("[ImgEditor] Structural remove: sin máscara persona — global agresivo")
        elif high_structural and structural_bias == "add" and final_mask is None:
            final_mask = self._build_person_addition_mask(img)
            if final_mask is None:
                print("[ImgEditor] Structural add: sin máscara protección — global")

        structural_inpaint = (
            high_structural and structural_bias in ("remove", "add") and final_mask is not None
        )

        # Intentar máscara de ropa si los ejes semánticos indican cambio de atributo (ropa/color)
        # y NO cambio de pose. Así la decisión es semántica, sin hardcoding de targets.
        # Para alta magnitud (>= 0.62) se salta la máscara: cambios de cuerpo necesitan
        # edición global para que el modelo genere piel/tono consistentes sin mismatch de color.
        skip_focal_cloth = (
            global_scene_realism
            or quality_primary
            or (quality_hybrid and is_global and has_quality_request)
        )
        if (
            final_mask is None
            and 0.45 <= mag_suggested < 0.62
            and not body_transform
            and not skip_focal_cloth
        ):
            axes = analysis.get("axis_scores", {})
            attr_score = axes.get("attribute", 0)
            pose_score = axes.get("pose", 0)
            is_attribute_change = attr_score > pose_score
            if is_attribute_change or (attr_score > 0 and pose_score == 0):
                try:
                    seg = self._get_clothing_segmenter()
                    if seg:
                        cloth_mask, _ = seg.segment_clothing(img, threshold=0.45)
                        if cloth_mask is not None and self._mask_coverage(cloth_mask) > 0.03:
                            final_mask = cloth_mask
                            print(f"[ImgEditor] Máscara de ropa para edit focalizado (attr={attr_score:.3f} > pose={pose_score:.3f})")
                except Exception as e:
                    print(f"[ImgEditor] ClothingSegmenter no disponible: {e}")

        # force_global solo cuando NO hay máscara y es cambio estructural o de alta magnitud.
        # La máscara de ropa ya se intentó arriba; si existe, se usará.
        force_global = final_mask is None and not background_focus and (
            high_structural
            or mag_suggested > 0.6
            or body_transform
            or (quality_hybrid and has_quality_request and not quality_primary)
        )

        use_flux_structural_inpaint = bool(structural_inpaint and structural_bias == "remove")

        if use_flux_structural_inpaint:
            print("[ImgEditor] Structural remove → Flux Fill inpaint (máscara, sin Kontext)")
        elif structural_inpaint:
            bias_label = structural_bias.upper()
            print(f"[ImgEditor] Structural {bias_label}: inpaint LongCat + denoise alto")
        elif force_global:
            reason = "cambio estructural (ancla)" if high_structural else "alta magnitud"
            bias_note = f", bias={structural_bias}" if high_structural else ""
            ref_label = "offset" if not high_structural else "index"
            print(f"[ImgEditor] Edición global ({reason}{bias_note}) — ref {ref_label}, denoise alto")
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

            quality_primary = bool(analysis.get("quality_primary", False))
            if quality_only or quality_primary:
                label = "calidad-primaria" if quality_primary else "mejora UI"
                print(
                    f"[ImgEditor] Pipeline TOP tier={enhance_tier} ({label}) — "
                    "LongCat realismo + Lanczos (sin rejilla ONNX)"
                )
                result = img.copy()
                msg = f"Mejora TOP tier={enhance_tier}"
                if progress_callback:
                    progress_callback({
                        "phase": "Mejorando imagen",
                        "progress": 0.04,
                        "detail": f"TOP {enhance_tier} {img.size[0]}×{img.size[1]}",
                    })
            else:
                client = None
                version = None
                flux_structural_override = False

                if use_flux_structural_inpaint and engine in ("longcat", "imagine", "longcat_full"):
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    from roop.img_editor.icedit_comfy_client import ICEditComfyClient
                    if self.flux_dev_abl_client is None:
                        self.flux_dev_abl_client = get_flux_edit_comfy_client()
                    client = self.flux_dev_abl_client
                    icedit = ICEditComfyClient()
                    fill_path = icedit._resolve_flux_fill(
                        os.path.join(
                            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                            "ui", "tob", "ComfyUI", "models",
                        )
                    )
                    version = os.path.basename(fill_path) if os.path.isfile(fill_path) else "flux1-dev-Q2_K.gguf"
                    flux_structural_override = True
                elif engine in ("longcat", "imagine"):
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    if structural_inpaint and not use_flux_structural_inpaint:
                        bias_label = structural_bias.upper()
                        version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                        print(
                            f"[ImgEditor] Structural {bias_label} → {version} "
                            "(inpaint, denoise alto)"
                        )
                    elif (mag_suggested >= 0.62 or body_transform) and not background_focus:
                        version = "LongCat-Image-Edit-Q4_K_S.gguf"
                        reason = "body transform" if body_transform else f"mag={mag_suggested:.2f}"
                        print(f"[ImgEditor] Alta intensidad ({reason}) → LongCat Full")
                    elif background_focus:
                        version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                        print("[ImgEditor] Fondo focal → LongCat Turbo (inpaint suave)")
                    else:
                        version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"
                elif engine == "longcat_full":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "LongCat-Image-Edit-Q4_K_S.gguf"
                elif engine == "flux_dev_abliterated":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_dev_abl_client is None:
                        self.flux_dev_abl_client = get_flux_edit_comfy_client()
                    client = self.flux_dev_abl_client
                    version = "T8-flux.1-dev-abliterated-V2-GGUF-Q4_K_M.gguf"
                elif engine == "qwen_edit":
                    from roop.img_editor.qwen_edit_comfy_client import get_qwen_edit_comfy_client
                    if self.qwen_edit_client is None:
                        self.qwen_edit_client = get_qwen_edit_comfy_client()
                    client = self.qwen_edit_client
                    version = "q2"
                elif engine == "omnigen2":
                    from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                    if self.omnigen2_client is None:
                        self.omnigen2_client = get_omnigen2_comfy_client()
                    client = self.omnigen2_client
                    version = None
                elif engine == "klein_base":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "flux-2-klein-base-4b-Q4_K_S.gguf"
                elif engine == "flux_q2":
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_dev_abl_client is None:
                        self.flux_dev_abl_client = get_flux_edit_comfy_client()
                    client = self.flux_dev_abl_client
                    version = "flux1-dev-Q2_K.gguf"
                elif engine == "hart":
                    from roop.img_editor.hart_edit_comfy_client import get_hart_edit_comfy_client
                    if not hasattr(self, 'hart_client') or self.hart_client is None:
                        self.hart_client = get_hart_edit_comfy_client()
                    client = self.hart_client
                    version = None
                else:
                    from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                    if self.flux_klein_client is None:
                        self.flux_klein_client = get_flux_edit_comfy_client()
                    client = self.flux_klein_client
                    version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

                if client and not (use_flux_structural_inpaint and result is not None):
                    if engine == "hart":
                        print(f"[ImgEditor] Generando con HART (autoregresivo) - prompt puro...")
                        hart_prompt = prompt_enhanced or prompt
                        try:
                            ok, load_msg = client.load()
                            if not ok:
                                return None, load_msg or "HART no disponible", final_mask
                        except Exception:
                            pass

                        result_obj, msg = client.generate(
                            prompt=hart_prompt,
                            num_inference_steps=params.get("num_inference_steps", 8),
                            guidance_scale=params.get("guidance_scale", 4.5),
                        )
                    else:
                        self._release_local_gpu_before_comfy()
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
                        if flux_structural_override:
                            gen_params["denoise"] = min(0.98, max(params["denoise"], 0.97))
                            gen_params["grow_mask_by"] = 48
                            gen_params["mask_image"] = final_mask
                            gen_params["num_inference_steps"] = max(26, params["num_inference_steps"])
                            gen_params["guidance_scale"] = 1.0
                        elif high_structural and engine in ("longcat", "imagine", "longcat_full"):
                            if structural_inpaint and structural_bias == "remove":
                                gen_params["reference_latents_method"] = "index"
                                gen_params["grow_mask_by"] = 40
                                gen_params["denoise"] = min(0.98, max(params["denoise"], 0.97))
                            elif structural_inpaint and structural_bias == "add":
                                gen_params["reference_latents_method"] = "offset"
                                gen_params["grow_mask_by"] = 30
                                gen_params["denoise"] = min(0.98, max(params["denoise"], 0.94))
                            else:
                                gen_params["reference_latents_method"] = "offset"
                        elif force_global and (body_transform or not high_structural):
                            gen_params["reference_latents_method"] = "offset"
                        if "qwen" not in engine and not flux_structural_override:
                            gen_params["mask_image"] = final_mask if mask_mode in ["manual", "global"] else None

                        if cancel_check and cancel_check():
                            return None, "Cancelado", final_mask
                        gen_params["cancel_check"] = cancel_check
                        result_obj, msg = client.generate(**gen_params)
                        if cancel_check and cancel_check():
                            return None, "Cancelado", final_mask
                    if result_obj:
                        result = result_obj.image
                        if result.size != img.size and engine != "hart":
                            result = result.resize(img.size, Image.LANCZOS)

                        if final_mask and engine != "hart":
                            if flux_structural_override or structural_inpaint:
                                soft_mask = self._feather_mask(final_mask, amount=25)
                                result = Image.composite(result, img.copy(), soft_mask)
                            else:
                                result = self._apply_skin_tone_match(img, result, final_mask)
                                feather_amt = 32 if (quality_hybrid or background_focus) else 20
                                soft_mask = self._feather_mask(final_mask, amount=feather_amt)
                                result = Image.composite(result, img.copy(), soft_mask)

            quality_primary = bool(analysis.get("quality_primary", False))
            if result is not None and (quality_only or quality_primary or quality_hybrid or auto_upscale):
                if cancel_check and cancel_check():
                    return None, "Cancelado", final_mask
                try:
                    q_gen = bool(analysis.get("quality_use_generative", True))
                    q_faces = bool(analysis.get("quality_preserve_faces", True))
                    if quality_only or quality_primary:
                        from roop.img_editor.top_tier_finish import run_top_tier_quality
                        result, fin_note, quality_analysis = run_top_tier_quality(
                            result,
                            enhance_tier=enhance_tier,
                            use_generative=q_gen,
                            preserve_faces=q_faces,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                    elif quality_hybrid:
                        from roop.img_editor.top_tier_finish import run_quality_enhance_after_edit
                        result, fin_note, quality_analysis = run_quality_enhance_after_edit(
                            result,
                            img,
                            enhance_tier=enhance_tier,
                            use_generative=q_gen,
                            preserve_faces=q_faces,
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                    else:
                        from roop.img_editor.image_quality_pipeline import get_quality_finisher
                        finisher = get_quality_finisher()
                        result, fin_note, quality_analysis = finisher.finish(
                            result,
                            upscale=auto_upscale,
                            enhance_tier="hd",
                            sharpen_image=True,
                            denoise=False,
                            ultra=False,
                            depixelize_image=False,
                            tile_restore=False,
                            cancel_check=cancel_check,
                            progress_callback=progress_callback,
                        )
                    self._last_quality_analysis = quality_analysis
                    if cancel_check and cancel_check():
                        return None, "Cancelado", final_mask
                    if quality_analysis and quality_analysis.get("summary"):
                        msg = f"{msg} | {quality_analysis['summary']}"
                    msg = f"{msg} | {fin_note}" if fin_note else msg
                    print(f"[ImgEditor] Post-calidad: {fin_note}")
                except Exception as e:
                    print(f"[ImgEditor] Post-calidad omitido: {e}")

            polish_analysis = None
            if result is not None and not quality_only and not quality_hybrid:
                try:
                    from roop.img_editor.top_tier_finish import (
                        run_top_tier_edit_finish,
                        should_edit_top_finish,
                    )
                    if should_edit_top_finish(analysis):
                        print("[ImgEditor] Acabado TOP post-edit (LongCat pulido + caras)")
                        result, polish_note, polish_analysis = run_top_tier_edit_finish(
                            result,
                            img,
                            analysis,
                            enhance_tier=enhance_tier or "hd",
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                        if polish_note and "omitido" not in polish_note:
                            msg = f"{msg} | {polish_note}" if msg else polish_note
                            print(f"[ImgEditor] Post-acabado TOP: {polish_note}")
                    else:
                        from roop.img_editor.hyperreal_polish import polish_result_image
                        result, polish_note, polish_analysis = polish_result_image(
                            result, tier=enhance_tier or "hd",
                        )
                        if polish_note and "omitido" not in polish_note:
                            msg = f"{msg} | {polish_note}" if msg else polish_note
                            print(f"[ImgEditor] Post-acabado: {polish_note}")
                except Exception as e:
                    print(f"[ImgEditor] Post-acabado omitido: {e}")

            if result is not None and enhance_faces and not quality_enhance:
                skip_cf = False
                try:
                    from roop.img_editor.top_tier_finish import should_edit_top_finish
                    skip_cf = bool(analysis and should_edit_top_finish(analysis))
                except Exception:
                    skip_cf = False
                if not skip_cf:
                    face_cap = None
                    qa = getattr(self, "_last_quality_analysis", None) or polish_analysis or {}
                    if qa.get("profile"):
                        face_cap = qa["profile"].get("face_enhance_cap")
                    result = self._enhance_faces(
                        result,
                        progress_callback=progress_callback,
                        quality_mode=False,
                        face_enhance_cap=face_cap,
                    )

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
