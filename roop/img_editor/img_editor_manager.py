#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, time, threading
from typing import Optional, Tuple, Dict
from PIL import Image
import numpy as np
import roop.globals
from roop.utils import get_vram_gb
from concurrent.futures import ThreadPoolExecutor

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
        self.prompt_rewriter = None
        self._analysis_cache = {}
        self._last_context = "normal"
        self._executor = ThreadPoolExecutor(max_workers=4)

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

    def _get_clipseg_masker(self):
        if self.clip_masker is None:
            from roop.img_editor.clipseg_masker import get_clipseg_masker
            self.clip_masker = get_clipseg_masker()
        return self.clip_masker

    def auto_detect_params(self, analysis: Dict, engine: str) -> Dict:
        """
        Calcula parámetros dinámicos basados puramente en el análisis semántico del LLM.
        Optimizado para velocidad en 8GB VRAM.
        """
        magnitude = analysis.get("magnitude", 0.5)
        
        # Escalar denoise (más conservador para evitar artefactos)
        if magnitude < 0.3:
            denoise = 0.25 + (magnitude * 0.5)
        elif magnitude < 0.6:
            denoise = 0.40 + ((magnitude - 0.3) * 0.5)
        else:
            denoise = 0.55 + ((magnitude - 0.6) * 0.5)
            
        denoise = min(denoise, 0.75) # Cap at 0.75 for realism
        
        # Optimización de pasos: 8 a 12 suele ser suficiente para Turbo/Klein
        steps = int(8 + (magnitude * 4))
        
        # Escalar guidance
        guidance = 3.0 + (magnitude * 4.0) # 3.0 to 7.0


        # Ajustes según motor
        if engine == "longcat":
            steps = 8
            guidance = 3.5
        elif engine == "longcat_full":
            steps = max(20, min(steps, 30))
            guidance = 4.5
        elif engine == "flux_schnell":
            steps = 4 # Schnell es extremadamente rápido
        elif "qwen" in engine:
            steps = min(steps, 8)
            guidance = min(guidance, 4.0)
        elif "klein" in engine:
            steps = min(steps, 10)
        elif "abliterated" in engine:
            steps = min(steps, 10)

        print(f"[ImgEditor] Optimization Applied: Denoise={denoise:.2f}, Steps={steps}, Guidance={guidance:.1f}")
        
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

        # Diccionario de traducción para comandos comunes
        translations = {
            "desnuda": "make nude",
            "desnudar": "make nude",
            "quita la ropa": "remove clothes",
            "ponle": "add",
            "cambia": "change",
            "fondo": "background",
            "personas": "people",
            "persona": "person",
            "cara": "face",
            "pelo": "hair",
            "traje": "suit",
            "vestido": "dress",
            "gafas": "glasses",
            "lentes": "glasses"
        }
        
        translated_prompt = user_prompt.lower()
        for es, en in translations.items():
            translated_prompt = translated_prompt.replace(es, en)

        # LIMPIEZA AGRESIVA de partículas del español que ensucian el prompt
        stop_words = [" a ", " la ", " las ", " el ", " los ", " de ", " un ", " una ", " para "]
        for word in stop_words:
            translated_prompt = translated_prompt.replace(word, " ")
        
        # Eliminar espacios extra
        translated_prompt = " ".join(translated_prompt.split())

        if is_longcat:
            # PARA LONGCAT: NUNCA usar contexto de MoonDream si hay una instrucción del usuario.
            # MoonDream describe la escena original y hace que LongCat intente RE-GENERARLA.
            # Al quitar el contexto, LongCat se ve forzado a usar SOLO la imagen visual y tu orden.
            base = f"Instruction: {translated_prompt}" if translated_prompt else img_context
            print(f"[ImgEditor] LongCat Prompt Optimizado: {base}")
        elif img_context:
            base = f"{img_context}. {user_prompt}"
        else:
            base = user_prompt

        if not is_longcat and not base.lower().startswith(("photo", "a photo", "a picture")):
            base = f"Photo of {base}"

        return base

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

    def _apply_skin_tone_match(self, original: Image.Image, generated: Image.Image, mask: Image.Image) -> Image.Image:
        """Ajusta el tono de la piel generada para que coincida con la original"""
        from PIL import ImageStat, ImageEnhance
        
        # Convertir a array para procesamiento rápido
        orig_arr = np.array(original.convert("RGB"))
        gen_arr = np.array(generated.convert("RGB"))
        mask_arr = np.array(mask.convert("L")) / 255.0
        
        # Redimensionar máscara si no coincide con las dimensiones de la imagen generada
        if mask_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            mask_arr = cv2.resize(mask_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Redimensionar imagen original para comparación si las dimensiones no coinciden
        # (Necesario para evitar IndexError cuando el motor de IA cambia el tamaño)
        orig_arr_comp = orig_arr
        if orig_arr.shape[:2] != gen_arr.shape[:2]:
            import cv2
            orig_arr_comp = cv2.resize(orig_arr, (gen_arr.shape[1], gen_arr.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Estimar color promedio de la zona original (donde está la máscara)
        # pero enfocado en el centro de la máscara para evitar bordes
        mask_center = mask_arr > 0.8
        if not np.any(mask_center): mask_center = mask_arr > 0.5
        
        if np.any(mask_center):
            avg_orig = orig_arr_comp[mask_center].mean(axis=0)
            avg_gen = gen_arr[mask_center].mean(axis=0)
            
            # Calcular corrección
            correction = avg_orig / (avg_gen + 1e-6)
            correction = np.clip(correction, 0.8, 1.2) # Limitar para no saturar
            
            # Aplicar corrección solo en la zona de la máscara
            for i in range(3):
                gen_arr[:,:,i] = np.clip(gen_arr[:,:,i] * correction[i], 0, 255)
                
            return Image.fromarray(gen_arr.astype(np.uint8))
        return generated

    def _feather_mask(self, mask: Image.Image, amount: int = 15) -> Image.Image:
        """Aplica un borde suave a la máscara"""
        from PIL import ImageFilter
        return mask.filter(ImageFilter.GaussianBlur(radius=amount))

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

        # Auto-switch a klein_base si no hay VRAM suficiente para Flux Dev
        if engine == "flux_dev_abliterated":
            vram = get_vram_gb()
            if 0 < vram <= 8:
                print(f"[ImgEditor] VRAM={vram}GB detectada. Flux Dev puede ser lento o fallar. Se recomienda Klein o LongCat si hay problemas.")
                # Ya no forzamos el cambio a klein_base
        elif engine == "qwen_edit":
            pass  # Qwen Edit Q2_K cabe bien en 8GB

        # 1. Análisis de la imagen original (describe qué hay para preservar contexto)
        img_description = ""
        if use_rewriter:
            try:
                # Obtener ID único de la imagen o usar el nombre si es archivo
                img_id = str(id(image))
                if hasattr(image, 'name'): img_id = image.name
                
                if img_id in self._analysis_cache:
                    print("[ImgEditor] Usando análisis de imagen cacheado")
                    img_description = self._analysis_cache[img_id]
                else:
                    from scripts.moondream_analyzer import MoonDreamImageAnalyzer
                    analyzer = MoonDreamImageAnalyzer()
                    import tempfile
                    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                        img_small = img.copy()
                        img_small.thumbnail((384, 384))
                        img_small.save(tmp.name)
                        res = analyzer.analyze(tmp.name)
                        img_description = res.get('positive', '')
                        self._analysis_cache[img_id] = img_description
                    
                    # LIBERAR VRAM DE MOONDREAM INMEDIATAMENTE
                    analyzer.unload()
                    del analyzer
                        
                    if img_description:
                        print(f"[ImgEditor] Imagen analizada: {img_description[:100]}...")
            except Exception as e:
                print(f"[ImgEditor] Warning: No se pudo analizar imagen: {e}")
        else:
            print("[ImgEditor] Análisis de imagen omitido (Use AI desactivado)")

        # 2. Análisis Semántico del prompt del usuario
        analysis = {"prompt": prompt, "magnitude": 0.5, "mask_target": "subject"}
        try:
            analyzer = self._get_prompt_analyzer()
            mode, confidence = analyzer.analyze(prompt)
            
            # Estimar magnitud basada en palabras clave de intensidad
            magnitude = 0.5
            text_lower = prompt.lower()
            strong_words = ["completamente", "totalmente", "mucho", "radical", "extremadamente", "todo", 
                            "full", "complete", "total", "radical", "extremely", "huge"]
            weak_words = ["un poco", "ligero", "suave", "sutil", "minimo", "algo",
                          "a bit", "slight", "soft", "subtle", "minimal", "some"]
            
            if any(w in text_lower for w in strong_words): magnitude += 0.25
            if any(w in text_lower for w in weak_words): magnitude -= 0.2
            
            # Detectar target de máscara simple
            mask_target = "subject"
            if any(w in text_lower for w in ["fondo", "background", "paisaje", "escenario"]):
                mask_target = "background"
            elif any(w in text_lower for w in ["ropa", "vestido", "traje", "camisa", "pantalones", "clothes", "outfit", "dress", "suit", "shirt"]):
                mask_target = "clothes"
            elif any(w in text_lower for w in ["cara", "rostro", "face", "ojos", "eyes", "pelo", "hair"]):
                mask_target = "face"
                
            analysis["magnitude"] = max(0.1, min(1.0, magnitude))
            analysis["mask_target"] = mask_target
            analysis["mode"] = mode.value
            print(f"[ImgEditor] Análisis: modo={mode.value}, mag={magnitude:.2f}, target={mask_target}")
        except Exception as e:
            print(f"[ImgEditor] Error en análisis semántico: {e}")

        # 2. Resolución de Parámetros basada en el Análisis
        params = self.auto_detect_params(analysis, engine)
        
        # Componer el prompt final (contexto visual + instrucción del usuario)
        prompt_enhanced = self._compose_generation_prompt(prompt, img_context=img_description, engine=engine)
        mask_target = analysis.get("mask_target", "subject").lower()

        print(f"[ImgEditor] Mask target final: {mask_target}", flush=True)
        print(f"[ImgEditor] Prompt final enviado: {prompt_enhanced}", flush=True)
        analysis["mask_target"] = mask_target
        
        if num_inference_steps: params["num_inference_steps"] = num_inference_steps
        if guidance_scale: params["guidance_scale"] = guidance_scale

        # 3. Máscara Automática
        final_mask = mask_image
        
        # Si no hay máscara manual, intentar generar una con CLIPSeg desde el prompt
        if final_mask is None:
            try:
                print(f"[ImgEditor] Intentando auto-máscara desde prompt: {prompt[:80]}")
                masker = self._get_clipseg_masker()
                auto_mask = masker.generate_mask(img, prompt)
                if auto_mask:
                    final_mask = auto_mask
                    print(f"[ImgEditor] Máscara automática GENERADA desde prompt")
            except Exception as e:
                print(f"[ImgEditor] Error en máscara: {e}")

        if final_mask is None and mask_target in ["body", "subject"]:
             print(f"[ImgEditor] Edición global (sin máscara) para {mask_target}")

        try:
            result = None
            msg = ""
            print(f"[ImgEditor] Iniciando generación con motor: {engine}")
            
            # Obtener el cliente adecuado
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
                version = "q2" # Q2_K para 8GB
            elif engine == "omnigen2":
                from roop.img_editor.omnigen2_gguf_comfy_client import get_omnigen2_comfy_client
                if self.omnigen2_client is None: self.omnigen2_client = get_omnigen2_comfy_client()
                client = self.omnigen2_client
                version = None # OmniGen2 no usa version en load()
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
                # Fallback a LongCat o Klein
                from roop.img_editor.flux_edit_comfy_client import get_flux_edit_comfy_client
                if self.flux_klein_client is None: self.flux_klein_client = get_flux_edit_comfy_client()
                client = self.flux_klein_client
                version = "LongCat-Image-Edit-Turbo-Q4_K_S.gguf"

            # Carga y generación unificada
            if client:
                load_params = {"flux_version": version} if "qwen" not in engine else {"qwen_version": version}
                success, msg = client.load(**load_params)
                if not success:
                    return None, msg, final_mask
                
                print(f"[ImgEditor] Enviando a ComfyUI ({engine}, version={version})...")
                gen_params = {
                    "image": img, "prompt": prompt_enhanced,
                    "num_inference_steps": params["num_inference_steps"],
                    "guidance_scale": params["guidance_scale"],
                    "seed": seed, "denoise": params["denoise"]
                }
                # Solo pasar mask_image si el cliente lo soporta (FluxClient lo soporta, QwenEdit no directamente así)
                if "qwen" not in engine:
                    gen_params["mask_image"] = final_mask if mask_mode in ["manual", "global"] else None
                
                result_obj, msg = client.generate(**gen_params)
                if result_obj: 
                    result = result_obj.image
                    
                    # Asegurar que el resultado coincide con el tamaño original (algunos motores redimensionan)
                    if result.size != img.size:
                        print(f"[ImgEditor] Redimensionando resultado {result.size} -> {img.size}")
                        result = result.resize(img.size, Image.LANCZOS)
                    
                    # APLICAR MEJORAS DE REALISMO POST-GENERACIÓN
                    if final_mask:
                        print("[ImgEditor] Aplicando afinamiento de tono de piel y bordes...")
                        # 1. Emparejar tono de piel con el original
                        result = self._apply_skin_tone_match(img, result, final_mask)
                        
                        # 2. Suavizar bordes (feathering) para integración perfecta
                        soft_mask = self._feather_mask(final_mask, amount=15)
                        
                        # Mezclar imagen generada con la original usando la máscara suave
                        # (Asegurar que el fondo original se mantiene intacto)
                        orig_bg = img.copy()
                        result = Image.composite(result, orig_bg, soft_mask)

            if result is not None and self._should_preserve_faces(face_preserve, analysis, prompt):
                print(f"[ImgEditor] Aplicando preservación de rostro (face_preserve={face_preserve})")
                try:
                    fp = self._get_face_preserver()
                    if fp:
                        result = fp.preserve_faces(img, result, method="swap")
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
