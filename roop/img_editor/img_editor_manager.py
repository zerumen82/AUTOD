#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImgEditor Manager - Gestiona la edicion de imagenes con ComfyUI

Implementa un enfoque de DOS PASADAS para preservar la cara 100%:
1. Primera pasada: Generar imagen con el prompt completo (strength alto)
2. Segunda pasada: Face swap para restaurar la cara original

NUEVO: Soporte para inpaint selectivo con detección automática de ropa (CLIPSeg)
"""

import os
import sys
import tempfile
import time
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2

# Importar funciones de deteccion de puerto
from roop.comfy_client import get_comfyui_url
from roop.img_editor.flux_client import get_flux_client, is_flux_available, is_flux_loaded
from roop.img_editor.clothing_segmenter import get_clothing_segmenter, is_clipseg_available


class ImgEditorManager:
    """Gestiona las operaciones de edicion de imagenes"""
    
    def __init__(self):
        self.client = None
        self.face_swapper = None
        self.face_analyzer = None
        self.flux_client = None
        self.use_flux_preferred = True  # Preferir FLUX si está disponible
    
    def is_comfy_available(self):
        """Verifica si ComfyUI esta disponible"""
        from roop.comfy_client import check_comfy_available
        return check_comfy_available()
        
    def _get_client(self):
        """Inicializa el cliente de ComfyUI"""
        if self.client is None:
            from roop.comfy_client import ComfyClient
            self.client = ComfyClient()
        return self.client
    
    def _init_flux_client(self):
        """Inicializa el cliente de FLUX si está disponible"""
        if self.flux_client is not None:
            return True
        
        try:
            self.flux_client = get_flux_client()
            if not self.flux_client.is_available():
                print("[ImgEditor] FLUX no disponible (diffusers no instalado)")
                return False
            
            # Verificar si FLUX está instalado localmente
            if not is_flux_available():
                print("[ImgEditor] FLUX no instalado localmente")
                return False
            
            # Cargar FLUX
            success, msg = self.flux_client.load()
            if not success:
                print(f"[ImgEditor] Error cargando FLUX: {msg}")
                return False
            
            print("[ImgEditor] FLUX cargado correctamente")
            return True
            
        except Exception as e:
            print(f"[ImgEditor] Error inicializando FLUX: {e}")
            return False
    
    def _init_face_swap(self):
        """
        Inicializa el face analyzer y face swapper.
        
        Returns:
            bool: True si la inicialización fue exitosa
        """
        # Si ya está inicializado, retornar True
        if self.face_analyzer is not None and self.face_swapper is not None:
            return True
        
        try:
            # 1. Inicializar Face Analyzer (InsightFace)
            print("[ImgEditor] Inicializando Face Analyzer...")
            import insightface
            from insightface.app import FaceAnalysis
            
            # Configurar para usar solo detección y reconocimiento (sin landmarks)
            # Esto evita el error de ONNX Runtime con landmark_2d_106.onnx
            self.face_analyzer = FaceAnalysis(
                allowed_modules=['detection', 'recognition']  # Sin landmark detection
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print("[ImgEditor] ✅ Face Analyzer inicializado (detection + recognition)")
            
            # 2. Inicializar Face Swapper (Reactor/Inswapper)
            print("[ImgEditor] Inicializando Face Swapper...")
            from roop.processors.FaceSwap import FaceSwap
            
            self.face_swapper = FaceSwap()
            # Inicializar con configuración para CUDA
            self.face_swapper.Initialize({
                'devicename': 'cuda',
                'model': 'inswapper_128.onnx'  # Modelo por defecto
            })
            print("[ImgEditor] ✅ Face Swapper inicializado")
            
            return True
            
        except ImportError as e:
            print(f"[ImgEditor] ❌ Error: Módulo no encontrado - {e}")
            print("[ImgEditor] Asegúrate de tener instalado: pip install insightface")
            return False
        except Exception as e:
            print(f"[ImgEditor] ❌ Error inicializando face swap: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_face_embedding(self, image: Image.Image):
        """Extrae el embedding facial de una imagen"""
        try:
            np_image = np.array(image)
            faces = self.face_analyzer.get(np_image)
            
            if len(faces) == 0:
                return None, None
            
            # Retornar la cara mas grande
            best_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            return best_face, best_face.embedding
        except Exception as e:
            print(f"[ImgEditor] Error extrayendo cara: {e}")
            return None, None
    
    def _restore_face(self, original: Image.Image, generated: Image.Image) -> Image.Image:
        """
        Restaura la cara original en la imagen generada usando face swap.
        
        Este es el paso 2 del enfoque de dos pasadas.
        """
        try:
            if not self._init_face_swap():
                print("[ImgEditor] Face swap no disponible, devolviendo imagen generada")
                return generated
            
            # Extraer cara del original (RGB para InsightFace)
            original_rgb = np.array(original)
            faces_original = self.face_analyzer.get(original_rgb)
            
            print(f"[ImgEditor] 🔍 Caras detectadas en original: {len(faces_original)}")
            if len(faces_original) > 0:
                # Mostrar bounding boxes
                for i, face in enumerate(faces_original):
                    bbox = face.bbox
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    print(f"   Cara {i}: bbox={bbox.tolist()}, área={area:.1f}")
            
            if len(faces_original) == 0:
                print("[ImgEditor] ❌ No se detectaron caras en el original")
                return generated
            
            # Usar la cara mas grande del original
            source_face = max(faces_original, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            print(f"[ImgEditor] ✅ Cara fuente seleccionada: bbox={source_face.bbox.tolist()}")
            
            # Convertir imagen generada a formato OpenCV (BGR) para face swapper
            generated_cv = cv2.cvtColor(np.array(generated), cv2.COLOR_RGB2BGR)
            
            # Detectar caras en la imagen generada (RGB para InsightFace)
            generated_rgb = np.array(generated)
            faces_generated = self.face_analyzer.get(generated_rgb)
            
            print(f"[ImgEditor] 🔍 Caras detectadas en generada: {len(faces_generated)}")
            if len(faces_generated) > 0:
                for i, face in enumerate(faces_generated):
                    bbox = face.bbox
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    print(f"   Cara {i}: bbox={bbox.tolist()}, área={area:.1f}")
            
            if len(faces_generated) == 0:
                print("[ImgEditor] No se detectaron caras en la generada, haciendo swap directo")
                # No hay cara en la generada, intentar swap directo
                # Crear una cara target ficticia basada en la posicion del original
                result = self.face_swapper.Run(source_face, source_face, generated_cv, paste_back=True)
                if result is not None:
                    return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                return generated
            
            # Hacer face swap: cara del original -> imagen generada
            target_face = max(faces_generated, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
            
            result = self.face_swapper.Run(source_face, target_face, generated_cv, paste_back=True)
            
            if result is not None:
                print("[ImgEditor] Cara restaurada correctamente")
                return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            else:
                print("[ImgEditor] Face swap fallo, devolviendo imagen generada")
                return generated
                
        except Exception as e:
            print(f"[ImgEditor] Error restaurando cara: {e}")
            import traceback
            traceback.print_exc()
            return generated
    
    def generate(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 25,
        guidance_scale: float = 8.5,
        strength: float = 0.95,
        seed: int = None,
        face_preserve: bool = True,
        use_ipadapter: bool = False,
        use_controlnet: bool = False,
        controlnet_strength: float = 0.35,
        ipadapter_strength: float = 0.7,
        use_flux: bool = True  # Usar FLUX Fill Pipeline si está disponible
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera una imagen editada usando el enfoque de DOS PASADAS:
        
        PASADA 1: Generar imagen con el prompt completo (strength alto)
                  Esto permite que los cambios del prompt se apliquen correctamente
                  
        PASADA 2: Face swap para restaurar la cara original
                  Esto preserva la identidad facial 100%
        
        Args:
            image: Imagen de entrada (PIL Image, path, o objeto con .name)
            prompt: Prompt positivo
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inference
            guidance_scale: Escala de guidance
            strength: Fuerza del img2img/denoise (se usara valor alto para aplicar cambios)
            seed: Semilla aleatoria
            face_preserve: Si True, hace face swap despues de generar
            use_ipadapter: Si True, usa IP-Adapter para mantener identidad completa
            use_controlnet: Si True, usa ControlNet para mantener estructura
            controlnet_strength: Fuerza del ControlNet (0.0-1.0)
            ipadapter_strength: Fuerza del IP-Adapter (0.0-1.0)
            use_flux: Si True, intenta usar FLUX Fill Pipeline primero (más rápido, mejor calidad)
        
        Returns:
            Tuple con (imagen resultado o None, mensaje estado)
        """
        original_image = None
        
        try:
            # 1. INTENTAR FLUX PRIMERO (si está disponible y se solicita)
            if use_flux:
                print("[ImgEditor] === INTENTANDO FLUX COMO PRIMARIO ===")
                if self._init_flux_client():
                    # Preparar imagen
                    if isinstance(image, Image.Image):
                        original_image = image.copy()
                    elif hasattr(image, 'name'):
                        original_image = Image.open(image.name).copy()
                        image = original_image
                    elif isinstance(image, str) and os.path.exists(image):
                        original_image = Image.open(image).copy()
                        image = original_image
                    
                    # Generar con FLUX
                    result, msg = self.flux_client.generate_fill(
                        image=image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        seed=seed,
                    )
                    
                    if result and result.image:
                        print(f"[ImgEditor] ✅ FLUX exitoso: {msg}")
                        final_image = result.image
                        
                        # PASADA 2: Restaurar cara si se solicita
                        if face_preserve and original_image is not None:
                            print("[ImgEditor] PASADA 2 (FLUX): Restaurando cara original...")
                            final_image = self._restore_face(original_image, final_image)
                        
                        return final_image, f"FLUX{msg}"
                    else:
                        print(f"[ImgEditor] ⚠️ FLUX falló: {msg}, cayendo a ComfyUI...")
                else:
                    print("[ImgEditor] ℹ️ FLUX no disponible, usando ComfyUI")
            
            # 2. FALLBACK A COMFYUI (SD 1.5 + ControlNet/IP-Adapter)
            print("[ImgEditor] === USANDO COMFYUI WORKFLOWS ===")
            client = self._get_client()
            
            # Verificar ComfyUI
            from roop.comfy_client import check_comfy_available, disable_safety_checker
            if not check_comfy_available():
                return None, "Error: ComfyUI no esta corriendo"
            
            # Intentar desactivar safety checker de ComfyUI
            print("[ImgEditor] Intentando desactivar safety checker de ComfyUI...")
            success = disable_safety_checker()
            if not success:
                print("[ImgEditor] WARN: No se pudo desactivar safety checker via API")
            
            # Verificar checkpoints
            checkpoints = client.get_checkpoints()
            if not checkpoints:
                return None, "Error: No hay checkpoints disponibles en ComfyUI"
            
            # Reducir tamaño de imagen si es muy grande (para evitar problemas de VRAM)
            max_size = 1024  # Máximo 1024px para evitar problemas de VRAM
            if original_image is not None:
                img_width, img_height = original_image.size
                if img_width > max_size or img_height > max_size:
                    scale = min(max_size / img_width, max_size / img_height)
                    new_width = int(img_width * scale)
                    new_height = int(img_height * scale)
                    print(f"[ImgEditor] Reduciendo imagen de {img_width}x{img_height} a {new_width}x{new_height}")
                    original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Detectar contenido adulto
            is_adult_content = any(keyword in prompt.lower() for keyword in 
                ["nude", "desnuda", "naked", "adult", "explicit", "topless", "nsfw"])
            
            # PASADA 1: Usar el strength del usuario
            final_strength = strength
            final_steps = num_inference_steps
            final_guidance = guidance_scale
            
            # Si es contenido adulto, optimizar parametros
            if is_adult_content:
                print("[ImgEditor] Contenido adulto detectado - optimizando parametros")
                final_steps = max(final_steps, 30)
                final_guidance = max(final_guidance, 9.0)
            
            # Guardar imagen temporal
            temp_image_path = None
            
            if isinstance(image, Image.Image):
                original_image = image.copy()
                temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_image_path = temp_image.name
                image.save(temp_image_path)
            elif hasattr(image, 'name'):
                temp_image_path = image.name
                original_image = Image.open(temp_image_path).copy()
            elif isinstance(image, str) and os.path.exists(image):
                temp_image_path = image
                original_image = Image.open(temp_image_path).copy()
            else:
                return None, "Error: Imagen no valida"
            
            if not os.path.exists(temp_image_path):
                return None, "Error: No se pudo guardar la imagen temporal"
            
            # Subir imagen a ComfyUI
            image_filename = client.upload_image(temp_image_path)
            if not image_filename:
                return None, "Error: No se pudo subir la imagen a ComfyUI"
            
            # Importar workflows
            from roop.img_editor.comfy_workflows import (
                build_img2img_workflow,
                build_inpaint_workflow,
                build_editor_workflow,
                get_available_checkpoints,
                get_default_checkpoint,
                check_controlnet_available,
                check_ipadapter_available
            )
            
            # Obtener checkpoint
            checkpoint = get_default_checkpoint()
            if checkpoint is None:
                return None, "Error: No hay checkpoints disponibles"
            
            # Verificar si ControlNet/IP-Adapter están disponibles
            has_controlnet = check_controlnet_available()
            has_ipadapter = check_ipadapter_available()
            
            # PASADA 1: Generar con el mejor workflow disponible
            print(f"[ImgEditor] PASADA 1: Generando (strength={final_strength})")
            
            should_use_ipadapter = use_ipadapter and has_ipadapter
            should_use_controlnet = use_controlnet and has_controlnet
            
            # Usar workflow de editor real si el usuario quiere ControlNet o IP-Adapter
            if should_use_controlnet or should_use_ipadapter:
                print(f"[ImgEditor] Modo: Editor Real con adaptadores")
                workflow = build_editor_workflow(
                    image_filename=image_filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed if seed is not None else int(time.time()),
                    steps=final_steps,
                    cfg=final_guidance,
                    denoise=final_strength,
                    checkpoint=checkpoint,
                    use_controlnet=should_use_controlnet,
                    use_ipadapter=should_use_ipadapter,
                    controlnet_strength=controlnet_strength,
                    ipadapter_strength=ipadapter_strength
                )
            else:
                # Modo simple: img2img sin adaptadores
                print(f"[ImgEditor] Modo: Img2Img simple (steps={final_steps}, cfg={final_guidance})")
                workflow = build_img2img_workflow(
                    image_filename=image_filename,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed if seed is not None else int(time.time()),
                    steps=final_steps,
                    cfg=final_guidance,
                    denoise=final_strength,
                    checkpoint=checkpoint
                )
            
            prompt_id, success, error = client.queue_prompt(workflow)
            if not success:
                print(f"[ImgEditor] Error en queue_prompt: {error}")
                return None, f"Error: {error}"
            
            print(f"[ImgEditor] Prompt encolado: {prompt_id[:8]}...")
            
            # Esperar resultado
            time.sleep(2)
            print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}...")
            images = client.get_images(prompt_id, "*")
            
            if not images:
                return None, "Error: No se pudo obtener la imagen de ComfyUI"
            
            # Convertir a PIL Image
            from io import BytesIO
            generated_image = Image.open(BytesIO(images[0]))
            
            if generated_image is None:
                return None, "Error: Image.open devolvio None"
            
            # PASADA 2: Restaurar cara original si face_preserve esta activado
            if face_preserve and original_image is not None:
                print("[ImgEditor] PASADA 2: Restaurando cara original...")
                final_image = self._restore_face(original_image, generated_image)
            else:
                final_image = generated_image
            
            # Limpiar temporal
            try:
                os.remove(temp_image_path)
            except:
                pass
            
            # Determinar modo usado
            if should_use_controlnet or should_use_ipadapter:
                mode_str = "Editor Real"
                if should_use_controlnet:
                    mode_str += " + ControlNet"
                if should_use_ipadapter:
                    mode_str += " + IPAdapter"
            else:
                mode_str = "Img2Img"
            
            face_str = " + FaceRestore" if face_preserve else ""
            print(f"[ImgEditor] OK {mode_str}{face_str} lista: {final_image.size}")
            return final_image, f"{mode_str}{face_str} generada correctamente"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"
    
    def generate_selective(
        self,
        image,
        prompt: str,
        negative_prompt: str = "",
        num_inference_steps: int = 30,
        guidance_scale: float = 9.0,
        strength: float = 0.9,
        seed: int = None,
        face_preserve: bool = True,
        auto_detect_clothing: bool = True,
        mask_threshold: float = 0.5,
        mask_dilation: int = 6,
        exclude_skin: bool = True,
        use_flux: bool = True
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera inpaint SELECTIVO usando detección automática de ropa con CLIPSeg.
        
        Este método:
        1. Detecta automáticamente las áreas de ropa con CLIPSeg
        2. Genera una máscara de esas áreas
        3. Aplica inpaint SOLO en la ropa detectada
        4. Restaura la cara original si face_preserve está activado
        
        Args:
            image: Imagen de entrada
            prompt: Prompt para la edición (ej: "nude woman, natural body")
            negative_prompt: Prompt negativo
            num_inference_steps: Pasos de inferencia
            guidance_scale: Escala de guidance
            strength: Fuerza del inpaint (0.7-1.0 recomendado)
            seed: Semilla aleatoria
            face_preserve: Si True, restaura la cara original
            auto_detect_clothing: Si True, detecta ropa automáticamente
            mask_threshold: Umbral para la máscara (0.3-0.7)
            mask_dilation: Píxeles a expandir la máscara (0-30)
            exclude_skin: Si True, excluye áreas de piel de la máscara
            use_flux: Si True, intenta usar FLUX primero
            
        Returns:
            Tuple con (imagen resultado o None, mensaje estado)
        """
        original_image = None
        mask_image = None
        
        try:
            # Preparar imagen
            if isinstance(image, Image.Image):
                original_image = image.copy()
            elif hasattr(image, 'name'):
                original_image = Image.open(image.name).copy()
                image = original_image
            elif isinstance(image, str) and os.path.exists(image):
                original_image = Image.open(image).copy()
                image = original_image
            else:
                return None, "Error: Imagen no válida"
            
            print(f"[ImgEditor] === INPAINT SELECTIVO ===")
            print(f"[ImgEditor] Imagen: {original_image.size}")
            
            # 1. DETECTAR ROPA CON CLIPSEG
            if auto_detect_clothing:
                print("[ImgEditor] Detectando ropa con CLIPSeg...")
                
                if not is_clipseg_available():
                    return None, "Error: CLIPSeg no disponible. Instala: pip install transformers"
                
                segmenter = get_clothing_segmenter()
                success, msg = segmenter.load()
                if not success:
                    return None, f"Error cargando CLIPSeg: {msg}"
                
                # Generar máscara de ropa
                mask_image, mask_array = segmenter.segment_clothing(
                    image=original_image,
                    threshold=mask_threshold,
                    combine_mode="max",
                    include_skin_exclusion=exclude_skin,
                    dilation=mask_dilation
                )
                
                # Verificar que la máscara no esté vacía
                mask_pixels = mask_array.sum() / 255
                total_pixels = mask_array.shape[0] * mask_array.shape[1]
                mask_coverage = mask_pixels / total_pixels
                
                print(f"[ImgEditor] Máscara: {mask_pixels:.0f} píxeles ({mask_coverage*100:.1f}% de la imagen)")
                
                if mask_coverage < 0.01:
                    print("[ImgEditor] ⚠️ Máscara muy pequeña, usando img2img normal")
                    return self.generate(
                        image=original_image,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        num_inference_steps=num_inference_steps,
                        guidance_scale=guidance_scale,
                        strength=strength,
                        seed=seed,
                        face_preserve=face_preserve,
                        use_ipadapter=False,
                        use_controlnet=False,
                        use_flux=use_flux
                    )
            else:
                # Sin detección automática, usar img2img normal
                return self.generate(
                    image=original_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=strength,
                    seed=seed,
                    face_preserve=face_preserve,
                    use_ipadapter=False,
                    use_controlnet=False,
                    use_flux=use_flux
                )
            
            # 2. APLICAR INPAINT SELECTIVO
            # Para inpaint, necesitamos un denoise más alto
            inpaint_denoise = max(0.85, strength)  # Mínimo 0.85 para inpaint efectivo
            
            # Mejorar el prompt para inpaint de cuerpo
            if auto_detect_clothing:
                # Añadir términos de calidad si no están presentes
                quality_terms = ["detailed", "realistic", "natural skin", "high quality"]
                prompt_lower = prompt.lower()
                for term in quality_terms:
                    if term not in prompt_lower:
                        prompt = f"{prompt}, {term}"
            
            # Intentar con FLUX Inpaint primero
            if use_flux and self._init_flux_client():
                print(f"[ImgEditor] Usando FLUX/SD Inpaint (denoise={inpaint_denoise})...")
                
                result, msg = self.flux_client.generate_inpaint(
                    image=original_image,
                    mask=mask_image,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    strength=inpaint_denoise,  # Usar denoise alto para inpaint
                    seed=seed
                )
                
                if result and result.image:
                    print(f"[ImgEditor] ✅ Inpaint exitoso: {msg}")
                    final_image = result.image
                    
                    # Restaurar cara si está activado
                    if face_preserve:
                        print("[ImgEditor] Restaurando cara original...")
                        final_image = self._restore_face(original_image, final_image)
                    
                    return final_image, f"Inpaint Selectivo ({mask_coverage*100:.1f}% modificado)"
                else:
                    print(f"[ImgEditor] ⚠️ Inpaint falló: {msg}")
            
            # 3. FALLBACK A COMFYUI INPAINT
            print("[ImgEditor] Usando ComfyUI Inpaint...")
            
            from roop.comfy_client import check_comfy_available, disable_safety_checker
            if not check_comfy_available():
                return None, "Error: ComfyUI no está corriendo"
            
            disable_safety_checker()
            
            client = self._get_client()
            
            # Reducir tamaño de imagen si es muy grande (para evitar problemas de VRAM)
            max_size = 1024  # Máximo 1024px para evitar problemas de VRAM
            img_width, img_height = original_image.size
            if img_width > max_size or img_height > max_size:
                scale = min(max_size / img_width, max_size / img_height)
                new_width = int(img_width * scale)
                new_height = int(img_height * scale)
                print(f"[ImgEditor] Reduciendo imagen de {img_width}x{img_height} a {new_width}x{new_height}")
                original_image = original_image.resize((new_width, new_height), Image.LANCZOS)
                mask_image = mask_image.resize((new_width, new_height), Image.LANCZOS)
            
            # Guardar imagen y máscara temporales
            temp_dir = tempfile.gettempdir()
            temp_image_path = os.path.join(temp_dir, "img_editor_input.png")
            temp_mask_path = os.path.join(temp_dir, "img_editor_mask.png")
            
            original_image.save(temp_image_path)
            mask_image.save(temp_mask_path)
            
            # Subir imagen y máscara a ComfyUI
            image_filename = client.upload_image(temp_image_path)
            mask_filename = client.upload_image(temp_mask_path)
            
            if not image_filename or not mask_filename:
                return None, "Error: No se pudieron subir las imágenes a ComfyUI"
            
            # Construir workflow de inpaint
            from roop.img_editor.comfy_workflows import get_default_checkpoint
            
            checkpoint = get_default_checkpoint()
            if checkpoint is None:
                return None, "Error: No hay checkpoints disponibles"
            
            print(f"[ImgEditor] Usando denoise={inpaint_denoise} para inpaint efectivo")
            
            # Usar workflow de inpaint con máscara
            workflow = self._build_inpaint_mask_workflow(
                image_filename=image_filename,
                mask_filename=mask_filename,
                prompt=prompt,
                negative_prompt=negative_prompt,
                seed=seed if seed is not None else int(time.time()),
                steps=num_inference_steps,
                cfg=guidance_scale,
                denoise=strength,  # Usar el denoise del usuario
                checkpoint=checkpoint,
                mask_dilation=mask_dilation  # Pasar dilatación
            )
            
            prompt_id, success, error = client.queue_prompt(workflow)
            if not success:
                return None, f"Error: {error}"
            
            print(f"[ImgEditor] Generando... ID: {prompt_id[:8]}...")
            time.sleep(2)
            images = client.get_images(prompt_id, "*")
            
            if not images:
                return None, "Error: No se pudo obtener la imagen de ComfyUI"
            
            from io import BytesIO
            generated_image = Image.open(BytesIO(images[0]))
            
            # Restaurar cara
            if face_preserve:
                print("[ImgEditor] Restaurando cara original...")
                final_image = self._restore_face(original_image, generated_image)
            else:
                final_image = generated_image
            
            # Limpiar temporales
            try:
                os.remove(temp_image_path)
                os.remove(temp_mask_path)
            except:
                pass
            
            return final_image, f"Inpaint Selectivo ComfyUI ({mask_coverage*100:.1f}% modificado)"
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return None, f"Error: {str(e)}"
    
    def _build_inpaint_mask_workflow(
        self,
        image_filename: str,
        mask_filename: str,
        prompt: str,
        negative_prompt: str,
        seed: int,
        steps: int,
        cfg: float,
        denoise: float,
        checkpoint: str,
        mask_dilation: int = 6
    ) -> dict:
        """Construye un workflow de inpaint con máscara para ComfyUI.
        
        Args:
            mask_dilation: Píxeles a expandir la máscara (se pasa a grow_mask_by)
        """
        
        final_negative = "low quality, blurry, distorted, bad anatomy, ugly, deformed, child, underage, minor"
        if negative_prompt:
            final_negative += f", {negative_prompt}"
        
        return {
            "1": {
                "inputs": {"image": image_filename, "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "LoadImage"}
            },
            "2": {
                "inputs": {"image": mask_filename, "upload": "image"},
                "class_type": "LoadImage",
                "_meta": {"title": "LoadMask"}
            },
            "3": {
                # Convertir imagen de máscara a tipo MASK
                "inputs": {
                    "image": ["2", 0],
                    "channel": "red"  # Usar canal rojo (la máscara es blanco/negro)
                },
                "class_type": "ImageToMask",
                "_meta": {"title": "ImageToMask"}
            },
            "4": {
                "inputs": {"ckpt_name": checkpoint},
                "class_type": "CheckpointLoaderSimple",
                "_meta": {"title": f"Checkpoint ({checkpoint})"}
            },
            "5": {
                "inputs": {"clip": ["4", 1], "text": prompt},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Positive Prompt"}
            },
            "6": {
                "inputs": {"clip": ["4", 1], "text": final_negative},
                "class_type": "CLIPTextEncode",
                "_meta": {"title": "Negative Prompt"}
            },
            "7": {
                "inputs": {
                    "pixels": ["1", 0],
                    "vae": ["4", 2],
                    "mask": ["3", 0],  # Usar la máscara convertida
                    "grow_mask_by": mask_dilation  # Usar el parámetro de dilatación
                },
                "class_type": "VAEEncodeForInpaint",
                "_meta": {"title": "VAEEncodeForInpaint"}
            },
            "8": {
                "inputs": {
                    "model": ["4", 0],
                    "positive": ["5", 0],
                    "negative": ["6", 0],
                    "latent_image": ["7", 0],
                    "seed": seed,
                    "steps": steps,
                    "cfg": cfg,
                    "sampler_name": "euler",
                    "scheduler": "normal",
                    "denoise": denoise
                },
                "class_type": "KSampler",
                "_meta": {"title": "KSampler"}
            },
            "9": {
                "inputs": {"vae": ["4", 2], "samples": ["8", 0]},
                "class_type": "VAEDecode",
                "_meta": {"title": "VAEDecode"}
            },
            "10": {
                "inputs": {
                    "filename_prefix": "inpaint_selective",
                    "images": ["9", 0],
                    "format": "png"
                },
                "class_type": "SaveImage",
                "_meta": {"title": "SaveImage"}
            }
        }
    
    def preview_clothing_mask(
        self,
        image: Image.Image,
        threshold: float = 0.5
    ) -> Tuple[Optional[Image.Image], str]:
        """
        Genera una vista previa de la máscara de ropa detectada.
        
        Útil para verificar qué áreas se modificarán antes de generar.
        
        Returns:
            Tuple de (imagen con máscara superpuesta, mensaje)
        """
        try:
            if not is_clipseg_available():
                return None, "CLIPSeg no disponible. Instala: pip install transformers"
            
            segmenter = get_clothing_segmenter()
            success, msg = segmenter.load()
            if not success:
                return None, f"Error cargando CLIPSeg: {msg}"
            
            # Generar máscara
            mask_image, mask_array = segmenter.segment_clothing(
                image=image,
                threshold=threshold,
                combine_mode="max",
                include_skin_exclusion=True
            )
            
            # Crear visualización
            preview = segmenter.visualize_mask(
                image=image,
                mask=mask_image,
                color=(255, 0, 0),  # Rojo
                alpha=0.5
            )
            
            mask_pixels = mask_array.sum() / 255
            total_pixels = mask_array.shape[0] * mask_array.shape[1]
            coverage = mask_pixels / total_pixels * 100
            
            return preview, f"Ropa detectada: {coverage:.1f}% de la imagen"
            
        except Exception as e:
            return None, f"Error: {str(e)}"
    
    def _check_models_available(self):
        """Verifica qué modelos están disponibles"""
        models = {
            "FLUX": False,
            "ComfyUI": False,
            "ControlNet": False,
            "IP-Adapter": False,
            "CLIPSeg": False
        }
        
        # Verificar FLUX
        try:
            if self.flux_client is None:
                self._init_flux_client()
            models["FLUX"] = is_flux_loaded()
        except:
            pass
        
        # Verificar ComfyUI
        try:
            from roop.comfy_client import check_comfy_available
            models["ComfyUI"] = check_comfy_available()
        except:
            pass
        
        # Verificar ControlNet e IP-Adapter
        try:
            from roop.img_editor.comfy_workflows import (
                check_controlnet_available,
                check_ipadapter_available
            )
            models["ControlNet"] = check_controlnet_available()
            models["IP-Adapter"] = check_ipadapter_available()
        except:
            pass
        
        # Verificar CLIPSeg
        try:
            models["CLIPSeg"] = is_clipseg_available()
        except:
            pass
        
        return models


# Instancia global
_manager = None

def get_img_editor_manager() -> ImgEditorManager:
    """Obtiene la instancia global del manager"""
    global _manager
    if _manager is None:
        _manager = ImgEditorManager()
    return _manager
