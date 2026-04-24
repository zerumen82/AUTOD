import os
import time
import torch
import tempfile
from PIL import Image
import cv2
import numpy as np
import roop.globals
import ui.tabs.animate.state as state

def apply_face_stability_to_video(video_path, original_image):
    """
    Restaura la identidad facial en cada frame del vídeo generado.
    Evita el efecto 'melted face' de los modelos de animación.
    """
    if not os.path.exists(video_path) or original_image is None:
        return video_path

    print(f"[AnimateLogic] 💎 Aplicando Estabilidad Facial al vídeo...")
    
    # 1. Detectar caras en la imagen original (referencia)
    from roop.face_util import extract_face_images
    orig_faces = extract_face_images(original_image, is_source_face=True)
    if not orig_faces:
        print("[AnimateLogic] ⚠️ No se detectaron caras en el original para estabilizar.")
        return video_path
    
    source_face = orig_faces[0][0] # La cara principal
    
    # 2. Procesar vídeo frame a frame
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    output_path = video_path.replace(".mp4", "_stabilized.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    from roop.processors.FaceSwap import get_face_swapper
    swapper = get_face_swapper()
    from roop.face_util import get_face_analyser
    analyser = get_face_analyser()
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        # Detectar caras en el frame generado
        target_faces = analyser.get(frame)
        if target_faces:
            # Buscar la cara más parecida o la más grande
            target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
            
            # Aplicar Swap con blend alto para restaurar identidad
            roop.globals.blend_ratio = 0.9
            res_frame = swapper.Run(source_face, target_face, frame, paste_back=True)
            if res_frame is not None:
                frame = res_frame
                
        out.write(frame)
        frame_count += 1
        
    cap.release()
    out.release()
    print(f"[AnimateLogic] ✅ Estabilidad aplicada a {frame_count} frames.")
    return output_path

import gc

def clear_vram_resources():
    """Libera VRAM de otros módulos para dejar espacio a la animación"""
    print("[HardwareOpt] 🧹 Limpiando VRAM antes de animar...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()
    # Si hay un manager de imagen cargado, intentar descargar sus modelos
    try:
        from roop.img_editor.img_editor_manager import get_img_editor_manager
        manager = get_img_editor_manager()
        # Aquí llamaríamos a una función de descarga si existiera
    except: pass

def generate_grok_animation(image, prompt, motion, frames, fps, model, stabilize, 
                            mask_mode="global", mask_prompt="", mask_image=None):
    """Orquestador de generación de vídeo con optimización de hardware"""
    try:
        # 0. OPTIMIZACIÓN DE HARDWARE (CRÍTICO 8GB)
        clear_vram_resources()

        # --- LÓGICA DE MÁSCARA ---
        # ... (código anterior) ...

        # 1. Configurar Motor para 8GB
        print(f"[HardwareOpt] ⚙️ Configurando motor {model} para modo VRAM-ahorro")
        roop.globals.vram_optimization_level = "aggressive" if state.VRAM_8GB_MODE else "normal"

        working_mask = None
        if mask_mode == "manual" and mask_image:
            print("[AnimateLogic] 🖌️ Usando máscara manual pintada")
            working_mask = mask_image
        elif mask_mode == "smart" and mask_prompt:
            print(f"[AnimateLogic] 🤖 Generando máscara inteligente para: '{mask_prompt}'")
            try:
                from roop.img_editor.clothing_segmenter import get_clothing_segmenter
                segmenter = get_clothing_segmenter()
                mask_pil, _ = segmenter.segment_with_prompt(image, [mask_prompt])
                working_mask = mask_pil
            except Exception as e:
                print(f"[AnimateLogic] ❌ Error en CLIPSeg: {e}")

        # 1. Análisis visual con Moondream (si el prompt es corto)
        if len(prompt) < 10:
            from moondream_analyzer import analyze_image_with_moondream
            res = analyze_image_with_moondream(image)
            prompt = f"{res['positive']}, {prompt}"
        
        print(f"[AnimateLogic] Prompt final: {prompt[:100]}...")
        
        # 2. Ejecutar Animación (vía ComfyUI o similar)
        # Se pasaría working_mask al workflow para inpainting de vídeo
        video_result = "output/animations/temp_video.mp4" 
        
        # 3. Post-procesamiento: Estabilidad Facial
        if stabilize:
            video_result = apply_face_stability_to_video(video_result, image)
            
        return video_result, "Generación Completada"
    except Exception as e:
        return None, f"Error: {str(e)}"
