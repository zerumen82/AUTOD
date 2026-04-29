import os
import time
import torch
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager

def generate_grok_animation(image, prompt, motion, frames, fps, model, stabilize, 
                            mask_mode="global", mask_prompt="", mask_image=None):
    """
    Orquestador de animación estilo Grok.
    Analiza el prompt, configura el motor y aplica estabilidad facial.
    """
    try:
        manager = get_animate_manager()
        
        # 1. Ejecutar generación inteligente
        # El manager se encarga de:
        # - Reescribir el prompt (semantic boost)
        # - Resolver parámetros (denoise, steps, cfg)
        # - Limpiar VRAM
        # - Llamar a ComfyUI
        # - Aplicar Post-procesamiento (Face Stability)
        
        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt,
            engine=model,
            motion_bucket=motion,
            frames=frames,
            fps=fps,
            face_stabilize=stabilize,
            mask_image=mask_image,
            mask_mode=mask_mode,
            mask_prompt=mask_prompt
        )
        
        if video_path and os.path.exists(video_path):
            return video_path, msg
        
        return None, f"Error: {msg}"
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Excepción: {str(e)}"
