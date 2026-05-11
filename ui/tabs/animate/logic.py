import os, time
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager
from roop.audio_generator import generate_audio

def generate_grok_animation(image, prompt, motion, frames, fps, model, stabilize, 
                            audio_text="", use_tts=False, language="Español", ref_voice=None,
                            mask_mode="global", mask_prompt="", mask_image=None):
    """
    Orquestador de animación.
    Analiza el prompt, genera audio si es necesario, configura el motor y aplica estabilidad facial.
    """
    try:
        manager = get_animate_manager()
        
        # 1. Manejo de Audio / TTS
        audio_path = None
        if audio_text and use_tts:
            print(f"[AnimateLogic] Generando audio TTS: '{audio_text[:30]}...'")
            try:
                audio_path = generate_audio(text=audio_text, lenguaje=language, speaker_wav=ref_voice)
            except Exception as e:
                print(f"[AnimateLogic] Error TTS: {e}")

        # 2. Ejecutar generación inteligente
        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else ("talking" if audio_path or ref_voice else "moving"),
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

def get_expression_prompt(action):
    """Devuelve el prompt técnico según el botón pulsado"""
    mapping = {
        "smile": "looking at camera, subtle natural smile, moving lips, happy expression",
        "wink": "looking at camera, blinking one eye, playful expression, winking",
        "angry": "serious face, intense look, no smile, cinematic expression",
        "wind": "hair blowing in the wind, realistic clothes movement, breeze effect"
    }
    return mapping.get(action, "")
