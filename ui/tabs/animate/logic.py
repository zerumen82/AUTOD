import os
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager

PRESET = {"frames": 49, "steps": 30, "cfg": 6.0, "chunks": 1}


def suggest_motion_prompt(image):
    """Sugiere un prompt de movimiento basado en el análisis de la imagen"""
    if image is None:
        return "moving, natural motion"
    
    try:
        from scripts.image_analyzer_for_prompt import ImageAnalyzer
        import tempfile
        import os
        
        analyzer = ImageAnalyzer()
        
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp.name)
            tmp_path = tmp.name
            
        analysis = analyzer.analyze(tmp_path)
        os.unlink(tmp_path)
        
        # Lógica de sugerencia de movimiento
        num_people = analysis.get('num_people', 0)
        scene = analysis.get('scene', 'indoor')
        lighting = analysis.get('lighting', 'natural')
        
        suggestions = []
        
        if num_people > 0:
            face = analysis['faces'][0]
            expr = face.get('expression', 'neutral')
            gender = face.get('gender', 'person')
            
            if 'smiling' in expr:
                suggestions.append("smiling and laughing, looking at camera")
            elif 'mouth open' in expr:
                suggestions.append("talking and gesturing, expressive face")
            else:
                suggestions.append("gentle head movement, blinking naturally, looking around")
                
            if num_people > 1:
                suggestions.append("people interacting and moving naturally")
        
        if 'outdoor' in scene:
            suggestions.append("wind blowing through hair and clothes, cinematic outdoor motion")
        elif 'dark' in lighting or 'night' in lighting:
            suggestions.append("subtle movements in the shadows, flickering lights")
            
        if not suggestions:
            suggestions.append("smooth cinematic camera movement, high quality motion")
            
        import random
        return random.choice(suggestions)
        
    except Exception as e:
        print(f"[AnimateLogic] Error sugiriendo prompt: {e}")
        return "cinematic motion, natural flowing movement"


def generate_grok_animation(image, prompt, stabilize=False, progress_callback=None, lora_name=None, lora_strength=1.0,
                            add_mmaudio=False, audio_prompt=""):
    try:
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "moving",
            engine="framepack" if not lora_name or lora_name == "None" else "wan_video", # Forzar WanVideo si hay LoRA
            frames=PRESET["frames"],
            fps=12,
            face_stabilize=stabilize,
            steps=PRESET["steps"],
            cfg=PRESET["cfg"],
            autoregressive_chunks=1,
            mask_image=None,
            mask_mode="global",
            mask_prompt="",
            progress_callback=progress_callback,
            lora_name=lora_name,
            lora_strength=lora_strength,
            add_mmaudio=add_mmaudio,
            audio_prompt=audio_prompt,
        )

        if video_path and os.path.exists(video_path):
            return video_path, msg

        return None, f"Error: {msg}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Excepción: {str(e)}"
