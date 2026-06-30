import os
from roop.animate.animate_manager import get_animate_manager

# AR automático estilo Imagine: bloques/decisión en animate_manager.resolve_autoregressive_plan
PRESET = {
    "frames": 37,
    "fps": 12,
    "steps": 24,
    "cfg": 5.5,
}


def suggest_motion_prompt(image):
    """Sugiere movimiento simple sin vision pesada."""
    if image is None:
        return "natural cinematic motion"
    return "gentle natural movement, subtle camera motion, realistic flow"


def generate_grok_animation(image, prompt, stabilize=False, progress_callback=None, lora_name=None, lora_strength=1.0,
                            add_mmaudio=True, audio_prompt="", cancel_check=None):
    try:
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "natural motion",
            engine="wan_video",
            frames=PRESET["frames"],
            fps=PRESET["fps"],
            face_stabilize=stabilize,
            steps=PRESET["steps"],
            cfg=PRESET["cfg"],
            autoregressive_chunks=0,
            mask_image=None,
            mask_mode="global",
            mask_prompt="",
            progress_callback=progress_callback,
            lora_name=lora_name if lora_name and lora_name != "None" else None,
            lora_strength=lora_strength,
            add_mmaudio=add_mmaudio,
            audio_prompt=audio_prompt,
            cancel_check=cancel_check,
        )

        if video_path and os.path.exists(video_path):
            return video_path, msg

        return None, f"Error: {msg}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Excepción: {str(e)}"