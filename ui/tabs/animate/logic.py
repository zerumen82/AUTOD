import os
from roop.animate.animate_manager import get_animate_manager

# AR automático estilo Imagine: bloques y continuidad en animate_manager.generate_video
# (_build_autoregressive_chunk_prompt + _concat_video_chunks).
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


def _bounded_int(value, default, low, high):
    """Entero dentro del rango que el motor instalado acepta; default si no."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def generate_grok_animation(image, prompt, engine="wan_video", stabilize=False, progress_callback=None, lora_name=None, lora_strength=1.0,
                            add_mmaudio=True, autoregressive_chunks=2, audio_prompt="", cancel_check=None,
                            frames=None, steps=None, fps=None):
    try:
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "natural motion",
            engine=engine,
            frames=_bounded_int(frames, PRESET["frames"], 17, 201 if engine == "ltx2_audio" else 49),
            fps=_bounded_int(fps, PRESET["fps"], 8, 30),
            face_stabilize=stabilize,
            steps=_bounded_int(steps, PRESET["steps"], 4, 40),
            cfg=PRESET["cfg"],
            autoregressive_chunks=autoregressive_chunks,
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