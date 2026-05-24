import os
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager

QUALITY_PRESETS = {
    "Rápido (1 bloque)":  {"frames": 49, "steps": 20, "cfg": 5.0, "chunks": 1},
    "Normal AR (2 bloques)":  {"frames": 49, "steps": 25, "cfg": 5.5, "chunks": 2},
    "Calidad AR (3 bloques)": {"frames": 49, "steps": 30, "cfg": 6.0, "chunks": 3},
}


def generate_grok_animation(image, prompt, quality="Normal AR (2 bloques)", stabilize=False, progress_callback=None):
    try:
        presets = QUALITY_PRESETS.get(quality, QUALITY_PRESETS["Normal AR (2 bloques)"])
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "moving",
            engine="wan_video",
            frames=presets["frames"],
            fps=12,
            face_stabilize=stabilize,
            steps=presets["steps"],
            cfg=presets["cfg"],
            autoregressive_chunks=presets["chunks"],
            mask_image=None,
            mask_mode="global",
            mask_prompt="",
            progress_callback=progress_callback
        )

        if video_path and os.path.exists(video_path):
            return video_path, msg

        return None, f"Error: {msg}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Excepción: {str(e)}"
