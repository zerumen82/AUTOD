import os
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager

PRESET = {"frames": 49, "steps": 30, "cfg": 6.0, "chunks": 3}


def generate_grok_animation(image, prompt, stabilize=False, progress_callback=None):
    try:
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "moving",
            engine="wan_video",
            frames=PRESET["frames"],
            fps=12,
            face_stabilize=stabilize,
            steps=PRESET["steps"],
            cfg=PRESET["cfg"],
            autoregressive_chunks=PRESET["chunks"],
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
