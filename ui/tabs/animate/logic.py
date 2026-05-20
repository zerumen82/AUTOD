import os
from PIL import Image
import roop.globals
from roop.animate.animate_manager import get_animate_manager


def generate_grok_animation(image, prompt, motion=127, frames=81, fps=16,
                            model="wan_video", stabilize=True):
    try:
        manager = get_animate_manager()

        video_path, msg = manager.generate_video(
            image=image,
            prompt=prompt if prompt else "moving",
            engine=model,
            motion_bucket=motion,
            frames=frames,
            fps=fps,
            face_stabilize=stabilize,
            mask_image=None,
            mask_mode="global",
            mask_prompt=""
        )

        if video_path and os.path.exists(video_path):
            return video_path, msg

        return None, f"Error: {msg}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Excepción: {str(e)}"
