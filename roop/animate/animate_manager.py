import os, time, tempfile
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import roop.globals

COMFY_URL = "http://127.0.0.1:8188"
MODELS_DIR = r"D:\PROJECTS\AUTOAUTO\ui\tob\ComfyUI\models\diffusion_models"


class AnimateManager:
    def __init__(self):
        self.face_swapper = None
        self.face_analyzer = None

    def _check_models(self, engine):
        available = [m.lower() for m in os.listdir(MODELS_DIR) if m.endswith(".gguf") or m.endswith(".safetensors")]
        if engine == "wan_video":
            needed = ["wan2.2"]
        elif engine == "svd_turbo":
            needed = ["svd", "stablevideo"]
        elif engine == "ltx_video":
            needed = ["ltx"]
        else:
            needed = []
        missing = [n for n in needed if not any(n in a for a in available)]
        return missing

    def resolve_params(self, engine, motion=127, frames=81, fps=16):
        base = {"engine": engine, "motion": motion, "frames": frames, "fps": fps, "steps": 25, "cfg": 5.0}
        if engine == "svd_turbo":
            base.update({"steps": 8, "cfg": 2.5})
        return base

    def rewrite_prompt(self, prompt):
        prompt_lower = prompt.lower()
        enhanced = prompt
        if not any(kw in enhanced for kw in ["high quality", "masterpiece", "detailed"]):
            enhanced = f"high quality, masterpiece, {enhanced}"
        if any(kw in prompt_lower for kw in ["viento", "wind", "sopla", "blow"]):
            enhanced += ", realistic wind effect, natural flowing movement"
        if any(kw in prompt_lower for kw in ["luz", "light", "iluminacion", "sun"]):
            enhanced += ", cinematic lighting, dynamic light"
        if any(kw in prompt_lower for kw in ["sonria", "sonrisa", "smile", "ria"]):
            enhanced += ", natural smile expression, subtle mouth movement"
        if any(kw in prompt_lower for kw in ["parpadee", "parpadeo", "blink", "ojos"]):
            enhanced += ", natural eye blink, subtle eye movement"
        return enhanced

    def generate_video(self, image, prompt, engine="wan_video", motion_bucket=127, frames=81, fps=16,
                       face_stabilize=True, mask_image=None, mask_mode="global", mask_prompt="",
                       progress_callback=None):
        t0 = time.time()
        if image.mode != "RGB":
            image = image.convert("RGB")

        print(f"[AnimateManager] Engine={engine} Frames={frames} FPS={fps}")

        missing = self._check_models(engine)
        if missing:
            return None, f"Modelos faltantes: {', '.join(missing)}"

        p = self.resolve_params(engine, motion_bucket, frames, fps)
        final_prompt = self.rewrite_prompt(prompt)
        print(f"[AnimateManager] Prompt: {final_prompt[:80]}...")

        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, f"anim_in_{int(t0)}.png")
        image.save(img_path)

        output_path = os.path.abspath(f"output/animations/video_{int(t0)}.mp4")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        try:
            from animate_photo import AnimatePhoto
            animator = AnimatePhoto()
            ok = animator.animate_image(
                image_pil=image, prompt=final_prompt,
                output_path=output_path, model=engine,
                frames=p["frames"], fps=p["fps"]
            )
            if ok and os.path.exists(output_path):
                if face_stabilize:
                    output_path = self.apply_face_stabilize(output_path, image)
                elapsed = time.time() - t0
                return output_path, f"OK ({elapsed:.0f}s)"
            return None, "Error en generación"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def _init_face_tools(self):
        if self.face_analyzer is not None:
            return True
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from roop.processors.FaceSwap import FaceSwap
            self.face_analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            import torch
            ctx = 0 if torch.cuda.is_available() else -1
            self.face_analyzer.prepare(ctx_id=ctx, det_size=(1024, 1024))
            self.face_swapper = FaceSwap()
            # Check for higher resolution model first (256x256 > 128x128)
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_256 = os.path.join(base_dir, 'models', 'inswapper_256.onnx')
            model_128 = os.path.join(base_dir, 'models', 'inswapper_128.onnx')
            if os.path.exists(model_256):
                model_path = model_256
            else:
                model_path = model_128
            self.face_swapper.Initialize({'devicename': 'cuda' if torch.cuda.is_available() else 'cpu',
                                          'model': model_path})
            return True
        except Exception as e:
            print(f"[AnimateManager] Face init: {e}")
            return False

    def apply_face_stabilize(self, video_path, original_image):
        if not self._init_face_tools():
            return video_path

        print("[AnimateManager] Stabilizing faces...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = video_path.replace(".mp4", "_stab.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

        orig_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        source_faces = self.face_analyzer.get(orig_cv)
        if not source_faces:
            cap.release()
            writer.release()
            return video_path
        source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        frame_count = 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"[Stabilize] {frame_count}/{total}", end="\r")
            target_faces = self.face_analyzer.get(frame)
            if target_faces:
                target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                old_blend = roop.globals.blend_ratio
                roop.globals.blend_ratio = 1.0
                res = self.face_swapper.Run(source_face, target_face, frame, paste_back=True)
                roop.globals.blend_ratio = old_blend
                if res is not None:
                    frame = res
            writer.write(frame)

        cap.release()
        writer.release()
        print(f"\n[Stabilize] Done. {frame_count} frames")
        return out_path


_manager = None


def get_animate_manager() -> AnimateManager:
    global _manager
    if _manager is None:
        _manager = AnimateManager()
    return _manager