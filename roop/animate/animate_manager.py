import os, time, tempfile
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import roop.globals

from roop.comfy_workflows import get_comfyui_url

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_dir():
    return os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models", "diffusion_models")

class AnimateManager:
    def __init__(self):
        self.face_swapper = None
        self.face_analyzer = None

    def _check_models(self, engine):
        available = []
        # Check diffusion_models for safetensors
        models_dir = get_models_dir()
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                available.extend(d.lower() for d in dirs)
                available.extend(f.lower() for f in files if f.endswith((".gguf", ".safetensors")))
        # Check unet/ for GGUF models
        unet_dir = os.path.join(os.path.dirname(models_dir), "unet")
        if os.path.exists(unet_dir):
            for root, dirs, files in os.walk(unet_dir):
                available.extend(f.lower() for f in files if f.endswith(".gguf"))
        if engine == "wan_video":
            needed = ["wan2.2", "wan2_2", "wan2.1", "wan2_1"]
        elif engine == "svd_turbo":
            needed = ["svd", "stablevideo"]
        elif engine == "ltx_video":
            needed = ["ltx"]
        else:
            needed = []
        if engine == "wan_video":
            missing = [] if any(n in a for n in needed for a in available) else ["wan2.2/wan2.1 (GGUF o safetensors)"]
        else:
            missing = [n for n in needed if not any(n in a for a in available)]
        return missing

    def resolve_params(self, engine, motion=127, frames=81, fps=16, magnitude=0.5):
        # Base dinámica escalada por magnitud detectada por el LLM
        # Magnitude (0.0 - 1.0)
        base = {
            "engine": engine,
            "motion": int(64 + (magnitude * 192)), # 64 a 256
            "frames": frames,
            "fps": fps,
            "steps": int(15 + (magnitude * 20)), # 15 a 35 pasos
            "cfg": 3.5 + (magnitude * 4.0) # 3.5 a 7.5
        }
        
        # Ajustes técnicos mínimos por motor
        if engine == "svd_turbo":
            # SVD Turbo es destilado, requiere pocos pasos y CFG bajo
            base.update({
                "steps": int(4 + (magnitude * 4)), # 4 a 8 pasos
                "cfg": 1.5 + (magnitude * 1.5)  # 1.5 a 3.0
            })
        elif engine == "wan_video":
            # Wan Video 14B escala bien con pasos moderados
            base["steps"] = int(20 + (magnitude * 10))
            
        print(f"[AnimateManager] Dynamic Params: Mag={magnitude:.2f}, Motion={base['motion']}, Steps={base['steps']}, CFG={base['cfg']:.1f}")
        return base

    def rewrite_prompt(self, prompt, image_context=""):
        prompt = (prompt or "").strip()
        
        # Usamos el rewriter LLM para una fusión semántica completa sin hardcoding
        try:
            from roop.img_editor.prompt_rewriter import get_prompt_rewriter
            rewriter = get_prompt_rewriter()
            
            # El rewriter devuelve un análisis completo incluyendo magnitud e intención
            analysis = rewriter.rewrite(prompt, image_context=image_context)
            enhanced = analysis.get("prompt", prompt)
            self._last_magnitude = analysis.get("magnitude", 0.5)
            
            print(f"[AnimateManager] Semantic boost: {enhanced[:60]}... (Mag: {self._last_magnitude})")
            return enhanced if enhanced else prompt
        except Exception as e:
            print(f"[AnimateManager] Rewriter no disponible: {e}")
            self._last_magnitude = 0.5
            return f"{prompt}, natural motion, high quality"

    def generate_video(self, image, prompt, engine="wan_video", motion_bucket=127, frames=81, fps=16,
                       face_stabilize=True, mask_image=None, mask_mode="global", mask_prompt="",
                       progress_callback=None):
        t0 = time.time()
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 1. Análisis y reescritura inteligente (obtiene magnitud semántica)
        img_description = ""
        try:
            from scripts.moondream_analyzer import MoonDreamImageAnalyzer
            analyzer = MoonDreamImageAnalyzer()
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                image.save(tmp.name)
                res = analyzer.analyze(tmp.name)
                img_description = res.get('positive', '')
            analyzer.unload()
            del analyzer
        except: pass

        final_prompt = self.rewrite_prompt(prompt, image_context=img_description)
        magnitude = getattr(self, "_last_magnitude", 0.5)

        # 2. Resolución dinámica de parámetros técnicos
        p = self.resolve_params(engine, motion_bucket, frames, fps, magnitude=magnitude)
        
        print(f"[AnimateManager] Engine={engine} Frames={frames} FPS={fps} Mag={magnitude}")
        print(f"[AnimateManager] Final Prompt: {final_prompt[:80]}...")

        missing = self._check_models(engine)


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
