#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Animate Manager - Generación de Video Inteligente (Grok Style)
Centraliza la lógica para SVD, Wan2.2 y LTX-Video.
"""

import os
import time
import json
import torch
import gc
import tempfile
from typing import Optional, Tuple, Dict, List
from PIL import Image
import numpy as np
import cv2
import requests
import roop.globals

class AnimateManager:
    """Gestiona la animación de imágenes con motores modernos (SVD, Wan2.2, LTXV)"""

    def __init__(self):
        self.comfy_url = "http://127.0.0.1:8188"
        self.face_swapper = None
        self.face_analyzer = None

    def _cleanup_vram(self):
        """Libera memoria de GPU para evitar OOM."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    def resolve_video_params(self, engine: str, motion_bucket: int, frames: int, fps: int) -> Dict:
        """Alinea los parámetros según el motor de vídeo."""
        params = {
            "engine": engine,
            "motion_bucket": motion_bucket,
            "num_frames": frames,
            "fps": fps,
            "steps": 20,
            "cfg": 3.0,
            "denoise": 1.0,
            "resolution": (720, 480)
        }
        if engine == "wan_video":
            params["steps"] = 30
            params["cfg"] = 6.0
            params["resolution"] = (720, 480)
        elif engine == "svd_turbo":
            params["steps"] = 8
            params["cfg"] = 2.5
        elif engine == "ltx_video":
            params["steps"] = 25
            params["cfg"] = 5.0
        return params

    def rewrite_video_prompt(self, prompt: str) -> str:
        """Optimiza el prompt para motores de vídeo."""
        prompt_lower = prompt.lower()
        enhanced = prompt
        if any(kw in prompt_lower for kw in ["viento", "wind", "sopla"]):
            enhanced += ", realistic wind effect, flowing hair, natural movement"
        if any(kw in prompt_lower for kw in ["luz", "light", "iluminacion"]):
            enhanced += ", cinematic lighting, dynamic shadows, high quality video"
        quality = "high quality, masterpiece, detailed motion, 4k video, stable movement"
        if quality not in enhanced:
            enhanced = f"{quality}, {enhanced}"
        return enhanced

    def generate_video(
        self,
        image: Image.Image,
        prompt: str,
        engine: str = "wan_video",
        motion_bucket: int = 127,
        frames: int = 81,
        fps: int = 16,
        face_stabilize: bool = True,
        mask_image: Optional[Image.Image] = None,
        mask_mode: str = "global",
        mask_prompt: str = "",
        progress_callback = None
    ) -> Tuple[Optional[str], str]:
        """Orquestador principal de generación de vídeo."""
        
        t0 = time.time()
        # Asegurar RGB para evitar errores de tensores
        if image.mode != "RGB":
            image = image.convert("RGB")

        print(f"[AnimateManager] 🎬 Iniciando animación con {engine}")

        # 1. Parámetros y Prompt
        p = self.resolve_video_params(engine, motion_bucket, frames, fps)
        final_prompt = self.rewrite_video_prompt(prompt)

        # 2. Auto-Mask
        prompt_lower = prompt.lower()
        if mask_mode == "global":
            if any(kw in prompt_lower for kw in ["pelo", "cabello", "hair"]):
                mask_mode, mask_prompt = "smart", "hair"
            elif any(kw in prompt_lower for kw in ["ojos", "mirada", "eyes", "parpadee", "blink"]):
                mask_mode, mask_prompt = "smart", "eyes"
            elif any(kw in prompt_lower for kw in ["boca", "sonría", "smile", "mouth"]):
                mask_mode, mask_prompt = "smart", "mouth"
            elif any(kw in prompt_lower for kw in ["fondo", "background", "escenario"]):
                mask_mode, mask_prompt = "smart", "background"
            
            if mask_mode == "smart":
                print(f"[AnimateManager] 🤖 Auto-Mask activado: {mask_prompt}")

        self._cleanup_vram()

        # 3. Preparar máscara
        working_mask = None
        if mask_mode == "manual":
            working_mask = mask_image
        elif mask_mode == "smart" and mask_prompt:
            try:
                from roop.img_editor.clothing_segmenter import get_clothing_segmenter
                segmenter = get_clothing_segmenter()
                working_mask, _ = segmenter.segment_with_prompt(image, [mask_prompt])
            except Exception as e:
                print(f"[AnimateManager] Error Smart Mask: {e}")

        # 4. Llamada a Motor
        try:
            from animate_photo import AnimatePhoto
            animator = AnimatePhoto()
            
            temp_dir = tempfile.gettempdir()
            img_path = os.path.join(temp_dir, f"animate_input_{int(t0)}.png")
            image.save(img_path)
            
            mask_path = None
            if working_mask:
                mask_path = os.path.join(temp_dir, f"animate_mask_{int(t0)}.png")
                working_mask.save(mask_path)
            
            output_video = os.path.abspath(f"output/animations/video_{int(t0)}.mp4")
            os.makedirs(os.path.dirname(output_video), exist_ok=True)

            success = animator.animate_image(
                model=engine, 
                image_path=img_path,
                prompt=final_prompt,
                output_path=output_video,
                frames=p["num_frames"],
                fps=p["fps"],
                mask_path=mask_path
            )

            if success and os.path.exists(output_video):
                if face_stabilize:
                    output_video = self.apply_face_stabilize(output_video, image)
                return output_video, f"Vídeo OK ({time.time()-t0:.1f}s)"
            
            return None, "Error en generación"

        except Exception as e:
            return None, f"Error: {str(e)}"

    def _init_face_tools(self):
        if self.face_analyzer is not None: return True
        try:
            import insightface
            from insightface.app import FaceAnalysis
            from roop.processors.FaceSwap import FaceSwap
            self.face_analyzer = FaceAnalysis(allowed_modules=['detection', 'recognition'])
            self.face_analyzer.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
            self.face_swapper = FaceSwap()
            self.face_swapper.Initialize({'devicename': 'cuda' if torch.cuda.is_available() else 'cpu', 'model': 'inswapper_128.onnx'})
            return True
        except: return False

    def apply_face_stabilize(self, video_path: str, original_image: Image.Image) -> str:
        """Restaura identidad frame a frame para evitar 'melting'."""
        if not self._init_face_tools(): return video_path
        
        print("[AnimateManager] 💎 Estabilizando Rostro...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        out_path = video_path.replace(".mp4", "_stabilized.mp4")
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        
        orig_cv = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
        source_faces = self.face_analyzer.get(orig_cv)
        if not source_faces: 
            cap.release(); writer.release()
            return video_path
        source_face = max(source_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            target_faces = self.face_analyzer.get(frame)
            if target_faces:
                target_face = max(target_faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))
                # Forzar identidad máxima
                old_blend = roop.globals.blend_ratio
                roop.globals.blend_ratio = 1.0
                res = self.face_swapper.Run(source_face, target_face, frame, paste_back=True)
                roop.globals.blend_ratio = old_blend
                if res is not None: frame = res
            
            writer.write(frame)
            
        cap.release(); writer.release()
        return out_path

_manager = None
def get_animate_manager() -> AnimateManager:
    global _manager
    if _manager is None: _manager = AnimateManager()
    return _manager
