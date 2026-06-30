import os, time, tempfile, subprocess, re
from typing import Optional, Tuple
from PIL import Image
import numpy as np
import cv2
import roop.globals

from roop.comfy_workflows import get_comfyui_url
from roop.utils import get_vram_gb
from roop.output_paths import get_animate_output_dir

def get_project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def get_models_dir():
    return os.path.join(get_project_root(), "ui", "tob", "ComfyUI", "models", "diffusion_models")

class AnimateManager:
    def __init__(self):
        self.face_swapper = None
        self.face_analyzer = None

    def _emit_progress(self, callback, message):
        if callback is None:
            return
        try:
            callback(message)
        except Exception as e:
            print(f"[AnimateManager] Progress callback error: {e}")

    def _check_models(self, engine):
        available = []
        models_dir = get_models_dir()
        if os.path.exists(models_dir):
            for root, dirs, files in os.walk(models_dir):
                available.extend(d.lower() for d in dirs)
                available.extend(f.lower() for f in files if f.endswith((".gguf", ".safetensors")))
        unet_dir = os.path.join(os.path.dirname(models_dir), "unet")
        if os.path.exists(unet_dir):
            for root, dirs, files in os.walk(unet_dir):
                available.extend(f.lower() for f in files if f.endswith(".gguf"))
        if engine == "framepack":
            needed = []
        elif engine == "wan_video":
            needed = ["wan2.2", "wan2_2", "wan2.1", "wan2_1"]
        elif engine == "svd_turbo":
            needed = ["svd", "stablevideo"]
        elif engine == "ltx_video":
            needed = ["ltx"]
        else:
            needed = []
        if engine == "wan_video":
            missing = [] if any(n in a for n in needed for a in available) else ["wan2.2/wan2.1 (GGUF o safetensors)"]
        elif engine == "framepack":
            missing = []
        else:
            missing = [n for n in needed if not any(n in a for a in available)]
        return missing

    def resolve_params(self, engine, motion=127, frames=33, fps=16, magnitude=0.5, steps=None, cfg=None):
        base = {
            "engine": engine,
            "motion": int(64 + (magnitude * 192)),
            "frames": frames,
            "fps": fps,
            "steps": steps if steps is not None else int(15 + (magnitude * 10)),
            "cfg": cfg if cfg is not None else 3.5 + (magnitude * 2.0)
        }

        if engine == "framepack":
            if steps is None:
                base["steps"] = int(25 + (magnitude * 10))
            if cfg is None:
                base["cfg"] = 4.5 + (magnitude * 2.0)
        elif engine == "wan_video":
            if steps is None:
                base["steps"] = int(20 + (magnitude * 15))
            if cfg is None:
                base["cfg"] = 4.0 + (magnitude * 2.0)
        elif engine == "svd_turbo":
            if steps is None:
                base["steps"] = int(4 + (magnitude * 4))
            if cfg is None:
                base["cfg"] = 1.5 + (magnitude * 1.5)

        vram_gb = get_vram_gb()
        if vram_gb > 0 and vram_gb <= 8:
            if engine == "wan_video":
                if base["frames"] > 81:
                    base["frames"] = 81
                if base["steps"] > 28:
                    base["steps"] = 28
            elif engine == "framepack":
                if base["frames"] > 49:
                    base["frames"] = 49
                if base["steps"] > 25:
                    base["steps"] = 25
            else:
                if base["frames"] > 33:
                    base["frames"] = 33
                if base["steps"] > 10:
                    base["steps"] = 10
            print(f"[AnimateManager] VRAM baja ({vram_gb}GB). Ajustando frames={base['frames']}, steps={base['steps']}.")

        print(f"[AnimateManager] Params: frames={base['frames']}, steps={base['steps']}, cfg={base['cfg']:.1f}")
        return base

    def _get_semantic_analyzer(self, full_ai: bool = False):
        if not hasattr(self, 'semantic_analyzer') or self.semantic_analyzer is None:
            from roop.img_editor.nlp.semantic_analyzer import get_semantic_analyzer
            self.semantic_analyzer = get_semantic_analyzer(full_ai=full_ai)
        return self.semantic_analyzer

    def rewrite_prompt(self, prompt, image_context=""):
        prompt = (prompt or "").strip()
        from roop.img_editor.prompt_translator import translate_prompt
        translated = translate_prompt(prompt)

        mag = 0.5
        self._last_speech_intensity = 0.0
        try:
            nlp = self._get_semantic_analyzer()
            mag = nlp.get_magnitude(translated)
            if hasattr(nlp, "get_axis_scores"):
                axes = nlp.get_axis_scores(translated)
                pose_score = float(axes.get("pose", 0.0))
                if pose_score > 0.08:
                    mag = max(mag, 0.70 + pose_score * 0.15)
            self._last_magnitude = mag
            if hasattr(nlp, "get_speech_intensity"):
                self._last_speech_intensity = float(nlp.get_speech_intensity(translated))
            print(
                f"[AnimateManager] Semantic: Magnitude={mag:.2f}, "
                f"Speech={self._last_speech_intensity:.2f}"
            )
        except Exception as e:
            print(f"[AnimateManager] NLP Error: {e}. Fallback to 0.5")
            self._last_magnitude = 0.5

        composed = self._compose_motion_prompt(translated, mag=mag)
        print(f"[AnimateManager] Motion prompt: {composed[:120]}...")
        self._last_user_motion_prompt = translated
        self._last_user_prompt_raw = prompt
        return composed

    def _compose_motion_prompt(self, user_prompt: str, mag: float = 0.5) -> str:
        raw = (user_prompt or "natural motion").strip()
        if mag > 0.65:
            return (
                f"Instruction: Animate this exact photograph. "
                f"Follow this action and motion completely and obviously: {raw}. "
                f"Natural cinematic movement, obey the instruction as strongly as possible."
            )
        return (
            f"Instruction: Animate this exact photograph. "
            f"Apply this motion naturally: {raw}. "
            f"Smooth realistic movement while keeping the scene coherent."
        )

    def _split_motion_clauses(self, prompt):
        prompt = (prompt or "").strip()
        if not prompt:
            return []

        parts = re.split(
            r"\s*(?:,|;|\.|\band then\b|\bthen\b|\bafter that\b|\bluego\b|\by luego\b|\bdespues\b|\bdespués\b)\s*",
            prompt,
            flags=re.IGNORECASE
        )
        clauses = [p.strip() for p in parts if len(p.strip()) > 2]

        if len(clauses) <= 1 and re.search(r"\s+and\s+", prompt, flags=re.IGNORECASE):
            clauses = [p.strip() for p in re.split(r"\s+\band\b\s+", prompt, flags=re.IGNORECASE) if len(p.strip()) > 2]

        return clauses or [prompt]

    def _build_autoregressive_chunk_prompt(self, final_prompt, chunk_index, total_chunks):
        motion_prompt = getattr(self, "_last_user_motion_prompt", "") or final_prompt
        clauses = self._split_motion_clauses(motion_prompt)

        if len(clauses) >= total_chunks:
            start = int(chunk_index * len(clauses) / total_chunks)
            end = int((chunk_index + 1) * len(clauses) / total_chunks)
            end = max(start + 1, end)
            chunk_action = ", ".join(clauses[start:end])
        elif len(clauses) > 1:
            if chunk_index == 0:
                chunk_action = clauses[0]
            elif chunk_index == total_chunks - 1:
                chunk_action = f"continuing the motion, {clauses[-1]}"
            else:
                action_index = min(len(clauses) - 1, chunk_index)
                previous_action = clauses[max(0, action_index - 1)]
                chunk_action = f"{previous_action}, continuing with {clauses[action_index]}"
        elif total_chunks > 1:
            temporal_hints = [
                "",
                "continuing the natural flow,",
                "the scene unfolds further,",
                "progressing the moment,",
            ]
            hint = temporal_hints[min(chunk_index, len(temporal_hints) - 1)]
            chunk_action = f"{hint} {motion_prompt}".strip().lstrip(",").strip()
        else:
            chunk_action = motion_prompt

        return chunk_action.strip()

    def _extract_last_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        last_frame = None
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            last_frame = frame
        cap.release()
        if last_frame is None:
            return None
        rgb = cv2.cvtColor(last_frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _concat_video_chunks(self, chunk_paths, output_path, fps):
        if not chunk_paths:
            return False

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self._concat_video_chunks_ffmpeg(chunk_paths, output_path, fps):
            return True

        first = cv2.VideoCapture(chunk_paths[0])
        width = int(first.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(first.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first.release()
        if width <= 0 or height <= 0:
            return False

        writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not writer.isOpened():
            print("[AnimateManager] OpenCV VideoWriter no disponible para concatenar MP4.")
            return False

        written = 0
        for chunk_index, chunk_path in enumerate(chunk_paths):
            cap = cv2.VideoCapture(chunk_path)
            frame_index = 0
            skip_first = chunk_index > 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_index += 1
                if skip_first and frame_index == 1:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                written += 1
            cap.release()
        writer.release()

        valid = self._video_has_frames(output_path)
        print(f"[AnimateManager] Autoregressive concat: {len(chunk_paths)} chunks, {written} frames -> {output_path}")
        return written > 0 and valid

    def _concat_video_chunks_ffmpeg(self, chunk_paths, output_path, fps):
        try:
            from roop.utils import get_ffmpeg_path
            ffmpeg = get_ffmpeg_path()

            cmd = [ffmpeg, "-hide_banner", "-y"]
            for chunk_path in chunk_paths:
                cmd.extend(["-i", chunk_path])

            filters = []
            labels = []
            for index in range(len(chunk_paths)):
                label = f"v{index}"
                if index == 0:
                    filters.append(f"[{index}:v]setpts=PTS-STARTPTS[{label}]")
                else:
                    filters.append(f"[{index}:v]trim=start_frame=1,setpts=PTS-STARTPTS[{label}]")
                labels.append(f"[{label}]")
            filter_complex = ";".join(filters) + ";" + "".join(labels) + f"concat=n={len(chunk_paths)}:v=1:a=0[outv]"

            cmd.extend([
                "-filter_complex", filter_complex,
                "-map", "[outv]",
                "-r", str(fps),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                output_path
            ])

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                print(f"[AnimateManager] FFmpeg concat failed: {result.stderr[-500:]}")
                return False

            if not self._video_has_frames(output_path):
                print("[AnimateManager] FFmpeg concat generó un vídeo ilegible.")
                return False

            print(f"[AnimateManager] Autoregressive concat via FFmpeg: {len(chunk_paths)} chunks -> {output_path}")
            return True
        except Exception as e:
            print(f"[AnimateManager] FFmpeg concat no disponible: {e}")
            return False

    def _video_has_frames(self, video_path):
        if not os.path.exists(video_path) or os.path.getsize(video_path) <= 0:
            return False
        cap = cv2.VideoCapture(video_path)
        try:
            return cap.isOpened() and int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
        finally:
            cap.release()

    def generate_video(self, image, prompt, engine="wan_video", motion_bucket=127, frames=33, fps=16,
                       face_stabilize=False, mask_image=None, mask_mode="global", mask_prompt="",
                       progress_callback=None, steps=None, cfg=None, autoregressive_chunks=1,
                       lora_name=None, lora_strength=1.0, add_mmaudio=False, audio_prompt="",
                       cancel_check=None):
        t0 = time.time()
        if image.mode != "RGB":
            image = image.convert("RGB")

        img_description = ""
        final_prompt = self.rewrite_prompt(prompt, image_context=img_description)
        magnitude = getattr(self, "_last_magnitude", 0.5)

        p = self.resolve_params(engine, motion_bucket, frames, fps, magnitude=magnitude, steps=steps, cfg=cfg)

        print(f"[AnimateManager] Engine={engine} Frames={p['frames']} FPS={fps} Steps={p['steps']} CFG={p['cfg']} LoRA={lora_name}")

        missing = self._check_models(engine)
        if missing:
            return None, f"Faltan modelos para {engine}: {', '.join(missing)}"

        temp_dir = tempfile.gettempdir()
        img_path = os.path.join(temp_dir, f"anim_in_{int(t0)}.png")
        image.save(img_path)

        output_path = os.path.join(get_animate_output_dir(), f"video_{int(t0)}.mp4")

        try:
            from animate_photo import AnimatePhoto
            animator = AnimatePhoto()
            autoregressive_chunks = max(1, int(autoregressive_chunks or 1))

            if engine == "framepack":
                self._emit_progress(progress_callback, "Generando animación con FramePack...")
                ok = animator.animate_image(
                    image_pil=image, prompt=final_prompt,
                    output_path=output_path, model="framepack",
                    frames=p["frames"], fps=p["fps"],
                    steps=p["steps"], cfg=p["cfg"],
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            elif engine == "wan_video" and autoregressive_chunks > 1:
                print(f"[AnimateManager] Autoregressive mode: {autoregressive_chunks} chunks x {p['frames']} frames")
                chunk_paths = []
                current_image = image
                for chunk_index in range(autoregressive_chunks):
                    if cancel_check and cancel_check():
                        elapsed = time.time() - t0
                        return None, f"Cancelado tras {elapsed:.0f}s"
                    chunk_path = output_path.replace(".mp4", f"_chunk{chunk_index + 1:02d}.mp4")
                    chunk_prompt = self._build_autoregressive_chunk_prompt(
                        final_prompt,
                        chunk_index,
                        autoregressive_chunks
                    )

                    print(f"[AnimateManager] AR chunk {chunk_index + 1}/{autoregressive_chunks}: {chunk_path}")
                    print(f"[AnimateManager] AR prompt {chunk_index + 1}: {chunk_prompt[:220]}...")
                    self._emit_progress(
                        progress_callback,
                        f"Generando bloque AR {chunk_index + 1}/{autoregressive_chunks}..."
                    )
                    ok = animator.animate_image(
                        image_pil=current_image, prompt=chunk_prompt,
                        output_path=chunk_path, model=engine,
                        frames=p["frames"], fps=p["fps"],
                        steps=p["steps"], cfg=p["cfg"],
                        lora_name=lora_name, lora_strength=lora_strength,
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if cancel_check and cancel_check():
                        elapsed = time.time() - t0
                        return None, f"Cancelado tras {elapsed:.0f}s"
                    if not ok or not os.path.exists(chunk_path):
                        elapsed = time.time() - t0
                        return None, f"Error en chunk AR {chunk_index + 1}/{autoregressive_chunks} tras {elapsed:.0f}s"

                    chunk_paths.append(chunk_path)
                    if chunk_index < autoregressive_chunks - 1:
                        self._emit_progress(
                            progress_callback,
                            f"Preparando continuidad desde el bloque {chunk_index + 1}..."
                        )
                        current_image = self._extract_last_frame(chunk_path)
                        if current_image is None:
                            elapsed = time.time() - t0
                            return None, f"No se pudo extraer último frame del chunk {chunk_index + 1} tras {elapsed:.0f}s"

                self._emit_progress(progress_callback, "Uniendo bloques AR...")
                ok = self._concat_video_chunks(chunk_paths, output_path, p["fps"])
            else:
                self._emit_progress(progress_callback, "Generando animación...")
                ok = animator.animate_image(
                    image_pil=image, prompt=final_prompt,
                    output_path=output_path, model=engine,
                    frames=p["frames"], fps=p["fps"],
                    steps=p["steps"], cfg=p["cfg"],
                    lora_name=lora_name, lora_strength=lora_strength,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
            if ok and os.path.exists(output_path):
                if face_stabilize:
                    self._emit_progress(progress_callback, "Estabilizando rostro...")
                    output_path = self.apply_face_stabilize(output_path, image)
                audio_note = ""
                if add_mmaudio:
                    from roop.animate.audio_pipeline import apply_animate_audio
                    speech_i = getattr(self, "_last_speech_intensity", 0.0)
                    user_raw = getattr(self, "_last_user_prompt_raw", prompt)
                    output_path, audio_msg = apply_animate_audio(
                        output_path,
                        user_prompt=audio_prompt or user_raw,
                        motion_prompt=final_prompt,
                        speech_intensity=speech_i,
                        progress_callback=progress_callback,
                    )
                    if audio_msg:
                        audio_note = f" | {audio_msg}"
                elapsed = time.time() - t0
                return output_path, f"OK ({elapsed:.0f}s){audio_note}"
            elapsed = time.time() - t0
            return None, f"Error en generación tras {elapsed:.0f}s"
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
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            model_256 = os.path.join(base_dir, 'models', 'inswapper_256.onnx')
            model_128 = os.path.join(base_dir, 'models', 'inswapper_128.onnx')
            model_path = model_256 if os.path.exists(model_256) else model_128
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
