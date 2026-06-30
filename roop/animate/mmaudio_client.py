import os
import time
import shutil
import subprocess
import requests
import cv2
from typing import Callable, Optional, Tuple

_CANCELLED = "__cancelled__"

from roop.comfy_workflows import get_comfyui_url
from roop.utils import get_ffmpeg_path, get_vram_gb

HF_BASE = "https://huggingface.co/Kijai/MMAudio_safetensors/resolve/main"

REQUIRED_MODELS = {
    "mmaudio_large_44k_v2_fp16.safetensors",
    "mmaudio_vae_44k_fp16.safetensors",
    "mmaudio_synchformer_fp16.safetensors",
    "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
}


def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _comfy_root():
    return os.path.join(_project_root(), "ui", "tob", "ComfyUI")


def _mmaudio_node_dir():
    return os.path.join(_comfy_root(), "custom_nodes", "ComfyUI-MMAudio")


def _mmaudio_models_dir():
    return os.path.join(_comfy_root(), "models", "mmaudio")


def _comfy_input_dir():
    return os.path.join(_comfy_root(), "input")


def _video_duration(video_path: str) -> Tuple[float, float]:
    cap = cv2.VideoCapture(video_path)
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 12.0)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if fps <= 0:
            fps = 12.0
        duration = frames / fps if frames > 0 else 4.0
        return max(0.5, min(duration, 30.0)), fps
    finally:
        cap.release()


def _installed_models() -> set:
    models_dir = _mmaudio_models_dir()
    if not os.path.isdir(models_dir):
        return set()
    found = set()
    for name in os.listdir(models_dir):
        if name.lower().endswith((".safetensors", ".pt", ".pth", ".ckpt")):
            found.add(name)
    return found


def is_node_installed() -> bool:
    init_py = os.path.join(_mmaudio_node_dir(), "__init__.py")
    nodes_py = os.path.join(_mmaudio_node_dir(), "nodes.py")
    return os.path.isfile(init_py) and os.path.isfile(nodes_py)


def is_models_ready() -> bool:
    installed = _installed_models()
    return REQUIRED_MODELS.issubset(installed)


def is_comfy_node_loaded(base_url: Optional[str] = None) -> bool:
    base = (base_url or get_comfyui_url()).rstrip("/")
    try:
        r = requests.get(f"{base}/object_info/MMAudioSampler", timeout=5)
        return r.status_code == 200 and "MMAudioSampler" in r.json()
    except Exception:
        return False


def is_available(check_comfy: bool = True) -> bool:
    if not is_node_installed() or not is_models_ready():
        return False
    if check_comfy:
        return is_comfy_node_loaded()
    return True


def get_status_message(check_comfy: bool = True) -> str:
    if not is_node_installed():
        return "MMAudio no instalado. Ejecuta scripts/install_mmaudio.ps1 y reinicia ComfyUI."
    missing = sorted(REQUIRED_MODELS - _installed_models())
    if missing:
        return f"Faltan modelos MMAudio: {', '.join(missing)}. Ejecuta scripts/install_mmaudio.ps1"
    if check_comfy and not is_comfy_node_loaded():
        return "Nodos MMAudio no cargados. Reinicia ComfyUI tras instalar ComfyUI-MMAudio."
    return "MMAudio listo"


def _resolve_sound_prompt(prompt: str, motion_prompt: str = "") -> str:
    text = (prompt or "").strip()
    if not text:
        motion = (motion_prompt or "").strip()
        if motion:
            text = f"natural ambient sound effects synchronized with: {motion}"
        else:
            text = "natural ambient sound effects synchronized with the video motion, cinematic atmosphere"
    try:
        from roop.img_editor.prompt_translator import translate_prompt
        text = translate_prompt(text)
    except Exception:
        pass
    return text


def _build_workflow(video_filename: str, duration: float, sound_prompt: str, seed: int) -> dict:
    models = _installed_models()
    main_model = "mmaudio_large_44k_v2_fp16.safetensors"
    if main_model not in models:
        main_model = next((m for m in sorted(models) if "large" in m and "44" in m and "fp16" in m), main_model)

    steps = 20 if get_vram_gb() <= 8 else 25

    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": video_filename}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {
            "class_type": "MMAudioModelLoader",
            "inputs": {"mmaudio_model": main_model, "base_precision": "fp16"},
        },
        "4": {
            "class_type": "MMAudioFeatureUtilsLoader",
            "inputs": {
                "vae_model": "mmaudio_vae_44k_fp16.safetensors",
                "synchformer_model": "mmaudio_synchformer_fp16.safetensors",
                "clip_model": "apple_DFN5B-CLIP-ViT-H-14-384_fp16.safetensors",
                "mode": "44k",
                "precision": "fp16",
            },
        },
        "5": {
            "class_type": "MMAudioSampler",
            "inputs": {
                "mmaudio_model": ["3", 0],
                "feature_utils": ["4", 0],
                "images": ["2", 0],
                "duration": round(duration, 3),
                "steps": steps,
                "cfg": 4.5,
                "seed": seed,
                "prompt": sound_prompt,
                "negative_prompt": "speech, voice, dialogue, talking, narration, music lyrics",
                "mask_away_clip": False,
                "force_offload": True,
            },
        },
        "6": {
            "class_type": "SaveAudioMP3",
            "inputs": {
                "audio": ["5", 0],
                "filename_prefix": "MMAudioAnim",
                "quality": "128k",
            },
        },
    }


def _copy_video_to_comfy_input(video_path: str) -> str:
    os.makedirs(_comfy_input_dir(), exist_ok=True)
    base = os.path.basename(video_path)
    name, ext = os.path.splitext(base)
    dest_name = f"mmaudio_{int(time.time())}_{name}{ext or '.mp4'}"
    dest_path = os.path.join(_comfy_input_dir(), dest_name)
    shutil.copy2(video_path, dest_path)
    return dest_name


def _interrupt_comfy(base_url: str, prompt_id: Optional[str] = None):
    try:
        requests.post(f"{base_url.rstrip('/')}/interrupt", timeout=3)
        if prompt_id:
            requests.post(f"{base_url.rstrip('/')}/queue", json={"delete": [prompt_id]}, timeout=3)
    except Exception:
        pass


def _wait_for_audio(
    base_url: str,
    prompt_id: str,
    timeout: int = 900,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Optional[dict]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if cancel_check and cancel_check():
            _interrupt_comfy(base_url, prompt_id)
            return None
        try:
            r = requests.get(f"{base_url}/history/{prompt_id}", timeout=30)
            if r.status_code != 200 or prompt_id not in r.json():
                time.sleep(1)
                continue
            hist = r.json()[prompt_id]
            status = hist.get("status", {})
            if isinstance(status, dict) and status.get("status_str") == "error":
                msgs = status.get("messages", [])
                return None
            outputs = hist.get("outputs", {})
            for node_out in outputs.values():
                for key in ("audio", "audios"):
                    items = node_out.get(key) or []
                    for item in items:
                        if isinstance(item, dict) and item.get("filename"):
                            return item
            if outputs:
                return None
        except Exception as e:
            print(f"[MMAudio] Poll error: {e}")
        time.sleep(1)
    return None


def _download_comfy_file(base_url: str, meta: dict, dest_path: str) -> bool:
    try:
        r = requests.get(
            f"{base_url}/view",
            params={
                "filename": meta["filename"],
                "subfolder": meta.get("subfolder", ""),
                "type": meta.get("type", "output"),
            },
            timeout=120,
        )
        if r.status_code != 200:
            return False
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        with open(dest_path, "wb") as f:
            f.write(r.content)
        return os.path.getsize(dest_path) > 0
    except Exception as e:
        print(f"[MMAudio] Download error: {e}")
        return False


def _mux_audio(video_path: str, audio_path: str) -> Optional[str]:
    out_path = video_path.replace(".mp4", "_audio.mp4")
    silent_backup = video_path.replace(".mp4", "_silent.mp4")
    ffmpeg = get_ffmpeg_path()
    try:
        if os.path.exists(silent_backup):
            os.remove(silent_backup)
        os.rename(video_path, silent_backup)
        cmd = [
            ffmpeg, "-hide_banner", "-y",
            "-i", silent_backup,
            "-i", audio_path,
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            out_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"[MMAudio] FFmpeg mux failed: {result.stderr[-400:]}")
            if not os.path.exists(video_path):
                os.rename(silent_backup, video_path)
            return None
        if os.path.exists(silent_backup):
            os.remove(silent_backup)
        if os.path.exists(video_path):
            os.remove(video_path)
        os.rename(out_path, video_path)
        return video_path
    except Exception as e:
        print(f"[MMAudio] Mux error: {e}")
        if not os.path.exists(video_path) and os.path.exists(silent_backup):
            os.rename(silent_backup, video_path)
        return None


def add_audio_to_video(
    video_path: str,
    sound_prompt: str = "",
    motion_prompt: str = "",
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> Tuple[Optional[str], str]:
    if not video_path or not os.path.exists(video_path):
        return None, "Video no encontrado para MMAudio"

    if cancel_check and cancel_check():
        return video_path, _CANCELLED

    status = get_status_message(check_comfy=True)
    if not is_available(check_comfy=True):
        return video_path, f"Audio omitido: {status}"

    base = get_comfyui_url().rstrip("/")
    duration, _fps = _video_duration(video_path)
    prompt = _resolve_sound_prompt(sound_prompt, motion_prompt)
    seed = int(time.time()) % (2**32)

    if progress_callback:
        progress_callback("Generando audio sincronizado (MMAudio)...")
    print(f"[MMAudio] duration={duration:.2f}s prompt={prompt[:120]}")

    try:
        if cancel_check and cancel_check():
            return video_path, _CANCELLED
        video_filename = _copy_video_to_comfy_input(video_path)
        workflow = _build_workflow(video_filename, duration, prompt, seed)
        r = requests.post(f"{base}/prompt", json={"prompt": workflow}, timeout=60)
        if r.status_code != 200:
            err = r.json().get("error", {}).get("message", r.text[:200]) if r.headers.get("content-type", "").startswith("application/json") else r.text[:200]
            return video_path, f"Audio omitido: ComfyUI rechazó MMAudio ({err})"

        prompt_id = r.json().get("prompt_id")
        if not prompt_id:
            return video_path, "Audio omitido: ComfyUI no devolvió prompt_id"

        audio_meta = _wait_for_audio(base, prompt_id, cancel_check=cancel_check)
        if cancel_check and cancel_check():
            return video_path, _CANCELLED
        if not audio_meta:
            return video_path, "Audio omitido: MMAudio no generó archivo de audio"

        temp_audio = video_path.replace(".mp4", "_mmaudio.mp3")
        if not _download_comfy_file(base, audio_meta, temp_audio):
            return video_path, "Audio omitido: no se pudo descargar el audio generado"

        if cancel_check and cancel_check():
            return video_path, _CANCELLED
        if progress_callback:
            progress_callback("Mezclando audio con el vídeo...")
        muxed = _mux_audio(video_path, temp_audio)
        if os.path.exists(temp_audio):
            try:
                os.remove(temp_audio)
            except OSError:
                pass

        if muxed:
            return muxed, "OK con audio MMAudio"
        return video_path, "Audio omitido: falló la mezcla FFmpeg"
    except Exception as e:
        print(f"[MMAudio] Error: {e}")
        return video_path, f"Audio omitido: {e}"