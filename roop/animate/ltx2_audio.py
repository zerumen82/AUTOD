"""LTX-2 destilado Q4 con audio nativo para Animate (ruta 8 GB).

Construye el grafo de ComfyUI de LTX-2 image-to-video con audio sincronizado
en un solo pase, usando el transformer GGUF Q4_K_M y Gemma 3 12B GGUF.

El motor está expuesto en `ui/tabs/animate/ui.py` (`ltx2_audio`) desde v5.1328,
con los parámetros medidos (8 pasos, CFG 1,0, 25 fps, fotogramas 8n+1). La
validación visual con un retrato real (cara, boca y diálogo) sigue pendiente:
los renders medidos son una escena de coche (MEMORY.md, 2026-11-10).
`diagnostics/real_animate_ltx2_audio_20261110.py` y
`diagnostics/ltx2_matrix_20261110.py` son la prueba de recurso y coste.

Ficheros (ver `download_ltx2_audio_q4.py`):
    diffusion_models/ltx-2-19b-distilled_Q4_K_M.gguf
    text_encoders/gemma-3-12b-it-Q4_K_M.gguf
    text_encoders/ltx-2-19b-embeddings_connector_distill_bf16.safetensors
    vae/LTX2_video_vae_bf16.safetensors
    checkpoints/LTX2_audio_vae_bf16.safetensors

ComfyUI 0.19.0 ya trae los nodos nativos de audio LTX (LTXVAudioVAELoader,
LTXVEmptyLatentAudio, LTXVConcatAVLatent, LTXVSeparateAVLatent,
LTXVAudioVAEDecode). El text encoder GGUF se carga con DualCLIPLoaderGGUF, que
solo lista `models/clip`; `ensure_clip_links()` enlaza ahí los dos ficheros.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(PROJECT_ROOT, "ui", "tob", "ComfyUI", "models")

UNET_NAME = "ltx-2-19b-distilled_Q4_K_M.gguf"
GEMMA_NAME = "gemma-3-12b-it-Q4_K_M.gguf"
CONNECTOR_NAME = "ltx-2-19b-embeddings_connector_distill_bf16.safetensors"
VIDEO_VAE_NAME = "LTX2_video_vae_bf16.safetensors"
AUDIO_VAE_NAME = "LTX2_audio_vae_bf16.safetensors"

MODEL_FILES = [
    ("diffusion_models", UNET_NAME, 12651789536),
    ("text_encoders", GEMMA_NAME, 7300778336),
    ("text_encoders", CONNECTOR_NAME, 2862983784),
    ("vae", VIDEO_VAE_NAME, 2445008522),
    ("checkpoints", AUDIO_VAE_NAME, 217740136),
]

# Resolución y duración de la prueba. 540p = 960x544 (múltiplo de 32).
DEFAULT_WIDTH = 960
DEFAULT_HEIGHT = 544
DEFAULT_FRAMES = 97          # múltiplo de 8 + 1 → 3,88 s a 25 fps
DEFAULT_FPS = 25

# Destilado: 8 pasos con CFG=1, misma programación de sigmas que la plantilla
# oficial `video_ltx2_i2v_distilled`.
DISTILLED_SIGMAS = [
    1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0,
]
DISTILLED_STEPS = len(DISTILLED_SIGMAS) - 1


def _path(folder, name):
    return os.path.join(MODELS_DIR, folder, name)


def model_status():
    """Estado por fichero: existe y tamaño coincide con el publicado en HF."""
    status = []
    for folder, name, expected in MODEL_FILES:
        path = _path(folder, name)
        size = os.path.getsize(path) if os.path.exists(path) else 0
        status.append({
            "folder": folder,
            "name": name,
            "path": path,
            "expected_bytes": expected,
            "bytes": size,
            "ok": size == expected,
        })
    return status


def missing_models():
    return [s["name"] for s in model_status() if not s["ok"]]


def ensure_clip_links():
    """DualCLIPLoaderGGUF solo lee `models/clip`; enlaza (o copia) ahí los dos ficheros."""
    clip_dir = os.path.join(MODELS_DIR, "clip")
    os.makedirs(clip_dir, exist_ok=True)
    linked = []
    for folder, name, _expected in MODEL_FILES:
        if name not in (GEMMA_NAME, CONNECTOR_NAME):
            continue
        src = _path(folder, name)
        dst = os.path.join(clip_dir, name)
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
            linked.append(dst)
            continue
        if os.path.exists(dst):
            os.remove(dst)
        try:
            os.link(src, dst)
        except OSError:
            import shutil
            shutil.copy2(src, dst)
        linked.append(dst)
    return linked


def fit_size(width, height):
    """Redondea a múltiplo de 32 (compresión espacial del VAE de LTX)."""
    def _fit(value):
        return max(32, int(round(value / 32.0)) * 32)
    return _fit(width), _fit(height)


def fit_frames(frames):
    """El modelo exige (múltiplo de 8) + 1; se toma el válido más cercano.

    48 → 49 (no 41): conserva la duración pedida en vez de recortarla.
    """
    frames = max(9, int(frames))
    lower = ((frames - 1) // 8) * 8 + 1
    upper = lower + 8
    return lower if (frames - lower) <= (upper - frames) else upper


def _trim_last_frame(video_path, output_path):
    """Extrae todos los fotogramas excepto el último para evitar el defecto de fundido en 49f.

    Solo se usa cuando frames < 97 (caso defectuoso). El último fotograma a 49f sale
    fundido e ilegible (confirmado con la misma semilla 1234). Recortar el último
    frame asegura el fotograma final visible, o el usuario puede elegir 97f.
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Contar fotogramas
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 1:
        cap.release()
        return False

    # Escribir todos excepto el último
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not out.isOpened():
        print(f"[LTX2] OpenCV no puede reescribir {os.path.basename(output_path)}")
        cap.release()
        return False

    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if written < total_frames - 1:
            out.write(frame)
        written += 1

    cap.release()
    out.release()
    return True


def compose_prompt(action, sounds="", max_words=200):
    """Prompt de LTX-2: un solo párrafo con acción, escena y sonidos.

    LTX-2 genera vídeo y audio juntos, así que el sonido (ambiente o diálogo)
    va descrito en el mismo párrafo, no como una lista. Se recorta a 200
    palabras, el límite práctico del modelo.
    """
    text = " ".join((action or "").split())
    if not text:
        text = "Natural, subtle movement in the scene."

    sounds = " ".join((sounds or "").split())
    if sounds and "audio" not in sounds.lower():
        text = f"{text.rstrip('.')}. Audio: {sounds.rstrip('.')}."
    elif sounds:
        text = f"{text.rstrip('.')}. {sounds.rstrip('.')}."

    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;:")
        if not text.endswith((".", "!", "?")):
            text += "."
    return text


def resolve_generation(vram_gb, frames, width=None, height=None):
    """Resolución y fotogramas que caben en esta GPU para LTX-2.

    540p es la ruta medida en 8 GB; por encima de 8 GB se permite 720p.
    """
    if width is None or height is None:
        width, height = (DEFAULT_WIDTH, DEFAULT_HEIGHT) if vram_gb and vram_gb <= 8 else (1280, 704)
    width, height = fit_size(width, height)
    return {
        "width": width,
        "height": height,
        "frames": fit_frames(frames),
        "fps": DEFAULT_FPS,
        "steps": DISTILLED_STEPS,
        "cfg": 1.0,
    }


def comfy_base_url():
    try:
        from roop.comfy_workflows import get_comfyui_url
        url = (get_comfyui_url() or "").rstrip("/")
        if url:
            return url
    except Exception:
        pass
    return "http://127.0.0.1:8188"


def generate(image_path, prompt, output_path, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
             frames=DEFAULT_FRAMES, fps=DEFAULT_FPS, seed=None, timeout=3600, progress=None):
    """Ejecuta el grafo en ComfyUI y devuelve (ruta_mp4, mensaje).

    `image_path` puede ser cualquier fichero local; se sube a `ComfyUI/input`.
    El MP4 resultante incluye la pista de audio nativa.
    """
    import json
    import shutil
    import time
    import uuid

    import requests

    def _say(message):
        print(f"[LTX2] {message}", flush=True)
        if progress is not None:
            try:
                progress(message)
            except Exception:
                pass

    missing = missing_models()
    if missing:
        return None, f"Faltan modelos LTX-2: {', '.join(missing)}"
    ensure_clip_links()

    base = comfy_base_url()
    # v5.1339: Animate arrancaba ComfyUI solo si el usuario lo hacia a
    # mano ("ComfyUI no esta activo. Inicia ComfyUI primero."), mientras que
    # el Editor de imagen lo arranca solo. El mismo criterio en las dos rutas
    # evita un paso manual que el usuario no tiene por que conocer: si el
    # backend no esta vivo, se levanta aqui.
    try:
        requests.get(f"{base}/system_stats", timeout=5)
    except Exception:
        try:
            from ui.tabs.comfy_launcher import is_comfyui_running, start as start_comfyui
            if not is_comfyui_running():
                _say("ComfyUI no estaba activo; arrancando el backend...")
                _comfy_ok, _comfy_msg, _ = start_comfyui(directly_run=True)
                if not _comfy_ok:
                    return None, f"ComfyUI no disponible: {_comfy_msg}"
                _say("ComfyUI listo")
            requests.get(f"{base}/system_stats", timeout=10)
        except Exception as _comfy_error:
            return None, (
                "ComfyUI no esta activo. Inicia ComfyUI primero "
                f"({_comfy_error})."
            )

    width, height = fit_size(width, height)
    frames = fit_frames(frames)
    stamp = int(time.time())
    image_name = f"ltx2_{stamp}_{uuid.uuid4().hex[:6]}.png"

    from PIL import Image
    comfy_input = os.path.join(PROJECT_ROOT, "ui", "tob", "ComfyUI", "input")
    os.makedirs(comfy_input, exist_ok=True)
    Image.open(image_path).convert("RGB").save(os.path.join(comfy_input, image_name))

    workflow = build_workflow(
        image_name, prompt, width=width, height=height, frames=frames, fps=fps,
        seed=int(seed if seed is not None else stamp % 2_147_483_647),
        filename_prefix=f"animate/ltx2_{stamp}",
    )

    _say(f"Enviando a ComfyUI: {width}x{height}, {frames} fotogramas, {fps} fps")
    response = requests.post(f"{base}/prompt", json={"prompt": workflow, "client_id": uuid.uuid4().hex}, timeout=60)
    if response.status_code != 200:
        return None, f"ComfyUI rechazó el grafo ({response.status_code}): {response.text[:300]}"
    prompt_id = response.json()["prompt_id"]

    started = time.time()
    last_note = started
    history = None
    while time.time() - started < timeout:
        time.sleep(3)
        # Un /history lento (VAE decodificando en el mismo proceso) no debe
        # matar una ejecución de 30 minutos: se reintenta en la siguiente vuelta.
        try:
            payload = requests.get(f"{base}/history/{prompt_id}", timeout=30).json()
        except requests.exceptions.RequestException:
            continue
        if prompt_id in payload:
            history = payload[prompt_id]
            break
        if time.time() - last_note > 20:
            _say(f"Generando… {time.time() - started:.0f}s")
            last_note = time.time()

    if history is None:
        return None, f"LTX-2 sin resultado tras {timeout}s"

    status = history.get("status", {})
    if status.get("status_str") != "success":
        errors = [m for m in status.get("messages", []) if m[0] in ("execution_error", "execution_interrupted")]
        return None, f"Error de ComfyUI: {json.dumps(errors)[:400]}"

    mp4 = None
    for node_out in history.get("outputs", {}).values():
        for entry in (node_out.get("images") or []) + (node_out.get("videos") or []):
            if entry.get("type") == "output" and entry.get("filename", "").endswith(".mp4"):
                mp4 = entry
                break
        if mp4:
            break
    if not mp4:
        return None, "ComfyUI no devolvió MP4"

    source = os.path.join(PROJECT_ROOT, "ui", "tob", "ComfyUI", "output",
                          mp4.get("subfolder", ""), mp4["filename"])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Recorte del último fotograma si frames < 97: el frame 49 fundido e
    # ilegible (misma semilla 1234 da el mismo defecto). Se recorta el último
    # fotograma para conservar el anterior visible; o elegir 97f para calidad.
    trim_output = None
    if frames < 97:
        trim_output = output_path + ".trim"
        if _trim_last_frame(source, trim_output):
            shutil.copy2(trim_output, output_path)
            os.unlink(trim_output)
            elapsed = time.time() - started
            _say(f"Listo en {elapsed:.0f}s: {os.path.basename(output_path)} (último frame recortado para evitar fundido)")
        else:
            shutil.copy2(source, output_path)
    else:
        shutil.copy2(source, output_path)

    elapsed = time.time() - started
    _say(f"Listo en {elapsed:.0f}s: {os.path.basename(output_path)}")
    return output_path, f"OK ({elapsed:.0f}s, vídeo + audio nativo LTX-2)"


def build_workflow(image_name, prompt, width=DEFAULT_WIDTH, height=DEFAULT_HEIGHT,
                   frames=DEFAULT_FRAMES, fps=DEFAULT_FPS, seed=0,
                   filename_prefix="animate/ltx2_audio"):
    """Grafo API de ComfyUI: imagen → vídeo + audio nativo en un pase.

    `image_name` es un fichero de `ComfyUI/input`. `prompt` debe describir la
    acción, la escena y los sonidos/diálogo; LTX-2 los genera juntos.
    """
    width, height = fit_size(width, height)
    frames = fit_frames(frames)
    sigmas = ", ".join(f"{s:g}" for s in DISTILLED_SIGMAS)

    return {
        "1": {"class_type": "UnetLoaderGGUF",
              "inputs": {"unet_name": UNET_NAME}},
        "2": {"class_type": "DualCLIPLoaderGGUF",
              "inputs": {"clip_name1": GEMMA_NAME, "clip_name2": CONNECTOR_NAME,
                         "type": "ltxv"}},
        "3": {"class_type": "VAELoader",
              "inputs": {"vae_name": VIDEO_VAE_NAME}},
        "4": {"class_type": "LTXVAudioVAELoader",
              "inputs": {"ckpt_name": AUDIO_VAE_NAME}},
        "5": {"class_type": "LoadImage",
              "inputs": {"image": image_name}},
        "6": {"class_type": "ImageScale",
              "inputs": {"image": ["5", 0], "upscale_method": "lanczos",
                         "width": width, "height": height, "crop": "center"}},
        "7": {"class_type": "LTXVPreprocess",
              "inputs": {"image": ["6", 0], "img_compression": 33}},
        "8": {"class_type": "CLIPTextEncode",
              "inputs": {"clip": ["2", 0], "text": prompt}},
        "9": {"class_type": "ConditioningZeroOut",
              "inputs": {"conditioning": ["8", 0]}},
        "10": {"class_type": "LTXVConditioning",
               "inputs": {"positive": ["8", 0], "negative": ["9", 0],
                          "frame_rate": fps}},
        "11": {"class_type": "EmptyLTXVLatentVideo",
               "inputs": {"width": width, "height": height,
                          "length": frames, "batch_size": 1}},
        "12": {"class_type": "LTXVImgToVideoInplace",
               "inputs": {"vae": ["3", 0], "image": ["7", 0],
                          "latent": ["11", 0], "strength": 1.0, "bypass": False}},
        "13": {"class_type": "LTXVEmptyLatentAudio",
               "inputs": {"audio_vae": ["4", 0], "frames_number": frames,
                          "frame_rate": fps, "batch_size": 1}},
        "14": {"class_type": "LTXVConcatAVLatent",
               "inputs": {"video_latent": ["12", 0], "audio_latent": ["13", 0]}},
        "15": {"class_type": "CFGGuider",
               "inputs": {"model": ["1", 0], "positive": ["10", 0],
                          "negative": ["10", 1], "cfg": 1.0}},
        "16": {"class_type": "KSamplerSelect",
               "inputs": {"sampler_name": "euler"}},
        "17": {"class_type": "ManualSigmas",
               "inputs": {"sigmas": sigmas}},
        "18": {"class_type": "RandomNoise",
               "inputs": {"noise_seed": int(seed)}},
        "19": {"class_type": "SamplerCustomAdvanced",
               "inputs": {"noise": ["18", 0], "guider": ["15", 0],
                          "sampler": ["16", 0], "sigmas": ["17", 0],
                          "latent_image": ["14", 0]}},
        "20": {"class_type": "LTXVSeparateAVLatent",
               "inputs": {"av_latent": ["19", 0]}},
        "21": {"class_type": "LTXVCropGuides",
               "inputs": {"positive": ["10", 0], "negative": ["10", 1],
                          "latent": ["20", 0]}},
        "22": {"class_type": "VAEDecode",
               "inputs": {"samples": ["21", 2], "vae": ["3", 0]}},
        "23": {"class_type": "LTXVAudioVAEDecode",
               "inputs": {"samples": ["20", 1], "audio_vae": ["4", 0]}},
        "24": {"class_type": "CreateVideo",
               "inputs": {"images": ["22", 0], "audio": ["23", 0],
                          "fps": float(fps)}},
        "25": {"class_type": "SaveVideo",
               "inputs": {"video": ["24", 0],
                          "filename_prefix": filename_prefix,
                          "format": "auto", "codec": "auto"}},
    }
