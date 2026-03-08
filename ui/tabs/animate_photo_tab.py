import os
import gradio as gr
import time
import tempfile
import threading
from pathlib import Path

# Obtener directorio base del proyecto (ui/tabs)
current_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))

# Puerto de ComfyUI
COMFYUI_PORT = os.environ.get('COMFYUI_PORT', '8188')

# ============================================================================
# CONFIGURACION - DETECCION DE MODELOS DISPONIBLES (lazy loading)
# ============================================================================

# Rutas de modelos
COMFY_MODELS_DIR = os.path.join(script_dir, "ui", "tob", "ComfyUI", "models")

# Variables globales (lazy loading)
AVAILABLE_MODELS = None
DEFAULT_MODEL = None
AUDIO_AVAILABLE = None
COMFY_AVAILABLE = None
LAUNCHER_AVAILABLE = None

def init_detect_models():
    """Inicializa la deteccion de modelos (solo se ejecuta una vez)"""
    global AVAILABLE_MODELS, DEFAULT_MODEL
    if AVAILABLE_MODELS is None:
        AVAILABLE_MODELS = detect_available_models()
        if AVAILABLE_MODELS["ltx_video"]:
            DEFAULT_MODEL = "ltx_video"
        elif AVAILABLE_MODELS["svd_turbo"]:
            DEFAULT_MODEL = "svd_turbo"
        elif AVAILABLE_MODELS["zeroscope_v2_xl"]:
            DEFAULT_MODEL = "zeroscope_v2_xl"
    return AVAILABLE_MODELS, DEFAULT_MODEL

def init_comfy_modules():
    """Inicializa los modulos de ComfyUI (lazy loading)"""
    global AUDIO_AVAILABLE, COMFY_AVAILABLE, LAUNCHER_AVAILABLE
    if AUDIO_AVAILABLE is None:
        try:
            import roop.audio_generator as audio_gen
            AUDIO_AVAILABLE = True
        except ImportError:
            AUDIO_AVAILABLE = False
    if COMFY_AVAILABLE is None:
        try:
            import roop.comfy_workflows as workflows
            import roop.comfy_client as comfy_client
            COMFY_AVAILABLE = True
        except ImportError:
            COMFY_AVAILABLE = False
    if LAUNCHER_AVAILABLE is None:
        try:
            from ui.tabs.comfy_launcher import start as start_comfy, stop as stop_comfy, kill_process_on_port
            LAUNCHER_AVAILABLE = True
        except ImportError:
            LAUNCHER_AVAILABLE = False
    return AUDIO_AVAILABLE, COMFY_AVAILABLE, LAUNCHER_AVAILABLE


# ============================================================================
# DETECCION DE MODELOS
# ============================================================================

def detect_available_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    COMFY_MODELS_DIR = os.path.abspath(os.path.join(current_dir, "..", "tob", "ComfyUI", "models"))
    
    available = {
        "ltx_video": False,
        "svd_turbo": False,
        "zeroscope_v2_xl": False
    }
    
    # Detectar LTX Video
    # Priorizar 0.9.5 sobre 0.9.1 (0.9.5 tiene VAE compatible con ComfyUI)
    ltx_paths = [
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "ltx-video-0.9.5"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "ltx-video-0.9.1"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "ltx-video"),
        os.path.join(COMFY_MODELS_DIR, "checkpoints", "ltx-video-0.9.5"),
        os.path.join(COMFY_MODELS_DIR, "checkpoints", "ltx-video-0.9.1"),
    ]
    
    for ltx_path in ltx_paths:
        if os.path.exists(ltx_path):
            # Verificar que tenga el archivo model.safetensors
            model_file = os.path.join(ltx_path, "model.safetensors")
            if os.path.exists(model_file):
                available["ltx_video"] = True
                break
    
    # Detectar SVD Turbo - buscar en checkpoints y diffusion_models
    svd_possible_names = ["svd", "svd_xt", "stable_video_diffusion", "stablevideodiffusion", 
                          "stable-diffusion-turbo", "stablediffusionturbo", "turbo"]
    
    # Buscar en checkpoints
    checkpoints_dir = os.path.join(COMFY_MODELS_DIR, "checkpoints")
    if os.path.exists(checkpoints_dir):
        for filename in os.listdir(checkpoints_dir):
            filename_lower = filename.lower()
            for name in svd_possible_names:
                if name in filename_lower:
                    available["svd_turbo"] = True
                    break
            if available["svd_turbo"]:
                break
    
    # Buscar también en diffusion_models
    diffusion_models_dir = os.path.join(COMFY_MODELS_DIR, "diffusion_models")
    if os.path.exists(diffusion_models_dir) and not available["svd_turbo"]:
        for item in os.listdir(diffusion_models_dir):
            item_lower = item.lower()
            for name in svd_possible_names:
                if name in item_lower:
                    available["svd_turbo"] = True
                    break
            if available["svd_turbo"]:
                break
    
    # Detectar Zeroscope V2 XL - buscar en checkpoints y diffusion_models
    zeroscope_possible_names = ["zeroscope", "zero-scope", "zero_scope"]
    
    # Buscar en checkpoints
    if os.path.exists(checkpoints_dir):
        for filename in os.listdir(checkpoints_dir):
            filename_lower = filename.lower()
            for name in zeroscope_possible_names:
                if name in filename_lower:
                    available["zeroscope_v2_xl"] = True
                    break
            if available["zeroscope_v2_xl"]:
                break
    
    # Buscar también en diffusion_models
    if os.path.exists(diffusion_models_dir) and not available["zeroscope_v2_xl"]:
        for item in os.listdir(diffusion_models_dir):
            item_lower = item.lower()
            for name in zeroscope_possible_names:
                if name in item_lower:
                    available["zeroscope_v2_xl"] = True
                    break
            if available["zeroscope_v2_xl"]:
                break
    
    print(f"[AnimatePhoto] Modelos detectados: {available}")
    return available


def check_comfy_status():
    """Verifica si ComfyUI esta corriendo y tiene modelos"""
    try:
        import requests
        response = requests.get(f"http://127.0.0.1:{COMFYUI_PORT}/system_stats", timeout=2)
        if response.status_code != 200:
            return "🔴 No conectado"
        
        # Verificar checkpoints
        response2 = requests.get(f"http://127.0.0.1:{COMFYUI_PORT}/object_info", timeout=5)
        if response2.status_code == 200:
            data = response2.json()
            if "CheckpointLoaderSimple" in data:
                node = data["CheckpointLoaderSimple"]
                if "input" in node and "required" in node["input"]:
                    ckpt_list = node["input"]["required"].get("ckpt_name")
                    if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                        return f"🟢 Listo ({len(ckpt_list[0])} modelos)"
        return "🟡 Conectado (cargando...)"
    except:
        return "🔴 No conectado"


# ============================================================================
# VARIABLES GLOBALES
# ============================================================================

_comfy_url = f"http://127.0.0.1:{COMFYUI_PORT}"
_log_messages = []


def log(message):
    """Agrega mensaje al log"""
    global _log_messages
    timestamp = time.strftime("%H:%M:%S")
    _log_messages.append(f"[{timestamp}] {message}")
    return "\n".join(_log_messages[-10:])


def merge_audio_video(video_path, audio_path, output_path):
    """Fusiona audio y video con ffmpeg"""
    import subprocess
    cmd = [
        "ffmpeg", "-y", "-i", video_path, "-i", audio_path,
        "-c:v", "copy", "-c:a", "aac", "-shortest", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


# ============================================================================
# GENERACION PRINCIPAL
# ============================================================================

def animate_photo(
    image,
    action_prompt,
    text_to_speak,
    reference_audio,
    model_version=None,
    frames=24,
    fps=24,
    width=512,
    height=512,
    progress=gr.Progress()
):
    """Genera video usando modelos locales con ComfyUI."""
    from PIL import Image
    from roop.comfy_client import ComfyClient
    import roop.comfy_workflows as workflows
    
    global_models, _ = init_detect_models()
    init_comfy_modules()
    
    original_image = image
    
    if image is None:
        return None, "❌ Sube una imagen", log("[X] No hay imagen")

    if not action_prompt or not action_prompt.strip():
        return None, "❌ Describe la accion", log("[X] No hay prompt")

    if not model_version or model_version == "none":
        return None, "❌ Selecciona un modelo", log("[X] No hay modelo")

    comfy_status = check_comfy_status()
    if "No conectado" in comfy_status:
        return None, f"❌ ComfyUI no conectado: {comfy_status}", log("[X] ComfyUI no disponible")
    
    if model_version == "ltx_video" and not global_models["ltx_video"]:
        return None, "❌ Modelo no encontrado", log("[X] Modelo no encontrado")
    elif model_version == "svd_turbo" and not global_models["svd_turbo"]:
        return None, "❌ Modelo no encontrado", log("[X] Modelo no encontrado")
    elif model_version == "zeroscope_v2_xl" and not global_models["zeroscope_v2_xl"]:
        return None, "❌ Modelo no encontrado", log("[X] Modelo no encontrado")

    temp_image = None
    temp_image_path = None
    voice_audio_path = None
    output_path = None
    output_path_final = None

    try:
        client = ComfyClient()
        
        if hasattr(image, 'name'):
            temp_image_path = image.name
        elif isinstance(image, Image.Image):
            temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            temp_image_path = temp_image.name
            image.save(temp_image_path)
        elif isinstance(image, str):
            temp_image_path = image
        else:
            try:
                temp_image = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
                temp_image_path = temp_image.name
                Image.fromarray(image).save(temp_image_path)
            except Exception as e:
                return None, f"❌ Error: {str(e)}", log(f"[X] Error imagen: {e}")
        
        if not os.path.exists(temp_image_path):
            return None, "❌ Error archivo temporal", log(f"[X] Temp file not found")
        
        image_filename = client.upload_image(temp_image_path)
        if not image_filename:
            return None, "❌ Error subiendo imagen", log("[X] Error upload imagen")

        log(f"[OK] Imagen: {image_filename}")

        if text_to_speak and text_to_speak.strip():
            log(f"[VOZ] Generando voz...")
            import roop.audio_generator as audio_gen
            
            if reference_audio:
                audio_file = audio_gen.generate_audio(text=text_to_speak, lenguaje="es", speaker_wav=reference_audio)
            else:
                audio_file = audio_gen.generate_audio(text=text_to_speak, lenguaje="es", model_name="tts_models/multilingual/multi-dataset/xtts_v2")

            if audio_file and os.path.exists(audio_file):
                voice_audio_path = audio_file
                log(f"[OK] Voz generada")
            else:
                log("[WARN] Error generando voz")

        timestamp = int(time.time())
        output_dir = os.path.abspath("output")
        os.makedirs(output_dir, exist_ok=True)
        
        if model_version == "ltx_video":
            # LTX Video usa parámetros específicos: 768x512, 49 frames (múltiplo de 8 + 1)
            ltx_width = 768
            ltx_height = 512
            ltx_frames = 49  # LTX requiere frames = 8*n + 1
            
            workflow = workflows.get_ltx_video_workflow(
                image_filename=image_filename,
                prompt=action_prompt,
                seed=int(timestamp),
                width=ltx_width, height=ltx_height, frames=ltx_frames, fps=fps
            )
            output_path = os.path.join(output_dir, f"ltx_video_{timestamp}.mp4")
        elif model_version == "svd_turbo":
            workflow = workflows.get_svd_turbo_workflow(
                image_filename=image_filename,
                prompt=action_prompt,
                seed=int(timestamp),
                width=width, height=height, frames=frames, fps=fps
            )
            output_path = os.path.join(output_dir, f"svd_turbo_{timestamp}.mp4")
        elif model_version == "zeroscope_v2_xl":
            workflow = workflows.get_zeroscope_v2_xl_workflow(
                image_filename=image_filename,
                prompt=action_prompt,
                seed=int(timestamp),
                width=width, height=height, frames=frames, fps=fps
            )
            output_path = os.path.join(output_dir, f"zeroscope_v2_xl_{timestamp}.mp4")
        else:
            return None, "❌ Modelo no reconocido", log("❌ Modelo no reconocido")

        log("[PROC] Procesando...")

        try:
            success, result = client.generate_video(temp_image_path, workflow, output_path)
        except Exception as e:
            log(f"[ERROR] {str(e)}")
            import traceback
            log(f"[ERROR] Stack: {traceback.format_exc()}")
            return None, "❌ Error generando", log("[X] Error generando video")

        if not success:
            log(f"[ERROR] {result}")
            return None, f"❌ {result}", log(f"[ERROR] {result}")

        log(f"[OK] Video generado")

        if voice_audio_path:
            log("[PROC] Añadiendo voz...")
            output_path_final = os.path.join(output_dir, f"imagine_{timestamp}_voz.mp4")
            merge_audio_video(output_path, voice_audio_path, output_path_final)
            log("[OK] Video con voz")
            return output_path_final, "✅ Completo (con voz)", log("[OK] Exito")
        else:
            return output_path, "✅ Completo (sin audio)", log("[OK] Exito")

    except Exception as e:
        log(f"[ERROR] {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"❌ {str(e)}", log(f"[ERROR] {str(e)}")

    finally:
        if temp_image and os.path.exists(temp_image.name):
            try:
                os.remove(temp_image.name)
            except:
                pass


# ============================================================================
# INTERFAZ MINIMA
# ============================================================================

def animate_photo_tab():
    """UI minima"""
    
    models_state = gr.State(value=None)
    default_model_state = gr.State(value=None)
    models_loaded = gr.State(value=False)
    
    def load_and_update_models(load_trigger):
        """Carga modelos y actualiza dropdown"""
        if load_trigger:
            models, default_model = init_detect_models()
            status_text = get_model_status_text(models, default_model)
            choices, value = get_choices_fn(models, "none")
            return models, default_model, True, status_text, gr.update(choices=choices, value=value)
        return None, None, False, "💡 Pulsa '🔍 Detectar Modelos'", gr.update(choices=[("Sin modelos", "none")], value="none")
    
    def get_choices_fn(models, current_value=None):
        if models is None:
            return [("Sin modelos", "none")], "none"
        choices = []
        default_value = "none"
        if models["ltx_video"]:
            choices.append(("LTX Video 0.9.1 (⭐ Recomendado)", "ltx_video"))
            default_value = "ltx_video"
        if models["svd_turbo"]:
            choices.append(("SVD Turbo", "svd_turbo"))
            if default_value == "none":
                default_value = "svd_turbo"
        if models["zeroscope_v2_xl"]:
            choices.append(("Zeroscope V2 XL", "zeroscope_v2_xl"))
            if default_value == "none":
                default_value = "zeroscope_v2_xl"
        if not choices:
            choices.append(("Sin modelos", "none"))
            default_value = "none"
        value_to_use = current_value if current_value in [c[1] for c in choices] else default_value
        return choices, value_to_use

    with gr.Tab("Imagine"):
        gr.Markdown("**Imagen + Accion = Video**")
        
        with gr.Row():
            comfy_status = gr.Textbox(
                value="🔴 No conectado",
                interactive=False,
                label="Estado",
                scale=4
            )
            btn_refresh_status = gr.Button("🔄", size="sm", scale=1)
        
        with gr.Row():
            model_info = gr.Textbox(
                label="Modelos",
                value="💡 Pulsa '🔍 Detectar Modelos'",
                interactive=False,
                lines=2
            )
            btn_detect_models = gr.Button("🔍", size="sm")
        
        model_selector = gr.Dropdown(
            label="Modelo",
            choices=[("Sin modelos", "none")],
            value="none",
            interactive=True
        )

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Imagen",
                    type="pil",
                    height=250,
                    sources=["upload", "clipboard"]
                )

                action_prompt = gr.Textbox(
                    label="Accion",
                    placeholder="Ej: 'personaje caminando'",
                    lines=2,
                )

                text_to_speak = gr.Textbox(
                    label="Voz (opcional)",
                    placeholder="Texto a decir...",
                    lines=2,
                )

                reference_audio = gr.Audio(
                    label="Voice Cloning",
                    type="filepath"
                )

                # Configuracion de video con tooltips
                gr.Markdown("### Configuracion de Video")
                
                with gr.Row():
                    frames = gr.Slider(
                        label="Frames ⓘ",
                        minimum=10,
                        maximum=100,
                        value=24,
                        step=1,
                        info="Numero de fotogramas del video (mas frames = video mas largo)"
                    )
                    fps = gr.Slider(
                        label="FPS ⓘ",
                        minimum=8,
                        maximum=30,
                        value=24,
                        step=1,
                        info="Fotogramas por segundo (24 = velocidad estandar)"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        label="Ancho ⓘ",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        info="Ancho del video en pixeles (mas alto = mas detalle)"
                    )
                    height = gr.Slider(
                        label="Alto ⓘ",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        info="Alto del video en pixeles"
                    )

                btn_animate = gr.Button("✨ Animar", variant="primary", size="lg")

            with gr.Column(scale=1):
                output_video = gr.Video(
                    label="Resultado",
                    height=350,
                    autoplay=True
                )

                status = gr.Textbox(
                    label="Estado",
                    value="💡 Sube una imagen",
                    interactive=False,
                    lines=2
                )

                log_box = gr.Textbox(
                    label="Log",
                    value="",
                    lines=4,
                    interactive=False
                )

        # Boton de refresh
        btn_refresh_status.click(fn=check_comfy_status, outputs=[comfy_status])

        btn_detect_models.click(
            fn=load_and_update_models,
            inputs=btn_detect_models,
            outputs=[models_state, default_model_state, models_loaded, model_info, model_selector]
        )

        btn_animate.click(
            fn=animate_photo,
            inputs=[input_image, action_prompt, text_to_speak, reference_audio, model_selector, frames, fps, width, height],
            outputs=[output_video, status, log_box]
        )

        def on_image_loaded(img):
            if img is not None:
                return "✅ Imagen cargada"
            return "💡 Sube una imagen"

        input_image.change(
            fn=on_image_loaded,
            inputs=[input_image],
            outputs=[status]
        )


def get_model_status_text(models=None, default_model=None):
    """Retorna texto con el estado de los modelos"""
    lines = []
    
    if models is None:
        models, _ = init_detect_models()
    if default_model is None:
        _, default_model = init_detect_models()
    
    if models["ltx_video"]:
        lines.append("[OK] LTX Video 0.9.1 (⭐)")
    else:
        lines.append("[X] LTX Video 0.9.1")
    
    if models["svd_turbo"]:
        lines.append("[OK] SVD Turbo")
    else:
        lines.append("[X] SVD Turbo")
    
    if models["zeroscope_v2_xl"]:
        lines.append("[OK] Zeroscope V2 XL")
    else:
        lines.append("[X] Zeroscope V2 XL")
    
    return "\n".join(lines)
