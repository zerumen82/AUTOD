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
        # PRIORIDAD: WanVideo > LTX > Zeroscope > CogVideo (SVD deshabilitado)
        if AVAILABLE_MODELS["wan_video"]:
            DEFAULT_MODEL = "wan_video"
        elif AVAILABLE_MODELS["ltx_video"]:
            DEFAULT_MODEL = "ltx_video"
        elif AVAILABLE_MODELS["zeroscope"]:
            DEFAULT_MODEL = "zeroscope"
        elif AVAILABLE_MODELS["cogvideo"]:
            DEFAULT_MODEL = "cogvideo"
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
            import roop.comfy_workflows_fixed as workflows_ltx
            from roop.comfy_workflows import check_required_nodes, check_model_available
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
        "cogvideo": False,
        "ltx_video": False,
        "svd_turbo": False,
        "svd_xt": False,
        "wan_video": False,
        "zeroscope": False
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
            # Verificar que tenga el archivo transformer.safetensors o model.safetensors
            model_file = os.path.join(ltx_path, "transformer.safetensors")
            if not os.path.exists(model_file):
                model_file = os.path.join(ltx_path, "model.safetensors")
            if os.path.exists(model_file):
                available["ltx_video"] = True
                break
    
    # Detectar SVD Turbo - buscar en checkpoints y diffusion_models
    svd_possible_names = ["svd", "stable_video_diffusion", "stablevideodiffusion", 
                          "stable-diffusion-turbo", "stablediffusionturbo", "turbo"]
    
    # Buscar en checkpoints
    checkpoints_dir = os.path.join(COMFY_MODELS_DIR, "checkpoints")
    if os.path.exists(checkpoints_dir):
        for filename in os.listdir(checkpoints_dir):
            filename_lower = filename.lower()
            for name in svd_possible_names:
                if name in filename_lower and "svd_xt" not in filename_lower:
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
                if name in item_lower and "svd_xt" not in item_lower:
                    available["svd_turbo"] = True
                    break
            if available["svd_turbo"]:
                break
    
    # SVD DESHABILITADO - No funciona con ComfyUI (model type not detected)
    # svd_xt_path = os.path.join(COMFY_MODELS_DIR, "diffusion_models", "svd_xt")
    # if os.path.exists(svd_xt_path):
    #     unet_file = os.path.join(svd_xt_path, "unet", "diffusion_pytorch_model.safetensors")
    #     if os.path.exists(unet_file) and os.path.getsize(unet_file) > 100_000_000:
    #         available["svd_xt"] = True
    #         available["svd_turbo"] = True
    
    # Detectar WanVideo - buscar en diffusion_models (archivo GGUF o carpeta wan)
    # IMPORTANTE: WanVideo requiere VAE compatible con la version actual de ComfyUI-WanVideoWrapper
    wan_paths = [
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "Wan2.1-I2V-14B-480P-Q4_K_M"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "Wan2.2-I2V-Q4_K_M"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "wan"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "Wan2.2"),
    ]
    
    wan_gguf_found = False
    for wan_path in wan_paths:
        if os.path.exists(wan_path):
            wan_gguf_found = True
            break
    
    # Verificar si el VAE de WanVideo es compatible
    # El VAE de WanVideo ( Wan2.1/Wan2.2) tiene estructura anidada (downsamples.X.downsamples.Y)
    # Esta es la estructura CORRECTA y compatible
    wan_vae_compatible = False
    vae_path = os.path.join(COMFY_MODELS_DIR, "vae")
    vae_file = os.path.join(vae_path, "Wan2.2_VAE.safetensors")
    if os.path.exists(vae_file):
        try:
            import safetensors.torch
            import re
            with safetensors.torch.safe_open(vae_file, framework="pt") as f:
                keys = list(f.keys())
                
                # La estructura correcta del VAE de WanVideo tiene downsamples.X.residual
                # No requiere dos 'downsamples' anidados
                has_downsamples = any("downsamples" in k for k in keys)
                
                # Tambien verificar que tenga la estructura basica de VAE
                has_encoder = any(k.startswith("encoder.") for k in keys)
                has_decoder = any(k.startswith("decoder.") for k in keys)
                
                if has_downsamples and has_encoder and has_decoder:
                    # Estructura con downsamples + encoder/decoder = WANVIDEO VAE COMPATIBLE
                    wan_vae_compatible = True
                    print(f"[AnimatePhoto] WanVideo VAE compatible detectada (estructura downsamples)")
                elif has_encoder and has_decoder:
                    # Estructura plana - verificar canales
                    # Buscar conv2.weight para verificar canales
                    conv2_shape = None
                    for k in keys:
                        if k == "conv2.weight":
                            conv2_shape = f.get_tensor(k).shape
                            break
                    if conv2_shape and len(conv2_shape) == 4:
                        in_channels = conv2_shape[1]
                        out_channels = conv2_shape[0]
                        print(f"[AnimatePhoto] VAE canales: in={in_channels}, out={out_channels}")
                        # WanVideoVAE38 tiene 48 canales, WanVideoVAE (original) tiene 16 canales
                        if in_channels == 48 or out_channels == 48 or in_channels == 16 or out_channels == 16:
                            wan_vae_compatible = True
                            print(f"[AnimatePhoto] WanVideo VAE compatible detectada ({in_channels} canales)")
                        else:
                            print(f"[AnimatePhoto] WanVideo VAE INCOMPATIBLE - canales no reconocidos")
                    else:
                        print(f"[AnimatePhoto] WanVideo VAE INCOMPATIBLE - no se pudieron verificar canales")
                else:
                    print(f"[AnimatePhoto] WanVideo VAE INCOMPATIBLE - falta estructura encoder/decoder")
        except Exception as e:
            print(f"[AnimatePhoto] Error verificando VAE WanVideo: {e}")
    
    # WanVideo disponible si tiene modelo GGUF Y VAE compatible
    if wan_gguf_found and wan_vae_compatible:
        available["wan_video"] = True
    elif wan_gguf_found and not wan_vae_compatible:
        print(f"[AnimatePhoto] WanVideo detectado pero VAE incompatible - no disponible")
    
    # Detectar Zeroscope - buscar en diffusion_models
    zeroscope_paths = [
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "zeroscope_v2_XL"),
        os.path.join(COMFY_MODELS_DIR, "diffusion_models", "zeroscope"),
    ]
    for zs_path in zeroscope_paths:
        if os.path.exists(zs_path):
            # Verificar que tenga los componentes necesarios (UNET, TEXT_ENCODER)
            unet_path = os.path.join(zs_path, "UNET")
            text_encoder_path = os.path.join(zs_path, "TEXT_ENCODER")
            if os.path.exists(unet_path) and os.path.exists(text_encoder_path):
                available["zeroscope"] = True
                print(f"[AnimatePhoto] Zeroscope detectado en: {zs_path}")
                break
    
    # Verificar CogVideoX
    cogvideo_path = os.path.join(COMFY_MODELS_DIR, "CogVideo", "CogVideoX-5b-1.5")
    if os.path.exists(cogvideo_path):
        # Verificar componentes necesarios
        transformer_path = os.path.join(cogvideo_path, "transformer_I2V", "diffusion_pytorch_model.safetensors")
        vae_path = os.path.join(cogvideo_path, "vae", "diffusion_pytorch_model.safetensors")
        text_encoder_path = os.path.join(cogvideo_path, "text_encoder", "model-00001-of-00002.safetensors")
        
        if os.path.exists(transformer_path) and os.path.exists(vae_path) and os.path.exists(text_encoder_path):
            available["cogvideo"] = True
            print(f"[AnimatePhoto] CogVideoX-5B-I2V detectado!")
    
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
            
            # Verificar nodos requeridos para animación
            required_animate_nodes = [
                "UNETLoader", "VAELoader", "CLIPVisionLoader",
                "SVD_img2vid_Conditioning", "VHS_VideoCombine",
                "LoadImage", "KSampler", "VAEDecode"
            ]
            missing_nodes = []
            for node_name in required_animate_nodes:
                if node_name not in data:
                    missing_nodes.append(node_name)
            
            if missing_nodes:
                return f"🔴 Faltan nodos: {', '.join(missing_nodes)}"
            
        return "🟡 Conectado (cargando...)"
    except:
        return "🔴 No conectado"


def check_required_nodes_for_animation():
    """Verifica los nodos requeridos para animate_image y retorna un diccionario de estado"""
    import requests
    
    required_nodes = {
        # Nodos base requeridos para cualquier workflow
        "base": ["LoadImage", "CheckpointLoaderSimple", "UNETLoader", "VAELoader", "KSampler", "VAEDecode", "CLIPTextEncode"],
        # Nodos para WanVideo (disponibles en ComfyUI-WanVideoWrapper + ComfyUI-GGUF)
        "wan_video": [
            "WanVideoDecode", "WanVideoImageToVideoEncode", "WanVideoTextEncode",
            "UnetLoaderGGUF", "CLIPLoaderGGUF"
        ],
        # Nodos para SVD Turbo
        "svd": ["CLIPVisionLoader", "SVD_img2vid_Conditioning", "VHS_VideoCombine"],
        # Nodos para LTX Video (disponibles en ComfyUI_LTXVideo)
        "ltx": ["LTXVideoCanny", "LTXVideoDiffusionCompression", "LTXVideoInterpolate", "LTXVideoDecode", "LTXVideoEncode", 
                 "LTXVLinearOverlapLatentTransition", "LTXVAddGuideAdvanced", "LTXVAddLatentGuide", "LTXVAdainLatent", 
                 "LTXVImgToVideoConditionOnly", "LTXVApplySTG", "LTXVBaseSampler", "LTXVInContextSampler", 
                 "LTXVTiledVAEDecode", "LTXVImgToVideo", "ModelSamplingLTXV", "LTXVConditioning"]
    }
    
    try:
        response = requests.get(f"http://127.0.0.1:{COMFYUI_PORT}/object_info", timeout=5)
        if response.status_code != 200:
            return {"error": "ComfyUI no responde", "available_nodes": []}
        
        available = response.json()
        available_node_names = list(available.keys())
        
        result = {
            "available_nodes": available_node_names,
            "missing": {},
            "ok": {}
        }
        
        # Verificar nodos base
        for category, nodes in required_nodes.items():
            result["missing"][category] = []
            result["ok"][category] = []
            for node in nodes:
                if node in available_node_names:
                    result["ok"][category].append(node)
                else:
                    result["missing"][category].append(node)
        
        return result
        
    except Exception as e:
        return {"error": str(e), "available_nodes": []}


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
    import roop.comfy_workflows_fixed as workflows_ltx
    from roop.comfy_workflows import check_required_nodes, check_model_available
    
    global_models, _ = init_detect_models()
    init_comfy_modules()
    
    original_image = image
    
    if image is None:
        return None, "❌ Sube una imagen", log("[X] No hay imagen")

    # Todos los modelos soportan prompts
    # (SVD deshabilitado)
    if True:
        if not action_prompt or not action_prompt.strip():
            return None, "❌ Describe la accion", log("[X] No hay prompt")

    if not model_version or model_version == "none":
        return None, "❌ Selecciona un modelo", log("[X] No hay modelo")

    comfy_status = check_comfy_status()
    if "No conectado" in comfy_status:
        return None, f"❌ ComfyUI no conectado: {comfy_status}", log("[X] ComfyUI no disponible")
    
    # Verificar nodos requeridos para animación
    # NOTA: CogVideoX usa DownloadAndLoadCogVideoModel que descarga automaticamente, omitir verificacion
    try:
        if model_version != "cogvideo":
            node_check = workflows.check_required_nodes(model_version)
            if not node_check["ok"]:
                error_msg = node_check.get("error", "Error desconocido")
                missing = node_check.get("missing_nodes", [])
                return None, f"❌ {error_msg}", log(f"[X] {error_msg}")
        else:
            log("[INFO] Omitiendo verificacion de nodos para CogVideoX (usa DownloadAndLoadCogVideoModel)")
    except Exception as e:
        log(f"[WARN] No se pudo verificar nodos: {e}")
    
    # DEBUG: Log model_version received
    log(f"[DEBUG] model_version seleccionado: {model_version}")
    
    # Validar modelo seleccionado
    if model_version == "svd_xt":
        return None, "❌ SVD XT está deshabilitado (no funciona con ComfyUI)", log("[X] SVD XT deshabilitado")
    elif model_version == "svd_turbo":
        return None, "❌ SVD Turbo está deshabilitado (no funciona con ComfyUI)", log("[X] SVD Turbo deshabilitado")
    elif model_version == "cogvideo" and not global_models["cogvideo"]:
        return None, "❌ Modelo CogVideoX no encontrado. Verifica los archivos en: ui/tob/ComfyUI/models/CogVideo/", log("[X] Modelo CogVideoX no encontrado")
    elif model_version == "wan_video" and not global_models["wan_video"]:
        log("[DEBUG] WanVideo model check - global_models: " + str(global_models.get("wan_video", "NOT FOUND")))
        return None, "❌ Modelo WanVideo no encontrado en: ui/tob/ComfyUI/models/diffusion_models/", log("[X] Modelo WanVideo no encontrado")

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
        
        # Seleccionar el módulo de workflow correcto según el modelo
        # Verificar primero si el modelo está disponible
        log(f"[DEBUG] Iniciando selección de workflow para: {model_version}")
        
        if model_version == "ltx_video":
            # Verificar si el modelo LTX está instalado
            model_check = workflows.check_model_available("ltx")
            if not model_check["ok"] or "ltx_video" not in model_check["available_models"]:
                log(f"[WARN] Modelo LTX Video no encontrado: {model_check.get('error', 'Archivos no encontrados')}")
                log("[INFO] Buscando modelos alternativos disponibles...")
                
                # Ver qué modelos están disponibles
                all_models = workflows.check_model_available("all")
                available = all_models.get("available_models", [])
                
                # SVD deshabilitado - solo usar otros modelos
                if "wan_video" in available:
                    log("[INFO] Usando WanVideo como alternativa")
                    model_version = "wan_video"
                elif "ltx_video" in available:
                    log("[INFO] Usando LTX Video como alternativa")
                    model_version = "ltx_video"
                elif "zeroscope" in available:
                    log("[INFO] Usando Zeroscope como alternativa")
                    model_version = "zeroscope"
                else:
                    return None, "❌ No hay modelos de video instalados", log("❌ Error: Instala al menos un modelo de video (WanVideo, LTX o Zeroscope)")
        
        if model_version == "ltx_video":
            # LTX Video usa parámetros específicos y workflow corregido
            ltx_width = 768
            ltx_height = 512
            ltx_frames = 49  # LTX requiere frames = 8*n + 1
            wf_module = workflows_ltx
        else:
            # SVD y WanVideo usan workflow original
            wf_module = workflows
        
        try:
            log(f"[DEBUG] Creando workflow para: {model_version}")
            if model_version == "ltx_video":
                # Usar workflow corregido de comfy_workflows_fixed
                workflow = wf_module.get_ltxvideo2_workflow(
                    image_filename=image_filename,
                    prompt=action_prompt,
                    seed=int(timestamp),
                    width=ltx_width, height=ltx_height, frames=ltx_frames, fps=fps
                )
                output_path = os.path.join(output_dir, f"ltx_video_{timestamp}.mp4")
            elif model_version == "svd_turbo":
                workflow = wf_module.get_svd_turbo_workflow(
                    image_filename=image_filename,
                    prompt=action_prompt,
                    seed=int(timestamp),
                    width=width, height=height, frames=frames, fps=fps
                )
                output_path = os.path.join(output_dir, f"svd_turbo_{timestamp}.mp4")
            elif model_version == "svd_xt":
                return None, "❌ SVD XT deshabilitado", log("[X] SVD XT deshabilitado")
            elif model_version == "svd_turbo":
                return None, "❌ SVD Turbo deshabilitado", log("[X] SVD Turbo deshabilitado")
            elif model_version == "wan_video":
                # Verificar si WanVideo está disponible
                model_check = workflows.check_model_available("wan")
                if not model_check["ok"] or "wan_video" not in model_check["available_models"]:
                    log(f"[WARN] Modelo WanVideo no encontrado")
                    all_models = workflows.check_model_available("all")
                    available = all_models.get("available_models", [])
                    
                    # SVD deshabilitado - usar otros modelos
                    if "ltx_video" in available:
                        log("[INFO] Usando LTX Video como alternativa")
                        model_version = "ltx_video"
                    elif "zeroscope" in available:
                        log("[INFO] Usando Zeroscope como alternativa")
                        model_version = "zeroscope"
                    else:
                        return None, "❌ No hay modelos de video instalados", log("❌ Error: Instala al menos un modelo de video")
                
                workflow = wf_module.get_wan_video_workflow(
                    image_filename=image_filename,
                    prompt=action_prompt,
                    seed=int(timestamp),
                    width=width, height=height, frames=frames, fps=fps
                )
                output_path = os.path.join(output_dir, f"wan_video_{timestamp}.mp4")
            elif model_version == "cogvideo":
                # CogVideoX - Usar workflow especializado
                log("[INFO] Generando workflow para CogVideoX...")
                
                # Importar el módulo de workflows fixed que tiene CogVideoX
                import roop.comfy_workflows_fixed as workflows_fixed
                
                workflow = workflows_fixed.get_cogvideox_workflow(
                    image_filename=image_filename,
                    prompt=action_prompt,
                    seed=int(timestamp),
                    width=width, height=height, frames=min(frames, 49),  # CogVideoX max 49 frames
                    fps=fps,
                    model_version="cogvideo_1_5_5b"
                )
                output_path = os.path.join(output_dir, f"cogvideo_{timestamp}.mp4")
            elif model_version == "zeroscope":
                # Zeroscope V2 XL
                log("[INFO] Generando workflow para Zeroscope V2 XL...")
                
                import roop.comfy_workflows as workflows_zs
                
                # Zeroscope usa 576x320 por defecto
                zs_width = 576
                zs_height = 320
                zs_frames = frames  # Zeroscope puede manejar varios frames
                
                workflow = workflows_zs.get_zeroscope_v2_xl_workflow(
                    image_filename=image_filename,
                    prompt=action_prompt,
                    seed=int(timestamp),
                    width=zs_width, height=zs_height, frames=zs_frames, fps=fps
                )
                output_path = os.path.join(output_dir, f"zeroscope_{timestamp}.mp4")
            else:
                return None, "❌ Modelo no reconocido", log("❌ Modelo no reconocido")
        except Exception as e:
            log(f"[ERROR] Error generando workflow: {e}")
            return None, f"❌ Error: {str(e)}", log(f"[ERROR] {e}")

        log("[PROC] Procesando...")

        try:
            success, result = client.generate_video(temp_image_path, workflow, output_path)
        except Exception as e:
            log(f"[ERROR] {str(e)}")
            import traceback
            log(f"[ERROR] Stack: {traceback.format_exc()}")
            return None, "❌ Error generando", log("[X] Error generando video")

        if not success:
            # Verificar si es un error de VAE de WanVideo
            error_str = str(result).lower()
            is_vae_error = any(x in error_str for x in [
                "vae", "size mismatch", "load_state_dict", 
                "conv1.weight", "downsamples", "encoder", "decoder"
            ])
            
            # Si es error de VAE y estamos usando WanVideo, intentar con LTX
            if is_vae_error and model_version == "wan_video":
                log(f"[WARN] Error de VAE en WanVideo: {result}")
                log("[INFO] Intentando con LTX Video como alternativa...")
                
                # Reintentar con LTX
                try:
                    import roop.comfy_workflows_ltx as workflows_ltx
                    ltx_wf = workflows_ltx.get_ltx_video_workflow(
                        image_filename=image_filename,
                        prompt=action_prompt,
                        seed=int(timestamp),
                        width=768, height=512, frames=49, fps=24
                    )
                    output_path = os.path.join(output_dir, f"ltx_video_{timestamp}.mp4")
                    success2, result2 = client.generate_video(temp_image_path, ltx_wf, output_path)
                    if success2:
                        log("[OK] Video generado con LTX Video (fallback)")
                        if voice_audio_path:
                            log("[PROC] Añadiendo voz...")
                            output_path_final = os.path.join(output_dir, f"imagine_{timestamp}_voz.mp4")
                            merge_audio_video(output_path, voice_audio_path, output_path_final)
                            log("[OK] Video con voz")
                            return output_path_final, "✅ Completo (LTX fallback, con voz)", log("[OK] Exito")
                        else:
                            return output_path, "✅ Completo (LTX fallback, sin audio)", log("[OK] Exito")
                    else:
                        log(f"[ERROR] LTX fallback también falló: {result2}")
                except Exception as e2:
                    log(f"[ERROR] Error en fallback LTX: {e2}")
            
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
        # ORDEN: WanVideo > LTX > Zeroscope > CogVideo (SVD deshabilitado)
        if models["wan_video"]:
            choices.append(("WanVideo (Image-to-Video)", "wan_video"))
            if default_value == "none":
                default_value = "wan_video"
        # SVD deshabilitado porque no funciona con ComfyUI
        # if models["svd_xt"]:
        #     choices.append(("SVD XT (Image-to-Video)", "svd_xt"))
        # if models["svd_turbo"]:
        #     choices.append(("SVD Turbo", "svd_turbo"))
        if models["ltx_video"]:
            choices.append(("LTX Video (Image-to-Video)", "ltx_video"))
            if default_value == "none":
                default_value = "ltx_video"
        if models["zeroscope"]:
            choices.append(("Zeroscope (Image-to-Video)", "zeroscope"))
            if default_value == "none":
                default_value = "zeroscope"
        if models["cogvideo"]:
            choices.append(("CogVideoX (Image-to-Video)", "cogvideo"))
            if default_value == "none":
                default_value = "cogvideo"
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
                    placeholder="Ej: 'personaje caminando', 'movimiento suave'",
                    lines=2,
                    info="Todos los modelos soportan prompts de texto"
                )
                
                # Info label para el prompt
                prompt_info = gr.Markdown(
                    "**Nota:** Todos los modelos soportan prompts de texto",
                    visible=False
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
                gr.Markdown("""
                ### Configuracion de Video
                
                **Guia rapida:**
                - **Frames:** Numero de fotogramas (10-100)
                - **FPS:** Velocidad (24=estandar, 30=suave)
                - **Ancho/Alto:** Resolucion (multiplos de 8 para mejor calidad)
                """)
                
                with gr.Row():
                    frames = gr.Slider(
                        label="Frames ⓘ",
                        minimum=10,
                        maximum=100,
                        value=24,
                        step=1,
                        info="Frames: 21-33 recomendado. WanVideo: multiples de 4n+1 | LTX: 8n+1 | SVD: max 25"
                    )
                    fps = gr.Slider(
                        label="FPS ⓘ",
                        minimum=8,
                        maximum=30,
                        value=24,
                        step=1,
                        info="FPS: 8=lento, 16=normal, 24=rapido. Recomendado 8-16 para mejor calidad"
                    )
                
                with gr.Row():
                    width = gr.Slider(
                        label="Ancho ⓘ",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        info="Ancho (recomendado): WanVideo 512-720 | LTX 512-768 | SVD 1024. Multiplos de 8"
                    )
                    height = gr.Slider(
                        label="Alto ⓘ",
                        minimum=256,
                        maximum=1024,
                        value=512,
                        step=64,
                        info="Alto (recomendado): WanVideo 512 | LTX 512 | SVD 576. Multiplos de 8"
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
        
        # Función para actualizar visibilidad del prompt según modelo
        def on_model_change(model):
            if model in ["svd_turbo", "svd_xt"]:
                return {
                    action_prompt: gr.update(placeholder="Escribe un prompt para controlar el video", interactive=True),
                    prompt_info: gr.update(visible=True)
                }
            else:
                return {
                    action_prompt: gr.update(placeholder="Ej: 'personaje caminando'", interactive=True),
                    prompt_info: gr.update(visible=False)
                }
        
        model_selector.change(
            fn=on_model_change,
            inputs=[model_selector],
            outputs=[action_prompt, prompt_info]
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
    
    lines.append("[X] SVD (deshabilitado)")
    
    if models["cogvideo"]:
        lines.append("[OK] CogVideoX-5B-I2V ⭐")
    else:
        lines.append("[X] CogVideoX-5B-I2V")
    
    if models["wan_video"]:
        lines.append("[OK] WanVideo")
    else:
        lines.append("[X] WanVideo")
    
    return "\n".join(lines)
