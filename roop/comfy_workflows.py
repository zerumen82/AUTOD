
"""
ComfyUI Workflows para Video AI
"""

import os

# Modelos disponibles
SVD_TURBO_MODEL = "stable_video_diffusion"  # Nombre base del modelo SVD
LTX_VIDEO_MODEL = "ltx-video-0.9.1"  # LTX Video 0.9.1 (VAE compatible con ComfyUI)
ZEROSCOPE_V2_XL_MODEL = "zeroscope_v2_XL"


def get_comfyui_url() -> str:
    """Obtiene la URL de ComfyUI dinámicamente"""
    try:
        from ui.tabs.comfy_launcher import get_comfy_url
        return get_comfy_url()
    except:
        return "http://127.0.0.1:8188"


def get_available_models(category="checkpoints"):
    """
    Obtiene la lista de modelos disponibles en ComfyUI para una categoría específica.
    
    Categorías disponibles:
    - checkpoints: Modelos principales (ckpt/safetensors)
    - clip: Text encoders (Gemma 3, T5-XXL, etc.)
    - vae: Variational Autoencoders
    """
    try:
        import requests
        # Usar /object_info para obtener información de los nodos (URL dinámica)
        comfy_url = get_comfyui_url()
        response = requests.get(f"{comfy_url}/object_info", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            # Buscar el nodo CheckpointLoaderSimple para obtener checkpoints
            if category == "checkpoints" and "CheckpointLoaderSimple" in data:
                checkpoint_node = data["CheckpointLoaderSimple"]["input"]["required"]
                if "ckpt_name" in checkpoint_node:
                    return checkpoint_node["ckpt_name"][0]
            
            # Buscar el nodo CLIPLoader para obtener CLIP models
            elif category == "clip" and "CLIPLoader" in data:
                clip_node = data["CLIPLoader"]["input"]["required"]
                if "clip_name" in clip_node:
                    return clip_node["clip_name"][0]
    except Exception as e:
        print(f"Error obteniendo modelos: {e}")
    return []


# =============================================================================
# WORKFLOW SVD Turbo - Velocidad máxima
# =============================================================================

def get_svd_turbo_workflow(
    image_filename, prompt, seed=None,
    width=720, height=480, frames=32, fps=24
):
    """
    WORKFLOW SVD Turbo - Velocidad máxima para tu GPU
    
    ⚠️  REQUIERE: Modelo SVD en diffusion_models/StableDiffusionTurbo/ o checkpoints/
    
    NOTA: SVD NO soporta prompts de texto. El prompt se ignora.
    Solo genera movimiento a partir de la imagen de entrada.
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic"

    negative_prompt = "low quality, blurry, distorted, bad anatomy"
    
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(script_dir, "ui", "tob", "ComfyUI", "models")
    diffusion_models_path = os.path.join(models_path, "diffusion_models")
    checkpoints_path = os.path.join(models_path, "checkpoints")
    
    # Buscar modelo SVD en diffusion_models/StableDiffusionTurbo
    svd_model_name = None
    stable_diffusion_turbo_path = os.path.join(diffusion_models_path, "StableDiffusionTurbo")
    
    if os.path.exists(os.path.join(stable_diffusion_turbo_path, "svd_xt.safetensors")):
        svd_model_name = "StableDiffusionTurbo/svd_xt.safetensors"
        print(f"[SVD_TURBO] Modelo encontrado en diffusion_models: {svd_model_name}")
    elif os.path.exists(os.path.join(stable_diffusion_turbo_path, "svd_xt_image_decoder.safetensors")):
        svd_model_name = "StableDiffusionTurbo/svd_xt_image_decoder.safetensors"
        print(f"[SVD_TURBO] Modelo encontrado en diffusion_models: {svd_model_name}")
    
    # Si no está en diffusion_models, buscar en checkpoints
    if not svd_model_name:
        for ckpt in os.listdir(checkpoints_path) if os.path.exists(checkpoints_path) else []:
            if "svd" in ckpt.lower():
                svd_model_name = ckpt
                print(f"[SVD_TURBO] Modelo encontrado en checkpoints: {svd_model_name}")
                break
    
    if not svd_model_name:
        svd_model_name = "StableDiffusionTurbo/svd_xt.safetensors"
        print(f"[SVD_TURBO] WARN Usando modelo por defecto: {svd_model_name}")
    
    # Buscar VAE
    vae_path = os.path.join(models_path, "vae")
    svd_vae = "svd_xt_image_decoder.safetensors" if os.path.exists(os.path.join(vae_path, "svd_xt_image_decoder.safetensors")) else "pixel_space"
    
    print(f"[SVD_TURBO] Usando modelo: {svd_model_name}, VAE: {svd_vae}")
    
    return {
        # 1: Cargar imagen
        "1": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        
        # 2: Cargar modelo SVD Turbo (UNETLoader para diffusion models)
        "2": {"inputs": {"unet_name": svd_model_name, "weight_dtype": "default"}, "class_type": "UNETLoader"},
        
        # 3: Cargar VAE
        "3": {"inputs": {"vae_name": svd_vae}, "class_type": "VAELoader"},
        
        # 4: Cargar CLIP Vision
        "4": {"inputs": {"clip_name": "open_clip_pytorch_model.bin"}, "class_type": "CLIPVisionLoader"},
        
        # 5: Condicionamiento para SVD
        "5": {"inputs": {
            "clip_vision": ["4", 0],
            "init_image": ["1", 0],
            "vae": ["3", 0],
            "width": width,
            "height": height,
            "video_frames": frames,
            "motion_bucket_id": 127,
            "fps": fps,
            "augmentation_level": 0.0
        }, "class_type": "SVD_img2vid_Conditioning"},
        
        # 6: Sampler - SVD Turbo es un modelo distilled, fewer steps needed
        "6": {"inputs": {
            "model": ["2", 0],
            "positive": ["5", 0],
            "negative": ["5", 1],
            "latent_image": ["5", 2],
            "seed": seed,
            "steps": 10,  # SVD Turbo: 10 steps son suficientes (modelo distilled)
            "cfg": 1.0,
            "sampler_name": "euler_ancestral",  # Faster sampler
            "scheduler": "normal",
            "denoise": 1.0
        }, "class_type": "KSampler"},
        
        # 7: Decodificar video
        "7": {"inputs": {"vae": ["3", 0], "samples": ["6", 0]}, "class_type": "VAEDecode"},
        
        # 8: VHS_VideoCombine - convertir frames a video (MP4)
        "8": {"inputs": {
            "images": ["7", 0],
            "frame_rate": float(fps),
            "loop_count": 0,
            "format": "mp4",
            "output_format": "mp4",
            "pix_fmt": "yuv420p",
            "crf": 20,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True
        }, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine (MP4)"}},
    }


# =============================================================================
# WORKFLOW LTX Video (0.9.1) - RECOMENDADO para 8GB VRAM
# =============================================================================

def get_ltx_video_workflow(
    image_filename, prompt, seed=None,
    width=768, height=512, frames=49, fps=24
):
    """
    WORKFLOW LTX Video - Usando nodos de ComfyUI-LTXVideo
    
    ✅ VENTAJAS:
    - Calidad excelente
    - Soporta prompts de texto
    - Usa LTXVConditioning y nodos nativos
    
    ⚠️  REQUIERE: 
    - Modelo LTX Video en diffusion_models/ltx-video-0.9.1/
    - VAE de LTX Video en vae/ltx-video-0.9.1_vae.safetensors
    - Custom node ComfyUI-LTXVideo instalado
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic"

    negative_prompt = "low quality, blurry, distorted, bad anatomy"
    
    # Ajustar frames para LTX (debe ser múltiplo de 8 + 1)
    frames = ((frames // 8) * 8) + 1
    
    # Ajustar dimensiones para ser múltiplos de 64
    width = (width // 64) * 64
    height = (height // 64) * 64
    
    print(f"[LTX VIDEO] Parametros: {width}x{height}, {frames} frames, {fps} fps")
    
    # Verificar y descargar VAE de LTX si no existe
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    vae_path = os.path.join(script_dir, "ui", "tob", "ComfyUI", "models", "vae")
    ltx_vae_path = os.path.join(vae_path, "ltx-video-0.9.1_vae.safetensors")
    ltx_vae_source = os.path.join(script_dir, "ui", "tob", "ComfyUI", "models", "diffusion_models", "ltx-video-0.9.1", "vae.safetensors")
    
    if not os.path.exists(ltx_vae_path):
        # Intentar copiar desde diffusion_models si existe
        if os.path.exists(ltx_vae_source):
            import shutil
            os.makedirs(vae_path, exist_ok=True)
            shutil.copy(ltx_vae_source, ltx_vae_path)
            print(f"[LTX VIDEO] VAE copiado a: {ltx_vae_path}")
        else:
            print(f"[LTX VIDEO] ADVERTENCIA: VAE de LTX no encontrado. Descarga con: python tools/download_ltx_video.py")
    
    # Obtener UNETs disponibles desde ComfyUI (diffusion_models)
    try:
        import requests
        comfy_url = get_comfyui_url()
        response = requests.get(f"{comfy_url}/object_info/UNETLoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "UNETLoader" in data:
                available_unets = data["UNETLoader"]["input"]["required"].get("unet_name", [[]])[0]
                print(f"[LTX VIDEO] UNETs disponibles: {available_unets}")
            else:
                available_unets = []
        else:
            available_unets = []
    except Exception as e:
        print(f"[LTX VIDEO] Error obteniendo UNETs: {e}")
        available_unets = []
    
    # Buscar modelo LTX en UNETs (diffusion_models)
    ltx_unet = None
    for unet in available_unets:
        unet_lower = unet.lower()
        # Priorizar 0.9.1 sobre 0.9.5 (0.9.1 tiene VAE compatible con ComfyUI después de correcciones)
        # NOTA: 0.9.5 usa transformer.safetensors que no es detectado por ComfyUI
        if "0.9.1" in unet_lower and "ltx" in unet_lower and "vae" not in unet_lower:
            ltx_unet = unet
            print(f"[LTX VIDEO] UNET LTX 0.9.1 encontrado: {ltx_unet}")
            break
        elif "ltx" in unet_lower and "vae" not in unet_lower:
            ltx_unet = unet
            print(f"[LTX VIDEO] UNET LTX encontrado: {ltx_unet}")
            # No break, seguir buscando 0.9.1
    
    # Si no hay UNET en la lista, usar el path por defecto (0.9.1)
    if not ltx_unet:
        ltx_unet = "ltx-video-0.9.1/model.safetensors"
        print(f"[LTX VIDEO] Usando UNET por defecto: {ltx_unet}")
    
    # Obtener VAEs disponibles
    try:
        response = requests.get(f"{comfy_url}/object_info/VAELoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "VAELoader" in data:
                available_vaes = data["VAELoader"]["input"]["required"].get("vae_name", [[]])[0]
                print(f"[LTX VIDEO] VAEs disponibles: {available_vaes}")
            else:
                available_vaes = []
        else:
            available_vaes = []
    except:
        available_vaes = []
    
    # Buscar VAE adecuado para LTX Video
    # LTX Video requiere un VAE específico con downscale_index_formula
    # El VAE pixel_space NO funciona con LTXVImgToVideo
    # NOTA: VAE 0.9.1 es compatible con ComfyUI después de correcciones de arquitectura
    # El VAE de 0.9.5 tiene arquitectura diferente (no soportado aún)
    ltx_vae = None
    
    # Primero buscar VAE de LTX en la lista de VAEs disponibles
    for vae in available_vaes:
        vae_lower = vae.lower()
        # Priorizar 0.9.1 sobre 0.9.5 (0.9.1 funciona con las correcciones aplicadas)
        if "0.9.1" in vae_lower and "ltx" in vae_lower:
            ltx_vae = vae
            print(f"[LTX VIDEO] VAE LTX 0.9.1 encontrado: {ltx_vae}")
            break
        elif "ltx" in vae_lower:
            ltx_vae = vae
            print(f"[LTX VIDEO] VAE LTX encontrado: {ltx_vae}")
            # No break, seguir buscando 0.9.1
    
    # Si no encontramos VAE de LTX, verificar si existe el archivo copiado
    if not ltx_vae:
        # El VAE de LTX Video 0.9.1 es compatible con ComfyUI
        ltx_vae_filename = "ltx-video-0.9.1_vae.safetensors"
        if ltx_vae_filename in available_vaes:
            ltx_vae = ltx_vae_filename
            print(f"[LTX VIDEO] VAE LTX encontrado: {ltx_vae}")
    
    # Si aún no tenemos VAE, usar el primero disponible (no pixel_space)
    if not ltx_vae:
        for vae in available_vaes:
            if vae != "pixel_space":
                ltx_vae = vae
                print(f"[LTX VIDEO] Usando VAE: {ltx_vae}")
                break
    
    # Último recurso: pixel_space (puede causar errores)
    if not ltx_vae:
        ltx_vae = "pixel_space"
        print(f"[LTX VIDEO] ADVERTENCIA: Usando pixel_space - puede no funcionar correctamente")
    
    # Obtener CLIPs disponibles
    try:
        response = requests.get(f"{comfy_url}/object_info/CLIPLoader", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if "CLIPLoader" in data:
                clip_info = data["CLIPLoader"]["input"]["required"]
                available_clips = clip_info.get("clip_name", [[]])[0]
                clip_types = clip_info.get("type", [[]])[0]
                print(f"[LTX VIDEO] CLIPs disponibles: {available_clips}")
            else:
                available_clips = []
                clip_types = []
        else:
            available_clips = []
            clip_types = []
    except:
        available_clips = []
        clip_types = []
    
    # Buscar CLIP para LTX (T5 o Gemma)
    ltx_clip = None
    clip_type = "ltx"
    for clip in available_clips:
        clip_lower = clip.lower()
        if "t5" in clip_lower or "gemma" in clip_lower or "ltx" in clip_lower:
            ltx_clip = clip
            print(f"[LTX VIDEO] CLIP encontrado: {ltx_clip}")
            break
    
    if not ltx_clip and available_clips:
        ltx_clip = available_clips[0]
        print(f"[LTX VIDEO] Usando CLIP disponible: {ltx_clip}")
    
    if not ltx_clip:
        ltx_clip = "t5xxl_fp16.safetensors"
        print(f"[LTX VIDEO] WARN Usando CLIP por defecto: {ltx_clip}")
    
    # Determinar tipo de CLIP
    if "ltx" in clip_types:
        clip_type = "ltx"
    elif "t5" in clip_types:
        clip_type = "t5"
    elif clip_types:
        clip_type = clip_types[0]
    
    # Construir workflow LTX usando UNETLoader + CLIPLoader + VAELoader
    # Esto permite usar modelos de diffusion_models/
    workflow = {
        # 1: Cargar imagen
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage"
        },
        
        # 2: Cargar modelo LTX (UNET desde diffusion_models)
        "2": {
            "inputs": {
                "unet_name": ltx_unet,
                "weight_dtype": "default"
            },
            "class_type": "UNETLoader"
        },
        
        # 3: Cargar CLIP (T5/Gemma para LTX)
        "3": {
            "inputs": {
                "clip_name": ltx_clip,
                "type": clip_type
            },
            "class_type": "CLIPLoader"
        },
        
        # 4: Cargar VAE
        "4": {
            "inputs": {
                "vae_name": ltx_vae
            },
            "class_type": "VAELoader"
        },
        
        # 5: CLIPTextEncode - prompt positivo
        "5": {
            "inputs": {
                "clip": ["3", 0],
                "text": prompt
            },
            "class_type": "CLIPTextEncode"
        },
        
        # 6: CLIPTextEncode - prompt negativo
        "6": {
            "inputs": {
                "clip": ["3", 0],
                "text": negative_prompt
            },
            "class_type": "CLIPTextEncode"
        },
        
        # 7: LTXVConditioning - configurar frame rate
        "7": {
            "inputs": {
                "positive": ["5", 0],
                "negative": ["6", 0],
                "frame_rate": float(fps)
            },
            "class_type": "LTXVConditioning"
        },
        
        # 8: LTXVImgToVideo - combinar imagen + conditioning + latent
        # Este nodo combina EmptyLTXVLatentVideo + LTXVPreprocess + LTXVAddGuide
        "8": {
            "inputs": {
                "positive": ["7", 0],
                "negative": ["7", 1],
                "vae": ["4", 0],
                "image": ["1", 0],
                "width": width,
                "height": height,
                "length": frames,
                "batch_size": 1,
                "strength": 1.0
            },
            "class_type": "LTXVImgToVideo"
        },
        
        # 9: KSampler - sampling
        "9": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2],
                "seed": seed,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler_ancestral",
                "scheduler": "normal",
                "denoise": 1.0
            },
            "class_type": "KSampler"
        },
        
        # 10: VAEDecode - decodificar latentes a frames
        "10": {
            "inputs": {
                "vae": ["4", 0],
                "samples": ["9", 0]
            },
            "class_type": "VAEDecode"
        },
        
        # 11: VHS_VideoCombine - convertir frames a video (MP4)
        # Usamos VHS_VideoCombine en lugar de CreateVideo+SaveVideo para mejor compatibilidad
        "11": {
            "inputs": {
                "images": ["10", 0],
                "frame_rate": float(fps),
                "loop_count": 0,
                "format": "mp4",
                "output_format": "mp4",
                "pix_fmt": "yuv420p",
                "crf": 20,
                "save_metadata": True,
                "pingpong": False,
                "save_output": True
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {"title": "Video Combine (MP4)"}
        },
    }
    
    print(f"[LTX VIDEO] Workflow creado con {len(workflow)} nodos")
    
    # DEBUG: Mostrar todos los nodos del workflow
    print(f"[LTX VIDEO] DEBUG Nodos del workflow:")
    for node_id, node_data in workflow.items():
        class_type = node_data.get("class_type", "UNKNOWN")
        print(f"  - Nodo {node_id}: {class_type}")
    
    return workflow


# =============================================================================
# WORKFLOW Zeroscope V2 XL (576p) - CORREGIDO
# =============================================================================

def get_zeroscope_v2_xl_workflow(
    image_filename, prompt, seed=None,
    width=576, height=320, frames=48, fps=24
):
    """
    WORKFLOW Zeroscope V2 XL - Prototipos rapidos
    
    ⚠️  REQUIERE: Modelo Zeroscope V2 XL en diffusion_models/zeroscope_v2_XL
    O un checkpoint SD 1.5 como fallback
    
    NOTA: Zeroscope es un modelo text-to-video. Para image-to-video,
    usamos el checkpoint como modelo base y el prompt para guiar.
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic"

    negative_prompt = "low quality, blurry, distorted, bad anatomy"
    
    # Rutas del modelo Zeroscope
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_path = os.path.join(script_dir, "ui", "tob", "ComfyUI", "models")
    diffusion_models_path = os.path.join(models_path, "diffusion_models")
    checkpoints_path = os.path.join(models_path, "checkpoints")
    zeroscope_path = os.path.join(diffusion_models_path, "zeroscope_v2_XL")
    
    # Verificar que la estructura del modelo existe
    unet_path = os.path.join(zeroscope_path, "UNET")
    
    # Buscar checkpoint SD 1.5 para fallback
    sd15_checkpoint = None
    for ckpt in os.listdir(checkpoints_path) if os.path.exists(checkpoints_path) else []:
        if "v1-5" in ckpt.lower() or "sd15" in ckpt.lower():
            sd15_checkpoint = ckpt
            break
    
    if not os.path.exists(unet_path):
        print(f"[ZEROSCOPE] No se encontro UNET en {zeroscope_path}")
        print(f"[ZEROSCOPE] Usando checkpoint SD 1.5 como fallback")
        
        if sd15_checkpoint:
            return _get_zeroscope_fallback_workflow(
                image_filename, prompt, negative_prompt, seed, 
                width, height, frames, fps, sd15_checkpoint
            )
        else:
            # Usar cualquier checkpoint disponible
            available_checkpoints = os.listdir(checkpoints_path) if os.path.exists(checkpoints_path) else []
            if available_checkpoints:
                return _get_zeroscope_fallback_workflow(
                    image_filename, prompt, negative_prompt, seed, 
                    width, height, frames, fps, available_checkpoints[0]
                )
    
    # Construir nombres de modelos para los loaders
    unet_name = "zeroscope_v2_XL/UNET/diffusion_pytorch_model.bin"
    clip_name = "zeroscope_v2_XL/TEXT_ENCODER/pytorch_model.bin"
    
    print(f"[ZEROSCOPE] Usando UNET: {unet_name}")
    print(f"[ZEROSCOPE] Usando CLIP: {clip_name}")
    
    # Workflow para Zeroscope usando UNETLoader + CLIPLoader
    return {
        # 1: Cargar imagen
        "1": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        
        # 2: Cargar UNET de Zeroscope
        "2": {"inputs": {"unet_name": unet_name, "weight_dtype": "default"}, "class_type": "UNETLoader"},
        
        # 3: Cargar VAE estandar (usar sd-vae-ft-mse o taesd)
        "3": {"inputs": {"vae_name": "taesd"}, "class_type": "VAELoader"},
        
        # 4: Cargar CLIP de Zeroscope
        "4": {"inputs": {"clip_name": clip_name, "type": "stable_diffusion"}, "class_type": "CLIPLoader"},
        
        # 5: Encode prompt positivo
        "5": {"inputs": {"clip": ["4", 0], "text": prompt}, "class_type": "CLIPTextEncode"},
        
        # 6: Encode prompt negativo
        "6": {"inputs": {"clip": ["4", 0], "text": negative_prompt}, "class_type": "CLIPTextEncode"},
        
        # 7: Encode imagen a latent
        "7": {"inputs": {"pixels": ["1", 0], "vae": ["3", 0]}, "class_type": "VAEEncode"},
        
        # 8: Sampler
        "8": {"inputs": {
            "model": ["2", 0],
            "positive": ["5", 0],
            "negative": ["6", 0],
            "latent_image": ["7", 0],
            "seed": seed,
            "steps": 25,
            "cfg": 12.0,  # Zeroscope usa CFG alto
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.8  # Denoise moderado para preservar algo de la imagen
        }, "class_type": "KSampler"},
        
        # 9: Decodificar video
        "9": {"inputs": {"vae": ["3", 0], "samples": ["8", 0]}, "class_type": "VAEDecode"},
        
        # 10: VHS_VideoCombine - convertir frames a video (MP4)
        "10": {"inputs": {
            "images": ["9", 0],
            "frame_rate": float(fps),
            "loop_count": 0,
            "format": "mp4",
            "output_format": "mp4",
            "pix_fmt": "yuv420p",
            "crf": 20,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True
        }, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine (MP4)"}},
    }


def _get_zeroscope_fallback_workflow(
    image_filename, prompt, negative_prompt, seed,
    width, height, frames, fps, checkpoint_name
):
    """
    Workflow fallback para Zeroscope usando AnimateDiff con checkpoint SD 1.5
    """
    print(f"[ZEROSCOPE_FALLBACK] Usando checkpoint: {checkpoint_name}")
    
    return {
        # 1: Cargar imagen
        "1": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        
        # 2: Cargar checkpoint SD 1.5
        "2": {"inputs": {"ckpt_name": checkpoint_name}, "class_type": "CheckpointLoaderSimple"},
        
        # 3: Encode prompt positivo
        "3": {"inputs": {"clip": ["2", 1], "text": prompt}, "class_type": "CLIPTextEncode"},
        
        # 4: Encode prompt negativo
        "4": {"inputs": {"clip": ["2", 1], "text": negative_prompt}, "class_type": "CLIPTextEncode"},
        
        # 5: Encode imagen a latent
        "5": {"inputs": {"pixels": ["1", 0], "vae": ["2", 2]}, "class_type": "VAEEncode"},
        
        # 6: AnimateDiff loader (si esta disponible)
        "6": {"inputs": {
            "model_name": "mm_sd_v15_v2.ckpt",
            "format": "AnimateDiff"
        }, "class_type": "AnimateDiffLoader"},
        
        # 7: Sampler
        "7": {"inputs": {
            "model": ["6", 0] if True else ["2", 0],  # Usar modelo con AnimateDiff si disponible
            "positive": ["3", 0],
            "negative": ["4", 0],
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": 20,
            "cfg": 8.0,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": 0.8
        }, "class_type": "KSampler"},
        
        # 8: Decodificar
        "8": {"inputs": {"vae": ["2", 2], "samples": ["7", 0]}, "class_type": "VAEDecode"},
        
        # 9: VHS_VideoCombine - convertir frames a video (MP4)
        "9": {"inputs": {
            "images": ["8", 0],
            "frame_rate": float(fps),
            "loop_count": 0,
            "format": "mp4",
            "output_format": "mp4",
            "pix_fmt": "yuv420p",
            "crf": 20,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True
        }, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine (MP4)"}},
    }


def get_lipsync_workflow(image_filename, audio_filename, seed=None):
    """Lip sync con LivePortrait"""
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    return {
        "1": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        "2": {"inputs": {"audio": audio_filename, "upload": "audio"}, "class_type": "VHS_AudioLoader"},
        "3": {"inputs": {
            "model": "liveportrait_onnx.safetensors", "with_expression": True,
            "lip_zero": True, "eye_retargeting": True, "lip_retargeting": True,
            "stitching": True, "relative_motion": True,
            "source_image": ["1", 0], "audio": ["2", 0]
        }, "class_type": "LivePortraitProcess"},
        # VHS_VideoCombine - convertir frames a video con audio (MP4)
        "4": {"inputs": {
            "images": ["3", 0],
            "frame_rate": 25.0,
            "loop_count": 0,
            "format": "mp4",
            "output_format": "mp4",
            "pix_fmt": "yuv420p",
            "crf": 20,
            "save_metadata": True,
            "pingpong": False,
            "save_output": True,
            "audio": ["2", 0]
        }, "class_type": "VHS_VideoCombine", "_meta": {"title": "Video Combine (MP4 + Audio)"}},
    }
