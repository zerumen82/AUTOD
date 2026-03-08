"""
ComfyUI Workflows para Video AI - VERSION CORREGIDA

OPCIONES DISPONIBLES:

1. Vidu (ByteDance - API Cloud)
   - Pros: Optimizado para NVIDIA GPUs, excelentes resultados con caras y motion
   - Contras: API paid, pero opciones low-cost
   - Recomendado: Vidu2 con modelos 'viduq2-pro-fast' o 'viduq2-turbo' (ideal para tu GPU)

2. WAN (Alibaba Cloud - API Cloud)
   - Pros: Alta velocidad, soporte para audio + video
   - Contras: API paid, variada calidad
   - Recomendado: WAN 2.6 para video generación

3. Kling (Baidu Paddle - API Cloud)
   - Pros: Excelente motion control, soporte para caras
   - Contras: API paid, requiere ajustes
   - Recomendado: Kling 2.6 con modelos 'kling-v2-6'

4. LTX-Video 2 (requiere Gemma 3)
   - Pros: Compatible con hardware limitado, menor tamaño (~10GB)
   - Contras: Requiere Gemma 3 (~12GB, formato safetensors)

5. CogVideoX (funciona con T5-XXL que ya tienes)
   - Pros: Funciona con tu T5-XXL actual
   - Contras: Solo video (sin audio automatico)
"""

# Modelos
LTX2_FP8_MODEL = "ltx-2-19b-distilled-fp8.safetensors"
LTX2_FP8_CKPT_PATH = "ltx-2-19b-distilled-fp8.safetensors"
LTX2_FP4_MODEL = "ltx-2-19b-dev-fp4.safetensors"
GEMMA_MODEL = "gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors"
# CogVideoX - Versiones disponibles:
# - CogVideoX/ (carpeta con model.safetensors.index.json) - Version 1.0/1.5/2.0 divida
# - cogvideo-9b.safetensors (~19GB) - Grande, requiere mucha VRAM
# - cogvideo-5b.safetensors (~10GB) - CogVideoX-2, nuevo pero menos probado
# - CogVideoX1.5-5B.safetensors (~10GB) - CogVideoX-1.5, mas estable y probado
COGVIDEO_MODEL_9B = "CogVideoX"  # Carpeta con modelo dividido
COGVIDEO_MODEL_5B = "cogvideo-5b.safetensors"
COGVIDEO_MODEL_1_5_5B = "CogVideoX"  # Carpeta con modelo dividido (model.safetensors.index.json)


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
        # Usar /object_info para obtener información de los nodos
        response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
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
# WORKFLOW LTX-2 (USA GEMMA 3)
# =============================================================================

def get_ltxvideo2_workflow(
    image_filename, prompt, seed=None,
    width=704, height=480, frames=121, fps=24, strength=0.6,
    model_version="ltx2_fp8"
):
    """
    ⚠️  REQUIERE: Gemma 3 en text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/
    
    Nodos LTX-Video disponibles (del codebase de ComfyUI):
    - CheckpointLoaderSimple: carga el modelo principal (MODEL, CLIP, VAE)
    - LTXVGemmaCLIPModelLoader: carga Gemma 3 para texto
    - LTXVImgToVideoAdvanced: genera video desde imagen
    - LTXVTiledVAEDecode: decodifica el video (con tileado temporal)
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic"

    negative_prompt = "low quality, blurry, distorted, bad anatomy"

    # Obtener modelos disponibles
    available_checkpoints = get_available_models("checkpoints")
    available_clip = get_available_models("clip")

    # Seleccionar modelo válido - usar FP8 por defecto ya que FP4 no está disponible
    if model_version == "ltx2_fp4" and LTX2_FP4_MODEL in available_checkpoints:
        model_name = LTX2_FP4_MODEL
    else:
        model_name = LTX2_FP8_MODEL

    # Buscar el nombre correcto del CLIP Gemma 3 (usar barras invertidas de Windows)
    gemma_clip_name = None
    for clip in available_clip:
        if "gemma-3-12b-it-qat-q4_0-unquantized" in clip.lower():
            gemma_clip_name = clip
            break

    # Si no se encuentra Gemma 3, usar el primer archivo disponible
    if not gemma_clip_name and available_clip:
        gemma_clip_name = available_clip[0]

    return {
        # 1: Cargar modelo LTX-2 (CheckpointLoaderSimple existe y funciona con LTX-2)
        "1": {"inputs": {"ckpt_name": model_name}, "class_type": "CheckpointLoaderSimple"},
        
        # 2: Cargar imagen
        "2": {"inputs": {"image": image_filename, "upload": "image"}, "class_type": "LoadImage"},
        
        # 3: Cargar Gemma 3 (usa LTXVGemmaCLIPModelLoader)
        "3": {"inputs": {
            "clip_name": gemma_clip_name if gemma_clip_name else "gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors",
            "model_name": model_name,
            "max_length": 1024
        }, "class_type": "LTXVGemmaCLIPModelLoader"},
        
        # 4: Positive prompt
        "4": {"inputs": {"clip": ["3", 0], "text": prompt}, "class_type": "CLIPTextEncode"},
        
        # 5: Negative prompt
        "5": {"inputs": {"clip": ["3", 0], "text": negative_prompt}, "class_type": "CLIPTextEncode"},
        
        # 6: Conditioning (combina prompts con frame_rate)
        "6": {"inputs": {
            "positive": ["4", 0], 
            "negative": ["5", 0], 
            "frame_rate": fps
        }, "class_type": "LTXVConditioning"},
        
        # 7: Crear latents vacíos para video (EmptyLTXVLatentVideo existe)
        "7": {"inputs": {
            "width": width, 
            "height": height, 
            "length": frames, 
            "batch_size": 1
        }, "class_type": "EmptyLTXVLatentVideo"},
        
        # 8: Generar video desde imagen (LTXVImgToVideoAdvanced)
        "8": {"inputs": {
            "positive": ["6", 0],
            "negative": ["6", 1],
            "vae": ["1", 2],  # VAE del checkpoint (output 2 de CheckpointLoaderSimple)
            "image": ["2", 0],
            "width": width,
            "height": height,
            "length": frames,
            "batch_size": 1,
            "strength": strength,
            "crf": 29,
            "blur_radius": 0,
            "interpolation": "lanczos",
            "crop": "disabled"
        }, "class_type": "LTXVImgToVideoAdvanced"},
        
        # 9: Decodificar video (LTXVTiledVAEDecode maneja video VAEs con tileado temporal)
        "9": {"inputs": {
            "vae": ["1", 2], 
            "samples": ["8", 2],  # latent output de LTXVImgToVideoAdvanced
            "tile_size": 256,
            "overlap": 64,
            "temporal_size": 16,
            "temporal_overlap": 4
        }, "class_type": "LTXVTiledVAEDecode"},
        
        # 10: Guardar imágenes individuales
        "10": {"inputs": {
            "filename_prefix": "LTX-Video_Output",
            "images": ["9", 0]
        }, "class_type": "SaveImage", "_meta": {"title": "Guardar Imágenes"}},
    }


# =============================================================================
# WORKFLOW VIDU (BYTE DANCE - API CLOUD)
# =============================================================================

def get_vidu_workflow(
    image_filename, prompt, seed=None,
    model_version="viduq2-pro-fast", duration=5, resolution="720p",
    movement_amplitude="medium"
):
    """
    🔥 Mejor opción para tu GPU (RTX 3060 Ti 8GB VRAM)
    
    Ventajas:
    - API cloud (no usa VRAM local)
    - Optimizado para animación de caras humanas
    - Soporte para audio automatico
    - Alta calidad de motion y expresiones
    
    Modelos recomendados para tu GPU:
    - viduq2-pro-fast: Bajo costo, alta velocidad
    - viduq2-turbo: Más rápido, calidad aceptable
    - viduq2-pro: Mayor calidad, costo moderado
    
    Resolution: 720p (mejor balance para 8GB VRAM)
    Duration: 5s (optiomal para coste y calidad)
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic, high quality"

    workflow = {
        # 1: Cargar imagen
        "1": {
            "inputs": {"image": image_filename, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Cargar Imagen"}
        },
        # 2: Generar video desde imagen (Vidu2 Image to Video)
        "2": {
            "inputs": {
                "model": model_version,
                "image": ["1", 0],
                "prompt": prompt,
                "duration": duration,
                "seed": seed,
                "resolution": resolution,
                "movement_amplitude": movement_amplitude
            },
            "class_type": "Vidu2ImageToVideoNode",
            "_meta": {"title": "Vidu2 Image to Video"}
        },
        # 3: Guardar video
        "3": {
            "inputs": {
                "filename_prefix": f"Vidu_{model_version}_Output",
                "video": ["2", 0]
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Guardar Video"}
        }
    }
    
    return workflow


# =============================================================================
# WORKFLOW WAN (ALIBABA CLOUD - API CLOUD)
# =============================================================================

def get_wan_workflow(
    image_filename, prompt, seed=None,
    model_version="wan2.6-i2v", duration=5, resolution="720P"
):
    """
    🚀 Alta velocidad para video generación
    
    Ventajas:
    - API cloud (no usa VRAM local)
    - Soporte para audio + video
    - Buen rendimiento con contenido variado
    - Precio competitivo
    
    Modelos recomendados:
    - wan2.6-i2v: Versión estable, buena calidad
    - wan2.5-i2v-preview: Opcion low-cost
    
    Resolution: 720P (optiomal para rendimiento)
    Duration: 5s (balance coste/calidad)
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "high quality video, smooth motion, detailed"

    workflow = {
        # 1: Cargar imagen
        "1": {
            "inputs": {"image": image_filename, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Cargar Imagen"}
        },
        # 2: Generar video desde imagen (Wan Image to Video)
        "2": {
            "inputs": {
                "model": model_version,
                "image": ["1", 0],
                "prompt": prompt,
                "negative_prompt": "low quality, blurry, distorted, bad anatomy",
                "resolution": resolution,
                "duration": duration,
                "seed": seed,
                "generate_audio": True,
                "prompt_extend": True,
                "watermark": False,
                "shot_type": "single"
            },
            "class_type": "WanImageToVideoApi",
            "_meta": {"title": "Wan Image to Video"}
        },
        # 3: Guardar video
        "3": {
            "inputs": {
                "filename_prefix": f"Wan_{model_version}_Output",
                "video": ["2", 0]
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Guardar Video"}
        }
    }
    
    return workflow


# =============================================================================
# WORKFLOW KLING (BAIDU PADDLE - API CLOUD)
# =============================================================================

def get_kling_workflow(
    image_filename, prompt, seed=None,
    model_version="kling-v2-6", duration=5, resolution="720p"
):
    """
    🎬 Excelente control de motion
    
    Ventajas:
    - API cloud (no usa VRAM local)
    - Control de camara profesional
    - Soporte para motion control
    - Alta calidad para contenido técnico
    
    Modelos recomendados:
    - kling-v2-6: Última versión, mejor motion control
    - kling-v2-5-turbo: Rápido y económico
    
    Resolution: 720p (balance calidad/rendimiento)
    Duration: 5s (optiomal para coste)
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "smooth natural motion, cinematic, high quality"

    workflow = {
        # 1: Cargar imagen
        "1": {
            "inputs": {"image": image_filename, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Cargar Imagen"}
        },
        # 2: Generar video desde imagen (Kling Image to Video with Audio)
        "2": {
            "inputs": {
                "model_name": model_version,
                "start_frame": ["1", 0],
                "prompt": prompt,
                "mode": "pro",
                "duration": duration,
                "generate_audio": True
            },
            "class_type": "ImageToVideoWithAudio",
            "_meta": {"title": "Kling Image to Video"}
        },
        # 3: Guardar video
        "3": {
            "inputs": {
                "filename_prefix": f"Kling_{model_version}_Output",
                "video": ["2", 0]
            },
            "class_type": "SaveVideo",
            "_meta": {"title": "Guardar Video"}
        }
    }
    
    return workflow


# =============================================================================
# FUNCION PRINCIPAL PARA SELECCIONAR EL MEJOR WORKFLOW
# =============================================================================

def get_best_workflow(
    image_filename, prompt, seed=None,
    target_resolution="720p", target_duration=5,
    use_api=True  # True para usar APIs cloud (mejor para tu GPU)
):
    """
    Selecciona el mejor workflow según tu hardware:
    
    - RTX 3060 Ti (8GB VRAM): API cloud (Vidu > WAN > Kling)
    - RTX 4090+ (24GB VRAM): LTX-2 o CogVideoX local
    """
    if use_api:
        return get_vidu_workflow(
            image_filename, prompt, seed,
            model_version="viduq2-pro-fast",
            duration=target_duration,
            resolution=target_resolution,
            movement_amplitude="medium"
        )
    
    # Si no quieres usar API, usar LTX-2 (si tienes Gemma 3)
    try:
        available_clip = get_available_models("clip")
        has_gemma3 = any("gemma-3" in clip.lower() for clip in available_clip)
        
        if has_gemma3:
            return get_ltxvideo2_workflow(
                image_filename, prompt, seed,
                width=704, height=480, frames=121, fps=24, strength=0.6
            )
        else:
            return get_cogvideox_workflow(
                image_filename, prompt, seed,
                width=720, height=480, frames=49, fps=8
            )
    except Exception as e:
        print(f"Error al seleccionar workflow local: {e}")
        return get_vidu_workflow(image_filename, prompt, seed)


# =============================================================================
# WORKFLOW COGVIDEO-X (USA T5-XXL - YA LO TIENES)
# =============================================================================

def get_cogvideox_workflow(
    image_filename, prompt, seed=None,
    width=720, height=480, frames=49, fps=8,
    model_version="cogvideo_5b"
):
    """
    ⚠️  CogVideoX requiere nodos especializados que pueden no estar disponibles
    en tu instalación actual de ComfyUI.
    
    Modelos disponibles:
    - cogvideo_5b: ~10GB, CogVideoX-2, nuevo
    - cogvideo_1_5_5b: ~10GB, CogVideoX-1.5, mas estable
    - cogvideo_9b: ~19GB, mayor calidad pero necesita más VRAM
    
    Parametros:
    - frames: 49 (6 seg), 101 (12 seg)
    """
    if seed is None:
        import random
        seed = random.randint(0, 1000000000)

    if not prompt or not prompt.strip():
        prompt = "high quality video, smooth motion"
    
    # Obtener modelos disponibles
    available_checkpoints = get_available_models("checkpoints")
    available_clip = get_available_models("clip")
    
    # Seleccionar modelo CogVideoX (usar el primer archivo sharded)
    if model_version == "cogvideo_9b":
        model_name = COGVIDEO_MODEL_9B
    elif model_version == "cogvideo_1_5_5b":
        model_name = COGVIDEO_MODEL_1_5_5B
    else:
        model_name = COGVIDEO_MODEL_5B
    
    # Buscar el nombre correcto del checkpoint CogVideoX
    cogvideo_checkpoint = None
    for ckpt in available_checkpoints:
        if "cogvideo" in ckpt.lower() and "model-00001" in ckpt.lower():
            cogvideo_checkpoint = ckpt
            break
    
    # Si no se encuentra, usar el primer checkpoint disponible
    if not cogvideo_checkpoint and available_checkpoints:
        cogvideo_checkpoint = available_checkpoints[0]
    
    # Buscar T5-XXL (si no está disponible, usar el primer CLIP disponible)
    t5_clip = None
    for clip in available_clip:
        if "t5" in clip.lower():
            t5_clip = clip
            break
    
    # Si no se encuentra T5, usar el primer CLIP disponible
    if not t5_clip and available_clip:
        t5_clip = available_clip[0]

    workflow = {
        # 1: Cargar modelo CogVideoX (usar nodo especializado)
        "1": {
            "inputs": {"model": cogvideo_checkpoint if cogvideo_checkpoint else "CogVideoX\\model-00001-of-00004.safetensors"},
            "class_type": "CogVideoXModelLoader",
            "_meta": {"title": f"CogVideoX ({model_name})"}
        },
        # 2: Cargar imagen
        "2": {
            "inputs": {"image": image_filename, "upload": "image"},
            "class_type": "LoadImage",
            "_meta": {"title": "Cargar Imagen"}
        },
        # 3: Cargar T5-XXL (tu modelo actual)
        "3": {
            "inputs": {"clip_name": t5_clip if t5_clip else "gemma-3-12b-it-qat-q4_0-unquantized\\model-00001-of-00005.safetensors"},
            "class_type": "CLIPLoader",
            "_meta": {"title": "T5-XXL (ya disponible)"}
        },
        # 4: Positive prompt (usa CogVideoTextEncode)
        "4": {
            "inputs": {"clip": ["3", 0], "prompt": prompt},
            "class_type": "CogVideoTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        # 5: Negative prompt (usa CogVideoTextEncode)
        "5": {
            "inputs": {"clip": ["3", 0], "prompt": "low quality, blurry, distorted"},
            "class_type": "CogVideoTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        # 6: CogVideo Image Encode (usa el nodo correcto)
        "6": {
            "inputs": {"vae": ["1", 1], "start_image": ["2", 0]},
            "class_type": "CogVideoImageEncode",
            "_meta": {"title": "Encode Image"}
        },
        # 7: Sampler (usa CogVideoSampler)
        "7": {
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "num_frames": frames,
                "steps": 50,
                "cfg": 6.0,
                "seed": seed,
                "scheduler": "CogVideoXDDIM",
                "image_cond_latents": ["6", 0]
            },
            "class_type": "CogVideoSampler",
            "_meta": {"title": "CogVideo Sampler"}
        },
        # 8: Decode (usa CogVideoDecode con todos los inputs requeridos)
        "8": {
            "inputs": {
                "vae": ["1", 1],
                "samples": ["7", 0],
                "enable_vae_tiling": True,
                "auto_tile_size": 512,
                "tile_sample_min_width": 512,
                "tile_sample_min_height": 512,
                "tile_overlap_factor_width": 0.25,
                "tile_overlap_factor_height": 0.25
            },
            "class_type": "CogVideoDecode",
            "_meta": {"title": "Decode Video"}
        },
        # 9: Save images
        "9": {
            "inputs": {
                "filename_prefix": "CogVideoX_Video",
                "images": ["8", 0]
            },
            "class_type": "SaveImage",
            "_meta": {"title": "Guardar Imágenes"}
        }
    }
    
    return workflow


# =============================================================================
# LIPSYNC (LIVE PORTRAIT)
# =============================================================================

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
        "4": {"inputs": {
            "images": ["3", 0],
            "fps": 25.0,
            "audio": ["2", 0]
        }, "class_type": "CreateVideo"},
        "5": {"inputs": {
            "video": ["4", 0],
            "filename_prefix": "LipSync_Output",
            "format": "mp4",
            "codec": "auto"
        }, "class_type": "SaveVideo"},
    }