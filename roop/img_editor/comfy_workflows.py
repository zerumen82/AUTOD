#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ComfyUI Workflows - Plantillas de workflows para Stable Diffusion
"""

import os
import sys
import requests
import time
from typing import Dict, Any, Optional, Tuple

# Puerto dinamico con deteccion automatica
def get_comfyui_port() -> str:
    """Detecta el puerto de ComfyUI automaticamente"""
    possible_ports = ['8188', '8189', '8190', '8888', '8000']
    
    for port in possible_ports:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=1)
            if response.status_code == 200:
                return port
        except:
            continue
    
    # Si no se detecta, usar variable de entorno o defecto
    return os.environ.get('COMFYUI_PORT', '8188')


def get_comfyui_url() -> str:
    """Obtiene la URL de ComfyUI dinámicamente"""
    port = get_comfyui_port()
    return f"http://127.0.0.1:{port}"


# NOTA: No usar COMFY_URL estático, siempre llamar get_comfyui_url()


def get_available_checkpoints() -> list:
    """Obtiene lista de checkpoints disponibles en ComfyUI"""
    try:
        response = requests.get(f"{get_comfyui_url()}/object_info/CheckpointLoaderSimple", timeout=10)
        if response.status_code == 200:
            data = response.json()
            if "CheckpointLoaderSimple" in data:
                node = data["CheckpointLoaderSimple"]
                if "input" in node and "required" in node["input"]:
                    ckpt_list = node["input"]["required"].get("ckpt_name")
                    if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                        return ckpt_list[0]
        return []
    except Exception as e:
        print(f"[Workflows] Error obteniendo checkpoints: {e}")
        return []


def get_default_checkpoint() -> Optional[str]:
    """Obtiene el checkpoint por defecto - PRIORIZA modelos especializados en adultos"""
    checkpoints = get_available_checkpoints()
    if not checkpoints:
        return None
    
    print(f"[Workflows] Checkpoints disponibles: {checkpoints}")
    
    # PRIORIDAD: Modelos especializados en adultos primero
    priority_names = [
        # Modelos especializados en adultos (prioridad máxima)
        "pornmasterpro", "pornmaster_pro", "pornmaster",
        "porncraft", "pornvision",
        "rawcharm", "amateur",
        # Modelos realistas
        "absolutereality", "epicrealism", "realismillustrious",
        "realistic", "realism",
        # SD 1.5 base (fallback)
        "v1-5-pruned", "v1-5", "sd15", "sd1.5",
    ]
    
    for name in priority_names:
        for ckpt in checkpoints:
            if name.lower() in ckpt.lower():
                print(f"[Workflows] Seleccionado checkpoint: {ckpt} (prioridad: {name})")
                return ckpt
    
    print(f"[Workflows] No se encontro checkpoint preferido, usando primero: {checkpoints[0]}")
    return checkpoints[0] if checkpoints else None


def build_img2img_workflow(
    image_filename: str,
    prompt: str,
    negative_prompt: str,
    seed: int = 42,
    steps: int = 30,
    cfg: float = 9.5,
    sampler: str = "euler",
    scheduler: str = "simple",
    denoise: float = 0.95,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Construye workflow img2img optimizado para contenido adulto"""
    
    if checkpoint is None:
        checkpoint = get_default_checkpoint()
    
    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")
    
    # Negative prompt optimizado para contenido adulto
    final_negative_prompt = "low quality, blurry, distorted, bad anatomy, ugly, deformed"
    if negative_prompt:
        final_negative_prompt += f", {negative_prompt}"
    
    # No modificar el prompt del usuario, solo detectar para ajustar parámetros
    final_prompt = prompt
    
    return {
        "3": {
            "inputs": {
                "clip": ["2", 1],
                "text": final_prompt
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Positive)"
            }
        },
        "4": {
            "inputs": {
                "clip": ["2", 1],
                "text": final_negative_prompt
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Negative)"
            }
        },
        "5": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {
                "title": "VAEEncode"
            }
        },
        "6": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler"
            }
        },
        "7": {
            "inputs": {
                "vae": ["2", 2],
                "samples": ["6", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAEDecode"
            }
        },
        "8": {
            "inputs": {
                "filename_prefix": "sd_img2img_output",
                "images": ["7", 0],
                "format": "png"
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "SaveImage"
            }
        },
        "2": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": f"CheckpointLoaderSimple ({checkpoint})"
            }
        },
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {
                "title": "LoadImage"
            }
        }
    }


def build_inpaint_workflow(
    image_filename: str,
    mask_filename: str = None,
    prompt: str = "",
    negative_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 9.5,
    denoise: float = 0.7,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Construye workflow de Inpaint con máscara.

    Si mask_filename es None, usa máscara automática (toda la imagen).
    Si mask_filename es proporcionada, usa esa máscara específica.
    """

    if checkpoint is None:
        checkpoint = get_default_checkpoint()

    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")

    # Negative prompt optimizado para contenido adulto
final_negative_prompt = "low quality, blurry, distorted, bad anatomy, ugly, deformed"
    if negative_prompt:
        final_negative_prompt += f", {negative_prompt}"
    
    # No modificar el prompt del usuario
    final_prompt = prompt if prompt else "high quality, detailed, realistic"

    # Determinar si usamos máscara externa o generamos una
    use_external_mask = mask_filename is not None and mask_filename != ""

    return {
        # Cargar imagen
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "LoadImage"}
        },
        # Cargar máscara (si existe)
        "100": {
            "inputs": {
                "image": mask_filename if use_external_mask else image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {"title": "LoadMask"}
        },
        # Cargar checkpoint
        "2": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {"title": f"Checkpoint ({checkpoint})"}
        },
        # Prompt positivo
        "3": {
            "inputs": {
                "clip": ["2", 1],
                "text": final_prompt
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Positive Prompt"}
        },
        # Prompt negativo
        "4": {
            "inputs": {
                "clip": ["2", 1],
                "text": final_negative_prompt
            },
            "class_type": "CLIPTextEncode",
            "_meta": {"title": "Negative Prompt"}
        },
        # Convertir imagen a máscara (usando canal alpha o creando máscara blanca)
        # ImageToMask usa el canal especificado: red=0, green=1, blue=2, alpha=3
        # Para máscara en escala de grises, usar 'red' funciona porque R=G=B
        "10": {
            "inputs": {
                "image": ["100", 0],
                "channel": "red"  # Usar canal rojo (funciona para máscaras B&W)
            },
            "class_type": "ImageToMask",
            "_meta": {"title": "ImageToMask"}
        },
        # Encode para inpaint con máscara
        "5": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2],
                "mask": ["10", 0],
                "grow_mask_by": 6
            },
            "class_type": "VAEEncodeForInpaint",
            "_meta": {"title": "VAEEncodeForInpaint"}
        },
        # KSampler
        "6": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": denoise
            },
            "class_type": "KSampler",
            "_meta": {"title": "KSampler (Inpaint)"}
        },
        # Decode
        "7": {
            "inputs": {
                "vae": ["2", 2],
                "samples": ["6", 0]
            },
            "class_type": "VAEDecode",
            "_meta": {"title": "VAEDecode"}
        },
        # Guardar
        "8": {
            "inputs": {
                "filename_prefix": "sd_inpaint_output",
                "images": ["7", 0],
                "format": "png"
            },
            "class_type": "SaveImage",
            "_meta": {"title": "SaveImage"}
        }
    }


def get_svd_turbo_workflow(
    image_filename: str,
    prompt: str = "",
    seed: int = 42,
    width: int = 720,
    height: int = 480,
    frames: int = 24,
    fps: int = 24,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Workflow para SVD Turbo (Stable Video Diffusion)"""
    
    if checkpoint is None:
        checkpoint = get_default_checkpoint()
    
    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")
    
    return {
        "3": {
            "inputs": {
                "clip": ["2", 1],
                "text": prompt if prompt else "high quality, detailed video"
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Positive)"
            }
        },
        "4": {
            "inputs": {
                "clip": ["2", 1],
                "text": "low quality, blurry, distorted, bad anatomy, ugly, deformed, low resolution, static, noisy"
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Negative)"
            }
        },
        "5": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {
                "title": "VAEEncode"
            }
        },
        "6": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 2,
                "cfg": 1.8,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler (SVD Turbo)"
            }
        },
        "7": {
            "inputs": {
                "samples": ["6", 0],
                "vae": ["2", 2],
                "upcast": False
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAEDecode"
            }
        },
        "8": {
            "inputs": {
                "images": ["7", 0],
                "filename_prefix": "svd_turbo_output",
                "format": "png",
                "lossless": False,
                "quality": 100,
                "path": ""
            },
            "class_type": "SaveImage",
            "_meta": {
                "title": "SaveImage"
            }
        },
        "9": {
            "inputs": {
                "images": ["7", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "format": "gif",
                "output_format": "gif",
                "width": width,
                "height": height
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine (GIF)"
            }
        },
        "10": {
            "inputs": {
                "images": ["7", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "format": "video/h264-mp4",
                "output_format": "mp4",
                "filename_prefix": "ComfyUI",
                "width": width,
                "height": height
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine (MP4)"
            }
        },
        "2": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": f"CheckpointLoaderSimple ({checkpoint})"
            }
        },
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {
                "title": "LoadImage"
            }
        }
    }


def get_wan2_2_animate_14b_workflow(
    image_filename: str,
    prompt: str = "",
    seed: int = 42,
    width: int = 720,
    height: int = 480,
    frames: int = 120,
    fps: int = 24,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Workflow para Wan2.2 Animate 14B"""
    
    if checkpoint is None:
        checkpoint = get_default_checkpoint()
    
    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")
    
    return {
        "3": {
            "inputs": {
                "text": prompt if prompt else "high quality, detailed video",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Positive)"
            }
        },
        "4": {
            "inputs": {
                "text": "low quality, blurry, distorted, bad anatomy, ugly, deformed, static, low resolution",
                "clip": ["2", 1]
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Negative)"
            }
        },
        "5": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {
                "title": "VAEEncode"
            }
        },
        "6": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 30,
                "cfg": 6.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler (Wan2.2)"
            }
        },
        "7": {
            "inputs": {
                "samples": ["6", 0],
                "vae": ["2", 2],
                "upcast": False
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAEDecode"
            }
        },
        "8": {
            "inputs": {
                "images": ["7", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "format": "video/h264-mp4",
                "output_format": "mp4",
                "width": width,
                "height": height
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine (MP4)"
            }
        },
        "2": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": f"CheckpointLoaderSimple ({checkpoint})"
            }
        },
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {
                "title": "LoadImage"
            }
        }
    }


def get_zeroscope_v2_xl_workflow(
    image_filename: str,
    prompt: str = "",
    seed: int = 42,
    width: int = 576,
    height: int = 320,
    frames: int = 48,
    fps: int = 24,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """Workflow para Zeroscope V2 XL"""
    
    if checkpoint is None:
        checkpoint = get_default_checkpoint()
    
    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")
    
    return {
        "3": {
            "inputs": {
                "clip": ["2", 1],
                "text": prompt if prompt else "high quality video"
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Positive)"
            }
        },
        "4": {
            "inputs": {
                "clip": ["2", 1],
                "text": "low quality, blurry, distorted, bad anatomy, ugly, deformed, static, low resolution"
            },
            "class_type": "CLIPTextEncode",
            "_meta": {
                "title": "CLIPTextEncode (Negative)"
            }
        },
        "5": {
            "inputs": {
                "pixels": ["1", 0],
                "vae": ["2", 2]
            },
            "class_type": "VAEEncode",
            "_meta": {
                "title": "VAEEncode"
            }
        },
        "6": {
            "inputs": {
                "model": ["2", 0],
                "positive": ["3", 0],
                "negative": ["4", 0],
                "latent_image": ["5", 0],
                "seed": seed,
                "steps": 25,
                "cfg": 12,
                "sampler_name": "euler",
                "scheduler": "simple",
                "denoise": 1.0
            },
            "class_type": "KSampler",
            "_meta": {
                "title": "KSampler (Zeroscope)"
            }
        },
        "7": {
            "inputs": {
                "samples": ["6", 0],
                "vae": ["2", 2],
                "upcast": False
            },
            "class_type": "VAEDecode",
            "_meta": {
                "title": "VAEDecode"
            }
        },
        "8": {
            "inputs": {
                "images": ["7", 0],
                "frame_rate": fps,
                "loop_count": 0,
                "format": "video/h264-mp4",
                "output_format": "mp4",
                "width": width,
                "height": height
            },
            "class_type": "VHS_VideoCombine",
            "_meta": {
                "title": "Video Combine (MP4)"
            }
        },
        "2": {
            "inputs": {
                "ckpt_name": checkpoint
            },
            "class_type": "CheckpointLoaderSimple",
            "_meta": {
                "title": f"CheckpointLoaderSimple ({checkpoint})"
            }
        },
        "1": {
            "inputs": {
                "image": image_filename,
                "upload": "image"
            },
            "class_type": "LoadImage",
            "_meta": {
                "title": "LoadImage"
            }
        }
    }


# =============================================================================
# WORKFLOW EDITOR REAL - ControlNet + IP-Adapter (SIN INPAINTING)
# =============================================================================

def check_controlnet_available() -> bool:
    """Verifica si los modelos de ControlNet están disponibles"""
    # Buscar desde el directorio raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    controlnet_dir = os.path.join(project_root, "ui", "tob", "ComfyUI", "models", "controlnet")
    
    required_models = [
        "control_v11f1e_sd15_tile.pth",
        "control_v11p_sd15_softedge.pth"
    ]
    
    for model in required_models:
        model_path = os.path.join(controlnet_dir, model)
        if os.path.exists(model_path):
            print(f"[ControlNet] [OK] Encontrado: {model}")
        else:
            print(f"[ControlNet] [MISSING] No encontrado: {model}")
            return False
    
    return True


def check_ipadapter_available() -> bool:
    """Verifica si los modelos de IP-Adapter están disponibles"""
    # Buscar desde el directorio raíz del proyecto
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ipadapter_dir = os.path.join(project_root, "ui", "tob", "ComfyUI", "models", "ipadapter")
    clip_vision_dir = os.path.join(project_root, "ui", "tob", "ComfyUI", "models", "clip_vision")
    
    # Verificar IP-Adapter (buscar varios nombres posibles)
    ipadapter_names = [
        "ip.adapter.plus.sd15.safetensors",  # Nombre correcto para preset PLUS
        "ip-adapter-plus_sd15.safetensors",   # Nombre alternativo
        "ip-adapter_sd15.safetensors",        # IP-Adapter base
    ]
    ipadapter_found = None
    for name in ipadapter_names:
        path = os.path.join(ipadapter_dir, name)
        if os.path.exists(path):
            ipadapter_found = name
            break
    
    if not ipadapter_found:
        print(f"[IPAdapter] [MISSING] No encontrado: ip-adapter_sd15.safetensors")
        return False
    print(f"[IPAdapter] [OK] Encontrado: {ipadapter_found}")
    
    # Verificar CLIP Vision (buscar varios nombres posibles)
    clip_vision_names = [
        "ViT-H-14.s32B.b79K.safetensors",     # Nombre correcto para IPAdapter
        "CLIP-ViT-H-14.safetensors",          # Nombre alternativo
        "open_clip_pytorch_model.bin",        # OpenCLIP model
    ]
    clip_vision_found = None
    for name in clip_vision_names:
        path = os.path.join(clip_vision_dir, name)
        if os.path.exists(path):
            clip_vision_found = name
            break
    
    if not clip_vision_found:
        print(f"[IPAdapter] [MISSING] No encontrado: CLIP Vision model")
        return False
    print(f"[IPAdapter] [OK] Encontrado: {clip_vision_found}")
    
    return True


def build_editor_workflow(
    image_filename: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 7.0,
    denoise: float = 0.75,
    checkpoint: Optional[str] = None,
    use_controlnet: bool = True,
    use_ipadapter: bool = True,
    controlnet_strength: float = 0.35,
    ipadapter_strength: float = 0.7
) -> Dict[str, Any]:
    """
    Construye workflow de EDICIÓN REAL usando ControlNet + IP-Adapter.
    
    Este workflow permite:
    - Mantener la estructura de la imagen original (ControlNet Tile)
    - Mantener los bordes suaves (ControlNet SoftEdge)
    - Mantener la identidad de la persona (IP-Adapter)
    
    NO usa inpainting - es edición directa con control estructural.
    
    Args:
        image_filename: Nombre del archivo de imagen subido a ComfyUI
        prompt: Prompt positivo para la edición
        negative_prompt: Prompt negativo
        seed: Semilla aleatoria
        steps: Pasos de inference
        cfg: Escala de guidance
        denoise: Fuerza del denoise (0.5-0.8 recomendado para edición)
        checkpoint: Checkpoint a usar
        use_controlnet: Si True, usa ControlNet para mantener estructura
        use_ipadapter: Si True, usa IP-Adapter para mantener identidad
        controlnet_strength: Fuerza del ControlNet (0.5-1.0)
        ipadapter_strength: Fuerza del IP-Adapter (0.5-1.0)
    
    Returns:
        Workflow dict para ComfyUI
    """
    
    if checkpoint is None:
        checkpoint = get_default_checkpoint()
    
    if checkpoint is None:
        raise ValueError("No hay checkpoints disponibles en ComfyUI")
    
    # Verificar modelos disponibles
    has_controlnet = use_controlnet and check_controlnet_available()
    has_ipadapter = use_ipadapter and check_ipadapter_available()
    
    if use_controlnet and not has_controlnet:
        print("[Editor] WARN: ControlNet no disponible, ejecuta tools/download_editor_models.py")
    
    if use_ipadapter and not has_ipadapter:
        print("[Editor] WARN: IP-Adapter no disponible, ejecuta tools/download_editor_models.py")
    
    # Negative prompt optimizado
    final_negative_prompt = "low quality, blurry, distorted, bad anatomy, ugly, deformed, watermark, text, signature"
    if negative_prompt:
        final_negative_prompt += f", {negative_prompt}"
    
    # Prompt base
    final_prompt = prompt if prompt else "high quality, detailed, realistic"
    
    # Construir workflow base
    workflow = {}
    
    # Nodo 1: Cargar imagen
    workflow["1"] = {
        "inputs": {
            "image": image_filename,
            "upload": "image"
        },
        "class_type": "LoadImage",
        "_meta": {"title": "LoadImage"}
    }
    
    # Nodo 2: Cargar checkpoint
    workflow["2"] = {
        "inputs": {
            "ckpt_name": checkpoint
        },
        "class_type": "CheckpointLoaderSimple",
        "_meta": {"title": f"Checkpoint ({checkpoint})"}
    }
    
    # Nodo 3: Prompt positivo
    workflow["3"] = {
        "inputs": {
            "clip": ["2", 1],
            "text": final_prompt
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Positive Prompt"}
    }
    
    # Nodo 4: Prompt negativo
    workflow["4"] = {
        "inputs": {
            "clip": ["2", 1],
            "text": final_negative_prompt
        },
        "class_type": "CLIPTextEncode",
        "_meta": {"title": "Negative Prompt"}
    }
    
    # Nodo 5: VAEEncode
    workflow["5"] = {
        "inputs": {
            "pixels": ["1", 0],
            "vae": ["2", 2]
        },
        "class_type": "VAEEncode",
        "_meta": {"title": "VAEEncode"}
    }
    
    # Model input para KSampler (puede ser modificado por ControlNet/IP-Adapter)
    model_input = ["2", 0]
    positive_input = ["3", 0]
    negative_input = ["4", 0]
    
    next_node = 6
    
    # IP-Adapter (si está disponible)
    if has_ipadapter:
        print(f"[Editor] Agregando IP-Adapter (strength={ipadapter_strength})")
        
        # IPAdapterUnifiedLoader - carga el modelo IP-Adapter y CLIP Vision
        # El preset "PLUS (high strength)" es ideal para edición
        workflow[str(next_node)] = {
            "inputs": {
                "model": model_input,
                "preset": "PLUS (high strength)"
            },
            "class_type": "IPAdapterUnifiedLoader",
            "_meta": {"title": "IPAdapter Unified Loader"}
        }
        ipadapter_loader_node = str(next_node)
        next_node += 1
        
        # IPAdapterAdvanced - aplica el IP-Adapter a la imagen
        workflow[str(next_node)] = {
            "inputs": {
                "model": [ipadapter_loader_node, 0],
                "ipadapter": [ipadapter_loader_node, 1],
                "image": ["1", 0],
                "weight": ipadapter_strength,
                "weight_type": "linear",
                "combine_embeds": "concat",
                "start_at": 0.0,
                "end_at": 1.0,
                "embeds_scaling": "V only"
            },
            "class_type": "IPAdapterAdvanced",
            "_meta": {"title": "IPAdapter Advanced"}
        }
        model_input = [str(next_node), 0]
        next_node += 1
    
    # ControlNet Tile (si está disponible)
    if has_controlnet:
        print(f"[Editor] Agregando ControlNet Tile (strength={controlnet_strength})")
        
        # Cargar ControlNet Tile
        workflow[str(next_node)] = {
            "inputs": {
                "control_net_name": "control_v11f1e_sd15_tile.pth"
            },
            "class_type": "ControlNetLoader",
            "_meta": {"title": "ControlNet Tile Loader"}
        }
        controlnet_tile_node = str(next_node)
        next_node += 1
        
        # Preprocesador Tile (TilePreprocessor)
        workflow[str(next_node)] = {
            "inputs": {
                "image": ["1", 0],
                "pyrUp_iters": 1
            },
            "class_type": "TilePreprocessor",
            "_meta": {"title": "Tile Preprocessor"}
        }
        tile_preproc_node = str(next_node)
        next_node += 1
        
        # Aplicar ControlNet Tile (usando ControlNetApplyAdvanced)
        workflow[str(next_node)] = {
            "inputs": {
                "positive": positive_input,
                "negative": negative_input,
                "control_net": [controlnet_tile_node, 0],
                "image": [tile_preproc_node, 0],
                "strength": controlnet_strength,
                "start_percent": 0.0,
                "end_percent": 1.0
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {"title": "ControlNet Tile Apply"}
        }
        positive_input = [str(next_node), 0]
        negative_input = [str(next_node), 1]
        next_node += 1
        
        # ControlNet SoftEdge
        workflow[str(next_node)] = {
            "inputs": {
                "control_net_name": "control_v11p_sd15_softedge.pth"
            },
            "class_type": "ControlNetLoader",
            "_meta": {"title": "ControlNet SoftEdge Loader"}
        }
        controlnet_softedge_node = str(next_node)
        next_node += 1
        
        # Preprocesador SoftEdge (HEDPreprocessor)
        workflow[str(next_node)] = {
            "inputs": {
                "image": ["1", 0],
                "resolution": 512,
                "safe": "enable"
            },
            "class_type": "HEDPreprocessor",
            "_meta": {"title": "HED SoftEdge Preprocessor"}
        }
        softedge_preproc_node = str(next_node)
        next_node += 1
        
        # Aplicar ControlNet SoftEdge (usando ControlNetApplyAdvanced)
        workflow[str(next_node)] = {
            "inputs": {
                "positive": positive_input,
                "negative": negative_input,
                "control_net": [controlnet_softedge_node, 0],
                "image": [softedge_preproc_node, 0],
                "strength": controlnet_strength * 0.8,  # Un poco menos fuerte
                "start_percent": 0.0,
                "end_percent": 1.0
            },
            "class_type": "ControlNetApplyAdvanced",
            "_meta": {"title": "ControlNet SoftEdge Apply"}
        }
        positive_input = [str(next_node), 0]
        negative_input = [str(next_node), 1]
        next_node += 1
    
    # KSampler
    workflow[str(next_node)] = {
        "inputs": {
            "model": model_input,
            "positive": positive_input,
            "negative": negative_input,
            "latent_image": ["5", 0],
            "seed": seed,
            "steps": steps,
            "cfg": cfg,
            "sampler_name": "euler",
            "scheduler": "normal",
            "denoise": denoise
        },
        "class_type": "KSampler",
        "_meta": {"title": "KSampler (Editor)"}
    }
    sampler_node = str(next_node)
    next_node += 1
    
    # VAEDecode
    workflow[str(next_node)] = {
        "inputs": {
            "vae": ["2", 2],
            "samples": [sampler_node, 0]
        },
        "class_type": "VAEDecode",
        "_meta": {"title": "VAEDecode"}
    }
    decode_node = str(next_node)
    next_node += 1
    
    # SaveImage
    workflow[str(next_node)] = {
        "inputs": {
            "filename_prefix": "editor_output",
            "images": [decode_node, 0],
            "format": "png"
        },
        "class_type": "SaveImage",
        "_meta": {"title": "SaveImage"}
    }
    
    print(f"[Editor] Workflow creado con {len(workflow)} nodos")
    print(f"[Editor] ControlNet: {'Sí' if has_controlnet else 'No'}")
    print(f"[Editor] IP-Adapter: {'Sí' if has_ipadapter else 'No'}")
    
    return workflow


def build_editor_workflow_simple(
    image_filename: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 7.0,
    denoise: float = 0.75,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Workflow simplificado que usa solo ControlNet Tile.
    
    Más ligero que el workflow completo, ideal para ediciones rápidas.
    """
    return build_editor_workflow(
        image_filename=image_filename,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        denoise=denoise,
        checkpoint=checkpoint,
        use_controlnet=True,
        use_ipadapter=False,  # Sin IP-Adapter para ser más ligero
        controlnet_strength=0.85
    )


def build_editor_workflow_identity(
    image_filename: str,
    prompt: str,
    negative_prompt: str = "",
    seed: int = 42,
    steps: int = 30,
    cfg: float = 7.0,
    denoise: float = 0.65,
    checkpoint: Optional[str] = None
) -> Dict[str, Any]:
    """
    Workflow optimizado para preservar identidad.
    
    Usa IP-Adapter con fuerza alta y ControlNet suave.
    Ideal para editar fotos de personas manteniendo su apariencia.
    """
    return build_editor_workflow(
        image_filename=image_filename,
        prompt=prompt,
        negative_prompt=negative_prompt,
        seed=seed,
        steps=steps,
        cfg=cfg,
        denoise=denoise,
        checkpoint=checkpoint,
        use_controlnet=True,
        use_ipadapter=True,
        controlnet_strength=0.4,  # Más suave para permitir cambios del prompt
        ipadapter_strength=0.85   # Más fuerte para identidad
    )
