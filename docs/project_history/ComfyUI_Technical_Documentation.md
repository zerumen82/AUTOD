# Documentación Técnica de ComfyUI para Video AI

## Estado Actual del Proyecto

### Descripción General
Este proyecto integra **ComfyUI** como motor de generación de video AI, proporcionando una interfaz de usuario para crear videos a partir de imágenes estáticas. Incluye soporte para múltiples modelos de difusión video y nodos personalizados optimizados para hardware con limitaciones de VRAM.

## Arquitectura de ComfyUI

### Conceptos Clave

#### 1. Nodos (Nodes)
Los nodos son componentes reutilizables que encapsulan funcionalidades específicas. Cada nodo tiene:
- **Input Types**: Definición de parámetros de entrada
- **Output Types**: Resultados generados por el nodo
- **Function**: Código que ejecuta la lógica
- **Category**: Clasificación para organización

#### 2. Workflows
Secuencias de nodos conectados que representan un proceso completo de generación. Los workflows se definen en Python o JSON y se ejecutan en el servidor ComfyUI.

#### 3. Modelos
Los modelos se organizan en categorías predefinidas:
- `checkpoints/`: Modelos principales (formato .safetensors o .ckpt)
- `text_encoders/`: CLIP, T5-XXL, Gemma 3 para procesamiento de texto
- `diffusion_models/`: Modelos de difusión para video (SVD, Zeroscope)
- `vae/`: Variational Autoencoders para decodificación de latentes
- `clip_gguf/`: Modelos CLIP en formato GGUF (cuantizados)
- `unet_gguf/`: Modelos UNET en formato GGUF (cuantizados)

## Estado de los Nodos Personalizados

### Nodos Disponibles

#### 1. ComfyUI-GGUF (Nodes)
- **UnetLoaderGGUF**: Carga modelos UNET en formato GGUF
- **CLIPLoaderGGUF**: Carga modelos CLIP en formato GGUF
- **DualCLIPLoaderGGUF**: Carga dos modelos CLIP GGUF
- **TripleCLIPLoaderGGUF**: Carga tres modelos CLIP GGUF
- **QuadrupleCLIPLoaderGGUF**: Carga cuatro modelos CLIP GGUF
- **UnetLoaderGGUFAdvanced**: Carga UNET GGUF con opciones avanzadas (dequantización, dtype)

#### 2. WanVideoWrapper
- **WanImgToVideo**: Genera video desde imagen usando modelo Wan2.2
- **GGUFModelLoader**: Carga modelos completos en formato GGUF (contiene UNET + CLIP + VAE)

#### 3. LTX-Video (Nodes Internos)
- **LTXVImgToVideoAdvanced**: Genera video con modelo LTX-2
- **LTXVGemmaCLIPModelLoader**: Carga Gemma 3 para LTX-2
- **LTXVSpatioTemporalTiledVAEDecode**: Decodificación con tileado para LOW VRAM

#### 4. Otros Nodos Importantes
- **DiffusersLoader**: Carga modelos en formato Diffusers (carpetas con subdirectorios UNET/TEXT_ENCODER)
- **ImageOnlyCheckpointLoader**: Carga modelos optimizados para generación de imágenes
- **SVD_img2vid_Conditioning**: Condicionamiento para Stable Video Diffusion
- **CLIPVisionLoader**: Carga modelos CLIP Vision para SVD

### Nodos con Problemas Detectados

#### 1. GGUFModelLoader
**Estado**: No detectado en la API de ComfyUI  
**Ubicación**: `/custom_nodes/ComfyUI-WanVideoWrapper/`  
**Error**: El nodo no se lista en `/object_info`  
**Implicación**: El workflow de Wan2.2-Animate-14B no funcionará sin este nodo

#### 2. DiffusersLoader
**Estado**: No confirmado  
**Ubicación**: `/custom_nodes/ComfyUI-DiffusersLoader/`  
**Error**: No verificado en la API  
**Implicación**: Modelos en formato Diffusers (SVD Turbo, Zeroscope) no se pueden cargar

## Estado de los Modelos

### Modelos Descargados y Detectados

#### 1. SVD Turbo
- **Ubicación**: `ui/tob/ComfyUI/models/diffusion_models/stable_video_diffusion/`
- **Formato**: Diffusers (carpeta con UNET, TEXT_ENCODER, config.json)
- **Estado**: Detectado (via `os.listdir`)
- **Problema**: No se puede cargar sin DiffusersLoader
- **Requerimiento de VRAM**: ~4-6GB

#### 2. Wan2.2-Animate-14B
- **Ubicación**: `ui/tob/ComfyUI/models/checkpoints/Wan2.2-Animate-14B-Q2_K.gguf`
- **Formato**: GGUF (cuantizado Q2_K)
- **Estado**: Detectado (en checkpoints)
- **Problema**: No se puede cargar sin GGUFModelLoader
- **Requerimiento de VRAM**: ~8GB

#### 3. Zeroscope V2 XL
- **Ubicación**: `ui/tob/ComfyUI/models/checkpoints/zeroscope_v2_XL/`
- **Formato**: Diffusers (carpeta con UNET, TEXT_ENCODER, VAE)
- **Estado**: Detectado (via `os.listdir`)
- **Problema**: No se puede cargar sin DiffusersLoader
- **Requerimiento de VRAM**: ~4GB

### Modelos Adicionales

#### LTX-2
- **LTX2 FP8**: `ltx-2-19b-distilled-fp8.safetensors` (~10GB)
- **LTX2 FP4**: `ltx-2-19b-dev-fp4.safetensors` (~5GB)
- **Requerimiento**: Gemma 3 en text_encoders/

#### CogVideoX
- **Modelos disponibles**:
  - CogVideoX-1.5-5B (~10GB)
  - CogVideoX-2-5B (~10GB)
  - CogVideoX-9B (~19GB)
- **Formato**: Sharded safetensors (model-00001-of-00005.safetensors)
- **Requerimiento de VRAM**: 8GB-12GB

## Workflows Implementados

### 1. SVD Turbo (get_svd_turbo_workflow)
**Características**:
- Velocidad máxima (~1-2s/video)
- Resolución: 720x480, 24 frames @ 24 fps
- VRAM: ~4-6GB

**Nodos Principales**:
- LoadImage
- UNETLoader (para SVD)
- VAELoader (svd_xt_image_decoder)
- CLIPVisionLoader (open_clip)
- SVD_img2vid_Conditioning
- KSampler
- VAEDecode
- CreateVideo / SaveVideo

### 2. Wan2.2-Animate-14B (get_wan2_2_animate_14b_workflow)
**Características**:
- Mejor detalle en rostros y movimiento
- Resolución: 720x480, 120 frames @ 24 fps
- VRAM: ~8GB

**Nodos Principales**:
- LoadImage
- GGUFModelLoader (problema: no disponible)
- CLIPTextEncode (2x para positive/negative)
- VAEDecode (para encode de imagen)
- WanImgToVideo (para generación)
- CreateVideo / SaveVideo

### 3. Zeroscope V2 XL (get_zeroscope_v2_xl_workflow)
**Características**:
- Rápido y ligero para prototipos
- Resolución: 576x320, 48 frames @ 24 fps
- VRAM: ~4GB

**Nodos Principales**:
- LoadImage
- ImageOnlyCheckpointLoader (o DiffusersLoader si disponible)
- SVD_img2vid_Conditioning
- KSampler
- VAEDecode
- CreateVideo / SaveVideo

### 4. LTX-2 FP8/FP4
**Características**:
- Alta calidad para hardware más potente
- Resolución: 320x192, 25 frames @ 24 fps
- VRAM: ~8GB-10GB

**Nodos Principales**:
- CheckpointLoaderSimple
- LTXVAudioVAELoader
- LTXVGemmaCLIPModelLoader
- LTXVImgToVideoAdvanced
- LTXVSpatioTemporalTiledVAEDecode

## Problemas y Soluciones

### 1. Modelos en Formato Diffusers No Detectados
**Causa**: Falta el nodo DiffusersLoader
**Solución**: Instalar el nodo via ComfyUI Manager o descargar modelos en formato .safetensors

### 2. GGUFModelLoader No Disponible
**Causa**: Nodo no registrado correctamente
**Solución**: Verificar instalación de ComfyUI-WanVideoWrapper y reiniciar ComfyUI

### 3. Modelos GGUF No Detectados en el Dropdown
**Causa**: El nodo ComfyUI-GGUF usa categorías específicas (unet_gguf, clip_gguf)
**Solución**: Verificar que los modelos estén en las carpetas correctas

### 4. Error en la Generación de Video
**Causa**: Problemas de compatibilidad con numpy/scipy
**Solución**: Instalar versiones específicas:
```bash
pip install numpy==1.23.5 scipy==1.9.3
```

## Verificación de Estado

### Script de Verificación de Nodos
```python
import requests
try:
    response = requests.get("http://127.0.0.1:8188/object_info", timeout=5)
    if response.status_code == 200:
        nodes = list(response.json().keys())
        print("Nodos disponibles:", len(nodes))
        print("\nNodos importantes:")
        important_nodes = [
            "CheckpointLoaderSimple", "ImageOnlyCheckpointLoader", 
            "UNETLoader", "CLIPLoader", "VAELoader",
            "SVD_img2vid_Conditioning", "DiffusersLoader",
            "WanImgToVideo", "GGUFModelLoader"
        ]
        for node in important_nodes:
            status = "✅" if node in nodes else "❌"
            print(f"{status} {node}")
    else:
        print(f"Error: {response.status_code}")
except Exception as e:
    print(f"Error: {e}")
```

### Script de Verificación de Modelos
```python
import os

def check_models():
    base_path = "d:/PROJECTS/AUTOAUTO/ui/tob/ComfyUI/models"
    
    directories = {
        "checkpoints": os.path.join(base_path, "checkpoints"),
        "diffusion_models": os.path.join(base_path, "diffusion_models"),
        "text_encoders": os.path.join(base_path, "text_encoders"),
        "vae": os.path.join(base_path, "vae"),
        "clip_gguf": os.path.join(base_path, "clip_gguf"),
        "unet_gguf": os.path.join(base_path, "unet_gguf")
    }
    
    for name, path in directories.items():
        if os.path.exists(path):
            files = os.listdir(path)
            print(f"\n📁 {name} ({len(files)} archivos):")
            for f in files:
                print(f"  - {f}")
        else:
            print(f"\n❌ {name} no existe")

check_models()
```

## Conclusión

### Estado del Proyecto
- **ComfyUI**: Funciona (http://127.0.0.1:8188)
- **Nodos**: Falta DiffusersLoader y GGUFModelLoader
- **Modelos**: Todos los modelos descargados existen, pero no se pueden cargar sin los nodos correspondientes
- **Workflows**: Los workflows están implementados, pero necesitan nodos que no están disponibles

### Próximos Pasos
1. Instalar el nodo DiffusersLoader via ComfyUI Manager
2. Instalar el nodo GGUFModelLoader via ComfyUI Manager
3. Reiniciar ComfyUI
4. Verificar la disponibilidad de nodos
5. Probar los workflows

## Versión
**Documento**: v1.0  
**Fecha**: 03/02/2026  
**Proyecto**: AUTOAUTO Video AI
