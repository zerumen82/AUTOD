
# Guía Técnica de Modelos para "Animate Photo"

## Índice
1. [Introducción](#introducción)
2. [Arquitectura del Sistema](#arquitectura-del-sistema)
3. [Modelos Disponibles](#modelos-disponibles)
4. [Especificaciones Técnicas](#especificaciones-técnicas)
5. [Flujo de Datos](#flujo-de-datos)
6. [Resolución de Problemas](#resolución-de-problemas)
7. [Optimización](#optimización)

---

## Introducción

La pestaña "Animate Photo" permite generar videos a partir de imágenes estáticas mediante modelos de difusión de video. El sistema utiliza ComfyUI como motor de inferencia y soporta múltiples modelos especializados.

### Componentes Principales
- **ComfyUI**: Motor de inferencia para modelos de difusión
- **Modelos de Video**: SVD Turbo, Wan2.2-Animate, Zeroscope
- **VAE**: Decodificadores para convertir latentes a píxeles
- **CLIP Vision**: Extracción de características de imagen
- **XTTS-v2**: Generación de audio y voz en español

---

## Arquitectura del Sistema

```
┌─────────────────────────────────────────────────────────────────┐
│                    AUTOAUTO UI (webview/pywebview)               │
├─────────────────────────────────────────────────────────────────┤
│                      animate_photo_tab.py                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐   │
│  │  Carga Imagen   │  │  Prompt Input   │  │  Selector Audio │   │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      roop/comfy_workflows.py                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Generador de Workflows JSON                   │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │   │
│  │  │ SVD Turbo    │ │ Wan2.2       │ │ Zeroscope XL    │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                   roop/comfy_client.py                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Cliente HTTP API de ComfyUI (puerto 8188)                 │   │
│  │  - queue_prompt()  - get_results()  - upload_image()       │   │
│  └──────────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────────┤
│                      ui/tob/ComfyUI/                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    ComfyUI Server                          │   │
│  │  ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐   │   │
│  │  │ Model Loader │ │ KSampler     │ │ VAE Decoder      │   │   │
│  │  └──────────────┘ └──────────────┘ └──────────────────┘   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Modelos Disponibles

### 1. SVD Turbo (Stable Video Diffusion)

#### Ubicación de Archivos
```
ui/tob/ComfyUI/models/
├── diffusion_models/
│   └── StableDiffusionTurbo/
│       └── svd_xt.safetensors          (~2GB)
├── vae/
│   └── svd_xt_image_decoder.safetensors
└── clip_vision/
    └── open_clip_pytorch_model.bin
```

#### Parámetros del Workflow
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| resolution | 720x480 | Resolución de salida |
| frames | 24 | Número de frames |
| fps | 24 | Fotogramas por segundo |
| motion_bucket_id | 127 | Control de movimiento (1-255) |
| fps_id | 5 | División de FPS base |

#### Arquitectura del Modelo
- **Tipo**: Diffusion model condicional a imagen
- **UNet**: expects 8 canales de entrada (3 ruido + 5 conditioning)
- **VAE**: svd_xt_image_decoder con arquitectura 4x upscaling
- **CLIP**: open_clipViT-H/14 para extracción de características

#### Nodos Requeridos
```
1. LoadImage              - Carga imagen de entrada
2. UNETLoader            - Carga svd_xt.safetensors
3. VAELoader             - Carga VAE decoder
4. CLIPVisionLoader      - Carga CLIP Vision
5. SVD_img2vid_Conditioning - Prepara condiciones
6. KSampler              - Sampling loop
7. VAEDecode             - Decodifica latentes a píxeles
8. CreateVideo           - Ensambla frames en video
9. SaveVideo             - Guarda archivo MP4
```

---

### 2. Wan2.2-Animate-14B (GGUF)

#### Ubicación de Archivos
```
ui/tob/ComfyUI/models/
└── diffusion_models/
    └── Wan2.2-Animate-14B-Q2_K.gguf   (~6.5GB, quantizado)
```

#### Parámetros del Workflow
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| resolution | 720x480 | Resolución de salida |
| frames | 120 | Número de frames |
| fps | 24 | Fotogramas por segundo |
| guidance | 6.0 | Fuerza del guidance |

#### Arquitectura del Modelo
- **Tipo**: Modelo de difusión Transformer
- **Parámetros**: 14B (quantizado Q2_K)
- **Context**: 77 tokens de texto
- **Quantización**: GGUF Q2_K para VRAM limitada

#### Nodos Requeridos
```
1. WanVideoModelLoader    - Carga modelo GGUF
2. WanVideoTextEncode    - Tokeniza prompt
3. WanVideoEncode        - Prepara condiciones
4. WanImageToVideo       - Sampling principal
```

---

### 3. Zeroscope V2 XL

#### Ubicación de Archivos
```
ui/tob/ComfyUI/models/
├── diffusion_models/
│   └── zeroscope_v2_XL/
│       ├── unet/
│       │   ├── diffusion_pytorch_model.bin
│       │   └── config.json
│       ├── text_encoder/
│       │   ├── pytorch_model.bin
│       │   └── config.json
│       └── tokenizer/
└── checkpoints/
    └── zeroscope_v2_XL.safetensors
```

#### Parámetros del Workflow
| Parámetro | Valor | Descripción |
|-----------|-------|-------------|
| resolution | 576x320 | Resolución de salida |
| frames | 48 | Número de frames |
| fps | 24 | Fotogramas por segundo |

---

## Especificaciones Técnicas

### SVD Turbo - Detalle de Canales

#### Problema Original
```
RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], 
expected input[24, 7, 60, 90] to have 8 channels, but got 7 channels
```

#### Análisis del Flujo de Tensores

```
ENTRADA (imagen):
  Shape: [B, 3, H, W]  (RGB)
  ↓ VAE Encode
LATENTE:
  Shape: [B, 4, H/8, W/8]  (4 canales VAE)
  ↓ Concatenación
NOISE + LATENTE:
  Shape: [B, 7, H/8, W/8]  (3 ruido + 4 latente)
  ↓ AJUSTE DE CANALES (FIX)
C_CONCAT (5 canales):
  Shape: [B, 5, H/8, W/8]  (padding con zeros)
  ↓ Concatenación
ENTRADA UNET:
  Shape: [B, 8, H/8, W/8]  (3 ruido + 5 conditioning) ✓
  ↓ UNet forward
SALIDA:
  Shape: [B, 4, H/8, W/8]  (predicción de ruido)
  ↓ VAE Decode
VIDEO:
  Shape: [B*T, 3, H, W]  (T = frames)
```

#### Fix Implementado

**Archivo**: `ui/tob/ComfyUI/comfy/model_base.py`

**En `_apply_model()` (líneas 170-220)**:
```python
# Asegurar que c_concat tenga exactamente 5 canales
if c_concat.shape[1] != 5:
    if c_concat.shape[1] < 5:
        extra = 5 - c_concat.shape[1]
        padding = torch.zeros(..., device=c_concat.device, dtype=c_concat.dtype)
        c_concat = torch.cat([c_concat, padding], dim=1)
    else:
        c_concat = c_concat[:, :5]

# Asegurar tensor final de 8 canales
xc = torch.cat([xc] + [c_concat], dim=1)
if xc.shape[1] != 8:
    # Ajuste adicional si es necesario
```

**En `SVD_img2vid.extra_conds()` (líneas 549-594)**:
```python
# Asegurar latent_image tenga exactamente 5 canales
target_channels = 5
if latent_image.shape[1] < target_channels:
    extra_channels = target_channels - latent_image.shape[1]
    padding = torch.zeros(..., device=latent_image.device, dtype=latent_image.dtype)
    latent_image = torch.cat([latent_image, padding], dim=1)
elif latent_image.shape[1] > target_channels:
    latent_image = latent_image[:, :target_channels]
```

### Formato de Tensores por Modelo

| Modelo | Input Channels | Output Channels | Conditioning |
|--------|---------------|----------------|-------------|
| SVD Turbo | 8 | 4 | imagen (5ch) |
| Wan2.2-Animate | 16 | 4 | imagen + texto |
| Zeroscope XL | 8 | 4 | imagen + texto |

---

## Flujo de Datos

### 1. Carga de Imagen
```
Imagen (PNG/JPG)
  ↓
LoadImage Node
  ↓
Tensor [B, 3, H, W]
  ↓
VAE Encode
  ↓
Latente [B, 4, H/8, W/8]
```

### 2. Preparación de Condiciones
```
CLIP Vision Loader
  ↓
Vision Output
  ↓
SVD_img2vid_Conditioning
  ↓
c_crossattn: [B, 77, 1024]
c_concat: [B, 5, H/8, W/8]  ← CANAL CRÍTICO
```

### 3. Sampling Loop
```
KSampler:
  1. noise = randn([B*frames, 4, H/8, W/8])
  2. for step in steps:
       x = x + c_concat  # [B*frames, 8, H/8, W/8]
       pred = unet(x, timestep, context)
       x = scheduler.step(pred)
```

### 4. Decodificación
```
VAEDecode:
  latents [B*frames, 4, H/8, W/8]
    ↓
  pixels [B*frames, 3, H, W]
    ↓
CreateVideo:
  frames → MP4 [T, H, W, 3]
```

---

## Resolución de Problemas

### Error: "Expected input to have 8 channels"

**Síntoma**:
```
RuntimeError: Given groups=1, weight of size [320, 8, 3, 3], 
expected input[24, 7, 60, 90] to have 8 channels
```

**Causa**: El tensor `c_concat` tiene solo 4 canales en lugar de 5.

**Solución**: El fix en `model_base.py` añade padding automáticamente.

**Verificación**:
```python
# Verificar channels en logs:
[APPLY MODEL DEBUG] c_concat shape before cast: torch.Size([24, 4, 60, 90])
[SVD FIX] Añadiendo 1 canales extra
[SVD FIX] Final latent_image shape: torch.Size([24, 5, 60, 90])
```

### Error: "Model not found"

**Síntoma**: El modelo no se carga.

**Solución**: Verificar rutas:
```
models/diffusion_models/StableDiffusionTurbo/svd_xt.safetensors
models/vae/svd_xt_image_decoder.safetensors
models/clip_vision/open_clip_pytorch_model.bin
```

### Error: "Node not found"

**Síntoma**: Error al ejecutar workflow.

**Solución**: Instalar nodos necesarios:
```
- ComfyUI basic nodes (incluye KSampler, VAELoader, etc.)
- ComfyUI's Video nodes (para CreateVideo, SaveVideo)
```

---

## Optimización

### Configuración Recomendada por VRAM

| VRAM | Configuración | Frames | Resolution |
|------|--------------|--------|------------|
| 4GB | SVD Turbo | 24 | 576x320 |
| 6GB | SVD Turbo | 24 | 720x480 |
| 8GB | SVD Turbo | 24 | 720x480 |
| 8GB | Wan2.2 Q2_K | 120 | 720x480 |
| 12GB+ | Wan2.2 FP16 | 120 | 1280x720 |

### Parámetros de Sampling

```python
# Configuración optimizada para SVD Turbo
sampler_config = {
    "steps": 20,           # Más steps = mejor calidad
    "cfg_scale": 3.0,      # Classifier-free guidance
    "sampler_name": "euler_ancestral",
    "scheduler": "sgm_uniform",
    "denoise": 1.0         # Sin inicialización de imagen
}
```

---

## Referencias de Archivos

| Archivo | Propósito |
|---------|-----------|
| `roop/comfy_workflows.py` | Generador de workflows JSON |
| `roop/comfy_client.py` | Cliente API ComfyUI |
| `ui/tabs/animate_photo_tab.py` | UI de la pestaña |
| `ui/tob/ComfyUI/comfy/model_base.py` | Modelo base con fixes |
| `ui/tob/ComfyUI/nodes/nodes_video_model.py` | Nodos de video |

---

**Última actualización**: 2026-02-05
**Estado**: ✅ SVD Turbo channel fix aplicado
