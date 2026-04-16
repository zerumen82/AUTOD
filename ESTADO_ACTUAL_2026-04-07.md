# Estado Actual del Proyecto - 2026-04-07

## Hardware
- **GPU**: NVIDIA RTX 3060 Ti (8GB VRAM)
- **RAM**: 32GB
- **CUDA**: 12.4

## Resumen General
El proyecto ahora usa **ICEdit** como motor principal de edición de imágenes (autoregresivo, estilo G-rok). Es un modelo de edición instruccional que funciona con ~4-6GB VRAM usando ComfyUI-nunchaku.

---

## Motores del Image Editor - Estado

### ✅ ICEDIT (Nunchaku) - NUEVO MOTOR PRINCIPAL
- **Modelo base**: FLUX.1-fill-dev GGUF Q4 (~6.8GB)
- **LoRA**: ICEdit-normal-LoRA (~200MB)
- **VRAM**: ~4-6GB (funciona con tu RTX 3060 Ti)
- **Tipo**: Autoregresivo/instruccional (estilo G-rok)
- **Estado**: Implementado, esperando descarga de modelos
- **Instrucciones**: "cambia el fondo", "ponle gafas", etc.

### ❌ Qwen Image Edit GGUF - ELIMINADO
- Incompatible con 8GB VRAM
- VAE 3D no convertible a 2D
- Archivo eliminado: `qwen_comfy_client.py`

### ❌ FLUX Fill NF4 - DESCARTADO  
- Eliminado por extremadamente lento

### ✅ SD 1.5 img2img - FALLBACK
- Funciona pero calidad media

---

## Modelos Instalados (ICEdit)

| Modelo | Ubicación | Tamaño | Estado |
|--------|-----------|--------|--------|
| FLUX.1-fill-dev Q4 | `models/unet/` | 6.2GB | ✅ Descargado |
| T5 XXL Encoder Q8 | `models/text_encoders/` | 5.0GB | ✅ Descargado |
| ICEdit LoRA | `models/loras/` | 232MB | ✅ Descargado |
| AE (vae) | `models/vae/` | 335MB | ✅ Descargado |

**Total: ~11.7GB** - Listo para usar con ~4-6GB VRAM

---

## Modelos Instalados

### ComfyUI
| Modelo | Ubicación | Tamaño | Estado |
|--------|-----------|--------|--------|
| Qwen GGUF Q2_K | `ComfyUI/models/unet/` | 7GB | ✅ Descargado |
| Qwen CLIP shards | `ComfyUI/models/text_encoders/text_encoder/` | 16.5GB | ✅ 4 shards |
| Qwen VAE | `ComfyUI/models/vae/vae/` | 253MB | ❌ Incompatible |
| SD1.5 VAE | `ComfyUI/models/vae/` | 334MB | ✅ Funcional |

### Custom Nodes ComfyUI
| Node | Estado |
|------|--------|
| ComfyUI-GGUF (city96) | ✅ Funcional |
| ComfyUI-QwenImageEdit (custom) | ✅ Cargado pero VAE roto |

---

## Lo que FUNCIONA ahora

### FaceSwap Tab
- ✅ Face swap en imágenes y videos
- ✅ Batch processing
- ✅ Temporal smoothing para videos
- ✅ Selected Faces mode

### Image Editor Tab  
- ✅ Motor SD 1.5 img2img (funcional pero calidad media)
- ❌ Motor Qwen Image Edit (crash por incompatibilidad de VRAM/latentes)

### Animate Photo Tab
- ✅ Video generation con ComfyUI

---

## Archivos Modificados Recientemente

| Archivo | Cambio |
|---------|--------|
| `roop/img_editor/qwen_comfy_client.py` | Workflow para Qwen (no funcional) |
| `roop/img_editor/__init__.py` | Imports actualizados |
| `ui/tabs/img_editor_tab.py` | UI con selector de motor |
| `ComfyUI/custom_nodes/ComfyUI-QwenImageEdit/nodes.py` | QwenVAELoader 3D→2D (no funciona) |

---

## Eliminados

- ❌ `flux_client.py` - eliminado
- ❌ `qwen_gguf_client.py` - eliminado
- ❌ `FLUX.1-fill-dev-NF4` - modelo eliminado
- ❌ `FLUX.1-fill-dev-QUAN` - modelo eliminado
- ❌ `Qwen-Image-Edit-2509-Q8_0.gguf` - eliminado (21.7GB)

---

## Próximos Pasos Recomendados

1. **Opción rápida**: Configurar SD 1.5 como motor principal del Image Editor
2. **Opción calidad**: Buscar modelo de edición que funcione con 8GB VRAM
3. **Opción largo plazo**: Esperar actualización de custom nodes Qwen para ComfyUI

---

## Lecciones Aprendidas

- Qwen Image Edit requiere arquitectura de latentes de 16 canales
- Los VAE 3D no son convertibles trivialmente a 2D
- 8GB VRAM es insuficiente para modelos de edición de última generación
- El modelo GGUF Q2_K (7GB) no cabe + buffers de computación = crash OOM