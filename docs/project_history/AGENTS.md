# AutoAuto - Documentación del Proyecto

## Estructura del Proyecto

```
D:\PROJECTS\AUTOAUTO\
├── run.py                    # Punto de entrada principal
├── config.yaml              # Configuración
├── venv/                   # Entorno principal (Python 3.10, PyTorch 2.4.1+cu121)
├── venv_ext/               # Entorno extendido (Python 3.11, PyTorch 2.5.1+cu121, flash-attn)
├── roop/                   # Módulo principal de procesamiento
│   ├── img_editor/        # Clientes de edición de imagen (HART, Qwen, Z-Image, etc.)
│   ├── processors/      # Procesadores de cara (GFPGAN, CodeFormer, etc.)
│   └── ProcessMgr.py  # Gestor de procesamiento principal
├── ui/                    # Interfaz Gradio + ComfyUI
│   └── tob/ComfyUI/    # Servidor ComfyUI (puerto 8188)
├── models/               # Modelos IA
├── checkpoints/         # Checkpoints de modelos
├── output/             # Salida de procesamientos
├── scripts/            # Scripts auxiliares
└── tools/             # Herramientas
```

## Entornos Virtuales

| Entorno | Python | PyTorch | CUDA | Uso Principal |
|---------|--------|---------|------|-------------|
| venv | 3.10.6 | 2.4.1+cu121 | 12.1 | App principal, ComfyUI |
| venv_ext | 3.11.9 | 2.5.1+cu121 | 12.1 | HART con flash-attn |

### Paquetes instalados en venv

- torch 2.4.1+cu121
- torchvision 0.19.1+cu121
- torchaudio 2.4.1+cu121
- torchsde 0.2.6
- transformers 5.5.4
- comfyui-frontend-package 1.42.10

### Paquetes instalados en venv_ext

- torch 2.5.1+cu121
- flash-attn 2.7.0.post2
- transformers 4.37.2 (versión específica para HART)
- scipy 1.17.1

## Motores de Generación de Imagen

### HART (Hybrid Autoregressive Transformer)

- **Ubicación**: `ui\tob\ComfyUI\custom_nodes\hart\`
- **Cliente**: `roop\img_editor\hart_edit_comfy_client.py`
- **Entorno**: `venv_ext` (con flash-attn)
- **Gestión**: `img_editor_manager.py` - llama al cliente con `venv_path=r"D:\PROJECTS\AUTOAUTO\venv_ext"`

Para usar HART con aceleración flash-attn:
1. Seleccionar "hart" en ImageEditor
2. El código usa automáticamente `venv_ext`

### Qwen Image Edit

- **Cliente**: `roop\img_editor\qwen_edit_comfy_client.py`
- **Servidor**: ComfyUI (puerto 8188)
- **Versiones**: q2, q3, q4 (quantization)

### Z-Image Turbo

- **Cliente**: `roop\img_editor\zimage_edit_comfy_client.py`
- **Servidor**: ComfyUI (puerto 8188)
- **Versiones**: q2, q3, q4, q5

### Flux (via ComfyUI)

- Nodos en ComfyUI custom_nodes

## FaceSwap - Modos de Procesamiento

### Modos disponibles (en `roop/globals.py`)

| Modo | Descripción |
|------|-------------|
| all | Todas las caras detectadas |
| selected | Una cara seleccionada manualmente |
| selected_faces | Múltiples caras seleccionadas |
| selected_faces_frame | Tracking de video - misma cara固定 |

### Tracking en selected_faces_frame

- **Archivo**: `roop/ProcessMgr.py`
- **Umbral de matching**: 0.15 (subido de 0.03)
- **Línea**: ~810 (`if best_match and best_score > 0.15:`)
- **Propósito**: Evitar fallback a caras incorrectas cuando la detección fluctúa

## Archivos Temporales

- **Ubicación**: `output_path + ".temp.mp4"`
- **Limpieza**: Automática al finalizar procesamiento
- **Código**: `roop/ProcessMgr.py` líneas ~1645-1648

## Arranque de la Aplicación

```bash
python run.py
```

1. Carga DLLs de CUDA
2. Inicia PyTorch
3. Lanza ComfyUI (puerto 8188)
4. Abre interfaz Gradio (puerto 7860)

## Problemas Conocidos y Soluciones

### ComfyUI no inicia - torchvision missing

```bash
venv\Scripts\python.exe -m pip install torchvision==0.19.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

### ComfyUI no inicia - torchsde missing

```bash
venv\Scripts\python.exe -m pip install torchsde
```

### ComfyUI no inicia - simpleeval/OpenGL missing

```bash
venv\Scripts\python.exe -m pip install -r ui\tob\ComfyUI\requirements.txt
```

### Instalar flash-attn en venv_ext

```bash
venv_ext\Scripts\python.exe -m pip install flash_attn-2.7.0.post2+cu124torch2.4.1cxx11abiFALSE-cp311-cp311-win_amd64.whl
```

## Comandos Útiles

### Ver versión de PyTorch

```bash
venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.version.cuda)"
```

### Listar paquetes instalados

```bash
venv\Scripts\python.exe -m pip list
```

### Reinstalar requisitos de ComfyUI

```bash
venv\Scripts\python.exe -m pip install -r ui\tob\ComfyUI\requirements.txt
```

## Cambios Recientes (2025-04-15)

### Tracking de Caras en Video
- **Archivo**: `roop/ProcessMgr.py`
- **Umbral de matching**: Subido de 0.03 a 0.15 (líneas ~627, ~810)
- **Efecto**: Reduce el fallback a caras incorrectas durante el tracking

### Limpieza de Archivos Temporales
- **Archivo**: `roop/ProcessMgr.py`
- **Ubicación**: Línea ~1645-1648
- **Función**: Elimina `.temp.mp4` después de procesar video

### HART con venv_ext

- **Archivo**: `roop/img_editor/img_editor_manager.py`
- **Línea**: ~970 (`venv_path=r"D:\PROJECTS\AUTOAUTO\venv_ext"`)
- **Efecto**: HART usa flash-attn para generación más rápida

### HART y ComfyUI

HART necesita toda la VRAM disponible (~8GB). El código en `img_editor_manager.py`:
1. **Cierra ComfyUI** antes de cargar HART (libera ~2GB VRAM)
2. **Genera la imagen** con HART
3. **Reinicia ComfyUI** automáticamente después

```python
# img_editor_manager.py líneas ~960-1000
# Cierra ComfyUI antes de HART
from ui.tabs.comfy_launcher import stop
stop()

# Genera HART...

# Reinicia ComfyUI después
from ui.tabs.comfy_launcher import start as start_comfy
threading.Thread(target=start_comfy, daemon=True).start()
```

## Modelos Autoregresivos - VRAM 8GB

### Modelos VAR (Visual AutoRegressive)

| Modelo | Parámetros | Resolución | FID | VRAM (FP16) | Estado |
|--------|-----------|------------|-----|-------------|--------|
| VAR-d16 | 310M | 256px | 3.55 | ~2GB | ✅ viable |
| VAR-d20 | 600M | 256px | 2.95 | ~3GB | ✅ viable |
| VAR-d24 | 1.0B | 256px | 2.33 | ~5GB | ✅ borderline |
| VAR-d30 | 2.0B | 256px | 1.97 | ~8GB | ❌ ajusta |
| VAR-d36 | 2.3B | 512px | 2.63 | ~12GB | ❌ |

**Repositorio**: https://github.com/FoundationVision/VAR
**Descarga**: `huggingface.co/FoundationVision/var`

### BitDance (Binary Token AR)

- **Descripción**: AR con tokens binarios, predice 64 tokens en paralelo
- **VRAM**: ~2GB (versiones 260M-1B)
- **Ventaja**: 30x más rápido que AR tradicionales
- **Repositorio**: https://github.com/shallowdream204/BitDance

### OmniGen2 GGUF

- **Repositorio**: `calcuis/omnigen2-gguf` (HuggingFace)
- **VRAM**: ~6GB en Q4_K_M
- **Uso**: Requiere ComfyUI-GGUF + gguf-node

### Modelos NO disponibles para 8GB

| Modelo | VRAM Original | Notas |
|--------|---------------|-------|
| VAREdit 8B | ~50GB | Solo versiones 2B (8GB) |
| GLM-Image 9B | >16GB | Solo FP16, sin GGUF |
| VAREdit GGUF | ❌ | No existe versión cuantizada |

## Notas

- HART funciona independientemente de ComfyUI via subprocess
- El custom node hart en ComfyUI no tiene nodos compatibles - error ignorable
- venv_ext tiene flash-attn instalado para aceleración HART
- Para VAR en 8GB: usar VAR-d20 (600M) o VAR-d24 (1B) FP16/INT8
- Resolución recomendada: 256px inicial → upscaler después