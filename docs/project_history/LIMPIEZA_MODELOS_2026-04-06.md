# Limpieza de Código - 2026-04-06

## ❌ Eliminados (no servían)

### FLUX (demasiado lento con 8GB VRAM)
- `roop/img_editor/flux_client.py`
- Modelos `FLUX.1-fill-dev-NF4` y `FLUX.1-fill-dev-QUAN`

### VAR (no sirve para editing de imágenes)
- `D:\PROJECTS\VAR` (repo clonado)
- `D:\PROJECTS\models\var-d24` (modelos descargados)
- `roop/img_editor/var_client.py`
- Scripts de descarga

**Motivo**: VAR es class-conditioned generation, NO editing. Solo genera imágenes nuevas desde clases Imagenet, no modifica imágenes existentes.

## ✅ Estado Actual del Image Editor

El Image Editor ahora usa exclusivamente:
1. **ComfyUI** como motor principal
2. **SD1.5** como fallback local
3. **Prompt Analyzer** para mejorar prompts automáticamente
4. **Face Preserver** para mantener identidad en face swap

## 🔧 Pendiente: Motor de Edición Real

Para edición de imágenes como Grok se necesita:
- **Qwen-Image-Edit** (Alibaba) - Diseño específico para edición con texto
- Soporta: cambio de ropa, mejora de calidad, adición de accesorios
