# Image Editor - Documentación Técnica

**Versión:** 2026.1  
**Última actualización:** Abril 2026  
**Estado:** ✅ Producción

---

## 1. Descripción General

Image Editor es un sistema de edición de imágenes basado en IA que permite modificar imágenes existentes mediante prompts de texto natural.

### 1.1 Casos de Uso

- Mejora de calidad de imágenes
- Cambio de ropa y accesorios
- Modificación de fondos y entornos
- Retoques faciales y corporales
- Adición de elementos a escenas

### 1.2 Tecnologías

| Componente | Tecnología |
|------------|------------|
| Motor de generación | ComfyUI SD1.5 Inpainting |
| Análisis de prompts | Semantic Analyzer (propio) |
| Detección facial | InsightFace |
| Face Swap | Reactor/Inswapper |
| UI | Gradio 3.41.2 |
| Backend | Python 3.10+ |

---

## 2. Características Técnicas

### 2.1 Inpainting con ComfyUI

**Funcionamiento:**
1. Análisis semántico del prompt
2. Detección de área a modificar
3. Creación de máscara con gradiente
4. Generación con SD1.5 inpainting
5. Composición con imagen original

**Parámetros:**
- Steps: 30-40
- CFG Scale: 7.0-8.0
- Denoise: 0.85-0.92
- Checkpoint: Realistic (absolutereality, epicrealism)

### 2.2 Prompt Enhancer

**Función:** Mejora automática de prompts mediante análisis semántico.

**Proceso:**
1. Tokenización del prompt original
2. Detección de palabras clave
3. Inferencia de acciones (cambiar, eliminar, añadir, mejorar)
4. Detección de grupos semánticos (cuerpo, fondo, estilo)
5. Generación de prompt mejorado con quality tags

**Ejemplo:**
```
Original: "mejora calidad"
Mejorado: "UHD, 8k, sharp focus, professional lighting, realistic, detailed, 
           ultra high quality, best quality, masterpiece, mejora calidad"
```

### 2.3 Character Reference

**Función:** Mantener consistencia facial entre imágenes.

**Implementación:**
1. Extracción de embedding facial de imagen de referencia
2. Face swap en imagen generada
3. Blend con imagen original

**Parámetros:**
- Modelo: inswapper_128.onnx
- Face Analyzer: InsightFace (detection + recognition)

### 2.4 Batch Processing

**Función:** Procesar múltiples imágenes con el mismo prompt.

**Flujo:**
1. Usuario sube N imágenes
2. Escribe UN prompt
3. Sistema procesa secuencialmente
4. Resultados: N × variaciones

**Rendimiento:**
| Imágenes | Variaciones | Tiempo Total |
|----------|-------------|--------------|
| 5 | 4 | ~3 min |
| 10 | 4 | ~6 min |
| 20 | 4 | ~12 min |

### 2.5 Text Overlay

**Función:** Añadir texto/logos a imágenes generadas.

**Implementación:**
- Motor: PIL ImageDraw + ImageFont
- Posiciones: 7 (top-left, top-center, top-right, bottom-left, bottom-center, bottom-right, center)
- Estilos: modern (Arial), classic (Times), handwritten (Calibri), bold (Arial Bold)
- Tamaño: Automático según resolución

### 2.6 Face Preservation

**Función:** Preservar cara original en imagen editada.

**Implementación:**
1. Detección de cara en imagen original
2. Detección de cara en imagen generada
3. Face swap: original → generada
4. Blend: 95% cara original, 5% blending

---

## 3. Especificaciones Técnicas

### 3.1 Requisitos de Hardware

| Componente | Mínimo | Recomendado |
|------------|--------|-------------|
| GPU | NVIDIA 4GB | NVIDIA 8GB+ |
| VRAM | 4GB | 6-8GB |
| RAM | 16GB | 32GB |
| Almacenamiento | 10GB libres | 20GB libres |

### 3.2 Resoluciones Soportadas

| Resolución | Dimensiones | VRAM | Tiempo |
|------------|-------------|------|--------|
| 480p | 640×480 | 4-5GB | ~20s |
| 720p | 1280×720 | 5-6GB | ~35s |
| 1024p | 1024×1024 | 6-7GB | ~60s |

### 3.3 Formatos

**Entrada:**
- PNG
- JPG/JPEG
- WEBP
- BMP

**Salida:**
- PNG (lossless)
- JPG (compresión ajustable)

### 3.4 Rendimiento

| Métrica | Valor |
|---------|-------|
| Tiempo por imagen | 30-45s |
| Tiempo por variación | 30-45s |
| Batch (10 imgs) | ~6 min |
| VRAM uso pico | 5-6GB |

---

## 4. Estado de Funcionalidad

### 4.1 Características Principales

| Característica | Estado | Implementación |
|----------------|--------|----------------|
| Inpainting | ✅ Funcional | ComfyUI SD1.5 |
| Prompt Enhancer | ✅ Funcional | Semantic Analyzer |
| Character Reference | ✅ Funcional | InsightFace + Face Swap |
| Batch Processing | ✅ Funcional | Procesamiento secuencial |
| Text Overlay | ✅ Funcional | PIL ImageDraw |
| Face Preservation | ✅ Funcional | Reactor/Inswapper |
| Resolution Control | ✅ Funcional | 480p/720p/1024p |
| Quality Presets | ✅ Funcional | Fast/Balanced/High/Max |

### 4.2 Integraciones

| Integración | Estado | Notas |
|-------------|--------|-------|
| ComfyUI | ✅ Activa | Puerto 8188 |
| InsightFace | ✅ Activa | CUDA |
| FLUX Fill | ❌ No disponible | Requiere token gated |
| LlamaGen | ❌ No implementado | No soporta inpainting |

### 4.3 Métricas de Calidad

| Métrica | Puntuación | Método |
|---------|------------|--------|
| Calidad de generación | 6-7/10 | Evaluación subjetiva |
| Comprensión de prompts | 7/10 | Semantic Analyzer |
| Preservación facial | 7/10 | Face swap accuracy |
| Coherencia de edición | 6-7/10 | Inpainting consistency |
| Velocidad | 7/10 | 30-45s por imagen |

---

## 5. Configuración

### 5.1 Parámetros Recomendados (RTX 3060 Ti 8GB)

```yaml
# Calidad
steps: 30-40
cfg_scale: 7.0-8.0
denoise: 0.85-0.92

# Resolución
resolution: 720p  # Balance calidad/velocidad

# Variaciones
num_variations: 4  # Default

# Face Preservation
preserve_face: true
blend_ratio: 0.95
```

### 5.2 Optimización para 8GB VRAM

**Ajustes aplicados automáticamente:**
- Steps mínimo: 30
- CFG máximo: 8.0
- Denoise optimizado: 0.85-0.92
- Redimensionamiento automático si > 1536px

### 5.3 Solución de Problemas

| Problema | Causa | Solución |
|----------|-------|----------|
| "CUDA out of memory" | VRAM insuficiente | Reducir resolución a 480p |
| "No faces detected" | Mala iluminación/ángulo | Usar foto frontal |
| "ComfyUI not connected" | ComfyUI no iniciado | Iniciar desde Comfy Launcher |
| "Resultados inconsistentes" | Prompt muy vago | Usar Prompt Enhancer |
| "Proceso lento" | Calidad muy alta | Usar preset "Rápido" |

---

## 6. Arquitectura

### 6.1 Diagrama de Flujo

```
Usuario → UI (Gradio) → Prompt + Imagen
                              ↓
                    Semantic Analyzer
                              ↓
                    Prompt Enhancer
                              ↓
                    Mask Generator (gradiente)
                              ↓
                    ComfyUI Inpainting
                              ↓
                    Face Preservation (opcional)
                              ↓
                    Text Overlay (opcional)
                              ↓
                    Resultado → Galería
```

### 6.2 Componentes Principales

| Archivo | Función |
|---------|---------|
| `ui/tabs/img_editor_tab.py` | UI Gradio + callbacks |
| `roop/img_editor/img_editor_manager.py` | Lógica de negocio |
| `roop/img_editor/flux_client.py` | Cliente FLUX (fallback) |
| `roop/img_editor/clothing_segmenter.py` | CLIPSeg segmentación |
| `roop/img_editor/controlnet_utils.py` | ControlNet utilities |
| `roop/img_editor/prompt_areas.json` | Configuración áreas semánticas |

### 6.3 Integración con ComfyUI

**Flujo:**
1. Upload de imagen y máscara a ComfyUI
2. Construcción de workflow JSON
3. Queue submission
4. Polling de resultado
5. Download de imagen generada
6. Composición final

**Workflow:**
- Nodo: Load Image
- Nodo: Load Mask
- Nodo: CLIP Text Encode (prompt)
- Nodo: CLIP Text Encode (negative)
- Nodo: KSampler (inpaint)
- Nodo: VAE Decode
- Nodo: Save Image

---

## 7. API Reference

### 7.1 ImgEditorManager

```python
from roop.img_editor.img_editor_manager import ImgEditorManager

manager = ImgEditorManager()

result, msg = manager.generate_intelligent(
    image=pil_image,
    prompt="mejora calidad",
    num_inference_steps=40,
    guidance_scale=8.0,
    strength=0.88,
    face_preserve=True,
    auto_enhance=True
)
```

### 7.2 Parámetros

| Parámetro | Tipo | Default | Descripción |
|-----------|------|---------|-------------|
| image | PIL.Image | - | Imagen original |
| prompt | str | - | Prompt de edición |
| negative_prompt | str | None | Prompt negativo |
| num_inference_steps | int | 25 | Steps de generación |
| guidance_scale | float | 7.5 | CFG scale |
| strength | float | 0.65 | Denoise strength |
| seed | int | None | Seed para reproducibilidad |
| face_preserve | bool | True | Preservar cara |
| auto_enhance | bool | True | Auto-mejorar prompt |

### 7.3 Retorno

```python
# Éxito
(result_image, "Generación completada")

# Error
(None, "Error: descripción del error")
```

---

## 8. Logs y Debugging

### 8.1 Logs en Consola

```
[OnGenerate] Iniciando - img=True, prompt=mejora calidad
[OnGenerate] Procesando directamente...
[ImgEditor] === 🧠 ANALIZANDO PROMPT ===
[ImgEditor] === ✍️ MEJORANDO PROMPT ===
[SemanticAnalyzer] Palabras clave: ['mejora', 'calidad']
[SemanticAnalyzer] Intensidad detectada: medium
[ImgEditor] Imagen: (1280, 720)
[ImgEditor] Creando máscara: y_start=0px (0.0%), y_end=720px
[ImgEditor] === 🎨 GENERANDO (OPTIMIZADO 8GB VRAM) ===
[ImgEditor] Steps: 30, CFG: 8.0, Denoise: 0.88
[ComfyUI] Prompt executed in 21.42 seconds
[OnGenerate] ✅ Variación 1 completada
[OnGenerate] ✅ Completado en 180s
```

### 8.2 Comandos de Diagnóstico

```bash
# Verificar ComfyUI
python check_comfyui_status.py

# Verificar modelos
python check_available_models.py

# Test de generación
python test_img_editor.py
```

---

## 9. Historial de Versiones

| Versión | Fecha | Cambios |
|---------|-------|---------|
| 2026.1 | Abr 2026 | Optimización 8GB VRAM |
| 2025.2 | Mar 2025 | Batch Processing |
| 2025.1 | Feb 2025 | Character Reference |
| 2024.3 | Dic 2024 | Text Overlay |
| 2024.2 | Nov 2024 | Prompt Enhancer |
| 2024.1 | Oct 2024 | Versión inicial |

---

## 10. Referencias

- [ComfyUI Documentation](https://comfyui.org/)
- [Stable Diffusion Inpainting](https://huggingface.co/docs/diffusers/inpaint)
- [InsightFace](https://insightface.ai/)
- [Gradio Documentation](https://gradio.app/docs/)

---

**Fin del documento**
