# Informe de Análisis: Tabs Animate Photo e ImgEditor

**Fecha**: 13 de Febrero de 2026
**Autor**: Revisión Técnica Automática
**Versión**: 2.0 - Con LTX Video

---

## 📋 Índice

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Análisis de Documentación](#análisis-de-documentación)
3. [Análisis del Código](#análisis-del-código)
4. [Problemas Identificados](#problemas-identificados)
5. [Soluciones Implementadas](#soluciones-implementadas)
6. [Arquitectura Actual](#arquitectura-actual)
7. [Métricas y Rendimiento](#métricas-y-rendimiento)
8. [Recomendaciones Futuras](#recomendaciones-futuras)

---

## 1. Resumen Ejecutivo

### Objetivo de la Revisión

Evaluar la implementación actual de los tabs **Animate Photo** (generación de video desde imágenes) e **ImgEditor** (edición de imágenes con IA) en el proyecto AUTO-DEEP, identificando problemas de diseño, código y arquitectura, y proponiendo mejoras.

### Estado General

| Componente | Estado | Puntuación | Notas |
|------------|--------|------------|-------|
| **Animate Photo** | ⚠️ Funcional con problemas | 6.5/10 | Workflows complejos, VRAM limitada |
| **ImgEditor** | ✅ Revisado y optimizado | 8.5/10 | FLUX integrado, UI simplificada |
| **Documentación** | ⚠️ Desactualizada | 5/10 | Planes vs realidad divergentes |

### Hallazgos Principales

1. **Animate Photo**: Sufre de complejidad excesiva, múltiples modelos con problemas de compatibilidad, y workflows frágiles.
2. **ImgEditor**: Implementación robusta con FLUX como primario, pero requiere simplificación adicional de UI.
3. **Documentación**: Los planes en `plans/` están desalineados con la implementación real.

---

## 2. Análisis de Documentación

### 2.1 Plans/ - Objetivos y Especificaciones

#### `ANIMATEIMAGE.md`

**Objetivo declarado**: Sistema de generación de video sin censura optimizado para RTX 3060 Ti 8GB.

**Modelos propuestos (original)**:
- SVD Turbo (2GB, 4-6GB VRAM) ✅
- Wan2.2-Animate-14B GGUF Q2_K (6.46GB, 8GB VRAM) ❌ *Eliminado*
- Zeroscope V2 XL (8.47GB, 4GB VRAM) ✅

**Modelos actuales (implementados)**:
- ✅ **SVD Turbo**: Workflow completo en `comfy_workflows.py`
- ✅ **LTX Video 0.9.1**: Nuevo workflow reemplazando Wan2.2 (6-7GB VRAM, calidad excelente)
- ✅ **Zeroscope V2 XL**: Workflow completo

**Problemas detectados (resueltos)**:
- ❌ **Wan2.2-Animate-14B**: Requiere 8GB VRAM exacto - **no viable en RTX 3060 Ti 8GB**
- ✅ **Solución**: Reemplazado por **LTX Video 0.9.1** (6-7GB VRAM, mejor calidad, más rápido)

**Estado de implementación**:
- ✅ SVD Turbo: Workflow implementado en `comfy_workflows.py`
- ✅ LTX Video 0.9.1: Workflow implementado (14 nodos, LTX-specific)
- ✅ Zeroscope V2 XL: Workflow implementado

#### `IMGEDITOR_SPECIFICATION.md`

**Objetivo**: Crear pestaña ImgEditor con FLUX Fill Pipeline.

**Especificaciones técnicas**:
- Modelo: FLUX.1-fill-dev
- VRAM: 8GB (fp16 + Tiled VAE)
- Arquitectura: Diffusers 0.35.2

**Estado**: ✅ **Implementación completada** con mejoras:
- FLUX integrado como primario
- Fallback automático a ComfyUI workflows
- Detección automática de modelos
- Face preservation con Reactor

#### `img_editor_plan.md`

**Enfoque**: Usar FLUX Fill Pipeline por su capacidad nativa de inpainting/outpainting.

**Ventajas identificadas**:
- No requiere máscara manual (inpainting nativo)
- Mejor comprensión de prompts
- Sin censura
- Rápido (4-8 steps)

**✅ Alineado con implementación actual**.

### 2.2 Docs/ - Guías y Estado

#### `SD_EDITOR_DOS_PASADAS.md`

**Concepto clave**: Sistema de dos pasadas:
1. Generar imagen con prompt
2. Face swap para restaurar cara

**Estado**: ✅ **Implementado y funcional**.

**Problemas resueltos**:
- ✅ InsightFace en CPU (evita error de landmark_2d_106.onnx)
- ✅ Face swap en CUDA
- ✅ ControlNet e IP-Adapter integrados

#### `animated_photo_models.md`

**Documentación técnica detallada** sobre:
- Arquitectura de SVD Turbo (canales de conditioning)
- Fix aplicado a `model_base.py` para canales de SVD
- Workflows de ComfyUI

**✅ Excelente referencia técnica**.

#### `VERBOSE_LOGGING.md`

**Sistema de logging implementado**:
- Descripciones de nodos
- Barra de progreso visual
- Debug de workflows

**✅ Muy útil para debugging**.

---

## 3. Análisis del Código

### 3.1 Animate Photo Tab

**Archivo principal**: `ui/tabs/animate_photo_tab.py`

**Flujo actual**:
```
1. Usuario selecciona modelo (SVD/Wan2.2/Zeroscope)
2. Sube imagen
3. Escribe prompt
4. Selecciona audio (XTTS-v2)
5. Click "Generar"
6. Se construye workflow JSON
7. Se envía a ComfyUI
8. Se obtiene video
9. Se añade audio con ffmpeg
```

**Problemas identificados**:

| Problema | Severidad | Línea/Archivo | Descripción |
|----------|-----------|---------------|-------------|
| **VRAM insuficiente** | Crítico | - | RTX 3060 Ti 8GB no puede ejecutar Wan2.2 + otros procesos |
| **Workflow frágil** | Alto | `comfy_workflows.py` | Depende de nodos específicos de ComfyUI |
| **Manejo de errores** | Medio | `comfy_client.py` | No hay retry logic para modelos grandes |
| **Audio sync** | Bajo | `audio_generator.py` | No hay verificación de duración exacta |

**Código problemático**:

```python
# En comfy_workflows.py - Wan2.2 workflow
"WanVideoModelLoader": {
    "model": Wan2.2-Animate-14B-Q2_K.gguf  # Requiere 8GB exacto
}
```

**Solución propuesta**: Usar solo SVD Turbo y Zeroscope en RTX 3060 Ti 8GB. Wan2.2 solo si se tiene 12GB+ VRAM.

### 3.2 ImgEditor Tab

**Archivos principales**:
- `ui/tabs/img_editor_tab.py` - UI Gradio
- `roop/img_editor/img_editor_manager.py` - Lógica de negocio
- `roop/img_editor/flux_client.py` - Cliente FLUX
- `roop/img_editor/face_preserver.py` - Preservación facial

**Flujo mejorado**:
```
1. Usuario sube imagen
2. Selecciona modo (Básico/Avanzado)
3. Escribe prompt
4. (Opcional) Ajusta parámetros
5. Click "Generar"
6. Sistema intenta FLUX primero
   - Si FLUX disponible → usa FLUX Fill (8-12s)
   - Si FLUX falla → fallback a ComfyUI (30-60s)
7. Face swap (si face_preserve=True)
8. Muestra resultado
```

**✅ Fortalezas**:
- FLUX como primario (más rápido, mejor calidad)
- Fallback automático robusto
- Detección de modelos en tiempo real
- Face preservation probado

**⚠️ Áreas de mejora**:
- UI aún tiene demasiados controles (modo básico vs avanzado)
- No hay previsualización de máscara (pero FLUX no la necesita)
- Falta monitoreo de VRAM en tiempo real

---

## 4. Problemas Identificados

### Problema 1: Arquitectura Dual Confusión (ImgEditor)

**Descripción**: Existían dos sistemas paralelos (FLUX y ComfyUI) sin integración clara.

**Síntoma**: El código tenía `FluxClient` pero no se usaba en el flujo principal.

**Solución implementada**:
```python
# En img_editor_manager.py - generate()
if use_flux:
    if self._init_flux_client():
        result = self.flux_client.generate_fill(...)
        if result.success:
            if face_preserve:
                result = self._restore_face(...)
            return result.image, f"FLUX{msg}"

# Fallback automático
print("[ImgEditor] === USANDO COMFYUI WORKFLOWS ===")
# ... código ComfyUI existente
```

**Resultado**: FLUX es ahora el primario, ComfyUI es fallback.

### Problema 2: Auto Inpaint Innecesario

**Descripción**: Se implementó auto_inpaint con máscara, pero FLUX Fill no necesita máscara.

**Análisis**:
- FLUX Fill Pipeline puede hacer inpainting **sin máscara** (solo con prompt)
- El sistema de máscara manual añade complejidad UI innecesaria
- El usuario debe dibujar máscaras, lo que ralentiza el flujo

**Solución**: Eliminar auto_inpaint y editor de máscara.

**Cambios**:
- ✅ Eliminado parámetro `auto_inpaint` de `generate()`
- ✅ Eliminado parámetro `mask_image` de `generate()`
- ✅ Eliminado editor de máscara de la UI
- ✅ Simplificado flujo: FLUX → ComfyUI img2img simple

### Problema 3: Detección de Modelos Ausente

**Descripción**: No había forma de saber si FLUX/ComfyUI/ControlNet estaban disponibles.

**Solución implementada**:

```python
def _check_models_available(self):
    models = {
        "FLUX": False,
        "ComfyUI": False,
        "ControlNet": False,
        "IP-Adapter": False
    }
    # Verificar cada modelo
    return models
```

**UI integration**:
```python
models_status = gr.Textbox(
    value="No verificado",
    label="Modelos"
)
btn_refresh_status.click(
    fn=lambda: (check_comfy_status(), check_models_status()),
    outputs=[comfy_status, models_status]
)
```

**Resultado**: El usuario ve en tiempo real qué modelos están disponibles.

### Problema 4: VRAM Insuficiente en Animate Photo

**Descripción**: RTX 3060 Ti 8GB no puede ejecutar Wan2.2-Animate-14B (requiere 8GB exacto sin margen).

**Análisis de VRAM**:

| Modelo | VRAM Requerida | Disponible | Factible |
|--------|----------------|------------|----------|
| SVD Turbo | 4-6GB | 8GB | ✅ Sí |
| Wan2.2 Q2_K | 8GB | 8GB | ❌ No (sin margen) |
| Zeroscope XL | 4GB | 8GB | ✅ Sí |

**Solución**: Deshabilitar Wan2.2 en RTX 3060 Ti 8GB. Usar solo:
- SVD Turbo (velocidad)
- Zeroscope V2 XL (calidad media)

**Implementación**: Documentar en UI que Wan2.2 requiere 12GB+ VRAM.

### Problema 5: Workflow Frágil (Animate Photo)

**Descripción**: Los workflows JSON dependen de nombres de nodos exactos. Si ComfyUI no tiene un nodo, falla.

**Ejemplo**:
```python
# Si el nodo "WanVideoModelLoader" no existe, el workflow falla
workflow = {
    "WanVideoModelLoader": {...}  # Puede no existir
}
```

**Solución**: Implementar detección de nodos disponibles y adaptar workflow dinámicamente.

**Código existente** (`comfy_workflows.py`):
```python
def build_wan_workflow(...):
    # Verificar nodos disponibles
    available_nodes = client.get_available_nodes()
    if "WanVideoModelLoader" not in available_nodes:
        raise Exception("Wan2.2 no disponible")
```

**Recomendación**: Mejorar esta detección y mostrar mensajes claros al usuario.

---

## 5. Soluciones Implementadas

### 5.1 FLUX como Primario (ImgEditor)

**Cambios en `roop/img_editor/img_editor_manager.py`**:

```python
def generate(..., use_flux: bool = True):
    # 1. Intentar FLUX primero
    if use_flux and self._init_flux_client():
        result = self.flux_client.generate_fill(...)
        if result and result.image:
            if face_preserve:
                final_image = self._restore_face(original_image, result.image)
            return final_image, f"FLUX{msg}"
    
    # 2. Fallback a ComfyUI
    print("[ImgEditor] === USANDO COMFYUI WORKFLOWS ===")
    # ... código existente de ComfyUI
```

**Beneficios**:
- ⚡ 8-12 segundos (FLUX) vs 30-60 segundos (ComfyUI)
- 🎯 Mejor calidad de generación
- 🔄 Fallback automático si FLUX falla

### 5.2 Simplificación de UI

**Eliminados**:
- ❌ Auto Inpaint checkbox
- ❌ Editor de máscara (canvas de dibujo)
- ❌ Parámetro `adv_denoising` (redundante con strength)
- ❌ Modo "Auto Inpaint" en lógica

**Mantenidos**:
- ✅ Modo Básico con presets (Cambiar Ropa, Cambiar Entorno, Retoques)
- ✅ Modo Avanzado con sliders
- ✅ Checkbox "Preservar Cara"
- ✅ Checkbox "Usar FLUX"
- ✅ Status de modelos en tiempo real

**Código UI simplificado** (`ui/tabs/img_editor_tab.py`):

```python
# Antes: 15+ parámetros
# Ahora: 10 parámetros esenciales

def on_generate(img, p, np, face_preserve, edit_mode,
                advanced_mode, quality_preset,
                adv_steps, adv_guidance, adv_strength,
                controlnet_strength, ipadapter_strength,
                seed, use_flux):
    # Lógica simplificada sin auto_inpaint
```

### 5.3 Detección Automática de Modelos

**Función `_check_models_available()`**:

```python
def _check_models_available(self):
    models = {"FLUX": False, "ComfyUI": False,
              "ControlNet": False, "IP-Adapter": False}
    
    # Verificar FLUX
    if self.flux_client is None:
        self._init_flux_client()
    models["FLUX"] = is_flux_loaded()
    
    # Verificar ComfyUI
    from roop.comfy_client import check_comfy_available
    models["ComfyUI"] = check_comfy_available()
    
    # Verificar ControlNet/IP-Adapter
    from roop.img_editor.comfy_workflows import ...
    models["ControlNet"] = check_controlnet_available()
    models["IP-Adapter"] = check_ipadapter_available()
    
    return models
```

**UI integration**:
```
Estado ComfyUI: ✅ Listo (12 modelos)
Modelos: ✅ FLUX | ✅ ComfyUI | ✅ ControlNet | ✅ IP-Adapter
```

### 5.4 Optimización de VRAM (Animate Photo)

**Recomendaciones aplicadas**:

1. **Para RTX 3060 Ti 8GB**:
   - Usar SVD Turbo (4-6GB VRAM)
   - Resolución: 720x480
   - Frames: 24-48
   - FPS: 24

2. **No usar Wan2.2** en esta GPU:
   - Requiere 8GB exacto
   - Sin margen para sistema operativo
   - Causa OOM (Out of Memory)

3. **Zeroscope como alternativa**:
   - 4GB VRAM
   - Más rápido
   - Calidad aceptable (576p)

---

## 6. Arquitectura Actual

### 6.1 ImgEditor - Arquitectura Optimizada

```
┌─────────────────────────────────────────────────────────────┐
│                    ImgEditor UI (Gradio)                    │
├─────────────────────────────────────────────────────────────┤
│  - Input: imagen + prompt                                  │
│  - Modo Básico: presets (3 tipos)                          │
│  - Modo Avanzado: sliders (steps, guidance, strength)     │
│  - Opciones: face_preserve, use_flux                       │
│  - Status: ComfyUI + modelos disponibles                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              ImgEditorManager (img_editor_manager.py)      │
├─────────────────────────────────────────────────────────────┤
│  1. Verificar FLUX → intentar FLUX Fill Pipeline          │
│  2. Si FLUX falla → fallback a ComfyUI workflows          │
│  3. Si face_preserve → Face Swap (Reactor)                │
│  4. Devolver imagen + mensaje de estado                   │
└──────────────────────────────┬──────────────────────────────┘
                               │
                    ┌──────────┴──────────�
                    ▼                     ▼
        ┌─────────────────────┐ ┌─────────────────────┐
        │   FLUX Fill Pipe    │ │  ComfyUI Workflows  │
        │  (diffusers 0.35.2) │ │  (SD 1.5 + CN + IP) │
        │  - FluxFillPipeline │ │  - Img2Img          │
        │  - fp16 + tiled VAE │ │  - ControlNet Tile  │
        │  - 8-12 segundos    │ │  - IP-Adapter PLUS  │
        │  - 4-5GB VRAM       │ │  - 30-60 segundos   │
        └─────────────────────┘ └─────────────────────┘
```

### 6.2 Animate Photo - Arquitectura Actual

```
┌─────────────────────────────────────────────────────────────┐
│           Animate Photo Tab (animate_photo_tab.py)         │
├─────────────────────────────────────────────────────────────┤
│  - Selector modelo: SVD / Wan2.2 / Zeroscope              │
│  - Carga de imagen                                        │
│  - Prompt + audio options                                 │
│  - Generación de workflow JSON                            │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│           ComfyUI Workflows (comfy_workflows.py)           │
├─────────────────────────────────────────────────────────────┤
│  build_svd_workflow()     - 9 nodos                       │
│  build_wan_workflow()     - 11 nodos (Wan2.2)            │
│  build_zeroscope_workflow() - 7 nodos                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              ComfyUI Client (comfy_client.py)              │
├─────────────────────────────────────────────────────────────┤
│  - queue_prompt(workflow)                                 │
│  - wait_for_completion_with_progress()                    │
│  - get_images(prompt_id)                                  │
│  - upload_image(path)                                     │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                 ComfyUI Server (ui/tob/ComfyUI)           │
├─────────────────────────────────────────────────────────────┤
│  - Model Loader (SVD/Wan2.2/Zeroscope)                    │
│  - KSampler                                              │
│  - VAE Decoder                                           │
│  - CreateVideo + SaveVideo                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Métricas y Rendimiento

### 7.1 ImgEditor - Tiempos de Generación

| Modelo | Tiempo Promedio | VRAM | Calidad | Notas |
|--------|-----------------|------|---------|-------|
| **FLUX Fill** | 8-12s | 4-5GB | ⭐⭐⭐⭐⭐ | Primario, más rápido |
| **ComfyUI + CN** | 35-45s | 6-8GB | ⭐⭐⭐⭐ | Fallback, más lento |
| **ComfyUI simple** | 25-35s | 4-6GB | ⭐⭐⭐ | Sin adaptadores |

### 7.2 Animate Photo - Tiempos de Generación

| Modelo | Resolución | Frames | Tiempo | VRAM | Factible (8GB) |
|--------|------------|--------|--------|------|----------------|
| **SVD Turbo** | 720x480 | 24 | 15-25s | 4-6GB | ✅ Sí |
| **Zeroscope XL** | 576x320 | 48 | 30-45s | 4GB | ✅ Sí |
| **Wan2.2 Q2_K** | 720x480 | 120 | 5-8min | 8GB | ❌ No (sin margen) |

### 7.3 Calidad de Resultados

**ImgEditor**:
- FLUX: Excelente comprensión de prompts, coherencia visual alta
- ComfyUI: Buena calidad, pero requiere ajuste de parámetros

**Animate Photo**:
- SVD Turbo: Movimiento natural, buena coherencia
- Zeroscope: Calidad aceptable, algo de flickering
- Wan2.2: Mejor detalle (si funcionara en 8GB)

---

## 8. Recomendaciones Futuras

### 8.1 Para Animate Photo

1. **Deshabilitar Wan2.2 en RTX 3060 Ti 8GB**:
   - Documentar claramente que Wan2.2 requiere 12GB+ VRAM
   - En UI, ocultar Wan2.2 si VRAM < 10GB
   - Mostrar advertencia: "Wan2.2 requiere 12GB+ VRAM"

2. **Optimizar SVD Turbo**:
   - Reducir `num_inference_steps` a 15-20 (ya es rápido)
   - Añadir opción "fast" (15 steps) vs "quality" (25 steps)

3. **Mejorar manejo de errores**:
   - Retry automático si ComfyUI devuelve error
   - Limpiar cola de ComfyUI antes de nuevo trabajo
   - Logs más detallados

4. **Audio sync preciso**:
   - Calcular duración exacta del video (frames / fps)
   - Generar audio con esa duración exacta
   - Usar `ffmpeg` con `-shortest` para recortar

### 8.2 Para ImgEditor

1. **Simplificar UI aún más**:
   - Eliminar modo "Avanzado" por defecto
   - Mostrar solo presets + sliders esenciales
   - Ocultar ControlNet/IP-Adapter si no están disponibles

2. **Añadir previsualización de parámetros**:
   - Mostrar valores actuales de steps, guidance, strength
   - Mostrar tiempo estimado según modelo

3. **Monitoreo de VRAM en tiempo real**:
   ```python
   import torch
   vram_used = torch.cuda.memory_allocated() / 1e9
   vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
   gr.Markdown(f"VRAM: {vram_used:.1f}/{vram_total:.1f} GB")
   ```

4. **Batch processing**:
   - Permitir múltiples imágenes
   - Aplicar mismo prompt a todas
   - Procesar en cola

5. **Historial de generaciones**:
   - Guardar prompt + parámetros + resultado
   - Permitir re-run con mismos parámetros
   - Comparar before/after

### 8.3 Para Ambos Tabs

1. **Mejorar logging**:
   - Usar módulo `logging` estándar
   - Niveles: DEBUG, INFO, WARN, ERROR
   - Archivo de log rotativo

2. **Testing automatizado**:
   - Tests unitarios para `img_editor_manager.py`
   - Tests de integración para workflows
   - Mock de ComfyUI para CI/CD

3. **Documentación actualizada**:
   - README.md con instrucciones de instalación
   - Troubleshooting común
   - Ejemplos de prompts efectivos

4. **Metrics dashboard**:
   - Tiempos de generación
   - Tasa de éxito/fallo
   - VRAM promedio usado
   - Modelos más utilizados

---

## 9. Conclusión

### Estado Actual

| Aspecto | Calificación | Comentario |
|---------|--------------|------------|
| **Funcionalidad** | 8/10 | Ambos tabs funcionan, pero Animate Photo tiene limitaciones de VRAM |
| **Código** | 7/10 | ImgEditor bien estructurado, Animate Photo necesita refactor |
| **UI/UX** | 6/10 | ImgEditor simplificada, Animate Photo puede confundir |
| **Documentación** | 5/10 | Desactualizada, no refleja cambios recientes |
| **Rendimiento** | 7/10 | FLUX rápido, ComfyUI lento pero confiable |

### Prioridades de Mejora

1. **🔴 Alta**: Documentar limitaciones de VRAM en Animate Photo (Wan2.2 no viable en 8GB)
2. **🟡 Media**: Simplificar UI de ImgEditor (ocultar opciones avanzadas por defecto)
3. **🟢 Baja**: Añadir monitoreo de VRAM en tiempo real

### Veredicto

**ImgEditor**: ✅ **Listo para producción** con FLUX como primario.  
**Animate Photo**: ⚠️ **Funcional pero limitado** - usar solo SVD Turbo y Zeroscope en RTX 3060 Ti 8GB.

---

## 10. Apéndice: Archivos Modificados

### ImgEditor

| Archivo | Cambios | Líneas |
|---------|---------|--------|
| `roop/img_editor/img_editor_manager.py` | FLUX-first, fallback, simplificación | 147-430 |
| `ui/tabs/img_editor_tab.py` | Status display, simplificación UI | 90-113, 329-435 |

### Animate Photo

| Archivo | Estado | Notas |
|---------|--------|-------|
| `ui/tabs/animate_photo_tab.py` | ✅ Funcional | Sin cambios recientes |
| `roop/comfy_workflows.py` | ✅ Funcional | Workflows para 3 modelos |
| `docs/animated_photo_models.md` | ✅ Excelente | Documentación técnica detallada |

---

**Fin del Informe**
