# 🎯 Plan de Acción Completo - Sistema de Generación de Video Sin Censura

## Objetivo Final
Crear un sistema **profesional y funcional** para generar videos desde imágenes con soporte de audio y prompts completos, optimizado para tu GPU RTX 3060 Ti (8GB VRAM).

## 📋 Resumen de Cambios
- ✅ **Eliminados modelos chinos oficiales (LTX/CogVideoX)**
- ✅ **Modelos antiguos removidos** (≈50GB liberados)
- ⚠️ **Modelos a instalar**: SVD Turbo, Wan2.2-Animate-14B, Zeroscope V2 XL

---

## 🚀 PASO 1: Descarga y Instalación de Modelos (Tu Responsabilidad)

### 1.1 Modelo 1: Stable Video Diffusion Turbo (SVD Turbo)
- **Descripción**: Mejor opción para tu GPU (velocidad + calidad)
- **Tamaño**: ≈2GB
- **VRAM Requerida**: 4-6GB
- **Link de Descarga**: https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt
- **Carpeta de Instalación**: `ui/tob/ComfyUI/models/diffusion_models/`
- **Archivos Necesarios**: `model.safetensors`, `config.json`

### 1.2 Modelo 2: Wan2.2-Animate-14B (GGUF Q2_K)
- **Descripción**: Mejor detalle en rostros y motion (versión quantizada para 8GB VRAM)
- **Tamaño**: ≈6.46GB (GGUF Q2_K)
- **VRAM Requerida**: 8GB
- **Link de Descarga**: https://huggingface.co/QuantStack/Wan2.2-Animate-14B-GGUF
- **Archivo Necesario**: `Wan2.2-Animate-14B-Q2_K.gguf`
- **Carpeta de Instalación**: `ui/tob/ComfyUI/models/diffusion_models/`

### 1.3 Modelo 3: Zeroscope V2 XL (576p)
- **Descripción**: Prototipos rápidos y rendimiento máximo
- **Tamaño**: ≈8.47GB (completo)
- **VRAM Requerida**: 4GB
- **Link de Descarga**: https://huggingface.co/cerspense/zeroscope_v2_XL
- **Carpeta de Instalación**: `ui/tob/ComfyUI/models/diffusion_models/zeroscope_v2_XL/` (crea la carpeta)
- **Archivos Necesarios**:
  - `unet/diffusion_pytorch_model.bin` (2.82GB) - Modelo principal
  - `unet/config.json` (727 Bytes) - Configuración de unet
  - `text_encoder/pytorch_model.bin` (681MB) - Text encoder
  - `text_encoder/config.json` (609 Bytes) - Configuración de text encoder
  - `tokenizer/` (directorio completo) - Tokenizer
  - `scheduler/` (directorio completo) - Scheduler

---

## 🚀 PASO 2: Configuración del Sistema (Mi Responsabilidad)

### 2.1 Actualización de Workflows en `comfy_workflows_fixed.py`
- Crear workflow para SVD Turbo (con audio)
- Crear workflow para Wan2.2-Animate-14B (con audio)
- Crear workflow para Zeroscope V2 XL (con audio)
- Optimizar para 8GB VRAM (tiling, quantization)
- Incluir soporte para prompts complejos

### 2.2 Actualización de la UI en `animate_photo_tab.py`
- Agregar selector de modelos:
  - SVD Turbo (velocidad)
  - Wan2.2-Animate-14B (calidad)
  - Zeroscope V2 XL (prototipos)
- Incluir opciones de audio:
  - Generación de voz con XTTS-v2
  - Voice cloning
  - Añadir audio ambiental
- Optimizar interfaz para usabilidad

### 2.3 Pruebas y Optimizaciones
- Probar SVD Turbo con prompts complejos
- Probar Wan2.2-Animate-14B con motion
- Probar Zeroscope V2 XL para prototipos
- Ajustar parametros para 8GB VRAM
- Documentar errores y soluciones

---

## 📊 Comparativa de Modelos (Para Tu Referencia)

| Modelo | Tamaño | VRAM | Tiempo/Video | Calidad | Audio |
|--------|--------|------|--------------|---------|-------|
| SVD Turbo | 2GB | 4-6GB | 1-2s | Alta | Sí |
| Wan2.2-Animate-14B (GGUF Q2_K) | 6.46GB | 8GB | 5-8min | Excelente | Sí |
| Zeroscope V2 XL | 8.47GB | 4GB | 30s-1min | Buena | Sí |

---

## 🎯 Estrategia de Uso Recomendada

### Para Pruebas Rápidas:
1. **SVD Turbo** - Genera videos en segundos
2. **Prompts simples** (ej: "una persona caminando")
3. Resolución: 720p @ 24fps

### Para Proyectos Serios:
1. **Wan2.2-Animate-14B** - Mejor detalle en rostros (solo si tienes 16GB+ VRAM)
2. **Prompts complejos** (ej: "una mujer bailando tango en una plaza de Buenos Aires")
3. Resolución: 1080p @ 24fps

### Para Prototipos:
1. **Zeroscope V2 XL** - Rápido y ligero
2. **Prompts básicos**
3. Resolución: 576p @ 16fps

---

## 📋 Tareas Pendientes (Para Monitorizar)

- [ ] Descargar SVD Turbo
- [ ] Descargar Wan2.2-Animate-14B FP4
- [ ] Descargar Zeroscope V2 XL
- [ ] Instalar modelos en ComfyUI
- [ ] Actualizar workflows en `comfy_workflows_fixed.py`
- [ ] Actualizar UI en `animate_photo_tab.py`
- [ ] Probar SVD Turbo
- [ ] Probar Wan2.2-Animate-14B
- [ ] Probar Zeroscope V2 XL
- [ ] Documentar errores y soluciones
- [ ] Crear guías de uso

---

## ⚠️ Notas Importantes

1. **Tiempo de Descarga**: Los modelos son grandes, asegúrate de tener una conexión estable. 
   - SVD Turbo: ~2GB
   - Zeroscope V2 XL: ~8.47GB
   - Wan2.2-Animate-14B (GGUF Q2_K): ~6.46GB
2. **VRAM**: 
   - SVD Turbo: 4-6GB
   - Zeroscope V2 XL: 4GB
   - Wan2.2-Animate-14B (GGUF Q2_K): 8GB
3. **Audio**: El sistema usa XTTS-v2 para generar voz en español
4. **Censura**: Todos los modelos son **sin censura** y permiten prompts avanzados
5. **Wan2.2-Animate-14B**: Usar la versión GGUF Q2_K es la mejor opción para tu GPU de 8GB VRAM. Esta versión mantiene una calidad excelente con un tamaño reducido.
6. **Estructura de Archivos**: 
   - Zeroscope V2 XL requiere una estructura de directorios específica. Crea una carpeta `zeroscope_v2_XL` en `ui/tob/ComfyUI/models/diffusion_models/` y coloca los archivos en sus respectivos subdirectorios.
   - Wan2.2-Animate-14B (GGUF) se instala directamente en la carpeta `ui/tob/ComfyUI/models/diffusion_models/`.

---

## 🚀 Comenzar el Proceso

Una vez que hayas **descargado y instalado los 3 modelos**, avísame para:
1. Actualizar los workflows
2. Probar el sistema
3. Optimizar para tu hardware

¡Estamos listos para crear un sistema de generación de video **profesional y sin censura** para tu GPU!