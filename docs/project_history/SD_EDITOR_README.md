# 🎨 SD EDITOR - Edición Inteligente de Imágenes

## 📋 Descripción

**SD Editor** es un editor inteligente de imágenes con Stable Diffusion donde el usuario solo necesita:
1. **Subir una imagen**
2. **Escribir un prompt natural** (ej: "haz que esté bailando")
3. **Click en Generar**

La IA se encarga de **TODO automáticamente**:
- 🧠 Analiza tu prompt y decide qué técnicas usar
- 🎭 Cambia poses (OpenPose)
- ✨ Mejora calidad/resolución (Tile/Upscale)
- 👗 Cambia ropa (Inpaint automático con CLIPSeg)
- 👤 Mantiene identidad (IP-Adapter + Face Swap)
- 🔄 Genera múltiples variaciones

---

## 🚀 Características

### ✅ Análisis Inteligente de Prompts

El sistema detecta automáticamente qué necesitas:

| Palabras Clave | Técnica Usada |
|----------------|---------------|
| "bailando", "sentado", "de pie", "pose" | **OpenPose** |
| "mejora", "calidad", "resolución", "4K" | **Tile/Upscale** |
| "ropa", "vestido", "cambia", "desnuda" | **Inpaint + CLIPSeg** |
| "sonrisa", "feliz", "expresión" | **Face Edit** |

### ✅ Acciones Rápidas (Eliminadas)

**Nota:** Las acciones rápidas fueron eliminadas para simplificar la interfaz. 
Ahora solo necesitas escribir tu prompt directamente.

Ejemplos de prompts que puedes usar:
```
"haz que esté bailando, pose dinámica, movimiento, energía"
"cámbialo a sentado cómodamente, pose relajada, natural"
"mejora la calidad y resolución, más detalle, 4K, upscale"
"cambia la ropa por un vestido elegante, fashion, estilo"
"haz que sonría, expresión feliz, alegre, natural"
```

### ✅ Generación Múltiple

- **1-8 variaciones** a la vez
- Cada variación usa una seed diferente
- Ideal para elegir la mejor opción

### ✅ Preservación de Identidad

- **Face Swap automático** después de generar
- Mantiene los rasgos faciales originales
- Opcional (se puede desactivar)

---

## 📖 Ejemplos de Uso

### Ejemplo 1: Cambiar Pose
```
Imagen: Foto de persona de pie
Prompt: "haz que esta persona esté bailando"
Resultado: Persona bailando (misma cara)
```

### Ejemplo 2: Mejorar Calidad
```
Imagen: Foto borrosa o baja resolución
Prompt: "mejora la calidad y resolución de la imagen"
Resultado: Imagen mejorada, más nítida, upscale
```

### Ejemplo 3: Cambiar Ropa
```
Imagen: Persona con ropa casual
Prompt: "cambia la ropa por un vestido elegante"
Resultado: Persona con vestido elegante (misma cara)
```

### Ejemplo 4: Múltiples Cambios
```
Imagen: Cualquier foto
Prompt: "mejora calidad, cambia ropa, y haz que sonría"
Resultado: Imagen mejorada con todos los cambios
```

---

## ⚙️ Opciones

### Número de Variaciones
- **1-2**: Rápido, para testing
- **4-6**: Recomendado (mejor balance)
- **8**: Máximo (más tiempo, más opciones)

**Default: 6 variaciones**

### Modo de Calidad
| Modo | Pasos | CFG | Denoise | Tiempo | Calidad |
|------|-------|-----|---------|--------|---------|
| ⚡ Rápido | 20 | 7.0 | 0.65 | ~40s | Buena |
| ⚖️ Equilibrado | 30 | 8.0 | 0.70 | ~60s | Muy buena |
| 🎨 Alta Calidad | 40 | 9.0 | 0.75 | ~80s | Excelente |
| 💎 Máxima Calidad | 50 | 11.0 | 0.80 | ~100s | Perfecta |

**Default: Alta Calidad (40 pasos)**

### Preservar Cara Original
- **Activado** (recomendado): Mantiene tu cara original
- **Desactivado**: Usa la cara generada por SD

### Auto-Mejorar
- **Activado** (recomendado): Mejora automática de calidad
- **Desactivado**: Sin mejoras automáticas

---

## 🔧 Requisitos

### Modelos Necesarios

| Modelo | Tamaño | Uso |
|--------|--------|-----|
| **FLUX Fill** | ~20GB | Generación principal (recomendado) |
| **ControlNet OpenPose** | ~1.5GB | Cambiar poses |
| **ControlNet Tile** | ~1.5GB | Upscale/mejora calidad |
| **IP-Adapter** | ~0.7GB | Mantener identidad |
| **CLIPSeg** | ~0.6GB | Detección automática de ropa |

### Instalación

```bash
# 1. Asegúrate de tener ComfyUI instalado
cd D:/PROJECTS/AUTOAUTO
python ui/tob/ComfyUI/main.py

# 2. Instala modelos necesarios (desde ComfyUI Manager)
- ControlNet OpenPose
- ControlNet Tile
- IP-Adapter
- CLIPSeg (transformers)

# 3. Para FLUX (opcional pero recomendado)
pip install diffusers
pip install flux
```

---

## 🎯 Flujo de Trabajo Interno

### Paso 1: Análisis del Prompt
```python
analysis = analyze_prompt("haz que esté bailando")
# Resultado: {use_openpose: True, use_tile: False, ...}
```

### Paso 2: Intentar FLUX (si disponible)
```python
if flux_available:
    result = flux.generate(prompt, image, ...)
    # FLUX es más rápido y mejor calidad
```

### Paso 3: Fallback a ComfyUI
```python
if not flux or failed:
    workflow = build_intelligent_workflow(analysis)
    result = comfyui.queue(workflow)
```

### Paso 4: Restaurar Cara (Face Swap)
```python
if face_preserve:
    final_image = restore_face(original, generated)
```

---

## 📊 Comparativa

### VS SD Editor Antiguo

| Característica | Antiguo | Nuevo |
|----------------|---------|-------|
| Interfaz | Compleja (20+ controles) | Simple (5 controles) |
| Inpaint | Manual (dibujar máscara) | Automático (CLIPSeg) |
| Análisis prompt | No | Sí (IA decide) |
| Generación múltiple | No | Sí (1-8 variaciones) |
| Presets | No | Sí (6 acciones rápidas) |
| Face Swap | Opcional manual | Automático |

### VS Otras Herramientas

| Herramienta | Ventaja SD Editor |
|-------------|-------------------|
| Photoshop | Automático vs Manual |
| Online AI | Local (privacidad) |
| Otros SD | Más simple + Face Swap |

---

## 🐛 Troubleshooting

### Error: "ComfyUI no conectado"
**Solución:** Inicia ComfyUI primero
```bash
python ui/tob/ComfyUI/main.py
```

### Error: "No hay checkpoints disponibles"
**Solución:** Instala modelos en ComfyUI
- Usa ComfyUI Manager
- Descarga checkpoints de Civitai

### Error: "CLIPSeg no disponible"
**Solución:** Instala transformers
```bash
pip install transformers
```

### Generación muy lenta
**Solución:** 
1. Usa modo "Rápido" (15 pasos)
2. Reduce número de variaciones
3. Usa FLUX si está disponible

### Resultados no preservan cara
**Solución:**
1. Activa "Preservar Cara Original"
2. Asegúrate que haya una cara detectable
3. Usa IP-Adapter si está disponible

---

## 💡 Consejos

### Para Mejores Resultados

1. **Sé específico en el prompt:**
   - ✅ "haz que esté bailando salsa, movimientos dinámicos"
   - ❌ "bailando"

2. **Usa los presets como base:**
   - Click en preset → Edita el prompt → Genera

3. **Genera 4 variaciones:**
   - Más opciones para elegir
   - Tiempo razonable (~2 minutos)

4. **Activa Auto-Mejorar:**
   - Mejora calidad automáticamente
   - Sin costo adicional significativo

### Prompts que Funcionan Bien

```
# Cambios de pose
"haz que esté bailando, pose dinámica, movimiento"
"cámbialo a sentado cómodamente, pose relajada"
"ponla de pie, pose natural, confiada"

# Mejoras de calidad
"mejora la calidad y resolución, más detalle, 4K"
"hazla más nítida, mejora iluminación, profesional"

# Cambios de ropa
"cambia la ropa por un vestido de noche elegante"
"ponle un traje formal, corbata, ejecutivo"
"haz que esté desnuda, cuerpo natural, realista"

# Expresiones faciales
"haz que sonría, expresión feliz, alegre"
"cambia a expresión seria, mirada intensa"
```

---

## 📁 Archivos

| Archivo | Función |
|---------|---------|
| `ui/tabs/img_editor_tab.py` | Interfaz de usuario |
| `roop/img_editor/img_editor_manager.py` | Lógica inteligente |
| `roop/img_editor/flux_client.py` | Cliente FLUX |
| `roop/img_editor/clothing_segmenter.py` | CLIPSeg (ropa) |
| `roop/img_editor/comfy_workflows.py` | Workflows ComfyUI |

---

## 🎯 Estado Actual

### ✅ Implementado
- [x] Análisis automático de prompts
- [x] Detección de poses (keywords)
- [x] Detección de mejora de calidad (keywords)
- [x] Detección de cambios de ropa (CLIPSeg)
- [x] Generación múltiple (1-8 variaciones)
- [x] UI simplificada (sin presets)
- [x] Face Swap automático
- [x] Calidad alta por defecto (40 pasos)
- [x] 6 variaciones por defecto
- [x] Soporte FLUX (prioritario)
- [x] Fallback a ComfyUI

### ⏳ Pendiente (Futuro)
- [ ] ControlNet OpenPose real (actual: keywords)
- [ ] ControlNet Tile real (actual: keywords)
- [ ] Reescritor de prompts con LLM
- [ ] Vista previa de pose objetivo
- [ ] Editor de máscara manual (opcional)

---

**Versión**: 2026.1
**Última actualización**: Marzo 2026
**Estado**: ✅ Funcional (con limitaciones)

---

## 📞 Soporte

Para issues o preguntas:
1. Verifica que ComfyUI esté corriendo
2. Revisa los logs en consola
3. Asegúrate de tener los modelos instalados
