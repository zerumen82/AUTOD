# SD Editor - Progreso y Estado

## NUEVO: Inpaint Selectivo con CLIPSeg

### El problema anterior

El editor de imágenes con SD 1.5 **NO funcionaba bien para cambiar ropa** porque:

1. **Img2Img no es selectivo**: Modifica toda la imagen, no solo las áreas que quieres cambiar
2. **No había máscara automática**: Sin máscara, no se puede especificar qué áreas editar
3. **ControlNet/IP-Adapter preservaban demasiado**: No permitían cambios de cuerpo

### La solución: CLIPSeg + Inpaint Selectivo

Ahora el sistema:

1. **Detecta automáticamente la ropa** con CLIPSeg (modelo de segmentación por texto)
2. **Genera una máscara** de las áreas de ropa
3. **Aplica inpaint SOLO en esas áreas** (no modifica cara, piel, fondo)
4. **Restaura la cara original** con face swap

---

## Cómo usar el nuevo sistema

### Paso 1: Subir imagen y escribir prompt

```
Prompt: "nude woman, natural body, realistic skin, high quality"
Negative: "clothes, dressed, low quality, blurry, censored"
```

### Paso 2: Configurar opciones

| Opción | Valor recomendado | Descripción |
|--------|-------------------|-------------|
| **Detección Automática de Ropa** | ✅ ACTIVADO | Detecta ropa con CLIPSeg |
| **Sensibilidad de Detección** | 0.5 | 0.2=más área, 0.8=solo ropa obvia |
| **Preservar Cara** | ✅ ACTIVADO | Restaura cara original |
| **Usar FLUX** | ✅ ACTIVADO | Mejor calidad |

### Paso 3: Preview de máscara (opcional)

Click en **"👁️ Ver Máscara"** para ver qué áreas se modificarán:
- **Rojo** = área a modificar (ropa detectada)
- **Normal** = área a preservar

### Paso 4: Generar

Click en **"🎨 Generar"** y esperar el resultado.

---

## Arquitectura del Sistema

```
[Imagen Original] 
    --> [CLIPSeg: detectar "clothing, shirt, dress, pants..."] 
    --> [Máscara automática + exclusión de piel] 
    --> [Inpaint con FLUX/SD SOLO en áreas detectadas] 
    --> [Face Swap: restaurar cara original] 
    --> [Resultado final]
```

---

## Archivos Principales

| Archivo | Función |
|---------|---------|
| `ui/tabs/img_editor_tab.py` | Interfaz de usuario con nuevas opciones |
| `roop/img_editor/img_editor_manager.py` | Lógica de generación selectiva |
| `roop/img_editor/clothing_segmenter.py` | **NUEVO** - Detección de ropa con CLIPSeg |
| `roop/img_editor/comfy_workflows.py` | Workflows de ComfyUI |
| `roop/img_editor/flux_client.py` | Cliente FLUX/SD con inpaint |

---

## Modelos Requeridos

### Para Inpaint Selectivo
- **CLIPSeg** (`CIDAS/clipseg-rd64-refined`) - Se descarga automáticamente de HuggingFace
- **FLUX Fill** o **SD Inpaint** - Para generar el inpaint

### Para Preservar Cara
- **InsightFace** - Análisis facial
- **Inswapper** - Face swap

---

## Instalación de CLIPSeg

```bash
pip install transformers torch pillow
```

El modelo CLIPSeg se descarga automáticamente la primera vez que se usa (~400MB).

---

## Guía de Uso por Tipo de Edición

### Para cambiar ropa/desnudos:
```
✅ Detección Automática de Ropa: ACTIVADO
✅ Preservar Cara: ACTIVADO
Sensibilidad: 0.5
Strength: 0.85-0.95
```

### Para cambiar entorno/fondo:
```
❌ Detección Automática de Ropa: DESACTIVADO
✅ ControlNet: ACTIVADO (en Modo Avanzado)
Strength: 0.6-0.8
```

### Para retoques manteniendo identidad:
```
❌ Detección Automática de Ropa: DESACTIVADO
✅ ControlNet: ACTIVADO
✅ IP-Adapter: ACTIVADO
Strength: 0.4-0.6
```

---

## Estado de los Componentes

| Componente | Estado | Notas |
|------------|--------|-------|
| CLIPSeg | ✅ Funciona | Detección automática de ropa |
| Inpaint Selectivo | ✅ Funciona | FLUX/SD + máscara |
| Face Swap | ✅ Funciona | CPU análisis, CUDA swap |
| ControlNet | ✅ Funciona | Tile + SoftEdge |
| IP-Adapter | ✅ Funciona | PLUS para SD 1.5 |

---

## Conclusión

El editor ahora tiene **dos modos de operación**:

1. **Inpaint Selectivo** (nuevo): Para cambiar ropa/desnudos
   - Detecta ropa automáticamente
   - Modifica SOLO las áreas detectadas
   - Preserva cara y piel

2. **Img2Img con ControlNet/IP-Adapter**: Para otros tipos de edición
   - Cambiar entorno/fondo
   - Retoques sutiles
   - Mantener estructura/identidad
