# AUTO-DEEP v2.2.2 - Guía de FaceSwap y SD Tab

## 🆘 Solución a Problema Crítico: "No faces detected in this frame"

**Última actualización**: 8 de Febrero de 2026 (16:55)

**IMPORTANTE**: Esta versión tiene SOLO 3 modos de FaceSwap:
1. **All** - Procesa todas las caras
2. **Selected Faces** - Selección manual de la cara de cada imagen
3. **Selected Faces Frame** - Tracking de video con selección inicial

**Cambios recientes (v2.2.2 - Modos corregidos):**
1. **`ui/tabs/faceswap_tab.py`**: 
   - Eliminado modo "Selected" (singular) - solo 3 modos ahora
   - "Selected faces" = Selección manual (muestra galería para que usuario elija)
2. **`roop/ProcessMgr.py`**: 
   - Procesa la cara guardada en `selected_face_references` por el usuario
3. **`docs/FACESWAP_SD_TAB_GUIDE.md`**: Documentación actualizada

---

## 🎭 Modos de FaceSwap

### 1. Mode "All" (Automático)
- **Comportamiento**: Detecta TODAS las caras automáticamente
- **Target**: Todas las caras en la imagen/video
- **Source**: Cara de origen más grande/del primer faceset
- **Uso**: Cuando no necesitas controlar qué cara se cambia

### 2. Mode "Selected Faces" (Selección Manual por Imagen)
- **Comportamiento**: El usuario selecciona manualmente la cara de CADA imagen
- **Target**: Cara seleccionada por el usuario de la galería
- **Source**: **Aleatorio** del faceset de origen
- **Flujo**:
  1. El sistema detecta todas las caras en la imagen
  2. **Muestra una galería** con las caras detectadas (si hay múltiples caras)
  3. El usuario **selecciona manualmente** cuál cara quiere cambiar
  4. Sistema guarda la selección en `selected_face_references`
  5. ProcessMgr usa esa selección para hacer swap
- **Uso**: Cuando necesitas control preciso sobre qué cara se cambia en cada imagen

### 3. Mode "Selected Faces Frame" (Tracking de Video con Selección Inicial)
- **Comportamiento**: El usuario selecciona una cara del PRIMER frame (si hay múltiples)
- **Target**: Cara seleccionada + tracking durante todo el video
- **Source**: **MISMA CARA** para todo el video (la más grande del faceset)
- **Flujo**:
  1. El sistema detecta todas las caras en el frame 1
  2. **Muestra una galería** con las caras detectadas (si hay múltiples)
  3. El usuario **selecciona manualmente** la cara a trackear
  4. Sistema guarda embedding y posición
  5. En cada frame, busca la cara que coincida (IoU + embedding)
  6. Si se pierde, intenta re-adquisición con embedding original
- **Uso**: Videos donde necesitas consistencia de la misma cara en todo el video

---

## 🔄 Proceso de FaceSwap

### Flujo Principal

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Source Faceset │────▶│  Face Detection  │────▶│  Target Selection│
│  (144 faces)    │     │  (get_all_faces) │     │  (por modo)     │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Result Image   │◀────│  Paste Back      │◀────│  Inswapper      │
│  (output/)      │     │  (simple blend)  │     │  (warp + blend) │
└─────────────────┘     └──────────────────┘     └─────────────────┘
```

### Paso 1: Detección de Caras
- **Archivo**: `roop/face_util.py`
- **Función**: `extract_face_images()` o `get_all_faces()`
- **Proveedores**: CUDAExecutionProvider (GPU) o CPUExecutionProvider
- **Threshold**: `max(0.1, distance_threshold * 0.5)` (más permisivo)
- **Nuevo atributo**: `normed_embedding` - Versión normalizada del embedding facial (L2) para matching

### Paso 2: Selección de Target (Modo Selected Faces)
- **Archivo**: `ui/tabs/faceswap_tab.py`
- **Función**: `on_use_face_from_selected()`
- **Flujo**:
  1. Detecta todas las caras
  2. Muestra galería para selección manual
  3. Usuario selecciona una cara
  4. Guarda en `selected_face_references[key]`

```python
# Cuando usuario selecciona una cara manualmente
video_key = f"selected_face_ref_{os.path.basename(file_path)}"
roop.globals.selected_face_references[video_key] = {
    'bbox': face_obj.bbox,
    'embedding': face_obj.embedding,
    'face_obj': face_obj
}
```

### Paso 3: Procesamiento en ProcessMgr
- **Archivo**: `roop/ProcessMgr.py`
- **Función**: `process_frame()`

```python
if face_swap_mode == 'selected_faces':
    # Busca la cara seleccionada por el usuario
    filename = os.path.basename(file_path)
    video_key = f"selected_face_ref_{filename}"
    
    if hasattr(roop.globals, 'selected_face_references'):
        if video_key in roop.globals.selected_face_references:
            face_ref_data = roop.globals.selected_face_references[video_key]
            target_face = face_ref_data.get('face_obj')
```

### Paso 4: Selección de Source
- **Archivo**: `roop/ProcessMgr.py`
- **Función**: `_select_source_face()`

```python
if face_swap_mode == 'selected_faces_frame':
    # Video: MISMA CARA para todo el video (la más grande)
    source_face = max(candidate_faces, key=lambda f: area)
else:
    # Imagen (selected_faces): ALEATORIO
    source_face = random.choice(candidate_faces)
```

**Resumen:**
- `"selected_faces"`: Source = aleatorio, Target = usuario selecciona manualmente
- `"selected_faces_frame"`: Source = cara más grande, Target = tracking
- `"all"`: Source = cara más grande, Target = todas las caras

### Paso 5: Inswapper (Face Swap)
- **Archivo**: `roop/processors/FaceSwap.py`
- **Función**: `Run()`
- **Modelo**: `inswapper_128.onnx`

### Paso 6: Paste Back (Blending)
- **Archivo**: `roop/processors/FaceSwap.py`
- **Función**: Uso directo de `paste_back=True` del modelo inswapper

---

## ⚙️ Configuración

### Variables Globales (roop.globals)

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ui_blend_ratio` | 0.95 | Intensidad del blend (0.1-1.0) |
| `distance_threshold` | 0.6 | Threshold para matching de caras |
| `mouth_open_threshold` | 0.35 | Ratio para detectar boca abierta |
| `use_enhancer` | True | Usar GFPGAN |
| `face_swap_mode` | 'selected_faces' | Modo de swap |

---

## 🔧 Solución de Problemas

### Problema: "No faces detected in this frame"

**Causas posibles**:
1. Threshold demasiado alto
2. Imagen muy oscura o borrosa
3. Cara parcialmente oculta

**Solución**:
```python
# En roop/face_util_rotation.py
# El threshold efectivo ya es: max(0.1, distance_threshold * 0.5)
```

### Problema: Selected Faces no guarda la selección

**IMPORTANTE**: Para que "Selected Faces" funcione correctamente, el usuario DEBE:

1. **Hacer clic en "Use Face from this Frame"** para cada archivo
   - Esto detecta las caras y muestra la galería
   - Guarda automáticamente la primera cara como referencia

2. **Si hay múltiples caras, seleccionar manualmente** cuál cambiar
   - El usuario debe hacer clic en la cara deseada de la galería
   - Luego hacer clic en "Usar Seleccionada"

3. **Para cada imagen/archivo, repetir el proceso**
   - Navegar al siguiente archivo
   - Hacer clic en "Use Face from this Frame"
   - Seleccionar la cara deseada si hay varias

**Verificación**:
```bash
# Verificar que selected_face_references tiene datos
python -c "
import roop.globals
if hasattr(roop.globals, 'selected_face_references'):
    print('Referencias guardadas:', len(roop.globals.selected_face_references))
    for k, v in roop.globals.selected_face_references.items():
        print(f'  {k}: bbox={v.get(\"bbox\")}')
else:
    print('No hay referencias guardadas')
"
```

### Problema: Selected Faces Frame no hace tracking correcto

**Causas posibles**:
1. No se seleccionó la cara del primer frame correctamente
2. La cara se mueve mucho entre frames
3. Hay múltiples caras similares en el video

**Solución**:
1. Asegurarse de seleccionar la cara del PRIMER frame
2. Usar "Use Face from this Frame" en el frame 1
3. Verificar que el tracking esté activo en los logs:
```bash
[DEBUG] [TRACK] Primer frame: cara seleccionada, score=0.xx
[DEBUG] [TRACK] Frame XX: cara encontrada (score=0.xx)
```

### Problema: Face swap en cara wrong

**Causas posibles**:
1. Modo "selected_faces" no encuentra la cara correcta
2. IoU threshold muy bajo

**Solución**:
```python
# En roop/ProcessMgr.py, línea 368-369
if best_iou > 0.3:  # Aumentar a 0.5 si hay falsos positivos
```

---

## 📋 Checklist de Verificación

Antes de procesar, verificar que:

### Para "Selected Faces" (imágenes):
- [ ] "Use Face from this Frame" se ha pulsado para cada imagen
- [ ] Si hay múltiples caras, se ha seleccionado manualmente
- [ ] La galería muestra las caras detectadas
- [ ] Logs muestran: `[SELECTED_FACES] Cara guardada para xxx.jpg`

### Para "Selected Faces Frame" (videos):
- [ ] Se ha seleccionado la cara del PRIMER frame (si hay múltiples caras)
- [ ] "Use Face from this Frame" se ha pulsado en frame 1
- [ ] Si hay múltiples caras, se ha seleccionado manualmente de la galería
- [ ] Solo UNA cara se procesa en cada frame (tracking activo)
- [ ] Logs muestran: `[SELECT_SOURCE] Modo selected_faces_frame`

### Para "All" (automático):
- [ ] No se requiere selección manual
- [ ] Todas las caras detectadas se procesan

---

## 📁 Archivos Clave

| Archivo | Función |
|---------|---------|
| `roop/ProcessMgr.py` | Gestor de procesamiento principal |
| `roop/processors/FaceSwap.py` | Lógica de face swap |
| `roop/face_util.py` | Detección de caras y extracción |
| `roop/types.py` | Definición de tipos (incluye clase Face) |
| `ui/tabs/faceswap_tab.py` | Interfaz de usuario FaceSwap |

---

## 📞 Debug Mode

Para debug avanzado, activar logs:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

O buscar en los archivos de log:
- Consola de VS Code
- `output/` directory
