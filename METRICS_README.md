# 📊 PANEL DE MÉTRICAS EN TIEMPO REAL

## 🎯 Descripción

Panel visual que muestra información en tiempo real durante el procesamiento de videos:

- **Progreso**: Porcentaje completado
- **Tiempo Restante**: Estimación basada en velocidad actual
- **FPS**: Frames por segundo procesados
- **Frames**: Contador de frames procesados/totales

---

## 🎨 Vista del Panel

```
┌─────────────────────────────────────────────────────────┐
│  📊 Métricas en Tiempo Real                             │
├──────────────┬──────────────┬──────────────┬───────────┤
│   Progreso   │ Tiempo Rest. │     FPS      │  Frames   │
│    45.2%     │    02:34     │     3.6      │  135/300  │
└──────────────┴──────────────┴──────────────┴───────────┘
[████████████████░░░░░░░░░░░░░░░░░░░░] 45%
```

---

## 📊 Métricas Mostradas

### 1. Progreso (%)
- **Rango**: 0% - 100%
- **Colores**:
  - 0-25%: 🔵 Azul
  - 25-50%: 🔵 Cyan
  - 50-75%: 🟢 Verde
  - 75-100%: 🟠 Naranja

### 2. Tiempo Restante
- **Formato**: MM:SS o HH:MM:SS
- **Cálculo**: Basado en FPS actual y frames restantes
- **Precisión**: ±10 segundos después del 25% de progreso

### 3. FPS (Frames Por Segundo)
- **Tipo**: Promedio móvil (últimos 30 frames)
- **Rango típico**:
  - GPU 4GB: 2-4 FPS
  - GPU 8GB: 5-8 FPS
  - GPU 12GB+: 10-15 FPS

### 4. Contador de Frames
- **Formato**: procesados/totales
- **Ejemplo**: 135/300

---

## 🔧 Implementación Técnica

### Archivos Creados

1. `roop/metrics_tracker.py` - Clase MetricsTracker
2. Panel HTML en `ui/tabs/faceswap_tab.py`

### Uso en Código

```python
from roop.metrics_tracker import MetricsTracker, set_current_tracker

# Inicializar
metrics = MetricsTracker(total_frames=300)
set_current_tracker(metrics)
metrics.start()

# Actualizar en cada frame
metrics.update_frame_processed(frame_num=50)

# Obtener HTML actualizado
html = metrics.get_progress_html()
```

---

## 📈 Algoritmo de Estimación

### Tiempo Restante

```python
elapsed = tiempo_transcurrido
progress = frames_procesados / total_frames

if progress > 0:
    total_estimado = elapsed / progress
    restante = total_estimado - elapsed
```

### FPS (Promedio Móvil)

```python
# Mantener últimos 30 FPS
fps_history.append(instant_fps)
if len(fps_history) > 30:
    fps_history.pop(0)

fps_actual = sum(fps_history) / len(fps_history)
```

---

## 🎯 Beneficios

### Para el Usuario

1. ✅ **Sabe cuánto falta** - No hay sorpresas
2. ✅ **Planifica su tiempo** - Puede hacer otras cosas
3. ✅ **Monitorea rendimiento** - Detecta cuellos de botella
4. ✅ **Experiencia profesional** - Se siente como software premium

### Para Debugging

1. ✅ **Identifica problemas** - FPS muy bajos = problema
2. ✅ **Optimiza configuración** - Ajusta batch_size según FPS
3. ✅ **Compara hardware** - FPS en diferentes GPUs

---

## 🎨 Personalización

### Colores del Panel

El panel usa un gradiente moderno:

```css
background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
```

### Colores de Progreso

| Progreso | Color | Hex |
|----------|-------|-----|
| 0-25% | Azul | `#3b82f6` |
| 25-50% | Cyan | `#06b6d4` |
| 50-75% | Verde | `#10b981` |
| 75-100% | Naranja | `#f59e0b` |

---

## 📊 Ejemplo de Logs

```
Processing video: my_video.mp4 (0-300/300 frames)
[BATCH] Configuración: batch_size=4, max_workers=4
[BATCH] Leyendo frames para procesamiento en paralelo...
[BATCH] 300 frames cargados en memoria
[BATCH] Iniciando procesamiento paralelo con 4 hilos...
Procesando frames (BATCH): 45%|████░░░░░░| 135/300 [00:37<02:34, 3.60 fps]
[BATCH] Procesamiento completado en 83.45s (3.60 fps)
[BATCH] Velocidad: 0.12x tiempo real
```

---

## 🔍 Troubleshooting

### Métricas no se actualizan

**Causa:** Tracker no inicializado

**Solución:**
```python
from roop.metrics_tracker import set_current_tracker
set_current_tracker(metrics)
```

### Tiempo restante incorrecto

**Causa:** Muy pronto en el procesamiento

**Solución:** Esperar al menos 25% de progreso para estimación precisa

### FPS muestran 0

**Causa:** Primer frame aún no procesado

**Solución:** Normal al inicio, se actualiza después del primer frame

---

## 💡 Consejos

1. **Esperar 25%** antes de confiar en tiempo restante
2. **Monitorear FPS** para ajustar batch_size
3. **Si FPS < 1**, reducir batch_size a 2
4. **Si FPS > 5**, aumentar batch_size para más velocidad

---

## 🚀 Mejoras Futuras

### Posibles Adiciones

- [ ] Gráfico de FPS en tiempo real
- [ ] Uso de VRAM/memoria
- [ ] Temperatura de GPU
- [ ] Estimación de tamaño de archivo
- [ ] Comparativa con tiempo real (0.5x, 1x, 2x)

---

## 📋 Integración con Otras Mejoras

### Batch Processing

El panel muestra FPS real del procesamiento en paralelo:

```
FPS: 4.2 (con batch_size=4)
```

### Detección de Boca

El tiempo de procesamiento de MediaPipe se refleja en FPS:

```
Sin MediaPipe: FPS 5.0
Con MediaPipe: FPS 4.2 (16% más lento)
```

---

**Fecha**: Marzo 2025  
**Versión**: 2025.1  
**Estado**: ✅ Implementado y Listo para Usar
