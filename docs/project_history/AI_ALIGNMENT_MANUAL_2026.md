# 📘 Manual de Arquitectura y Mejoras (Abril 2026)
### Para uso de Agentes IA y Desarrolladores

Este documento resume la reingeniería aplicada al proyecto para asegurar que futuras intervenciones mantengan la estabilidad y no rompan las mejoras críticas implementadas.

---

## 1. 👤 Pestaña FaceSwap (Arquitectura Modular)
**Ubicación**: `ui/tabs/faceswap/`
*   **Estado**: Modularizado (anteriormente un archivo de 4600 líneas).
*   **Archivos clave**:
    *   `state.py`: Control de flags. **IMPORTANTE**: No quitar `_IS_UPDATING_GALLERY`. Se usa para bloquear clics accidentales durante la renderización de Gradio.
    *   `events.py`: Gestión de eventos. El `onclick` ahora es directo y aditivo (permite multiselección).
    *   `logic.py`: Procesamiento de IA. Incluye el Smart Tracking por embeddings.

**Mejoras de IA aplicadas**:
*   **Detección en Bordes/Verticales**: `roop/face_util.py` ahora usa `det_size=(640, 640)` y `det_thresh=0.15`. **NO VOLVER A 320px**, o se perderán caras en fotos verticales.
*   **Smart Tracking**: En vídeos (`Selected Faces Frame`), se comparan embeddings de caras para mantener la estabilidad identidad frame a frame.

---

## 2. 🎨 Image Editor (Estilo "Grok Imagine")
**Ubicación**: `ui/tabs/img_editor_tab.py` y `roop/img_editor/`
*   **Integración Moondream 2**: Usa `scripts/moondream_analyzer.py` para obtener descripciones visuales detalladas antes de editar.
*   **Analizador Semántico**: `img_editor_manager.py` deduce la intensidad del cambio.
    *   *Intensidad Alta* (Posturas/Acciones) → Sube Denoise automáticamente (0.92+).
    *   *Intensidad Media* (Ropa) → Denoise equilibrado (0.88).

**Flujo Crítico**:
*   **Pasada de Identidad**: Tras generar la imagen (pose nueva), el sistema ejecuta una **segunda pasada automática de FaceSwap** sobre todas las caras detectadas para que el cambio de ropa o pose no altere el parecido facial.

---

## 3. 🎬 Próxima Fase: Animate Image (Plan de IA)
Para futuras sesiones o agentes, el plan aprobado es:
1.  **Modularización**: Copiar la estructura de FaceSwap (`ui/tabs/animate/`).
2.  **Face Stability 2.0**: Implementar la restauración de identidad frame a frame tras la animación para evitar el efecto "cara derretida".
3.  **Masked Animation**: Integrar CLIPSeg para que el usuario pueda animar solo partes específicas (ej. solo el agua, solo el pelo).
4.  **Limpieza de VRAM**: Asegurar que `_cleanup_temp_and_vram()` se llame antes de cargar modelos de vídeo pesados.

---

**Nota para Agentes**: Respetad el flag `_IS_UPDATING_GALLERY` en las galerías de Gradio; de lo contrario, se dispararán eventos infinitos que congelarán la interfaz.
