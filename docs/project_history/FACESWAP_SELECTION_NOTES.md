# Faceswap Tab - Mejoras y Notas

## Uso de Selección de Caras

### Cómo seleccionar una cara en la galería de selección

1. **Galería superior** → Muestra las caras detectadas en el frame actual
2. **Slider** → Mover a la posición de la cara que quieres seleccionar (1, 2, 3, etc.)
3. **Botón [OK]** → Confirmar la selección y añadir la cara a "CARAS DE DESTINO"

### Por qué no click directo en la galería

**Problema conocido de Gradio**: Cuando la galería se actualiza dinámicamente varias veces, los eventos de click en las miniaturas dejan de funcionar correctamente.

Síntomas:
- El click funciona en las primeras 1-2 imágenes
- Después de varias actualizaciones, el click ya no dispara el evento
- El problema ocurre porque Gradio pierde los event handlers cuando se regenera el DOM de la galería

### Solución implementada

Se usa el patrón **Slider + Botón [OK]**:
- Slider selecciona la cara (sin confirmar automáticamente)
- Botón [OK] confirma y añade la cara

Este método es más estable y confiable que el click directo.

### Alternativas consideradas pero rechazadas

1. **allow_preview=True**: Funciona pero abre un modal grande que no es deseado
2. **Botones por cara**: Problemas similares de eventos
3. **HTML personalizado**: Complejo de actualizar dinámicamente

## Código relevante

- `faceswap_tab.py` - FaceSwapTab
- `face_selection` - Galería de caras detectadas
- `face_selector_slider` - Slider de selección
- `bt_use_selected_face` - Botón de confirmación

## Historial de cambios

- 2026-04-23: Implementado slider + botón como solución estable al problema de click en galería