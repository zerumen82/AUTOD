# Validación pendiente del código del 18/10/2026

Las comprobaciones de código no certifican la calidad de un modelo. Esta lista
recoge las pruebas reales que faltan para aceptar los flujos abiertos. No
interrumpir ni reiniciar un proceso con trabajo activo.

- [ ] **Generate / Chroma:** render con el workflow corregido, T5 único y CFG
  real de configuración. Comprobar prompt, anatomía, color y controles CFG /
  negativo. Las imágenes anteriores al fix usan otro workflow y no lo validan.
- [ ] **Cámara:** probar ángulo alto/bajo, detrás-izquierda, primer plano y plano
  general sobre una imagen adecuada. Comprobar movimiento real y conservación
  del contenido. Un encuadre ya coincidente puede no moverse; el LoRA no tiene
  una pose de primera persona real entre sus 96 combinaciones.
- [ ] **Editor / Kontext:** una instrucción compuesta completa, cambio de prenda
  con corte exacto, retirada de objeto completo y apariencia en varias fotos y
  semillas. Confirmar que lo no pedido se conserva. Entrada de imagen normal,
  sin lienzo de pintado de máscaras.
- [ ] **Editor / LongCat elegido explícitamente:** apariencia/recolor compuestos
  y varias personas, comprobando que no aparece una cara nueva, que la
  restauración no borra cambios anteriores y que cancelar detiene el reintento.
  Los contratos simulan la inferencia y no sustituyen esta prueba.
- [ ] **FaceSwap:** perfiles/oclusiones difíciles y vídeos representativos
  completos, incluidos cortes y readquisición. Comparar identidad fuente,
  proporciones, borde, boca, nitidez y otras personas. Los clips previos sin
  metadata final y los contratos CPU no cierran estos criterios.
- [ ] **Android:** probar la app instalada contra `/edit` o `/edit_options` y
  Generate; carga, resultado y controles. Los contratos 8/9 y 10 parámetros
  conservados no equivalen a esta prueba en teléfono.

Guardar muestras con nombres nuevos, sus originales y logs de la versión
probada. Si una prueba descubre un defecto reproducible del código, se reabre
la implementación correspondiente; no se marca como aprobada por haber
terminado el render.
