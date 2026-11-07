# Validación pendiente — contraste del 29/10/2026

Las comprobaciones de código no certifican la calidad de un modelo. Esta lista
recoge las pruebas reales que faltan para aceptar los flujos abiertos. No
interrumpir ni reiniciar un proceso con trabajo activo. El usuario autoriza
la GPU para verificar el 25/10, tras comprobar disponibilidad y cola. Las
ejecuciones nuevas se registran en MEMORY.md; terminar no implica PASS visual.

Los logs `logs/ui_start_20261025_000432.log` y
`logs/ui_start_20261025_003450.log` registran renders Kontext terminados en
169,81 s y 157,29 s, respectivamente, con región focal y modelo ejecutado.
También registran una salida Generate y un lote FaceSwap de 45 archivos en
4 min 33 s. Ya no se considera que falte toda ejecución posterior al fix;
falta inspección visual de las entregas y cobertura de los casos siguientes.

Pruebas nuevas autorizadas25/10: Kontext global sobre astronaut
`1792952153054322000` PASS acotado de calvicie sin halo luminoso evidente;
Grace `1792952455112538200` PASS de azul pero añade bob/flequillo, sin certificar
corte exacto. Root y tester inspeccionan entregas: ambas con exterior0. Chroma
`diagnostics/out/chroma_real_20261025/1792952781341099100/` PASS acotado de
workflow y escena neutra (tetera roja/taza azul),768x512/30steps/CFG3.8/T5único.
No sustituyen variedad de casos ni aceptación visual del usuario.

- [ ] **Generate / Chroma:** verificar un render con el workflow corregido,
  T5 único y CFG real de configuración; comprobar prompt, anatomía, color y
  controles CFG/negativo. `generate/gen_1792880307.png` corresponde a
  `flux_dev_abliterated`, no a Chroma: este último empieza después en el log.
  La salida FLUX y las imágenes anteriores al fix no validan Chroma.
  El render nuevo indicado arriba sí acredita esa escena y el workflow;
  siguen pendientes anatomía, otros prompts y efecto de variar CFG/negativo.
- [ ] **Cámara:** probar ángulo alto/bajo, detrás-izquierda, primer plano y plano
  general sobre una imagen adecuada. Comprobar movimiento real y conservación
  del contenido. Un encuadre ya coincidente puede no moverse; el LoRA no tiene
  una pose de primera persona real entre sus 96 combinaciones.
- [ ] **Editor / Kontext:** una instrucción compuesta completa, cambio de prenda
  con corte exacto, retirada de objeto completo y apariencia en varias fotos y
  semillas. Confirmar que lo no pedido se conserva. Entrada de imagen normal,
  sin lienzo de pintado de máscaras.
- [ ] **Editor / política de motor y compatibilidad:** Kontext es el modelo
  requerido al elegirlo, salvo la excepción explícita de la ruta medida con
  visión autorizada posteriormente (MEMORY.md, 25–28/10). Sus alias antiguos
  resuelven a Kontext; esa compatibilidad no elimina la excepción registrada
  en `edit_execution.py`. Hay rechazos del modelo alternativo
  registrados el 25/10; verificar también entrega, estado visible y
  cancelación. En apariencia/recolor compuestos y varias personas, comprobar
  que no aparece una cara nueva ni se borran cambios anteriores. Los
  contratos con inferencia sustituida no acreditan estas propiedades visuales.
- [ ] **FaceSwap:** perfiles/oclusiones difíciles y vídeos representativos
  completos, incluidos cortes y readquisición. Comparar identidad fuente,
  proporciones, borde, boca, nitidez y otras personas. Los clips previos sin
  metadata final y los contratos CPU no cierran estos criterios. El lote de
  fotos terminado el 25/10 no equivale a validar un vídeo representativo.
  El vídeo normal usado en octubre ya no existe en su ruta registrada;
  solicitada ruta vigente. Prueba nueva CPU en retratos ejecuta modelos reales,
  sin certificar parecido ni perfiles; detalle y límites en MEMORY.md.
- [ ] **Android:** probar la app instalada contra `/edit` o `/edit_options` y
  Generate; carga, resultado y controles. Los contratos 8/9 y 10 parámetros
  conservados no equivalen a esta prueba en teléfono. La app actual es nativa
  Kotlin, no una WebView; conexión y progreso se prueban en sus propios controles.
  Build debug y lintDebug/lintRelease ya pasan (0errores,68/65avisos).
  ADB no detecta dispositivos: no se ha instalado ni probado en teléfono.

Guardar muestras con nombres nuevos, sus originales y logs de la versión
probada. Si una prueba descubre un defecto reproducible del código, se reabre
la implementación correspondiente; no se marca como aprobada por haber
terminado el render.
