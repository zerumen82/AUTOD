# spec/03 — Pendientes y deudas declaradas

## Cierre de sesión (webview → X) — límites medidos 2026-10-20

- Las tres rutas de cierre están probadas con ComfyUI real (propietario,
  secundario y `atexit`), pero **no se ha cerrado una ventana de webview
  pulsando la X en una UI real**: la verificación de esa ruta es el cableado
  estático de `roop/core.py` más la ejecución real de la misma función. Falta
  una prueba de clic en ventana servida por el usuario.
- La VRAM liberada por ComfyUI en los escenarios del probe es pequeña (−2 a
  −117 MB) porque ComfyUI arranca frío, sin workflows ejecutados. Con un
  ComfyUI con modelos cargados la liberación observada durante el desarrollo
  fue de **4638 → 754 MB**, pero esa medición no está en un informe del probe.
  El cierre del proceso garantiza la reclaim del SO; lo acreditado es puerto
  cerrado, PIDs fuera de `nvidia-smi` e informe con `vram_delta_mb`.
- El tenedor local sí está probado con memoria real: `_release_local_gpu()`
  con FaceAnalysis residente devuelve 594 MB.
- Si el cierre ocurre con la cola ocupada, la política espera
  `queue_grace_seconds` (15 s) y luego corta igual; si sigue un batch local de
  FaceSwap se le pide cancelación con `roop.globals.processing = False`. Ese
  caso de interrupción forzada **no se ha reproducido** en una prueba real.
- La UI de Gradio y el servidor HTTP no se detienen al cerrar la ventana: la
  consola de `run.py` sigue abierta por diseño y el proceso sigue vivo. Solo se
  liberan GPU y ComfyUI.

## Selección regional — límites medidos 2026-10-20

- La protección del vecino queda ACREDITADA en las dos direcciones sobre foto
  real (`head-2` y `head-0`): 0 px cambiados en las cabezas no seleccionadas,
  0 px de fuga gris, 0 px fuera de la máscara efectiva. No certifica que TODA
  foto con caras solapadas quede protegida: el cegado depende de las elipses
  medidas por el detector.
- La ADHERENCIA de Kontext sigue siendo PARCIAL: al pedir «short black hair and
  a short beard» la cabeza editada conserva algo de pelo rubio en la parte
  alta. La protección y la obediencia son cosas distintas; este trabajo
  acredita la primera, no la segunda.
- El contexto de cabeza ceñido (0,15) reduce la ambigüedad pero no la elimina
  cuando dos caras se solapan de verdad: de ahí el cegado, no del recorte.
- Sin validación en navegador por el usuario de la selección con el nuevo
  cegado: la prueba fue por contrato y por render real, no por clic en la UI.

## Selector visual — límites medidos 2026-10-19

- SAM2 local autorizado expresamente por «SI»; evaluación aislada en tres
  fotos y grilla más densa. No promovido: scores altos admiten cuchara bajo
  etiqueta plato y depósito parcial bajo etiqueta moto. CLIP semántico local
  tampoco distingue todas las propuestas. Una alternativa de cuchara revisada
  manualmente sirve solo para diagnóstico de render, no como regla automática.
  GT coffee_v1 tiene núcleo spoon_neck disputado: consultar ERRATA, conservar
  anotaciones/hashes y no usarlo para admisión. GrabCut sigue excluido.

- Backend, selección múltiple, UI/sesiones/carreras y API implementados con
  reviewer APPROVE y contratos verdes. No equivalen a PASS visual general.
- Café real `1792420385485144000`: máscara «coffee cup» bbox169,17,422,227
  omite cuerpo/asa; máscara «saucer» mide una parte de cuchara. El render crudo
  vuelve azules también el plato y el asa; el pegado protege el plato/resto
  pero solo cambia la franja superior de taza. Resultado parcial, sin tocar
  umbrales por foto. Fuente/máscaras/raw/final/informe tester bajo
  `diagnostics/out/editor_selection_20261019/`. La segmentación sigue pudiendo
  omitir parte del objeto o etiquetar mal una región; comprobar el resaltado.
- Cabeza real inicial `1792420763483413500`: referente y pelo rojo correctos, exterior
  0 px cambiados y otra persona protegida; flequillo más largo de lo pedido
  y leve cambio de expresión. Con geometría corregida `1792441547925018000`
  el pelo rojo tiene laterales más cortos; la expresión exacta sigue sin
  certificarse. No certifica todas las cláusulas ni identidad.
- Cuchara real inicial `1792421243545046400`: máscaras guardadas del análisis real
  del café, huella comprobada y una generación Kontext; conserva exterior,
  pero permanece plateada pese a pedir oro. No repetir igual receta/prompt
  esperando certificar cumplimiento sin una hipótesis nueva contrastada.
- Corrección comprobada de contexto de objetos: `1792441271736905200`
  (A/B contexto) y `1792441449254040300` (wrapper nativo) muestran cazoleta
  dorada con el soporte original/exterior intactos. El raw del A/B dora también
  el mango trasero; el pegado lo conserva plateado porque CLIPSeg lo omite.
  Contexto2,0 no arregla cobertura. Geometría7/7, backend15/15 y contexto4/4
  PASS; reviewer APPROVE del código, tester PASS técnico y visual PARCIAL.
  Guía5, T5FP8, variantes multiescala/alias no solucionaron el fallo completo.
- UI HTTP real y aislamiento/reutilización comprobados tras corregir ndarray
  de salida; servidor de diagnóstico cerrado. Falta validación en navegador
  por el usuario; no se instaló un navegador ni Playwright para esta tarea.
- No instalar ni recuperar segmentadores rechazados para ocultar estas deudas;
  una ampliación de diseño/modelos requiere la decisión explícita del usuario.
  Se mantiene pendiente su validación visual. No cerrar H-EDIT-SELECT por
  métricas de exterior o un resultado de inferencia sin comprobar contenido.

## Kontext — revisión real y protección focal, 2026-10-18

- Retirada activa de LongCat, receta configurada y protección automática focal
  comprobadas por contratos CPU, lint/compilación y reviewer APPROVE del código.
- La revisión original confirma referente equivocado: mujer convertida en hombre
  calvo y hombre original con pelo. No inferir personas reales de detecciones.
- Prueba real no explícita: ruta global previa cambió ambas personas; tras los
  fixes, CLIPSeg no identifica con seguridad el sujeto en cuatro variantes y se
  conserva la entrada (0 píxeles cambiados). La protección funciona, pero no es
  una edición cumplida. Detalle `diagnostics/out/kontext_review_20261018/REVIEW.md`.
- Localización visual más capaz con un modelo local ya instalado: diseño
  consultado al usuario; no añadir listas por palabras, excepciones por foto,
  modelos nuevos ni recuperar SAM2/GrabCut rechazados sin decisión explícita.
- Se mantiene abierta la calidad general y la validación del usuario. El
  detector de postura ligero también conserva falsos positivos incidentales;
  los contratos de protección no certifican ese clasificador.

## Estado vigente — completar código, 2026-10-18

- Cámara: corregidas las regresiones de altura/distancia/orientación; 13/13
  casos y 96/96 frases entrenadas, POV 10/10. Señal y discriminación en
  `config/camera_pose.json`. Revisión independiente APPROVE del código.
  Falta prueba visual de encuadres; las 96 poses del LoRA no incluyen primera
  persona real y no garantizan movimiento cuando la entrada ya tiene ese ángulo.
- Chroma: corregido el workflow heredado FLUX. Usa T5 único, `type=chroma`,
  opciones de tokenización y CFG real 3,8 configurado, con overrides conservados.
  FLUX/SDXL mantienen sus contratos. Falta render real del workflow corregido;
  las imágenes del 18/10 anteriores a este fix pertenecen al workflow antiguo.
- Traducción: corregido el falso positivo de cognados como
  `PENETRACION→penetration` mediante el tokenizador destino ya instalado;
  12/12 casos y fallback sin recurso. No certifica traducciones arbitrarias ni
  corrige por catálogo el género que produzca Marian.
- Editor: default Kontext en configuración independiente. La UI conserva
  `gr.Image`; **no se reintroduce pintado de máscaras**. Se retiró el lienzo
  añadido por interpretación de una deuda antigua tras el recordatorio del
  usuario. Endpoints escritorio/Android 13/9/8 siguen recibiendo Image;
  el caso Kontext sin selección recibe `mask_image=None`, modo global.
  La transferencia PNG evita la compresión WebP al reutilizar resultados.
- FaceSwap: H-FS-005 ya tenía el código implementado; SHA de ProcessMgr coincide
  con APPROVE del 17/10. Auditoría actual 12/12 probes, tracking 453/453 e
  integración 27/27; corregidos solo dos probes con warmup de modelo/aserto
  obsoleto. Evidencia `diagnostics/out/faceswap_code_audit_20261018/`.
  Siguen pendientes perfiles, parecido, oclusiones y vídeos completos. No se
  atribuye finalización a los A/B antiguos que carecen de metadata final.
- Batería final de los bloques anteriores 9/9: `logs/code_completion_final_suite_20261018.json`.
  Generate 108/108 y contrato legacy APK 31/31. Lint crítico de roop/ui (sin
  código externo ComfyUI) y compileall verdes. Esto verifica código y contratos,
  no la calidad visual ni una prueba instalada en teléfono.
- Experimentos históricos rechazados de SAM2/GrabCut/Hyperswap no se reactivan
  para rellenar casillas. Klein en Generate y bigASP siguen siendo ampliaciones
  pendientes de decisión. No instalar ni cambiar motores por estas notas.

## Continuación 2026-10-18 — Kontext compuesto y revisión pendiente de cámara

- Corregido el splitter legado que dividía también las instrucciones Kontext:
  una petición sin máscara manual conserva sus cláusulas en una generación.
  Contratos CPU rojo→verde y LongCat Full 2/2 PASS; reviewer APPROVE acotado.
  Render real chaqueta roja: antes 1/2 regiones y delta rojo -0,3 pp; después
  una pasada Kontext y +44,2 pp. Salidas/trazas en
  `diagnostics/out/edit_strong_kontext_20261018/1792338057787951000/` (antes)
  y `1792338249643359400/` (después). Misma foto/petición, semillas no fijadas.
  Se observa cuero rojo con cara, pose y fondo conservados; el corte sigue
  parecido a la chaqueta original. Falta validación del usuario y más casos;
  no cierra H-EDIT-003/H-EDIT-004 ni cumplimiento exacto de «blazer».
- Revisión heredada de cámara CHANGES_REQUESTED (corregida en el estado vigente): `low angle shot`,
  `high angle shot`, `close-up shot`, `full body shot` y algunas combinaciones
  detrás/izquierda no conservan las poses que admitía la ruta anterior;
  `CAMERA_SIGNAL=0.12` sigue fuera de configuración. El probe POV 10/10 no
  cubre estos casos. Esta continuación no modifica esas decisiones/anclas,
  ni aprueba el lote heredado cámara/default/UI.

## Última actualización 2026-10-18 (v5.1227: modelo por defecto de Generate que obedece el prompt)

- **«QUIERO UN MODELO MEJOR POR DEFECTO EN GENERATE. Y QUE OBEDEZCA AL PROMPT. REVISAR LOS CONTROLES»**: controles revisados de punta a punta (todo cableado; el aviso «Negativo sin efecto: CFG=1» es correcto — FLUX.1 guía vía FluxGuidance, muestreo distilled a CFG=1, documentado en la UI). Default `flux_dev_q4` → **`flux_dev_abliterated`** (`generate_ui.default_engine`): misma familia FLUX.1-dev con mejor Following instalado y direcciones de rechazo eliminadas (obedece también prompts adultos), 25 pasos/guía 3.5/`explicit: true`. Contrato fail→pass `probe_gen_default_engine_20261018` (rojo con flux_dev_q4 → verde 7/7); `probe_gen_prompt_adherence` actualizado (108/0, guard de «auto-scan nunca elige modelo adulto» intacto con pony_realism); batería **34/34**.
- **Pendiente**: relanzar la UI para cargarlo (la sesión 00:17 no lo tiene); validación visual de generación = del usuario. **Deuda**: la APK Android sigue con `flux_dev_q4` como primera opción de su desplegable (preferencia propia de la app, sin tocar). **klein 4B/9B no están cableados en Generate** (solo en el Editor): cablearlos sería funcionalidad nueva, pendiente de decisión. Sin commitear (lote v5.1226+v5.1227, orden pendiente).

## Última actualización 2026-10-18 (v5.1226: «cambia el POV» elegía ángulo por regex; pose medida por conceptos)

- **«np haCE CAMBIO DE POV» + «NO QUIERO HARDCODING»** (log `ui_start_20261018_001751.log`, edit 00:47 «CAMBIA EL POV, CORNUDO POV», salida `edit_1792277509.png`): la ruta POV sí se activó (QwenCamera 231 s con el lienzo ~1 MP de v5.1222) pero `camera_pose()` caía por regex hardcodeados al default fijo `front-right quarter view elevated shot medium shot` — «CORNUDO POV» (primera persona) no tenía representación. El LoRA solo tiene 96 poses de tercera persona; la aproximación entrenada más cercana al POV es `front view …` (la cámara donde estarían los ojos del observador).
- Fix v5.1226 **sin hardcodeo**: `CAMERA_POSES` en `nlp/semantic_analyzer.py` (anclas de concepto EN/ES por candidata y eje, mismo patrón que `CONCEPTS`, sin tokens ambiguos sueltos) + `score_camera_pose()` con el MISMO `_score_texts` del analizador ligero + `camera_pose()` como argmax por eje con defaults diseñados solo bajo `CAMERA_SIGNAL=0.12`; los regex de cámara, borrados. Contrato fail→pass `probe_editor_camera_pose_pov_20261018.py`: **rojo demostrado** (5 casos al ángulo stock) → verde 10/10; contrato previo `probe_editor_viewpoint_dispatch` intacto. Suite CPU **27/27**, ruff+py_compile verdes (el fallo aislado de `probe_editor_wall_instructions` fue entorno con la otra app del usuario usando la GPU — relanzado PASS).
- **Pendiente**: ~~relanzar la UI~~ — relanzada 02:26, v5.1226 **en producción** (log 02:26: la pose medida `front view eye-level` se aplica en un edit «cambia el pov»). El usuario reporta «sigue sin cambiar el pov»: **límite del LoRA confirmado en uso real** — sus 96 poses son de tercera persona; en una foto ya frontal «front view eye-level» apenas mueve la cámara. Sin más recorrido en esa ruta para POV real (declarado). Sin commitear (lote v5.1226+v5.1227, orden pendiente).

## Última actualización 2026-10-17 (v5.1225: paredes a medio pintar por cobertura de zona en el Editor)

- **«Ha vuelto el mismo problema con las paredes» + «¿por qué lo hace dos veces?» — diagnosticados en el edit 22:39** (`ui_start_20261017_220140.log`, foto hotel 6395×4263, «el hombre es calvo, las paredes son de color gris»). (1) Los dos renders son el pase principal + el **Refino focal** (diseño v5.1210, pegado limitado a la zona ya escrita — verificado en código); no hay doble pegado. (2) La ruta de recolor v5.1209 SÍ se activó (consultas singulares «the wall» = 5,8 %) pero la zona = el **fragmento CLIPSeg**: `_continue_measured_region` solo rellena huecos dentro de esa máscara, el hueco del sampler y el pegado son la zona → la pared quedó gris SOLO en un parche difuminado junto a la lámpara y crema en el resto (`edit_1792269760.png`).
- Fix v5.1225: `_grow_recolor_surface` — un recolor de superficie sin parte distinta sigue el color de la propia superficie (mediana erosionada + envelope p90 + separación plana 18 + conectividad, guardas de personas/objetos medidos); cableado solo en `surface_recolor and requested is None`, resto de rutas intactas. Contrato `probe_editor_recolor_surface_zone_20261017.py` **rojo demostrado** (pared 8,4 %/roja 14,6 %) → verde (88,0 %/100 %, madera-sábanas-piel-vecina ≤5 %). Dos probes de la suite actualizados con frontera de color (escenarios planos cubiertos enteros por el flood): contratos conservados, suite CPU **26/26**, ruff+py_compile verdes.
- **Pendiente**: validación visual suya (paredes grises uniformes, sin parche junto a la lámpara). **Commiteado y pusheado en `4b6fcd4` (2026-10-18)** junto con v5.1224. Deuda sin tocar: AVISO de traducción con falso positivo (`penetracion→penetration`, declarada en v5.1223).

## Última actualización 2026-10-17 (v5.1224: gate adulto se caía con las traducciones reales de Generate)

- **«NSFW ON pero el prompt no pide contenido adulto» en corridas adultas — causa medida y fix v5.1224**: el gate mide la traducción EN (línea555 de `flux_gen_comfy_client.py`), y el analizador ligero da `coverage(matched/palabras del prompt)·0.75 + density(matched/palabras del concepto)·0.35`; con prompts largos (~23 tokens) hacen falta 2+ coincidencias exactas en un solo concepto para cruzar 0,13. Medido sobre las traducciones REALES del log 15:47 (mismo prompt ES, Marian variant): «practice SEX … EXPLICIT PENETRATION» → 0,1430 ON; «SEXUAL RELATIONS WITH … EXPLICITE PENETRATION» → **0,0715 OFF**; «practice dogging …» → **0,0999 OFF**. Impacto: 5 de 10 corridas adultas salieron sin prefijo adulto/anti-censor. Fix **solo config** (`config/generation_prompt.json`): una línea nueva en `adult_gate.concepts` con las salidas del traductor medidas en producción («sexual relations» por «relaciones sexuales», «penetration», «cuckold», «dogging»); umbral 0,13 intacto y `_score_texts` NO tocado (el mismo scorer rutea el Editor). Contrato fail→pass `probe_gen_adult_gate_translations_20261017.py` (**rojo demostrado**: 0,0715/0,0999 OFF) → verde (0,4104/0,2052/0,2114 ON) + 9 regresiones del contrato previo intactas. Suite CPU **25/25**, ruff+py_compile verdes.
- **Pendiente**: sin pérdida — commiteado en `4b6fcd4` (2026-10-18). Deuda sin tocar: AVISO de traducción con falso positivo (`penetracion→penetration`), ya declarada en v5.1223.

## Corrección H-FS-005 en verificación — 2026-10-17

- Los fallos de composición y controles de la auditoría inferior ya tienen correcciones en código, diez contratos CPU verdes y lint/compilación afectados verdes. Usuario confirma Similitud conectada preservando default y retirada realtime; revisiones independientes en curso.
- A/B CPU real del vídeo normal completo134frames en ejecución; versión de primeros fixes cargada antes de conectar Similitud. Prueba real multicaras de versión final preparada. No se interviene UI/GPU activa.
- Pendientes de calidad H-FS-003/004 se mantienen: perfiles/parecido, oclusión natural difícil y vídeo largo representativo con cortes/zoom. Un clip corto ni componentes sintéticos certifican esos casos.

## Última actualización 2026-10-17 — auditoría FaceSwap

- **P1**: composición de segunda cara contra `original_frame` borra el swap anterior (`ProcessMgr:6380`); reproducido con asignación exacta AST y máscaras disjuntas. **P2**: suavizado de vídeo forzado True, Auto-Rotar sin lector, slider «Similitud» sin lector del umbral en motor. **P3**: preview realtime no conectado. Sin correcciones todavía: el usuario pidió revisión.
- Rescate de oclusión frontal repite el cálculo inicial desde v5.1216 y no puede reducir una máscara >40 % con entradas idénticas. Halo arreglado en fotos no valida vídeo, que conserva blur de transición. No relajar gates ni cambiar modelo sin una decisión respaldada por pruebas reales.
- Último lote sí terminó (56/56, END), dos SIN_SWAP y 32/54 oclusiones descartadas; anterior al arreglo posterior de guía frontal. Fuente repetida y coseno bajo al destino no demuestran identidad errónea.
- Reviewer CHANGES_REQUESTED. Ocho probes CPU pasan; lint crítico/compilación verdes, lint completo18 diagnósticos. Informe/evidencia `diagnostics/out/faceswap_audit_20261017.md` y bitácora MEMORY. H-FS-003/004 abiertas; no inferencias nuevas ni intervención UI/GPU.

## Última actualización 2026-10-17 — Generate H-GEN-003

- Corregidos disponibilidad/compatibilidad de LoRAs, selecciones de plano/color/piel descartadas y CFG LongCat ignorado; UI aclara negativo CFG=1 y HD/4K como perfil de acabado. Dos adaptadores locales recuperados y hashes comprobados: Amateur (Pony) y sdsFILM (SDXL base), solo selección manual. FLUX/LongCat no son compatibles con esos adaptadores; las seis entradas históricas del catálogo siguen sin archivo, sin descarga autorizada.
- Reviewer APPROVE y tester PASS técnico independiente CPU (81 comprobaciones: 11 LoRA + 6 controles + 5 UI/HTTP + 31 APK/Generate + 28 WS). Adherencia root 108/0; Ruff/py_compile verdes. Evidencia completa y hashes en MEMORY.md. El probe legado contiene mensajes de deuda históricos; no describen el proceso actual.
- Pendientes: probar carga GPU/resultado visual con prompt neutro cuando termine el POV del usuario; cargar los cambios en la UI por relanzamiento voluntario, sin interferir con sus pruebas; APK instalada sigue sin verificar. H-GEN-003 permanece abierta por esas comprobaciones. No se cambiaron gates del prompt ni el código de POV/Editor en esta intervención.

## Última actualización 2026-10-17 (v5.1223: aviso «max_new_tokens vs max_length» en cada traducción)

- **Aviso de consola en cada prompt — causa medida y fix v5.1223**: el Marian local (compartido por Editor y Generate) traía `max_length=512` y la llamada pasa `max_new_tokens` → transformers avisaba en TODA traducción; `generate()` recalcula el valor justo después, así que era solo ruido. Fix: `model.generation_config.max_length = None` en la carga. Contrato fail→pass `probe_editor_translator_maxlen_20261017.py` (unidad + flujo con captura del logger, **rojo demostrado** con el aviso real). Traducción idéntica pre/post. Suite CPU **25/25**, ruff+py_compile verdes.
- **Deuda declarada del repaso de generaciones**: el AVISO de traducción marca como «sin traducir» tokens que sí se tradujeron (`penetracion→penetration`, similitud difusa ≥0,68 compara el crudo con su propia traducción); sin daño funcional (solo ruido; en modo lento desactiva el rewriter por prudencia) y no tocado por riesgo de romper la detección de residuos. Commit/push pendiendo orden explícita.

## Última actualización 2026-10-17 (v5.1222: «cambia el pov» no movía la cámara → ~1 MP con FluxKontextImageScale)

- **«Cambia el pov» sin movimiento — causa medida y fix v5.1222**: la ruta de cámara terminó bien («Qwen Camera OK (192s)», fix del 16/10) pero el crudo reprodujo la composición de la entrada; los 3 renders reales de producción (00:11, 02:38, 16:53) tampoco movieron (correlación 0,829 vs 0,354 de un movimiento real). Experimento aislador (misma foto/pose/semilla, un delta por corrida): **`FluxKontextImageScale` ~1 MP → REENCUADRADO (−0,0798, 197 s en 8 GB)**; CLIP fp8 sin escalar → sin movimiento. Causa: la workflow de producción entregaba 768×512 directo y la LoRA de ángulos reconstruye en lienzo pequeño; la referencia del autor escala a ~1 MP. Fix: nodo 9 `FluxKontextImageScale` + `UPLOAD_SIDE=2048` (solo acota gigantes; el lienzo lo decide node9; presupuestos intactos; CLIP GGUF se queda = decisión de 8 GB preservada). Contrato fail→pass `probe_editor_camera_scale_20261017.py` con **rojo demostrado** (sin nodo 9; subida 768×512). Suite CPU **24/24**, ruff+py_compile verdes. Hoja: `diagnostics/out/pov_experiments_20261017.png`.
- **Pendiente**: validación visual del usuario — su UI (15:47) corre con v5.1221 pero **sin** v5.1222; relanzarla (solo con su orden, consola visible, sin redirect) y repetir «cambia el pov». Si el movimiento se ve pero la pose por defecto (front-right + elevated) se queda corta, decidir pose más lejana sin inventar ángulos. Deudas cosméticas de Generate declaradas (warning `max_new_tokens/max_length`; AVISO de traducción con falso positivo `penetracion→penetration` — sin daño funcional). Commit/push pendiendo orden explícita.

## Última actualización 2026-10-17 (v5.1221: «no lleva» sin sujeto → el sintagma inicial nombra a la persona)

- **«La que cruza sigue con media» — causa medida y fix v5.1221**: en las tres peticiones de «la mujer no lleva medias, ni zapatos» (13:47, 13:55 y calidad) el log dice «zonas medidas=0, sin zona=2» → foto entera regenerada sin verificación. La media negra con encaje desapareció (cruce 102/69/63 → 169/109/96) pero el muslo que cruza salió 25-34 niveles más oscuro que su brazo desnudo (180/134/130) y liso → se lee como media nude; en la de 13:47 la pierna se recolocó y la silla ocupó su zona. Causa: `_PREDICATE_RE` no conoce `wears`/`lleva` → sin verbo de predicado no se proponía el sujeto al segmentador → sin sujeto no hay zona. Fix: fallback **solo gramatical** en `_visual_subject_description` (si la cláusula abre con sintagma determinante, `_determiner_phrases` propone el sujeto; sin listas de prendas; segmentador y compuerta de cara siguen decidiendo). Contrato fail→pass `probe_editor_wear_clause_zone_20261017.py` (flujo local anclado + unidad + guarda) con **rojo demostrado** (reprodujo el log real línea a línea). Suite CPU **23/23**, ruff+py_compile verdes. Verificado con CLIPSeg real sobre la foto de entrada: «la mujer» 10,6 % / «the woman» 12,7 %.
- **Deuda declarada**: el tono «piel vs prenda» no se verifica post-pase; con la ruta ahora local y acotada el riesgo baja, pero si el usuario vuelve a ver «media» con zona medida haría falta esa verificación + retry. La UI sigue apagada (7861/8188 sin escuchar; el log de la sesión terminó con un evento de cierre de ventana). Al relanzarla (con su orden) cargará v5.1221. Commit/push del lote pendiendo orden explícita.

## Última actualización 2026-10-17 (v5.1220: cara añadida por el modelo → retry + restauración)

- **Tercera cara sobre la coronilla — causa medida y fix v5.1220**: «el hombre debe ser calvo» sobre foto 2847×4263 (10:23, `edit_1792225412.png`) aplicó el calvo y corrió el refino focal, pero el modelo pintó una cara NUEVA sobre la coronilla (el arranque midió 2 caras; el resultado, 3). El pegado por cambio medido la conserva (textura nueva = cambio real) y ni el no-op ni el fantasma la cogen (contenido nuevo, correlación HF rota — clase contraria al halo de v5.1219). Fix: `_added_face_mask` mide cada cara del resultado contra las caras de la entrada (centro a >0.6 de su propia altura de toda referencia ⇒ nueva, sin referencia no se juzga) ⇒ retry Turbo con prompt «exactly the same people: no face may be added»; si el retry persiste, esa región se restaura de la foto original con impresión en consola. Contrato fail→pass `probe_editor_added_face_20261017.py` (unidad, flujo, persistencia, **rojo demostrado**). Suite CPU **22/22**, ruff+py_compile verdes. Deuda: solo el flujo rápido single-subject (los pases de sujeto múltiple no lo tienen).
- **UI a relanzar** con v5.1220 en consola visible (sin redirect) cuando el usuario lo desee: el calvo sobre foto grande debe salir sin cara en la coronilla; si hay defecto, la consola imprimirá «Cara añadida por el modelo…» y quizá «El segundo intento conserva una cara añadida…». Commit/push del lote pendiendo orden explícita.

## Última actualización 2026-10-17 (v5.1219: fantasma del material original → retry; consola visible)

- **«Borra imagen, deja halos» — causa medida y fix v5.1219**: la cáscara gris sobre el cuero cabelludo midió el tono de la pared ([141,143,156] vs [135,140,154]) = el modelo desvaneció el pelo original a opacidad parcial en vez de reemplazarlo; el pegado lo conservó porque midió cambio (no era no-op). Fix: `_edit_leaves_a_ghost` (correlación HF ≥0,45 + amplitud ≤0,75 dentro de los píxeles cambiados) dispara el segundo intento Turbo con prompt de reemplazo completo («no translucent remnant… fully opaque») y consola lo anuncia antes del render. Contrato fail→pass `probe_editor_ghost_retry_20261017.py` con **rojo demostrado** (detector desactivado → 1 intento, halo entregado). Suite CPU **21/21**, ruff+py_compile verdes.
- **«Sin logs» — causa propia resuelta**: el relanzamiento 01:47 escondió la consola (redirect a archivo + minimizado); `run.py` ya teea a `logs/ui_start_*.log` por sí solo → relanzado 02:21 con consola visible: 8 ediciones + 3 gens con todo impreso, incluido **«Refino focal: ventana … escala previa 3.82×/5.57×»** (v5.1218 por fin visible en consola).
- **`probe_editor_appearance_head_hole_20261015` pasó a SKIP legítimo**: su fixture `img_editor/edit_1792085759.png` fue borrado del disco y los PNG de `img_editor/` no se trackean en git (mismo patrón de skip que `bald_outside_detail`). Si se quiere volver a ejercitar esa ruta hay que regenerar la foto-fixture.
- **UI relanzada 03:1x** (la sesión 02:21 terminó sola a las 02:54, 19 min idle): PID 10844 en 7861, consola visible, ComfyUI 8188 intacto, carga v5.1218+v5.1219. **Validación del usuario pendiente**: (a) calvo en foto con pelo → sin cáscara fantasma y con «Resto fantasma…»/«Segundo intento aplicado» si hubo retry; (b) foto grande con «Refino focal:» en consola. Commit/push del lote pendiendo orden explícita.

## Última actualización 2026-10-20 (guard de visión Moondream, GPU libre)

- **Guard de visión implementado** (`roop/img_editor/vision_guard.py`): el
  preflight exige backend ggml con descarga a GPU y VRAM libre suficiente para
  el proyector, y aborta con mensaje antes de cargar modelos. Aplicado a
  `element_selection._moondream_description` y a
  `instruction_verifier.InstructionVerifier._load`. Parámetros en
  `config/editor_selection.json` → `vision_guard`.
- **Deuda declarada — el mensaje de rechazo no se ha visto en pantalla**: el
  callback de producción que la UI invoca (`selection_events.on_analyze`) sí
  está verificado — devuelve el mensaje de rechazo sin excepción, con la
  sesión intacta y salida limpia (`diagnostics/out/vision_guard_20261020/
  ui_busy_gpu.json`). Falta relanzar la UI para confirmar cómo se ve en
  pantalla. Pendiente de una sesión con la UI activa.
- **Deuda declarada — margen de VRAM sin calibrar contra ComfyUI**: el margen
  configurado es 1024 MiB. No se ha medido el pico de VRAM cuando ComfyUI
  mantiene un modelo cargado mientras corre el análisis, que es el caso real
  de uso. Ajustar con medición, no por suposición.
- **Deuda declarada — `nvidia-smi` como dependencia**: el guard depende de que
  `nvidia-smi` esté en el PATH. Si no está, cae a `torch.cuda.mem_get_info`,
  que **no** ve la VRAM de otros procesos y por tanto es más permisivo. Se
  prefiere un rechazo explícito antes que permitir una ejecución que no cabe.
- **Hallazgo reutilizable**: `torch.cuda.mem_get_info` subestima la VRAM de
  otros procesos (6400 MiB externos visibles como 3738 MiB). Cualquier
  preflight de VRAM del proyecto debe usar `nvidia-smi`.
  Evidencia: `diagnostics/out/vision_guard_busy_gpu_20261020/vram_probe_6.0.json`.

## Última actualización 2026-10-17 (Generate: gate adulto + coma pegada; UI relanzada 01:16)


- **Generate — dos fixes medidos** (log 00:22-00:42): (1) gate adulto a 0,121 vs 0,13 por cero cobertura de «orgy»/«explicit» → concepto añadido en `adult_gate` (config/generation_prompt.json), con cuidado medido del falso positivo «photo»⊂«photorealistic»; (2) coma pegada sin espacio → token sin traducir en la salida EN → `_normalize_glued_list_punctuation` en `prompt_translator.py`. Contrato fail→pass `probe_gen_adult_gate_20261017.py` (4 rojos→verde); regresiones verdes: `probe_gen_prompt_adherence` 108/0, `wall_instructions` PASS, suite Editor 19/19, ruff+py_compile+JSON OK.
- **Deuda declarada — traducción de género Marian**: «jovencitos» → «young girls» (género invertido por el modelo NMT); corregirlo exigiría vocabulario hardcodeado o post-edición frágil — NO hecho, queda como límite medido del opus-mt local.
- **UI relanzada 01:16** (PID 20468→22420, puerto 7861, icono OK, ComfyUI 8188 intacto) cargando TODOS los fixes: dirección por petición (sufijo), prints del segundo intento, coma pegada y gate adulto. La UI previa (00:59) murió sola tras el banner sin traceback — causa no determinada; se relanzó con `-u` para trazabilidad.
- **Validación del usuario pendiente**: relanzar calvo en la foto del sofá 00:1x (tras dos pases sigue con pelo corto) y confirmar en consola los nuevos prints «Cambio casi nulo…; segundo intento». Commit/push del lote (5 fixes tarde + run.py + dirección/prints + gate/coma) pendiendo orden explícita.

## Última actualización 2026-10-17 (auditoría de la sesión noche 23:17→00:15 + 2 fixes)

- **Auditoría completa de la sesión noche sobre el log real** (`ui_start_20261016_231655.log`): 20 imágenes, 0 Tracebacks, 0 «Sujeto no medido», todas las peticiones despacharon — los fixes de la tarde (puerta de caras, puente de idioma, receta calvo, sufijo corporal, origen=) **validados en producción**. Dos defectos nuevos arreglados: (1) typo «ddesnuda» caía el sufijo «is fully nude» (la dirección la medía la grafía cruda) y el print decía «intensity=0.70 < 0.30» — falso; fix: si la cláusula líder no midió dirección, gana la medida de la otra grafía de la misma petición + mensaje honesto (`dirección=`). (2) El retry por cambio casi nulo generaba un segundo render **sin ninguna línea en consola** (run 15, 23:54); fix: print «Cambio casi nulo…; segundo intento Turbo» antes de cada retry y «Segundo intento aplicado/falló» después (4 sitios). Contratos fail→pass en `probe_editor_compound_fast_dispatch`; suite **19/19**, ruff y py_compile verdes.
- **Pendiente: relanzar la UI** (la de 7861 corre con el código ANTERIOR a estos dos fixes) cuando el usuario lo autorice y sin render en curso — se cargan los prints del segundo intento y la dirección por petición.
- **Pendiente visual del usuario**: `edit_1792185595` (sofá/dungeon 23:17) → cabeza pelada ✓ a mi vista; `edit_1792188900`/`edit_1792188965` (sofá con dos pases de calvo seguidos, 00:12/00:14) → el hombre **sigue con pelo corto** y ambas salidas son casi idénticas: el cambio medido fue real (13,5 % en cabeza) pero no resultó calvo — decidir si es la foto o cumplimiento.
- **Commit/push del lote** (5 fixes de la tarde + run.py + auditoría: dirección por petición, mensaje honesto, 6 prints de retry) pendiendo orden explícita.

## Última actualización 2026-10-16 (sesión de noche)

- **Calvo 22:51 — foto sin editar: la puerta de credibilidad descartó la cara del hombre.** CLIPSeg midió `'Man'` 18,2% / `'el hombre'` 19,7% pero ambas salieron con «no coincide con una cara» → «Sujeto no medido», `edit_1792184313.png` salió intacta. Medido (`debug_subject_noface_20261016.py`): la cara del hombre (103×87 px) se detectó con det 0,31 y `_credible_faces` (0,5×0,89) la descartó; la máscara cubría esa cara al **93,9%** — sin cara creíble dentro, la puerta de `_grow_measured_subject` no tenía con qué confirmar. Fix: `_measured_subject_passes(..., detected_faces=None)` — la puerta del solape acepta cualquier cara DETECTADA (la máscara de la consulta es la corroboración), los anclajes de calor/head siguen con las creíbles; `detected_faces` se pasa desde `generate_intelligent` en el compuesto y en el flujo rápido. Contrato fail→pass en `probe_editor_subject_descriptors`; suite **19/19**, ruff y py_compile verdes. ~~Pendiente: relanzar la UI con consola visible y repetir el calvo en esa foto~~ → **validado en producción el 23:17 de la sesión noche**: la misma foto midió «Sujeto 1/1 medido cobertura=29.0%» y `edit_1792185595.png` salió con la cabeza pelada (validación visual final del usuario).

- **Calvo 22:25 — instrucción despachada sin traducir; fix del puente de idioma.** El usuario repitió el calvo y salió mal (`edit_1792182345.png`, pelo intacto): la receta sí viajaba, pero la instrucción llegó **en español crudo**. Causa medida: en esa foto CLIPSeg `'Man'` (traducida) midió 0,01% vacía y `'el hombre'` (cruda) 6,6% → el grupo de sujeto eligió la cláusula española (el match es por palabra de sujeto) y LongCat, sin traducir, apenas obedeció. Fix `_translated_subject_counterpart` en `_group_measured_subjects`: el nombre se empareja **por posición** con su traducción y el texto despachado es siempre el traducido. Contrato fail→pass en `probe_editor_subject_descriptors`; suite **19/19**, ruff y py_compile verdes. **Pendiente: relanzar la UI con consola visible y repetir el calvo en la misma foto.**

- **«Procesado dos veces» — fix de concurrencia medido.** En la sesión real del usuario (18:55-20:00, `ui_start_20261016_185559.log`) «el hombre debe ser calvo» corrió dos pasadas de ~3 min (18:56 `calidad=False` → `edit_1792169992`; segunda petición `calidad=True` arrancó el mismo segundo en que la primera terminó, sin «Clic rechazado» → `edit_1792170143`). Experimento directo con gradio 6.13.0: `concurrency_limit=None` corre en paralelo (gap 0,05 s); la concurrencia por defecto (1) **encola**. El botón de escritorio ya tenía `None`, pero los listeners de API (`edit_options`/`edit`, móvil) no → su cola ejecutaba el segundo envío a espaldas del usuario. Fix en `ui/tabs/img_editor/events.py`: `_bind_generate` pasa `concurrency_limit=None` a los tres listeners (fallback TypeError para Gradio 3.x) y los wrappers `_on_generate_desktop/_api_options/_api_legacy` etiquetan el origen (`origen=` en «Solicitud recibida»). Contrato fail→pass `probe_editor_ui_skip._check_wiring_concurrency`. Suite **19/19**, ruff y py_compile verdes. **Pendiente: relanzar la UI y repetir un envío en caliente para ver el rechazo en voz alta; confirmar en log el `origen=` del segundo envío.**
- **Sufijo de desnudez con sujeto equivocado — fix medido.** Log real 19:49: «el sofá debe ser gris, las paredes deben ser grises, la mujer debe ir desnuda» despachó «The sofa is fully nude» (el sufijo tomaba el primer sujeto del compuesto) → sofá sin pintar, zona dañada («muy mal» del usuario). Fix en `img_editor_manager.py`: en compuesto el sufijo nombra al sujeto de la cláusula con mayor señal corporal medida (`attribute_body`/`attribute_uncover`); cláusula sin sujeto («…calvo e ir desnudo») conserva el sujeto global, como hasta hoy. Contrato en `probe_editor_compound_fast_dispatch` (caso sofá: «sofa is fully nude» ausente, «the woman is fully nude» presente por cláusula corporal). **Pendiente: validar visualmente en la UI relanzada que sofá/paredes sí se pintan ahora de gris.**
- **H-EDIT-003 — «gorrito» del calvo: causa raíz real y receta ya implementada.** El usuario repitió el calvo en la UI nueva (21:51; sin doble procesado, `origen=desktop`) y salió idéntico. Diagnóstico sobre artefactos: el hombre llevaba una **diadema con pompones blancos**; el modelo quitó el pelo pero **conservó la diadema** y pintó el cuero cabelludo **gris** → cabeza de «gorrito» blanco. El anclaje estaba bien (cabeza medida 100 %, ventana local, denoise 1.0): faltaba **definir el estado calvo** en el prompt. Fix `_bald_recipe_tail` en los cinco puntos de despacho (composer LongCat + 4 retries/cumplimiento + plantillas híbrida/genérica): cuero cabelludo pelado, sin diadema/gorra/cubierta, tono de piel natural. Contrato fail→pass en `probe_editor_bald_compliance` (retry real pre-fix sin definición de estado); suite **19/19**, ruff y py_compile verdes. **Pendiente: relanzar la UI y repetir el calvo en foto no explícita para la validación visual.** La estrechación del agujero a cuero cabelludo (cara como negativo) queda como opción si el tono regenerado no convence. Ver MEMORY 2026-10-16.

## Última actualización 2026-10-16 (sesión de tarde)

- **H-GEN-002 — A/B real completado sobre el motor por defecto.** `probe_gen_castillo_real.py` quedó parametrizado por motor (`--engine`, default el de la UI); `flux_dev_q4` semilla 123456 con el checkbox NSFW OFF/ON da salidas byte-idénticas (SHA256 `7db9c50e…`; el ON lo sirve la caché de ComfyUI = mismo workflow) e imagen correcta: castillo con unicornios en estilo dibujo (`diagnostics/out/gen_castillo_style/castillo_flux_dev_q4_nsfw_*.png`). El par histórico `castillo_nsfw_off/on.png` (abliterated) se conserva con nombre propio: mismo prompt ON/OFF, pero ese motor renderiza basura adulta incluso OFF (cautela documentada; no es el default). Evidencia de texto: 108/0 + 31/0. Queda abierto solo lo del teléfono/APK (casilla 74 del requisito).
- **H-EDIT-003 — ancla al pie corregida (bug del árbol actual).** Diagnóstico de la edición real del 15/10 (`edit_1792015316.png`, «Man must be bald»): los metadatos del workflow confirman máscara = elipse de cabeza, y el pelo trasero quedó fuera (brecha conocida). Pero el código ACTUAL, probado offline con la misma foto (`diagnostics/debug_bald_zone_20261016.py`), anclaba la edición a una **falsa cara en los dedos del pie** (det 0.30; el calor «Man» la prefería a la real 0.69 por perfil) y enviaba una máscara de 1.608 px sobre el pie sin tocar la cabeza. Fix: `_credible_faces` en `img_editor_manager.py` (score ≥ 0.5×mejor; caras reales medidas ≥ 0.57× en fixtures, el pie 0.43×), aplicado en `source_faces`. Contrato nuevo `probe_editor_bald_backhair_20261016.py`: la máscara cubre ≥70 % de la elipse de cabeza y ≤200 px la banda del pie. Tras el fix el agujero vuelve a ser la cabeza del hombre (36.345 px ≈ elipse 15/10).
- **Brecha abierta (sin tocar, decisión del usuario):** en la ruta de cabeza el pelo trasero sigue fuera del agujero (elipse); `_complete_hair_surface` no actúa porque «calvo» no existe aún en la foto y su semilla es la región medida por CLIPSeg. La decisión sobre `_complete_hair_surface` (mantener/recortar/revertir, 2026-10-10) y la validación visual de H-EDIT-003/004 siguen pendientes; la foto del caso es contenido explícito → diagnóstico sobre artefactos, jamás renders nuevos.
- **«Cambia el POV y no lo hace» — ruta de cámara con presupuesto corregido.** La petición real de las 15:35 se detectó y enrutó bien, pero Qwen Camera murió por `stall_timeout=150` mientras la carga fría + prep tardaba **202 s** en llegar al KSampler (job `b69c5c2d`: `execution_interrupted` en node 15 con los 14 nodos previos ya ejecutados; el reintento «se descarga y se reintenta» no llegó a correr porque la cola seguía ocupada por el propio job) y la foto salió idéntica con «POV no aplicado». Fix en `roop/img_editor/qwen_camera_comfy_client.py`: stall 300 s (>202 medidos) / timeout 420 s / tope de ruta 720 s, reintento solo con presupuesto y su log cuando ocurre de verdad. Contrato nuevo en `probe_editor_viewpoint_dispatch` (fail→pass con los kwargs capturados: 150→300/420, attempts=2); suite Editor **16/16**, ruff y py_compile verdes. **Pendiente: relanzar la UI (ventana cerrada a las 15:36; el servidor en 7861 corre con el código viejo) y repetir «cambia el pov» en foto no explícita para la validación visual.** Ver MEMORY 2026-10-16.
- **«Gato bailando vestido» — el hecho de postura ahora usa el concepto medido.** Trazado real: la frase se resolvía como pase corporal anclado («exact same framing») → el gato se vestía pero no bailaba, porque `facts["pose"]` solo miraba la lista de palabras del analizador (sin bailar/dance) e ignoraba `CONCEPTS["pose"]` (medía 0,289 en «el gato está bailando»; «the cat is dancing» = 0,000 por el hueco dance↔dancing del matcher). Fix A: hecho de postura = keyword **o** score medido ≥0,20 y ≥ resto ×1,25; fix B: descripción EN con forma -ing en `CONCEPTS["pose"]`. Contrato fail→pass nuevo en `probe_editor_compound_fast_dispatch` (postura EN/ES True, «vestido» suelto False, caso gato exige `no_kontext=True`/`preserve_frame=False`/«postura nueva»); suite **19/16+3** verdes, ruff y py_compile verdes. **Pendiente: validación visual del usuario con su foto de gato tras relanzar la UI** (el resultado del modelo para «gato bailando con ropa» no está validado visualmente). Ver MEMORY 2026-10-16.

## Deudas del proyecto (de `MEMORY.md`, no introducidas por H-GEN-001)

- Recolor de parte 2026-10-06: revisadas las correcciones de `_measured_region_passes` que quedaron pendientes el 2026-10-05. Solo recolor usa la parte medida con el sujeto como contexto; add/remove conserva su ruta estructural y su reparación; parte no medida con solape facial real se abstiene sin repintar. Contratos CPU verdes (recolor_part_dispatch 4/4, compound/viewpoint/proposal/paste_zone y demás exit 0). Render real GPU del caso «pelo rojo» (`hair_red_part_v3`): una inferencia, 0,97 % cambiado, pelo rojo sin borrar a la mujer ni mover al hombre. PASS acotado a esa foto/petición/semilla; H-EDIT-003 y H-EDIT-004 siguen abiertas.

- Caja de contexto H-EDIT-004 2026-10-06: hipótesis medida con `probe_editor_object_context_real.py` (Sobel bajo la semilla ponderado por píxeles no excluidos, umbral 1,05) — el bbox de la semilla CLIPSeg (y_max=436) cortaba la consulta «plant» por encima del suelo (y≈522–608). Caja medida [877,234,1136,526] con suelo y=522. CLIPSeg en caja sube precisión 0,705→0,770 y recall 0,637→0,667 con 0 solape de silla; SAM2 con caja recupera tronco 0,473→0,828 pero maceta solo 0,094; GrabCut re-invasa silla (925–966 px). Calor CLIPSeg dentro de la maceta ~0 (máx 0,004) con y sin caja: el fallo no es de encuadre, la consulta no genera evidencia de «maceta». Sin candidato integrable hoy; integrar dejaría restos de maceta. Veredicto `diagnostics/out/editor_object_complete_20261006/root_context_diagnosis.md`. Sin cambios de producción, sin GPU, sin descargas. H-EDIT-004 abierta; pendiente decisión: ruido/generación en la caja hasta el síntoma de suelo, o consulta/soporte distinto del planner 3B.

- Consulta de maceta y caja extendida 2026-10-06: la caja [877,234,1136,526] no contenía la maceta (0/2916 px, maceta y=540–608). Barrido con `--extend-floor 110` ([877,234,1136,636]): «flower/plant pot» y «potted plant» dan calor de maceta real (máx 0,145–0,204 vs 0), pero ningún umbral separa maceta de silla (th 0,10 → pot 0,42 con 87 px de silla; th 0,05 → pot 0,72 con 462 px). SAM2 con caja extendida sube maceta 0,094→0,390 pero silla 644→1555 px; GrabCut maceta 0,864 con silla 2032 px y 3266 px fuera; visualmente SAM deja cuenco y toca reposabrazos, GrabCut invade silla/mesa. Sin candidato integrable; H-EDIT-004 abierta. Evidencia `pot_query_*_metrics.json`, `pot_threshold_sweep_metrics.json`, `root_contextbox636_metrics_plant.json`, zooms `*_review_zoom.png`. Siguiente hipótesis: silla medida como negativo en SAM2 en la caja extendida, o ruido/generación hasta el suelo.

- Silla como negativo 2026-10-06: pipeline medido completo (caja Sobel extendida + positivos CLIPSeg de maceta en banda bajo semilla + silla CLIPSeg negativa → SAM2+GrabCut → menos silla medida) da `context_box_excl_2_grabcut` IoU 0,646 y con silla restada **0,664**, maceta 0,848/0,842, silla 993→411 px; 0 de esos 993 px caen dentro de la maceta anotada. Visual con flecos de borde y tinte residual en reposabrazos. Ampliado `probe_editor_object_complete_real.py` con `--positive-heat`/`--negative-mask` (py_compile/Ruff verdes). **Sin integrar en el manager**; H-EDIT-004 abierta, validación visual del usuario pendiente. Evidencia `pot_exclusion_verdict.md` y `segmentation_1791314575237438400_65aac32a`.

- Tracking2026-10-06, tras «hazlo»: corregido reset duro convertido a suave en closeup y comprobación de corte sin caras antes de hold. Reviewer APPROVE35/0; tester PASS453 contratos +27 integración; lint/compilación afectados verdes. Vídeo normal CUDA completo corto134 frames:130 swaps/modelo por versión,0px antes/actual; continuidad básica sin regresión, sin corte nativo. Root inspecciona todos los fotogramas renderizados. Quedan vídeos largos/cortes nativos/perfiles/oclusión prolongada y validación visual del usuario; H-FS-004 abierta. Deuda real observada: frames26–29 sin nueva detección, hold26–28 y original29, readquisición30, comportamiento previo idéntico. No alargar hold ni atribuirlo al reset corregido sin medir movimiento. Evidencia `diagnostics/out/tracking_reset_fix_20261006/`.

- Nuevo lote oral14fotos2026-10-06: log contemporáneo504612489 ya aporta occ0,02704/rel0,367/ring0,312, mezcla/paste0,912; reproduce gateobject actual, falta validar máscara/objeto y calidad visual. 64876795/564017241 conservan mezcla0,850/0,880 por geometría, paste0. CatorceJPEG decodificables,11checks de auditoríaPASS y lint/compilación/probes verdes; no nuevo render ni cambio de producción. Evidencia `diagnostics/out/mouth_batch_review_20261006/audit.md`. H-FS-004 abierta; no interpretar pesos de composición como porcentaje real de píxeles originales.

- Continuación oral2026-10-06: corregidos escala del anillo y puntos incorrectos de abertura MediaPipe14/17→13/14. Muestras reales normales abiertas mejoran la abertura; controles cerrados01416/01947 y pelo03206 conservados. Escala reviewerAPPROVE/testerPASS56contratos/206verificaciones; apertura reviewerAPPROVE27controles/testerPASS147verificaciones, ochoA/B realesCPU, tres abiertas mejoran visualmente. No cerrar H-FS-004: falta validación visual contemporánea de504612489, vídeos completos, oblicuidad, perfiles/objetos difíciles y evitar que textura/doble composición preserve boca del destino cuando no corresponde. Persona parcial de fondo00815 excluida de calidad de la adulta principal. Evidencia `diagnostics/out/mouth_math_review_20261006/`.

- Revisión oral2026-10-06: aislamiento de EMA/histéresis/hold/paste oral fuera de vídeo suavizado corregido y logs de decisión por foto añadidos; reviewerAPPROVE acotado y ocho casos con inferenciasCPU reales pasan. El0,91 de504612489 es ramaobjeto, con evidencias ausentes del log viejo; causalidad/objeto real y defecto visual aún pendientes. No afirmar mejora visual de ese caso a partir del aislamiento. TesterPASS acotado52/0 y tres comparaciones0px con ajustes iguales; la calidad oral visual continúa pendiente. General smoothingTrue aplica colorLAB aun en foto y difiere133793px del controlFalse: confusor preexistente registrado, no modificado fuera del alcance oral. Evidencia `diagnostics/out/mouth_review_20261006/`.

- Actualización2026-10-06 tras «DALE»: las dos entradas siguientes documentan la auditoría previa, no deudas de código actuales. Hyperswap incompleto retirado, rescate128 restaurado, escala/plantilla/config corregidas, elipse de recuperación retirada; lint afectado/crítico y compilación ampliada verdes. Extracción real6/7 (P306 realmente cortada), reviewerAPPROVE acotado; evidencia `diagnostics/out/source_fix_20261006/`. TesterPASS acotado: dos retratos normales con inferencias128/XSeg/GFPGANCPU reales y comparación antes/actual idéntica; sin regresión sana. H-FS-004 y calidad visual/vídeos siguen abiertos (perfil, cara pequeña, máscara colapsada y oclusiones difíciles sin validación nueva). No renderizar1994859933 por contenido explícito; seleccionar referencias normales de perfil/oclusión.

- 2026-10-06, extracción FaceSwap: reparada sintaxis que bloqueaba import de `get_gender` y descartaba caras ya detectadas; tres fotos reales vuelven a1 cara en wrapper UI. El añadido Hyperswap previo mantiene `_hyperswap_256_get_result` indefinido y atributo `roop.globals.providers` inexistente, pendiente decisión del usuario sobre retirar fallback/restaurar rescate128. Ruff global no verde. `P1070610.JPG` tiene rostro real visible pero sigue rechazada por segunda pasada sim0,462<0,50; no relajar gate sin medir falsos positivos. Evidencia `diagnostics/out/source_import_20261006/`.
- Auditoría2026-10-06 (`source_validation_20261006_004726`): compilación principal y11probes verdes; lint completo FaceSwap162diagnósticos, lint crítico2F821 y compilación ampliada falla2copias UI sin imports. Tester FAIL global. `estimate_constrained_norm` ya está en Run128/256 pero acepta geometría degenerada en caso reproducido; configuración escala sin lectores y plantilla ArcFace omite shift128/256. Sigue elipse en broad_floor y fallback. No cerrar H-FS-004 con helper aislado verde; falta corregir y verificar integración visual. Siete fotos reales:3extraídas,4rechazadas por gates; revisar casos completos/falsos positivos antes de cambiar umbrales.

- Editor2026-10-05: modelo3B autorizado/descargado y planner local medido; taza/libro sobre una sola escena funcionan. Traducción3B de ruta ambigua aprobada e integrada: RAW mantiene medición, diez traducciones CPU reales/contratos en PASS y reviewer APPROVE acotado. El manager real planta ya manda plant en vez de floor, pero sigue eliminación parcial y bloques gris/rosa (91s/1,25%, personas conservadas). Traducción no basta; pendiente receta/cobertura/generación y generalización. Comprensión de referencias espaciales todavía falla en algunos prompts ingleses. No cerrar versatilidad ni calidad visual universal.
- Máscara de planta solo contiene núcleo de copa; Otsu global conectado apenas cambia0,4→0,392 y no amplía cobertura. Pendiente otra medición de objeto completo e inpaint coherente, preservando personas; no ampliar solo paste.
- Receta8/referencia espacial regional aprobada por «quiero que sigas para solucionarlo todo» e integrada con configuración `longcat.region_edit`. Manager completo planta44s/1,32% y taza45s/0,19%; reviewer APPROVE y tester PASS técnico. Taza blanca pequeña visible, pero anillo claro adicional; planta sigue parcial con hojas/tronco/maceta. Rutas globales conservan12/no_ref. No cerrar calidad ni versatilidad general.
- Ruff cliente/manager ahora verde: limpieza acotada de33fallos previos (imports, variables sin uso y sentencias múltiples); compile/diff-check verdes y reviewer APPROVE. Contratos regional/compuesto, Generate72/0+31/0 y viewpoint pasan. Esto no equivale a lint global de todo el repositorio ni a historia visual cerrada.
- Cobertura de planta, alternativas descartadas CPU: ISNet general conectado a semilla selecciona ambas personas28,82%; CLIPSeg visual condicionado con máscara selecciona62,70%escena; visual de recorte vuelve a copa1,72%; umbralesCLIP0,3/0,2/0,1/0,05 excluyendo foreground medido no alcanzan tronco/maceta. Query `potted plant` ~2% no completa objeto. Moondream2 devuelve coordenadas erróneas[0,61,0,13,0,78,0,34]; no usarlas como máscara.
- SAM2.1Tiny156MB (Meta) e Hyperswap1a256384MB (FaceFusion) autorizados explícitamente/descargados con SHAoficial y probados localmente, sin paquetesnuevos. SAM2CPU2,1–2,4s mejora contorno, pero no objeto completo; 1a pierde identidad frente1c y altera bocaoclusión. FAIL visual para integración, producción sin esos modelos.
- Replayruido ventana completa8/refretira silla/planta en bruto, pero redibuja escena y cambia pose/texturas; no pegar ventanaentera. Delta conectada+ISNetprotección/source+generated pierde fidelidad visual (sombra silla/costura brazo/mueveobjetosvecinos), aunque pixelsprotegidos=0error. Rechazado como candidato de producción. RefinadoGrabCut de silla en diagnóstico, aún sin aprobación/integración ni calidad universal.
- Retirada de silla derecha en español se abstiene: propuesta`'silla a la derecha'` no medida. Diagnóstico CPU equivalente`'chair on the right'` sí mide solo silla derecha3,26%, aunque ruedas/puntas quedan fuera; pendiente prueba manager en inglés y corrección de consulta de idioma. No extender traducción3B fuera del alcance aprobado ni usar bboxVLM inventado.
- FaceSwap: restaurar pelo nuevo medido no quita marca; área68 con alineación intacta y variantePoisson eliminan núcleo pero crean transición y bajan identidad a0,5737/0,5171 y0,5585/0,5014. Diagnósticos rechazados, no sustituir producción128 con esos parches.
- FaceSwap: Hyperswap1c256 descargado y comparado con aprobación explícita. Mejora similitud medida (0,6566/0,6004 frente a 0,4972/0,5211) y definición en perfil/frontal, pero conserva marca deformada de frente ya en salida bruta; sin enhancer tampoco se corrige. Alineación68RGB reduce identidad a0,3280/0,3096. No se ha aprobado cambiar producción128. Primeras dos corridas fueron híbridas CPU/CUDA por la selección interna de ProcessMgr; probe corregido y últimas comparaciones CPU reales. Pendientes proporciones/parecido visual y seis vídeos completos.

- **FaceSwap**: renders completos de los seis vídeos recientes, banco de sources y pareja concreta para el mal tamaño/colocación en fotos de baja calidad; el caso `1994859933.jpg` sigue saliendo mal.
- **Image Editor «calvo»**: falta la validación VISUAL del usuario (hojas `sh_bath_personwin_full_s8.png`, `sh_bath_pw_zoom_u6.png`, `sh_bath800_check_v5.png`); la estabilidad de generación depende de la semilla (ver bitácora 2026-10-03, A/B `bath800_seedA..D`).
- **UI**: pestañas reordenadas en `ui/main.py` → hace falta reiniciar AutoAuto (comprobrando antes que no haya render activo).
- ~~**Probes preexistentes en rojo**~~ → **resuelto 2026-10-16**: `probe_editor_paste_guard.py` ya se había reescrito al contrato vigente (2026-10-07); `probe_editor_compound_people_fast_route` se actualizó (2026-10-08); los cinco que piden `img_editor/edit_1790808153.png` dan ahora **SKIP exit 0** con aviso de artefacto histórico ausente; `ui_skip` se reescribió al contrato vigente (la petición focal ambigua genera anclada y acotada; exit 0, sin GPU). Ver MEMORY 2026-10-16.
- Salvaguarda de proceso (propuesta tras el lote de 93 fotos muerto sin rastro) → **implementada 2026-10-16**: `run.py` vuelca stdout+stderr a `logs/ui_start_<ts>.log` con PID y marca de fin (ausencia = muerte externa); `roop/core.py` y `ProcessMgr.py` registran PID en cabecera y marca `=== END ===` de sus logs de lote.

## Deudas conocidas de Generate (detectadas en la revisión 2026-10-03)

- `lora_catalog` de `config/generation_prompt.json` apunta a ficheros SDXL que no existen en `models/loras/` → el autoloreador resuelve vacío (`gen_lora_resolver.py:21-26`).
- El catálogo de LoRAs disponibles en disco es de Qwen/ICEdit, no de SDXL: hay que alinear catálogo ↔ disco o ampliar a motores DiT.
- Este ComfyUI no registra `/progress` (`ui/tob/ComfyUI/server.py:501-1027`) → resuelto en Fase 3b: ETA derivada de `/history` (`history_mean_seconds`) + fallback `/queue` (verificado 2026-10-04, ETA `0:23` en la barra).
- ComfyUI cachea la ejecución con parámetros idénticos: `SaveImage` no re-ejecuta y `history` queda **sin outputs** → segunda corrida idéntica devolvía «terminó sin imagen». Mitigado con `filename_prefix` único por corrida (`GenFlux_{ts}` / `GenSDXL_{ts}` en `flux_gen_comfy_client.py`); la imagen sale del caché (mismos píxeles, diff = 0 con semilla 777).
- `roop/comfy_workflows.py` y `comfy_workflows_fixed.py` son legacy: solo se importa `get_comfyui_url`.
- El pintado de máscaras no forma parte de la UI vigente: el usuario recordó
  su retirada el 18/10. No recuperarlo a partir de esta deuda histórica.

## Riesgos de H-GEN-001

- **R1**: enrutar edición desde Generate duplica lógica de UI → mitigación: reutilizar `generate_intelligent` sin copiar código.
- **R2**: img2img con denoise<1 puede no cumplir la instrucción → el resultado debe pasar por la misma medición del Editor.
- **R3**: añadir campos rompe el contrato de la APK (`GradioClient.kt:472-491`, `generate_image` con 6-7 parámetros) → los parámetros nuevos van **detrás** de los existentes y con defaults compatibles, o se añade un endpoint nuevo.
- **R4**: VRAM 8 GB compartida → toda prueba real comprueba antes si hay render activo.

## Verificación final de H-GEN-001 (2026-10-04)

- `probe_v51140_gen.py` → **30 OK / 0 FAIL** (contrato APK de 10 posicionales intacto).
- `smoke_gen_ui.py` → `SMOKE OK` (UI construye, `/generate_image` = 10 inputs, `bt_to_editor` conectado).
- Fase 3 real (`test_fase3.py`): unload `7080MB libres de 8192MB`, media histórica `24.0 s`, ETA `0:23` en la barra, diff semilla 777 = **0**, `samples=2` → `gen_1791138065.png` + `gen_1791138065_2.png`.
- Rutas reales: edición enrutada → `generate/gen_1791137177.png` (25 s); img2img FLUX y LongCat en `diagnostics/out/gen_img2img_*.png`.
- Resta: **reinicio de la UI** (con permiso del usuario y sin render activo) para ver los cambios, y bitácora en `MEMORY.md`.

## Auditoría 2026-10-04

- Actualización2026-10-05: selección de parte y receta8/referencia espacial mejoran pelo rojo/paredes azules en pruebas reales, pero persisten mechones oscuros, borde claro y cambios de planta/silla. Calvicie sigue fallando en fotos adicionales; historia abierta.
- Diseño de propuestas de zona por modelo local integrado y revisado (APPROVE acotado), con Qwen3B aprobado y configurado en CPU. Taza/libro reales conservan personas en la misma escena (PASS acotado); retirada de planta continúa FAIL. Contratos negativos evitan propuestas inventadas/saturadas/ambiguas, soporte facial y cambio silencioso Turbo→Full. Los experimentos de prompt todavía confunden algunos objetos con sus ubicaciones; no cerrar calidad ni versatilidad.
- FaceSwap256 con entradas correctas y plantillaArcFace no conserva identidad fuente según cosenos reales; plantillaFFHQ alternativa en diagnóstico, sin sustitución del modelo128 en producción. Los seis vídeos completos siguen pendientes.
- Generate H-GEN-002 está especificada en spec/02-REQUISITOS.md. El A/B real del castillo con flux_dev_abliterated falla visualmente tanto con NSFW OFF como ON, pese a los probes de prompt en verde; flux_dev_q4 mostró castillo y unicornios con tres semillas. El 2026-10-05 se cambió el default a flux_dev_q4 y el checkbox a OFF en `config/generation_prompt.json` tras la instrucción del usuario de continuar los arreglos (las preguntas de Codex no le aparecían). Android debug compiló con Gradle 8.7 local; reviewer dio APPROVE y sd-tester dio PASS final para el alcance Generate (72/0, 31/0, PNG y APK comprobados). El root generó `generate/gen_1791179741.png` por `/generate_image_ui` servido e inspeccionado visualmente. Pendientes prueba en teléfono y regresión visual de Editor/FaceSwap, que siguen abiertos.
- Image Editor Turbo: la máscara ampliada para pelo produce franja gris translúcida en una semilla y ambas máscaras pueden producir segunda cabeza. Comparación en diagnostics/out/ab800_seedpairs_y1.png. Un recorte diagnóstico del borde inferior de la zona añadida no quitó la franja (`diagnostics/out/bath800_edgeclipA/result_real.png`), así que no se integró. La semilla 424243 evita la cabeza extra pero deja pelo en media cabeza; ampliar genéricamente la instrucción empeora el resultado. Pendientes solución visual, validación en más fotos y guardia contra cabezas extra.
- FaceSwap: la foto real 1994859933.jpg sigue mal tras ALIGN_REJECT y rescate de referencia; diagnostics/faceswap_batch_first_rescued.jpg. Corrección de la auditoría: los lanzadores usan `venv/Scripts/python.exe` con ONNX Runtime 1.23.2 y CUDAExecutionProvider activo. La prueba de la misma foto en CPU y CUDA difiere 0,039 niveles de media y conserva el defecto. Ni `inswapper_256.onnx`, CodeFormer local frente a GFPGAN, ni erosiones offline de máscara de 11/21 px resolvieron visualmente este perfil. Los logs del enhancer ya reflejan el motor elegido. Pendientes los vídeos completos y una solución de identidad/colocación para perfiles extremos.
# FaceSwap — revisión de salidas, 2026-10-13

- Guard general de contención de landmarks y reset de máscara tras corte duro corregidos; mantienen modelos y márgenes previos. Evidencia en `diagnostics/out/project_fix_20261013/faceswap_latest_review/`. El lote del usuario terminó su log tras 12 salidas de 93 anunciadas, sin END; no se conoce la causa de interrupción.
- La comprobación real CPU tiene cuatro casos y ocho inferencias por modelo, con 0 píxeles antes/actual. Alcance visual: tres caras principales (gimnasio, perfil y mano junto al pelo) y una persona pequeña de fondo en 02457. La adulta principal de 02457 no fue extraída: no cuenta como perfil validado.
- El guard evita inferencias con coordenadas incoherentes; cuando no existe una referencia sana conserva el resultado previo. No demuestra resolver falta de textura, halo o parecido, ni vídeos completos. H-FS-004 continúa abierta.
- La miniatura de fuente azul del probe previo era RGB guardado como BGR: el hash coincide con el crop RGB devuelto por la extracción. Corregido el writer del probe para futuras corridas; no atribuir ese artefacto al modelo o al color de producción.
- El usuario priorizó esta revisión sobre las otras correcciones iniciadas. Los cambios locales previos de Editor/Animate siguen pendientes de completar y revisar como lote separado; el bloqueo de opciones incompatibles de Generate quedó **resuelto 2026-10-16** con la decisión de default `img2img` (Generate y Editor paralelos e independientes; `editor` manual). No declarar «todo resuelto».

# Image Editor - refino focal del pegado grande (v5.1218), 2026-10-17

- Defecto medido: con foto de 27 MP (6390x4263) el sampler capa a ~1 MP y el pegado sube x3.34-x5.12; las zonas medidas salen suaves ("las ediciones pierden foto", log ui_start_20261017_011634). El refino focal existente no disparaba: ruta cuerpo lo descartaba en silencio, retry y compuesto no lo tenian cableado.
- v5.1218: prints en toda salida silenciosa del refino, refino tras el segundo intento y refino en el pase compuesto (vision_reference=None heredado del compuesto). Suite editor 20/20, adult_gate PASS, prompt_adherence 108/0, ruff limpio.
- Pendiente: validacion visual del usuario con foto grande (zonas medidas con textura fotografica, sin aspecto pintado). Lote completo sin commitear hasta orden explicita.

# Image Editor - ángulo/postura obedecidos (v5.1238/v5.1239), 2026-10-22

- Resuelto medido: «CAMBIA EL ANGULO, IMAGEN DE LATERAL» (y perfil/lado/de pie) miden POSE=True y salen por la rama `pose_change` (motor con visión, foto completa, sin ancla, marco reinterpretado, caras ajenas protegidas); contrato `pose_change_uses_the_vision_engine` → suite selección 9/9. Causa raíz: el sustantivo del medio («imagen/foto/image») en los conceptos `structural`/`structural_remove`/`attribute_appearance` daba +0,25 a toda petición que mencionara la foto y hacía imposible ganar la competición de pose. Ancla POV borrada por la sesión del 19/10 restaurada → probe cámara 10/10.
- Pendiente: render real del ángulo sobre la foto del usuario + inspección visual; requiere reinicio de la app (Generate v5.1237 + pose + conceptos + ancla POV en proceso).
- **Deuda preexistente (commit `9b5b6d0`) — RESUELTA 2026-10-22 (v5.1240)**: el alias `longcat→kontext` (línea 6104) normaliza antes de `engine in ("longcat","imagine")` (6391) → ruta Turbo 6388-7634 inalcanzable para kontext. La compuerta de caras añadidas **v5.1220 quedó sin efecto para kontext** (todos sus call sites en la ruta muerta; la integridad visual ordinary solo avisa, no restaura). Arreglo: compuerta portada a la ruta ordinary de kontext (tras `result = result_obj.image`, generación a foto completa, `_instruction_preserves_face_layout` como exención, reintento único con `_same_people_retry_prompt` y restauración por `Image.composite` si persiste o falla). Los 2 contratos viejos actualizados al contrato vivo sin perder aserciones de protección: `probe_editor_added_face_routes_20261018` (compound = pasada global) PASS y `probe_editor_compound_fast_dispatch` ( despacho vía `engine="qwen_camera"`, la única entrada viva a la ruta Turbo: fallback 6288→6294→6391) rc=0 con 63 OK.
