# spec/02 — Requisitos (historias)

Historia activa: **H-GEN-001 — Generate versátil** (aprobada por el usuario el 2026-10-03: «Todo: 0+1+2+3»; motor por defecto sin cambiar).

## H-GEN-001.1 — Controles avanzados en la UI (Fase 1)

**Como** usuario **quiero** fijar seed, steps, CFG y negative prompt **para** poder reproducir y ajustar corridas.

- [x] Aparecen 4 controles en `ui/tabs/gen/ui.py` (seed vacío = aleatorio; steps/CFG vacíos = valor de `config/model_configs.json`; negative vacío = el de config).
- [x] `on_generate_image` los pasa a `client.generate(...)` y llegan al `KSampler`/`CLIPTextEncode`.
- [x] Misma seed + mismos parámetros ⇒ misma imagen (verificado con 2 corridas reales y diff de píxeles ≈ 0).
- [x] Sin controles, el comportamiento previo es idéntico (probe existente en verde).

## H-GEN-001.2 — `color_grade` y resolución libre (Fase 1)

- [x] Desplegable de `color_grade` con las opciones de `gen_prompt_modifiers.py:63-69`; `auto` por defecto.
- [x] Sliders `width`/`height` visibles (512-1536, paso 64) sustituyen/augmentan la orientación.
- [x] El preview de modificadores refleja el `color_grade` elegido (no `"auto"` fijo).

## H-GEN-001.3 — Motores (Fase 1)

- [x] El desplegable incluye los alias ya soportados por `load()`: `longcat`, `longcat_full`, `flux_q2` además de los actuales.
- [x] El default fue `flux_dev_abliterated` hasta H-GEN-002; el 2026-10-05 se sustituyó por `flux_dev_q4` tras el fallo visual reproducido del modelo anterior y la instrucción del usuario de continuar los arreglos.
- [x] Un motor no instalado no aparece (comportamiento actual conservado).

## H-GEN-001.4 — Puente Gen→Edit (Fase 1)

- [x] Botón «Editar este resultado» junto a la imagen de Generate que carga esa imagen en la entrada del Editor y salta a su pestaña.

## H-GEN-001.5 — Imagen de entrada en Generate (Fase 2)

- [x] Campo de imagen opcional en Generate.
- [x] Si hay imagen y **no** hay instrucción de edición: img2img con referencia (`LoadImage` + `VAEEncode`, denoise < 1 tomado de config).
- [x] Si hay imagen **e** instrucción de edición (añadir/quitar persona u objeto, ropa, fondo, POV/ángulo, pose): se enruta a `generate_intelligent()` del Editor y el resultado se muestra en Generate.
- [x] Sin imagen: txt2img como hasta ahora (regresión cero).

## H-GEN-001.6 — VRAM, progreso, calidad y muestras (Fase 3)

- [x] Tras generar, Generate libera la VRAM del servidor (`POST /autoauto/unload`) igual que el editor.
- [x] La barra de progreso muestra paso/ETA derivados de `/history` (este ComfyUI no expone `/progress`).
- [x] Selector de calidad de salida (hd / 4K) usando `image_quality_pipeline` como el Editor.
- [x] Selector de nº de muestras (1-4) con `batch_size` parametrizado.

## Criterios comunes de la historia

- `py_compile` + probes de Generate existentes en verde; Image Editor y FaceSwap sin regresión.
- Prueba real de Generate tras cada fase, con imagen guardada en `generate/` y anotada en `MEMORY.md`.
- Parámetros nuevos con default en `config/`, sin valores mágicos.

## H-GEN-002 — Adherencia al prompt y control NSFW

**Como** usuario quiero que Generate respete el sujeto y estilo pedidos, y que el contenido NSFW solo se active con un control manual, para evitar escenas no solicitadas.

Decisiones ya aprobadas por el usuario el 2026-10-04: checkbox NSFW manual, gate de estilo por conceptos medidos sin listas de palabras y verificación mediante probe y render real.

- [x] El checkbox NSFW llega al ensamblado de prompt y filtra prefijo y LoRAs adultas cuando está apagado (probe de ensamblado 52/0).
- [x] El gate de estilo omite añadidos fotorreales ante una petición no fotorreal (probe de ensamblado 52/0).
- [x] La petición CASTILLO CON UNICORNIOS, ESTILO DIBUJO produce visualmente castillo y unicornios en estilo dibujo con `flux_dev_q4` y NSFW apagado: semillas 123456, 234567 y 345678, salidas en `diagnostics/out/gen_castillo_style/`. Con `flux_dev_abliterated`, OFF/ON fallaron con la misma semilla.
- [x] La configuración normal de la UI selecciona `flux_dev_q4` y abre NSFW apagado; ambos defaults viven en `config/generation_prompt.json` y la UI construida devolvió esos valores.
- [ ] El contrato de diez parámetros de la APK, Image Editor y FaceSwap siguen funcionando; el endpoint `/generate_image` respondió por HTTP con diez argumentos y la APK debug compiló, pero falta instalarla en un teléfono y la validación visual de Editor/FaceSwap continúa abierta.
- [x] Render real por la ruta servida `/generate_image_ui`: `generate/gen_1791179741.png`, 768×512, seed 123456, castillo y unicornio en ilustración, inspeccionado visualmente; 20/20 pasos ComfyUI y estado «Listo».
- [x] Review final y verificación independiente de Generate: reviewer APPROVE y sd-tester PASS de la versión final (72/0 y 31/0 repetidos; PNG abierto y verificado; APK debug y SHA256 comprobados). El tester no repitió el HTTP servido ni probó la APK instalada en un teléfono.

## H-FS-003 — Perfil, halo y selección del enhancer

Alcance confirmado el 2026-10-05 por el usuario: «TODO» y «sigue+», tras presentar FaceSwap, Editor y revisión de logs.

Comparación aislada aprobada explícitamente: descargar Hyperswap1c256 oficial y compararlo con la fuente y el perfil de referencia. Descarga y comparación completadas; similitud medida y definición mejoran, pero la marca deformada de frente permanece incluso antes del enhancer. La prueba no autoriza cambiar el modelo de producción ni cierra la historia.

- [x] La vista previa inicializa el enhancer elegido aunque la selección global anterior fuese otra; ejecución real y salida guardada en `diagnostics/out/faceswap_preview_CodeFormer_1791184533405190500.png`; reviewer APPROVE y sd-tester PASS (2026-10-05).
- [x] Los logs de calidad nombran el procesador ejecutado, incluido el fallback, y la dimensión real de su salida; derivación desde `enhancer_key` revisada y CodeFormer 512 px verificado en el log real.
- [ ] El perfil de `1994859933.jpg` mejora visualmente en proporción, nitidez y parecido con la fuente; se conservan oclusiones y personas ajenas.
- [ ] El cambio visual se verifica en fotos adicionales y los vídeos representativos completos; un clip corto o una métrica de cambio no cierran el defecto.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS; evidencia y límites en MEMORY.md.

## H-EDIT-003 — Edición localizada estable con Turbo

Mismo alcance confirmado el 2026-10-05. Se conserva la elección previa de Turbo y la localización por conceptos y regiones medidos.

Recordatorio del usuario el 2026-10-05: el Editor debe ser versátil porque puede pedir cualquier edición. «Calvo» es un caso de reproducción del defecto; la corrección debe conservar la capacidad general de interpretar instrucciones.

Diseño aprobado el 2026-10-05: cuando una edición de región no encuentra zona existente, el modelo de lenguaje ya instalado propone una zona mencionada en la instrucción y el segmentador la comprueba en la foto. El usuario eligió «Sí, modelo local y comprobación de la zona» tras revisar la alternativa de añadir una taza sobre la mesa. No sustituye la instrucción ni descarga modelos.

Ampliación aprobada explícitamente después: descargar y probar Qwen2.5-3B-Instruct Q4_K_M para interpretar zonas (2,1GB, repositorio oficial Qwen). Si mejora la evidencia real, su configuración se limita a propuestas de región; los clientes de Generate y el rewriter general conservan sus ajustes. Parámetros y ruta del modelo en configuración, sin descarga automática desde el flujo del editor.

Continuación aprobada por el usuario el 2026-10-05 con «continua», tras presentar la propuesta y solicitar aprobación: usar el mismo modelo3B para traducir la instrucción completa únicamente en la ruta de región ambigua. Diagnóstico CPU inicial de nueve instrucciones conserva objetos/ubicaciones y evita planta→floor, pero suaviza modalidad en dos casos; un prompt posterior mantiene must/must not en diez casos reales (`region_llm_1791222472492290000.json`). Mantener medición de la región desde RAW y abstención si falla, sin sustituir el traductor general ni afirmar calidad visual sin render real.

- [x] En la ruta ambigua, la región se mide desde RAW y solo la instrucción completa traducida con el modelo local configurado llega al generador. Modelo cerrado antes de ComfyUI; ausencia, error, respuesta vacía o truncada conservan la foto con mensaje explícito. Resto de rutas conserva su traductor. Reviewer APPROVE acotado y tester PASS de integración/contratos (2026-10-05).
- [x] La traducción integrada conserva acción, objetos, ubicaciones, obligatoriedad y negación en diez pruebas CPU reales (`region_llm_1791223077064804600.json`); manager real `versatility_remove_plant_translated3b_20261005` manda plant y conserva personas, pero eliminación parcial con manchas, FAIL visual (91s/1,25 %). Este criterio verifica la conexión y su evidencia; no cierra la calidad de la historia.

Continuación aprobada por el usuario tras la propuesta: «quiero que sigas para solucionarlo todo». Se explicó que añadir/quitar objetos son las ediciones estructurales: extender la receta8/referencia espacial ya usada para apariencia y recolor a esas ediciones con región medida. A/B aislados con misma ventana/máscara/prompt/seed: planta45s elimina mejor copa y manchas, pero conserva restos; taza43s añade taza con asa más visible, personas conservadas. Tester PASS acotado de mejora en una escena, sin calidad universal. Evidencia `plant_native_recipe_1791223360498195100`, `cup_native_recipe_1791223582924147400` y sus composiciones `local_replay_compare_1791223496827217600`/`1791223740393913300`. Parámetros en configuración de región, sin cambiar rutas globales ni decidir por vocabulario; verificar manager completo y continuar cobertura del objeto.

Ampliación de pruebas aprobada explícitamente el2026-10-05: «Sí, descargar y probar ambos» para SAM2.1HieraTiny156MB (repositoriooficialMeta) y Hyperswap1a256384MB (FaceFusionoficial). SAM2 se compara desde puntos derivados de máscaras reales y guarda todas las alternativas: comprobar objeto completo, exclusión de otraspersonas y tiempo, con pesoslocales y sin paquetesnuevos. Hyperswap1a reutiliza alineación/normalización/probeCPU revisado y se compara con1c/baseline sobre perfil/frontal e identidad. Esta aprobación de descarga/prueba no declara mejora ni sustituye producción; integrar solo tras evidencia visual/review y decisión concreta del usuario.

- [ ] Si la medición directa de una edición estructural falla, se aceptan únicamente propuestas del modelo local presentes en la instrucción y medidas por el segmentador; una región pequeña válida puede servir de soporte. Parámetros en configuración y abstención si no hay zona.
- [ ] La ruta integrada añade un objeto en una foto real sin sustituir las personas ni el resto de la escena; se prueba otra petición estructural y los rechazos de propuestas inventadas, ausentes o saturadas.
- [ ] Un fallo de carga o generación Turbo devuelve el error sin cambiar silenciosamente al motor Full.
- [ ] Una propuesta de zona que cubre una cara existente no puede usar a esa persona como superficie para insertar otro objeto; la continuación conserva el mismo rechazo.
- [ ] El router corporal recibe los conceptos de vista/postura medidos por el router general, sin variables no definidas al usar Kontext.
- [ ] Si los conceptos medidos sugieren una región pero empatan con apariencia corporal sin pedir cuerpo/postura, se comprueba una propuesta basada en la instrucción original antes de generar. Sin región verificada, no se repinta la foto entera; una traducción incorrecta no sustituye esa instrucción.

- [ ] La decisión de ruta se apoya en conceptos y regiones medidos, sin listas de palabras, prendas, géneros ni tipos de petición escritos a mano. Se verifican ediciones reales distintas de quitar pelo, además de los contratos de despacho.
- [ ] Si el segmentador mide una parte solicitada dentro del sujeto, un recolor usa esa parte y conserva el sujeto restante; no extiende su máscara a toda la persona ni elimina la parte por pertenecer a una persona.

- [ ] «El hombre debe ser calvo» cambia toda la superficie pedida sin pelo residual, franja translúcida ni cabeza añadida.
- [ ] Se conserva la pose y la otra persona en la foto de 800×533 y su versión 1248×832, con varias semillas y fotos adicionales.
- [x] Una corrección de workflow se compara con la misma foto, máscara y semilla antes de integrarse; control 8 pasos sin referencia frente a 8 con referencia (424242), otras dos semillas y manager real 800×533; parámetros en `longcat.appearance_edit` y `longcat.steps`.
- [x] Generate, ediciones compuestas y cambios de vista conservan sus contratos y probes verdes: Generate 72/0 y 31/0; dispatch compuesto y viewpoint exit 0; workflow localizado verifica VAE de foto y referencia `index_timestep_zero`.
- [ ] Compilación Python, revisión APPROVE y verificación PASS; evidencia visual real, validación del usuario y bitácora.

## H-FS-004 — Causa raíz warp/modelo (nueva, aprobada 2026-10-05)

Continuación de tracking autorizada «hazlo»2026-10-06: respetar corte duro solicitado aunque closeup esté activo, limpiar boca/oclusión/fotograma/hold del plano anterior y conservar referencias/fuente y estado de otros vídeos. Reinicio suave explícito de zoom conserva continuidad. Comprobar cortes en frames sin detección antes de mantener composición antigua. Exigir contrato antesFAIL/despuésPASS, probes/lint/compilación, reviewerAPPROVE/testerPASS y secuencia normal real para tracking/bocas; vídeo completo con evidencia y revisión visual para cerrar calidad. No cambiar modelos ni umbrales sin causa reproducida; no certificar tracking con fixtures de estado o clips cortos. Registro `diagnostics/out/tracking_reset_fix_20261006/`.

Verificación de esta corrección acotada2026-10-06: reviewer APPROVE35/0; tester PASS453 reset +27 integración, baseline54FAIL; lint/compilación afectados verdes. Inferencia CUDA completa antes/actual sobre vídeo normal134frames (130swaps por versión),0px de diferencia en clip sin cortes; inspección root de todos los renders. Defecto de estado reproducido/corregido, continuidad corta sin regresión. H-FS-004 permanece abierta: falta vídeo largo/cortes nativos y validación visual del usuario; pérdida de detección26–29 observada y declarada en spec/03.

Continuación de revisión de bocas solicitada2026-10-06 («revisa los logs para mejorarlo»): corregir uso de EMA/histéresis/hold oral en fotos y frames sin smoothing, conservar protección de objetos medida y mostrar evidencias/motivo por foto. Verificar código real y retratos normales; no inferir de ratio0,15/restore0,91 que el objeto sea falso sin occ/rel/ring. No cambiar umbrales ni modelos con ese log incompleto.

Continuación «continua»: corregir inconsistencia geométrica del anillo peri-bucal al ampliar el crop con GFPGAN. Misma huella256→512 cambia ring0,065→0,349 y activa gate0,30 sin evidencia nueva. Escalar radios del anillo con dimensiones reales raw→referencia; tamaños base al config local, umbrales/paste/modelos vigentes conservados. Exigir regresión de escala con controles de labios aislados y oclusión que cruza labios, comparaciones con fotos normales/modelos reales y reviewer/tester independientes. Esto no acredita semántica universal ni el destino504612489.

Diagnóstico posterior de bocas realmente abiertas: MediaPipe medía puntos14→17 (grosor del labio inferior), no abertura13→14. Cuatro fotos normales con inferenciaCPU real confirman ratios erróneos0,10–0,15 frente a abertura0,25–0,61, corroborado por topología oficial. Corregir constantes y campos de labios0/13/14/17, mantener denominador, umbrales, contorno de composición y modelos; verificar abiertas/cerradas y protección real, reviewer/tester y comparación visual sin declarar identidad universal.

Verificación acotada de esta continuación: escala APPROVE/PASS56contratos y206verificaciones reales; apertura APPROVE27controles/PASS147verificaciones, ochoA/BCPU con128/XSeg/GFPGAN y MediaPipe reales. Tres bocas abiertas mejoran visualmente; dos cerradas y pelo03206 conservan0px, positivo parcial conserva paste0,895. Primer control P407 mal etiquetado cerrado y primeras corridas con hash previo distinguidos en bitácora. H-FS-004 permanece abierta por vídeos completos, perfiles/identidad/oclusión difíciles y caso0,91. Evidencia `diagnostics/out/mouth_math_review_20261006/summary.json`.

Verificación oral acotada2026-10-06: reviewerAPPROVE/testerPASS52/0; lint/compilación/probes verdes y ocho casos con128/XSeg/GFPGANCPU reales. Tres controles con igual flag producen0px distintos entre historia limpia/previa; vídeo suavizado conserva memoria. Primera aserción fallida y confusor de colorLAB documentados. Pendiente causalidad del0,91, objetos orales y calidad visual/vídeos; H-FS-004 permanece abierta. Evidencia `diagnostics/out/mouth_review_20261006/summary.json`.

Continuación aprobada el2026-10-06 por «DALE», tras auditoría reviewer CHANGES_REQUESTED: retirar fallback Hyperswap incompleto y restaurar rescate128 de referencias sanas, corregir convención/escala de alineación y conectar configuración, eliminar regeneración elíptica de las máscaras fallidas, y revisar falsos rechazos de fuentes con modelos locales. Corregir lint/compilación identificados y verificar extracción real, regresiones, reviewer y tester. No instalar modelos/dependencias ni sustituir producción128. La aprobación de reparar no cierra calidad visual ni vídeos.

Corrección acotada verificada2026-10-06: extracción real6/7 (una cara realmente cortada), lint/compilación y11probes verdes; reviewerAPPROVE y testerPASS acotados. Composición sana en dos retratos normales con modelosCPU reales, antes/actual idénticos; evidencia `diagnostics/out/source_fix_20261006/summary.json`. H-FS-004 permanece abierta por los criterios visuales siguientes; la referencia explícita no se utiliza en nuevos renders, se requieren muestras normales de perfil/oclusión.

**Como** usuario **quiero** que el perfil `1994859933.jpg` y los vídeos completos mejoren proporción/nitidez/parecido **para** cerrar el defecto que los gates/máscaras no corrigieron.

Diagnóstico que la motiva (MEMORY.md + código): la alineación `norm_crop2` usa solo 5 `kps` (`roop/ProcessMgr.py:4747-4776`); con `eye_rel=0,028` genera warp `1,6×` desplazado y proyecta `512 px` sobre ventana `~1246 px`. El gate `0,05<eye_rel<1,20` + rescate `_swap_from_saved_reference:4662-4712` detecta pero no re-estima `M`; `diagnostics/faceswap_batch_first_rescued.jpg` sigue defectuosa. Los fallbacks recrean elipse (`boost 0,92`, `broad_floor 0,65`, `safety recovery`, `create_soft_mask` en `roop/quality_enhancements.py:275-304`) y el doble `_limit_face_paste` solo contiene (`12 %/16 %` frontal, `6 %/10 %` perfil). `128` estándar `MAE 0` vs `facefusion`; `256` no conserva identidad (`-0,016` vs `0,450`); CodeFormer/GFPGAN y erosiones `11/21 px` no corrigen visualmente.

- [ ] Re-estimar `M` con `bbox+kps` (o `FFHQ/68pts`) o cambiar el modelo `128` por uno con identidad en perfil probada; no cerrar con otro gate/máscara.
- [ ] El fallback de máscara colapsada no puede ser elipse; sin halo/blur elíptico fuera del rostro.
- [ ] El perfil de `1994859933.jpg` mejora visualmente en proporción, nitidez y parecido con la fuente; se conservan oclusiones y personas ajenas.
- [ ] El cambio visual se verifica en fotos adicionales y los vídeos representativos completos; un clip corto, una sola foto o una métrica de píxeles/cambio no cierran el defecto.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS visual del tester; un probe verde sin PASS visual no cierra la historia.
- [ ] Parámetros nuevos en `config/`, evidencia y límites en `MEMORY.md`, validación visual del usuario.

## H-EDIT-004 — Objeto completo antes de receta (nueva, aprobada 2026-10-05)

**Como** usuario **quiero** retiradas/añadidos con objeto completo y una sola receta `remove/add` **para** no dejar restos ni redibujar la escena.

Diagnóstico que la motiva: `CLIPSeg` mide núcleo (`planta copa ~2 %`, `silla 3,26 %` sin ruedas, `mesa 1,2 %` rechazada por piso `4 %`); `ISNet 1,93 %→28,82 %` toma ambas personas; `visual_forward 62,7 %` escena; `SAM2 CPU 2,1-2,4 s` mejora contorno sin objeto completo; `GrabCut` solo diagnóstico. Receta `8 pasos/ref espacial regional` conserva personas pero deja `tronco/maceta/hojas`, `respaldo hueco`, `anillo claro`; `12/no_ref global` retira pero mueve escena; `ruido ventana completa` redibuja. `Calvo 800×533`: contar caras `2 vs 3` no prueba calvicie; `424243` evita cabeza extra pero deja pelo parcial; `fullsurface` crea cabeza ajena.

- [ ] Cerrar objeto completo (`SAM2+GrabCut` integrado) con `IoU/cobertura` medida y exclusión de otras personas antes de tocar `denoise/steps`; no ampliar solo `paste`.
- [ ] Una sola receta `remove/add` con preservación medida: retira/añade sin restos, sin mover personas/objetos vecinos, sin segunda cabeza ni franja gris.
- [ ] Verificación en varias semillas y fotos adicionales además de la escena de referencia; contratos `CPU/dispatch` en verde no equivalen a calidad visual.
- [ ] Probes pertinentes y compilación Python verdes, revisión APPROVE y verificación PASS visual del tester; evidencia visual real, validación del usuario y bitácora.
