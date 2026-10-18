# spec/02 — Requisitos (historias)

Decisión H-FS-005 (2026-10-17): el usuario confirma conectar Similitud al matching destino conservando exactamente los umbrales vigentes en el valor por defecto. Menor valor exige mayor similitud; selección de fuente y márgenes de ambigüedad intactos. Confirma retirar «Previsualizar Swap (Realtime)», conservando los previews existentes.

Trabajo activo: corrección de los fallos del proyecto autorizada el 2026-10-13 («arreglalo todo», confirmado: fallos del proyecto). Las historias siguientes conservan sus criterios y límites de validación.

## Completar código pendiente — 2026-10-18

Instrucción del usuario: «SIGUE HASTA DEJARLO SOLO PRUEBAS. TODO EL CÓDIGO».
Completar las correcciones pendientes en los flujos existentes y dejar una lista
verificable de pruebas reales y validaciones visuales. Conservar elecciones
vigentes de motor/modelo, compatibilidad Android y cómputo local. No instalar
paquetes/modelos, hacer commits ni reiniciar procesos activos.

- [ ] Cámara: las poses ya admitidas de altura, distancia y combinaciones de
  orientación se conservan mediante medición de conceptos, sin regex de usuario;
  parámetros de decisión en configuración y contratos de regresión.
- [ ] Editor/Generate: corregir fallos reproducidos del código pendiente y
  completar configuración de los defaults ya elegidos. Preservar máscaras
  manuales, instrucción completa, personas/regiones no solicitadas y APK.
  Aclaración posterior del usuario: las máscaras se habían eliminado de la
  edición global Kontext. No reintroducir pintado manual en la UI; escritorio
  y Android conservan su entrada Image y sus trece/ocho/nueve parámetros.
- [ ] FaceSwap: contrastar deudas con implementación vigente; completar los
  defectos de código reproducidos sin cambiar modelo/identidad ni recuperar
  soluciones rechazadas. Separar comprobaciones de código y calidad del modelo.
- [ ] Lint, compilación, contratos pertinentes, revisión independiente y bitácora;
  actualizar spec con pendientes actuales en vez de marcar como resueltos
  defectos visuales todavía observados. Las pruebas pueden descubrir más código
  necesario: un defecto de calidad conocido no se convierte en «solo pruebas».

## Continuación del Editor — 2026-10-18: contrato compuesto de Kontext

Continuación solicitada con «CONTINUA». Se verifica el contrato previo documentado
en `diagnostics/probe_editor_kontext_default_20261018.py`: Kontext sin máscara
manual recibe la instrucción compuesta completa, como edición global por prompt.
El splitter legado no debe convertirla en varias subediciones que omiten cláusulas
sin región confirmada. Se conservan el gate de cámara y las máscaras manuales.

- [x] Contrato CPU rojo→verde: rewriter con varias regiones, una generación
  Kontext y ambas cláusulas presentes; sin recursión por regiones.
- [x] Prueba GPU no explícita de sustitución de chaqueta por blazer roja:
  guardar traza, salida única y medición de rojo; la variación del lienzo por sí
  sola no acredita obediencia. Inspección visual y límites en MEMORY.md.
- [x] Lint y compilación afectados, regresión pertinente, reviewer y tester.
  No cerrar calidad general ni la revisión de cámara con esta comprobación.
- [ ] Blazer de corte exacto, casos adicionales y validación visual del usuario:
  la muestra genera cuero rojo, pero conserva corte y costuras de cazadora.

## Continuación de correcciones — 2026-10-13

Dentro de H-EDIT-003/H-EDIT-004 y las deudas ya declaradas, sin sustituir Turbo ni el modelo FaceSwap y sin instalar dependencias:

- [ ] Toda edición corporal localizada conserva el ancla espacial, tanto con parte medida como con persona de respaldo; se comprueba el workflow real.
- [ ] Las superficies no consultan objetos fijos ajenos a la instrucción; la protección de otras personas usa la evidencia visual existente. Ningún mensaje atribuye SAM2/GrabCut a una operación que no los ejecuta.
- [ ] Los compuestos conservan todas las zonas medidas de cada cláusula; una petición explícita de postura mantiene el contrato de permitir nueva geometría.
- [ ] El reintento corporal comprueba cambio dentro de la zona pedida; cambios colaterales no cuentan como aplicación. Los fixtures respetan el tamaño real de la ventana.
- [ ] Los controles existentes de Generate, FaceSwap y Animate corresponden a capacidades ejecutadas o explican su límite antes de iniciar trabajo.
- [ ] Lint/compilación, probes pertinentes, reviewer APPROVE y tester PASS, con logs de la versión final. Las mejoras visuales requieren muestras reales adicionales y validación del usuario; los contratos no las certifican.

Prioridad confirmada después: revisar FaceSwap por caras y calidad incorrectas en las salidas de `output`. Corregir el contrato general de geometría de H-FS-004: los cinco landmarks deben pertenecer a su bbox, independientemente de confianza y orientación. Reutilizar el margen de contención vigente, declarado en configuración; rechazar o rescatar desde una referencia válida antes de estimar la matriz. No sustituir modelo, identidad, escala ni umbrales de boca. Verificación con contratos CPU y retratos no explícitos; la revisión del lote no autoriza regenerar imágenes sexuales explícitas.


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
- [x] Si hay imagen **e** instrucción de edición (añadir/quitar persona u objeto, ropa, fondo, POV/ángulo, pose): se enruta a `generate_intelligent()` del Editor y el resultado se muestra en Generate. **Actualización 2026-10-16**: la ruta por defecto pasó de `auto` a `img2img` (`generate_ui.default_route` en `config/generation_prompt.json`) porque el usuario quiere Generate y Editor **paralelos e independientes**: Generate ya no enruta sola al Editor; `editor`/`auto` son elección manual en el desplegable. Verificado: UI construida con `route.value='img2img'`, adherencia 108/0 y contrato APK 31/31.
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
- [x] A/B real sobre el motor por defecto (2026-10-16): `probe_gen_castillo_real.py` parametrizado por motor (`--engine`, default `get_default_generation_engine()`); `flux_dev_q4` semilla 123456 con NSFW OFF/ON produce salidas **byte-idénticas** (SHA256 `7db9c50e…`; el ON lo sirve la caché de ComfyUI porque el workflow final es el mismo) y la imagen es el castillo con unicornios en estilo dibujo. El checkbox ya no secuestra el prompt no adulto en la ruta por defecto. Con `flux_dev_abliterated` el prompt también llegó idéntico ON/OOFF, pero ese motor renderiza una mancha adulta incluso con OFF (cautela documentada; no es el default).

## H-GEN-003 — LoRAs instaladas y compatibles en Generate (2026-10-17)

Continuación autorizada: revisar las LoRAs locales antes de descartarlas/copiar (sesión OpenCode del 17/10) y «continua y solucionemos los problemas de Generate». Corrección de disponibilidad y compatibilidad; no cambia el motor por defecto ni los gates del prompt. El usuario está probando POV: no reiniciar UI ni ocupar la GPU para estas comprobaciones.

- [x] El selector ofrece únicamente archivos instalados compatibles con el motor elegido; cambiar de motor conserva una selección válida o vuelve a Auto.
- [x] Una selección manual ausente, inválida o incompatible devuelve error antes de subir imágenes o enviar workflows. La elección válida conserva la fuerza del catálogo y no depende del clasificador de intención.
- [x] Las dos LoRAs de estilo ya existentes en WebUI se copian sin sobrescribir a la carpeta de ComfyUI, con hashes comprobados; el catálogo declara SDXL/Pony y selección manual, sin añadirlas automáticamente a escenas.
- [x] Auto explica por consola cuándo el motor no admite el catálogo automático SDXL; ninguna selección manual se ignora en rutas delegadas.
- [x] Los cuatro modificadores extra seleccionados llegan al ensamblado (límite en configuración); no desaparecen plano/color por los defaults de luz/piel. Las opciones incompatibles se ajustan en los cuatro selectores y el preview inicial muestra los defaults.
- [x] LongCat conserva CFG real del JSON/override; FLUX.1 mantiene FluxGuidance y su negativo a CFG=1 se explica en UI/logs. HD/4K se identifica como perfil de acabado, sin prometer una resolución final que el flujo no garantiza.
- [ ] Contratos de disponibilidad/compatibilidad rojo→verde, regresión Generate/APK, lint y compilación verdes; reviewer APPROVE y tester independiente. La carga GPU y el efecto visual siguen sin verificarse hasta prueba real en un momento que no interfiera con el Editor.

## H-FS-005 — composición y controles FaceSwap (2026-10-17)

Alcance autorizado por «SOLUCIONALO» tras la auditoría del 17/10. Conservar modelos, identidad origen, selección destino, umbrales vigentes y funcionamiento de Generate/Editor/Android; no instalar dependencias ni reiniciar la UI activa. Las decisiones sobre el slider «Similitud» y preview realtime se consultan al usuario mientras avanzan las correcciones independientes.

- [x] Componer una segunda cara conserva swaps previos fuera de su zona, también ante saltos/errores de esa cara; el original sigue siendo la referencia para detección, oclusión y color (contratos CPU).
- [x] Vídeo principal y fallback reciben el valor real de suavizado temporal desde UI/core; llamadores antiguos conservan el default existente (contratos CPU).
- [x] Auto-Rotar desactivado omite solo pasadas rotadas, conservando detección original y rescates de escala; activado conserva el comportamiento anterior (contratos CPU).
- [x] El rescate frontal localiza componentes de la máscara ya medida, exige evidencia oral/anillo y cobertura vigentes y se abstiene si no hay un único candidato; no repite el mismo detector ni amplía el gate del 40 % (contratos CPU).
- [x] Código de vídeo usa máscara alfa suavizada conservando textura del parche; no desenfoca los píxeles del swap en la transición (contratos CPU).
- [ ] Verificar un vídeo completo permitido, sin equiparar un clip corto con certificación de perfiles/cortes largos.
- [x] Controles de similitud/preview corregidos según la decisión del usuario del encabezado, sin un éxito simulado ni controles que aparenten funcionar (Gradio y callback reales aislados de inferencia).
- [ ] Probes rojo→verde, lint y compilación afectados verdes, reviewer APPROVE y tester PASS. Registrar inferencias reales y evidencia visual permitida, límites y pendientes en MEMORY.md.

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
