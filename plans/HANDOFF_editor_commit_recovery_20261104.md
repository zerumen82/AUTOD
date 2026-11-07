# Relevo — recuperar el arreglo previo de Image Editor, 2026-11-04

## ESTADO FINAL MÁS RECIENTE — 05/11, v5.1308

La recuperación y las comprobaciones nativas terminaron. Los apartados
inferiores con renders activos o decisiones pendientes son historia.

- Auto vuelve a la ruta focal Kontext para apariencia pura medida. Manual,
  selección, cámaras, cuerpos, objetos, compuestos y motores explícitos
  conservan sus rutas. Plantilla regional genérica ya aplicada.
- También se corrigió la máscara sobrescrita por el auxiliar vacío, que se
  eliminó junto a la definición shadowed sin alterar el extractor efectivo.
- Prueba productiva FINAL, sin override: carpeta
  `appearance_reference_20261023/1793866121580301900`. Exec64041 exit0.
  Root PASS acotado: cabeza rapada, sin bloque/halo amplio, exterior0;
  39.725 píxeles cambiados, una llamada de20 pasos, 283s/371,297s callback.
  Ver `dispatch_0.json`, `metrics.json` y `real_auto_astronaut_production_1308.log`.
  Tester independiente final **PASS acotado**, guardado en
  `auto_appearance_recovery_20261104/native_auto_production_1308_verdict.json`.
  Confirma imagen FINAL, prompt productivo sin override, exterior0 y hashes.
  Brillo/textura lisa y vello corto no se presentan como perfección.
- Grace controlada mejoró con la plantilla aplicada: PASS acotado de
  calvicie/gafas/rasgos/no halo y exterior0. Sigue retirando gorra y dejando
  vello muy corto; no es preservación absoluta ni rasurado0pelo.
- Build1308, Ruff/compile pertinentes,10 suites compartidas y28 controles
  engine_status verdes. Reviewer APPROVE técnico. Veredictos/evidencia en
  `diagnostics/out/auto_appearance_recovery_20261104/`.
- Cola ComfyUI0/0 al terminar; no hay prueba propia activa. No reiniciar ni
  repetir renders por leer los identificadores antiguos inferiores.
- Sin commit/push ni instalaciones. Cargar v5.1308 al reabrir AutoAuto si
  había instancia anterior. Mantener límites y aceptación del usuario abiertos,
  sin volver a pedirle que elija detalles internos de motor.

## MÁS RECIENTE — v5.1308 (05/11)

- CandidataGrace1793865362633754700 PASS **ACOTADO** independiente: calvicie,
  gafas/rasgosmejorpreservados,nohaloamplio,exterior0. Gorradesaparece y hay
  fuzzlateralmuycorto; no es conservaciónabsoluta/rasurado0pelo. BaselineFAIL
  no se borra. Controls independiente confirma source/reference/maskidénticos,
  seed864801;WFdifieresolotextoyfilenameequivalente.
- Exacta plantilla genérica ensayada YA APLICADA a edit.prompt_template,
  versiónv5.1308 en AGENTS/run/core; no cambio de modelos/default/máscaras.
  ReviewerAPPROVE técnico. Snapshotbefore_prompt_scope_20261105.
- **ACTIVO exec64041**: Autoastronautnativo SINoverride CLI, log
  auto_appearance_recovery_20261104/real_auto_astronaut_production_1308.log.
  Localizar carpeta nueva, leer dispatch0y métricas, abrirFINAL ytester.
- Testerejecuta regresionescompartidasporplantilla(apari/selección/recolor/
  add/remove/multiclause/Auto), no repetir otrasporcostumbre.
- Build1308pendiente(verlognuevo si ya se lanzó), Ruff/compile probe verdes.
- Importante observaciónreview: candidataen47871nohabía persistidoJSONantes
  inferencia(solo callsenmemo y metricsalfinal); suworkflowreal se guardó en
  marcha. Probeahoraescribedispatch_N.json ANTESgenerateparapruebasnuevas.
- Concluir solo trasAutoactualfinalyverificación técnica; aceptaciónusuario/
  calidaduniversalsiguenabiertas. Nosehizo commit/push ni reinicio/instalación.

## ACTUALIZACIÓN 05/11 — v5.1307 y prueba de segunda foto

- Auto astronaut FINAL1793822106826892700 **PASS acotado independiente**:
  38877changed/exterior0,1llamada20pasos165s/251,015scallback;frentebrillante.
- Auto Grace FINAL1793864753014300900 **FAIL visual**: cara cambiada,
  gafas perdidas,pelo conservado congorra. Exterior0 noimplica cumplimiento.
  No cerrar calidad general ni afirmar dosfotosverdes.
- **v5.1307**: corregida máscara que se perdía por auxiliar léxico vacío en
  selected_region_edit. Retirada reasignación+stub+definición shadowed; métodos
  efectivos de extracción intactos. Fixture editado representa cambio real;
  guard no-op no se desactiva. Nuevoscontratosarriba/abajo/objetivofuera soporte.
- Tester28engine_status/15Auto/13body/15selectionPASS, reviewerAPPROVE acotado,
  Ruff/compile ybuildaislado verdes. Evidencia en auto_appearance_recovery_20261104,
  `target_mask_results_20261105.json`, `v51307_*.log`, `build_1307_20261105.log`.
- **ACTIVO exec47871**: variante neutralGrace con misma semilla864801 del
  historyoriginal y restricción genérica de mantener atributos no solicitados.
  `real_auto_grace_preserve_20261105.log` dentroauto_appearance_recovery_20261104.
  CLI probeextiende --seed/--prompt-template; se registranlosparámetrosreales.
  No cambió plantilla productiva. Fuente inglés equivalente vs españolanterior:
  comparar RGBreferences/masks/workflowsantesafirmarA/B de soloprompt.
- Snapshot anteserrorlocal: before_target_mask_20261105. No volver a introducir
  el auxiliar vacío ni dar por resuelto objeto local solo porque máscara pasa.
- Pendiente: terminar variante, inspeccionar final, comparar control si posible,
  valorar si mejora genérica mereceaplicarse. Si falla, no acumular prompts ni
  cambiar a otro motor sin evidencia. Mantener límites y usuario informado.

## LEER PRIMERO — continuación v5.1306 (supera preguntas pendientes inferiores)

El usuario respondió «no sé… funcionaba y ahora cada vez peor» ante elegir
motores. Se retoma el arreglo mínimo solicitado con la recomendación probada,
sin seguir cargándole decisiones internas. **La recuperación ya está aplicada**;
los apartados inferiores que dicen «no aplicado/pregunta pendiente» son historia.

- Versión actual **v5.1306**, no v5.1305.
- Config `editor_ui.automatic_appearance_engine=kontext`; default sigue Auto.
- Helper `_automatic_appearance_engine(raw, translated, current_engine)` y
  redirección después de described_subject, antes de Turbo. Solo apariencia
  pura medida, sin máscaras/compuestos/intenciones incompatibles; permiso
  automático puntual. Explicit engines y manual/cámara/otras rutas intactos.
- Snapshot before_current de 5 archivos para revisar el cambio de este turno:
  `diagnostics/out/auto_appearance_recovery_20261104/before_current/`.
- Nuevo probe `diagnostics/probe_editor_auto_appearance_recovery_20261104.py`:
  RED previo,15tests GREEN. Ruff/compile verdes; reviewer **APPROVE acotado**.
- Build exec18611, log `auto_appearance_recovery_20261104/build.log`; confirmar
  terminación antes de reportar verde.
- Tester ejecuta regresiones; engine_status mantiene2fallos corporales que
  debe separar causalmente, no ocultar ni ampliar esta reparación para tocarlos.
- **Prueba REAL Auto activa exec16398**, log
  `diagnostics/out/auto_appearance_recovery_20261104/real_auto_astronaut.log`.
  Callbackreal astronaut512, modelo elegido por Auto, no intervención de motor
  en el generador. Probe real existente añade --engine y métricaengine_requested.
  Encontrar carpeta fresca por EVIDENCE/log o último directorio del conjunto
  appearance_reference_20261023; inspeccionar FINAL y pedir tester independiente.
  No confundir PASS Context explícito anterior con esta prueba Auto.
- No reiniciar/interrumpir UI/renders sin comprobar primero estado/propiedad.
  Si se toca más código, actualizar versión/bitácora según instrucción usuario.
- Terminar pruebas, actualizar spec/02 H-EDIT-AUTO-RECOVER, MEMORY y este relevo.
  Si Auto pasa, comunicar corrección concreta y límites de una muestra; no
  afirmar resolver universalmente el mapa de versatilidad.

## Petición vigente y prioridad

El usuario insiste: «pero no lo has arreglado. hay un commit que dije que
pusiera esta semana creo que arregló Image Edit». Después pide anotar todo
antes de agotar tokens. Esto NO cancela el arreglo: la documentación es una
salvaguarda para seguir, no el resultado final del trabajo.

Hay que recuperar el comportamiento visual bueno del arreglo anterior.
NO volver a cerrar la tarea diciendo solo que HEAD no importa o que pasan
contratos CPU. El usuario tiene razón: el defecto visual continúa abierto.

Leer AGENTS.md global/proyecto, spec/ y MEMORY.md. Skill fix ya aplicada.
No instalar paquetes/modelos, no hacer reset/checkout sobre archivos actuales,
no hacer commit/push nuevo sin orden, no interrumpir UI/renders compartidos.
No generar ediciones de las fotos explícitas: las pruebas usan astronaut/Grace
neutrales instaladas. Mantener todas las rutas de edición y personas.

## Workspace y cambios ya hechos

- CWD `D:/PROJECTS/AUTOAUTO`; rama `experiment/klein-9b`.
- HEAD `4d7ce3c75caa5fc15f3bd366f83f8b0603f20a44`, v5.1280, 03/11 09:07.
- MUCHOS cambios locales previos sin commit: conservarlos.
- Versión actual de código **v5.1305**, coherente en AGENTS.md, run.py y
  roop/core.py. No modificar otra versión en silencio si se vuelve a tocar
  producción: el usuario exige versionar y registrar los cambios.
- Corrección anterior aplicada en manager: refino conserva latente foto y
  receta `appearance_sample`; retira override de no_kontext y argumentos
  muertos. Conserva pegado completo de región y guardias faciales.
- Retirados helper `_expand_measured_region` y única llamada: dilataba sin
  límite de sujeto y excluía todas las cajas faciales, incluso la objetivo.
- Dos probes: `diagnostics/probe_editor_focused_refine.py` ampliado y nuevo
  `diagnostics/probe_editor_measured_scope_dispatch_20261104.py`.
- Reviewer APPROVE y tester PASS CPU acotados al cambio anterior; 9 probes
  finales, Ruff crítico/ambos probes completos, compilación y build aislado
  verdes. No aprobación global: deuda Ruff previa y un probe added_face
  falla antes del refino. Leer evidencia, no inventar suite global verde.

## Resultado visual anterior: sigue FAIL

Carpeta `diagnostics/out/refine_recipe_20261104/`:

- `neutral_real/`: pelo rojo, máscara ampliada al 87 %, resultado incorrecto.
- `neutral_appearance/`: calva alrededor de bloque frontal de pelo original;
  refino omitido por máscara demasiado grande. Raw ya tiene el defecto.
- `neutral_without_expansion/`: mismo astronaut ampliado a 2048 y semilla
  424242; máscara baja de 1.212.918 a 364.076 px, reaparece refino real
  1064x1268 con mask=foto/ref=index_timestep_zero. Dos inferencias 82+87 s.
  Desaparece bloque frontal, PERO quedan restos claros y orla: FAIL visual.
- `tester/final_results.json`, `tester/verdict.json`, `reviewer.json`,
  `build_final.log`: evidencia técnica anterior.
- El tester corrigió un harness con Ran 0 tests; intento inválido preservado.
  El final ejecuta 1 test/6 subtests de despacho y 30 checks de refino.

## Búsqueda del commit que recuerda el usuario

Se revisaron git log --all y reflog desde 26/10. No limitarse al último.

| Commit | Fecha | Contenido relevante |
| --- | --- | --- |
| `d3b4f8a` | 27/10 10:40 | selected_region_edit: cabeza completa, inclusión en selección y `_blend_head_boundary` NORMAL_CLONE |
| **`68ad5fb`** | **27/10 11:17** | **Mejor candidato respaldado por evidencia:** ruta GLOBAL focal Kontext, escala al presupuesto, plantilla regional, protección del referente, crop_max_area y armonización del borde |
| `1dacf32` | 27/10 11:35 | pasa recipe a `_vision_zone_pass` |
| `00d1924` | 03/11 00:20 | lote manager+memoria v5.1273; INTRODUCE en Git Auto→LongCat y deja de normalizar longcat→Kontext |
| `4d1cb41` | 03/11 00:28 | «Image Editor P0 mejoras v5.1274»: fallback/umbral/validación posteriores; no origen del arreglo Kontext validado |
| `4d7ce3c` | 03/11 09:07 | solo versión en run.py |

Reviewer confirmó de forma independiente: los arreglos Kontext siguen en
el código actual, pero Auto→LongCat en GPU <=10GB entra en otra ruta que
devuelve antes; no ejecuta los bloques `engine == kontext` de escala/plantilla/
armonización. Esta es una pista causal más útil que «HEAD no importa».

MEMORY Oct24/25 conserva PASS visual acotado REAL con Kontext:

- Global astronaut `diagnostics/out/appearance_reference_20261023/1792858682799844700/result.png`
  (inspeccionado otra vez, cabeza rapada sin parche/elipse notoria).
- Repetición global `1792952153054322000`, motor Kontext registrado.
- Selección `1792855675823830800`, mismo conjunto de carpetas.
- Métricas guardadas en cada `metrics.json`; referencia global 753x768,
  20 pasos, prompt regional completo, modelo Kontext Q4_K_M.
- `plans/HANDOFF_editor_global_20261024.md` detalla implementación/historia.

Estos PASS NO acreditan todas las categorías ni dos fotos perfectas. Grace
azul cumplía color pero cambiaba peinado. No confundir pruebas históricas
con una ejecución nueva.

## Histórico exacto que NO se pudo renderizar

Worktree detached limpio conservado en
`diagnostics/out/refine_recipe_20261104/head_v51280`.
HEAD exacto falla al importar el Editor: en su gen_prompt_config faltan
`get_longcat_model_config`, `get_longcat_full_model_config` y
`SubjectPromptConfigError` importados por su manager. Ruff detecta además
hood antes de definir en línea 2360; esa rama NO fue ejecutada.
No parchear ese snapshot y llamarlo HEAD intacto. No demuestra que nunca
funcionara el directorio local de entonces: pudo contener archivos sin commit.
Pruebas: `head_v51280_import.log`, `head_v51280_ruff.log`,
`head_v51280_provenance.json` (5 hashes coinciden normalizando CRLF).
La versión actual importa con mismo Python: `current_v51305_import.log`.

Se buscaron órdenes previas de commit en historial Codex relevante al proyecto:
aparece 24/10 «ACTUALIZA DOCUMENTACION Y COMITRE Y PUSH Y PRUEBO YO» y respuesta
«Solo documentación». Eso corresponde a dd7ec11, solo docs. No afirmar que
sea la orden exacta que recuerda esta semana. OpenCode/Copilot no aportaron
otra orden en la búsqueda acotada realizada. No seguir explorando historiales
indefinidamente: ya hay candidato de código y evidencia visual concreta.

## Prueba real Kontext: TERMINADA; leer actualización más reciente

### Estado final (18:35)

Exec84899 terminó **exit0**. Cola ComfyUI running0/pending0. No queda render
de este diagnóstico activo. No arrancar otro solo por leer el apartado viejo.

`result.png`, `mask.png` y `metrics.json` existen en
`diagnostics/out/appearance_reference_20261023/1793812998803975900/`.
Root inspeccionó la imagen FINAL compuesta: cabeza rapada sin bloque de pelo
ni orla del control LongCat; **PASS visual acotado a esta muestra**, no Auto
arreglado ni validación universal. 1llamada20steps, referencia753x768,
440s modelo/606,641s callback;39748px cambiados y0fuera máscara. Verificador
automático dio señal positiva. Tester independiente confirmó **PASS acotado
Kontext explícito** y recalculó39748px cambiados, exterior0 y maxdelta0 donde
máscara==0. Soporte máscara41129px; núcleo>=96:33506. Veredicto y hashes en
`diagnostics/out/refine_recipe_20261104/historic_kontext_current_verdict.json`.
No acredita Auto, rapidez, otra foto ni calidad universal.

**Pregunta al usuario todavía pendiente**: recuperar ruta en Auto para
apariencia medida manteniendo manualLongCat y otras rutas. La autorización
no llega por dejar pasar tiempo. No modificar producción hasta respuesta.

### Diseño mínimo revisado, NO aplicado (reviewer, 18:35)

- Mantener default_engine=auto: cambiar default global a Kontext perjudica
  lienzo manual. No activar restauración de toda la revisión histórica.
- Configurable `editor_ui.automatic_appearance_engine` (Kontext o LongCat),
  validación estricta de alias; candidato aprobado por reviewer para proponer
  al usuario, no autorizado aún: kontext.
- Despacho después de cámara y `edit_described_subject`, ANTES del bloque
  Turbo: solo auto_requested+engineactualLongCat, sin máscara, depth0,
  apariencia medida pura. Sin calidad sola/acabado generativo, pose/body/
  scenario/structural_add/structural_remove/region_ambiguous/recolor.
- Excluir compuestos con `_split_compound_instruction`: los facts agregados
  pueden ocultar una región en mezcla «calva y pared azul». No inventar otro
  dispatcher ni listas de palabras; múltiples/mezclas siguen ruta previa.
- Si el alias configurado es Kontext, permitir su archivo mediante
  `permit_automatic_model(get_kontext_model_config()['model_file'])` justo al
  decidir la ruta. No ampliar permisos genéricos en begin_execution.
- Respetar elecciones explícitas de motor y toda selección/manual/cámara/
  cuerpo/objetos. Abstenerse como la ruta Kontext vigente si no hay región
  inequívoca; no sustituir por una foto completa regenerada.
- Tests conductuales: Auto apariencias diversas→Kontext y permiso correcto;
  configLongCat→ruta previa; configinválida→error antes de inferir; máscara
  manual, selección, cámara, postura, cuerpo, escenario, añadir/quitar,
  recolor y compuestos→rutas previas; explicitLongCat/Kontext respetados;
  calidad sin prompt/cancelación intactos y modelo ajeno rechazado.
- Si usuario acepta, especificar antes de editar, siguiente versión v5.1306,
  implementar/probar/review/tester/render real. No dar por arreglado Auto por
  el PASS de escoger Kontext explícitamente.

### Actualización tras la espera (18:33)

La inferencia SÍ terminó, sin interrumpirla. Tardó 438 s; KSampler fase369s
con arranque muy lento antes del primer paso y luego ~3,8s/paso. No atribuir
el tiempo entero al coste por paso ni declarar bloqueo permanente.
`raw_0.png` ya existe y se inspeccionó: cabeza rapada, sin el bloque frontal
ni la orla vistos con LongCat. El log confirma «Borde de cabeza focal
armonizado», integración, autoacabado y exterior restaurado. A esta
actualización el callback aún carga el verificador VL; final/metrics pendientes.
El raw NO sustituye una validación de la imagen final compuesta.

Se ha enviado UNA pregunta concreta al usuario y está pendiente:
«¿Recupero en Auto la ruta Kontext del commit68ad5fb para cambios de apariencia
medidos, manteniendo LongCat para el lienzo manual y las demás rutas?»
Opciones: Sí recuperar / No conservar despacho. **No aplicar cambio de motor
hasta respuesta**, por las reglas del usuario. No se ha cambiado producción.
El reviewer está revisando solo el diseño mínimo; no implementa ni genera.

### Detalle de lanzamiento y observaciones anteriores

Se lanzó el probe existente, sin doblar inferencia ni cambiar producción:

    python -u -c "... run_case(SimpleNamespace(mode='global', sample='astronaut', prompt='La persona debe ser calva'))"

Usa `diagnostics/probe_editor_appearance_real_20261023.py`, callback real
`on_generate`, engine=kontext explícito, sin máscara manual, calidad=False.
Fuente astronaut 512 instalada. No se fijó semilla en este primer replay:
es repetición de ruta histórica, NO A/B de semilla controlada frente a LongCat.

- Sesión exec unificada: **84899**; consultar con write_stdin, no wait.
- Log: `diagnostics/out/refine_recipe_20261104/historic_kontext_route_current.log`.
- Evidencia fresca: `diagnostics/out/appearance_reference_20261023/1793812998803975900/`.
- Prompt ComfyUI observado: **f8bdb519-68d1-40bf-b3bd-bce8c6c946ba**.
- WS conectado 18:24:05 local; a 18:25:04 entra KSampler y aún no había
  primer paso a la última consulta (18:26:43: GPU 13 %, VRAM 3729MB usada).
- Cola solo ese trabajo, pending=0. No interrumpir/reiniciar por suponer fallo.
  Consultar /queue, /history/id y log antes de actuar; respetar el timeout real.
- El log confirma recorte 753x768, plantilla regional, modelo físico
  flux1-kontext-dev-Q4_K_M.gguf, guidance3.5,20 pasos, ref=offset.
- Inicializó rewriter pequeño CPU y escupió plantilla extensa en log; filtrar
  líneas que empiecen por [ImgEditor]/[FluxClient]/[ComfyUI] para ver progreso.
- Hasta ahora NO hay result.png/metrics.json nuevo; solo source/reference.

## Siguiente paso exacto

1. Esperar y comprobar el proceso/prueba activa; abrir imagen NEUTRA nueva al
   terminar. Confirmar armonización del borde, cumplimiento y exterior.
2. Si esa ruta vuelve a pasar, ya se habrá localizado comportamiento bueno y
   bypass actual. Preparar corrección mínima de despacho/restauración, NO
   rollback de todo el repo. Mantener cámaras/posturas/cuerpo/objetos/personas,
   selección y lienzo manual (Kontext no admite el manual actual).
3. La configuración actual mantiene default editor_ui=auto, auto usa LongCat.
   No cambiar a ciegas el default global: puede perjudicar lienzo y otras rutas;
   el usuario exige decidir motores/diseño. Si hay que elegir, presentar una
   sola decisión concreta, con resultado visual y diff preparados, no otra
   pregunta genérica para posponer el arreglo.
4. Un A/B adicional útil sería mismo astronaut 2048/prompt/semilla424242 de la
   prueba LongCat con Kontext explícito. Documentar diferencia de modelo; no
   llamarlo aislamiento de una sola línea de código.
5. Si hay cambio de producción: especificación previa, versión siguiente,
   prueba que reproduzca fallo→pase, regresiones pertinentes, reviewer APPROVE,
   tester PASS acotados y prueba real con evidencia. Registrar fallos también.
6. No finalizar otra vez solo con «pendiente» o «último commit no importa»:
   la petición vigente es arreglar el Image Editor usando el arreglo recordado.
