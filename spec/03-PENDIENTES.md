# spec/03 — Pendientes y deudas declaradas

## 2026-11-10 — v5.1328: LTX-2 Q4 con audio nativo (prueba real en 8 GB)

- Grafo y utilidades en `roop/animate/ltx2_audio.py`; probe de contrato
  (`probe_ltx2_audio_contract_20261110.py`, **16/16** re-ejecutado) y runner con
  medición (`real_animate_ltx2_audio_20261110.py`).
- Medido en RTX 3060 Ti 8 GB + 34,3 GB RAM + pagefile 78 GB: 960×544, 97
  fotogramas, 8 pasos → **12 min 28 s** en frío y 1 min 58 s con caché
  caliente; pico VRAM 7 535 MB (92 %), mínimo
  1,8 GB de RAM libre, ComfyUI 20,6 GB RSS. La ruta cabe por el pagefile, no por
  VRAM: es el requisito que hay que declarar.
- **720p medido** (`97f_720p`): 1280×704, 97f → **815,8 s** en frío y
  **185,9 s** en caliente (prompt 179,82 s), pico VRAM **6916/7218 MB**, pico
  RSS **25,7/24,2 GB**, MP4 de 825 KB con AAC 24 kHz y 3,88 s. Último
  fotograma limpio. Evidencia en
  `diagnostics/out/ltx2_matrix_20261110/`.
- Coste a 49 frames (960×544): 1,96 s de vídeo, MP4 273 KB; re-ejecutado en
  frío hoy: **828,7 s** (prompt 13:13), pico **7728 MB**, RSS **26,3 GB**; audio
  intacto. El último fotograma sale fundido (recortar o usar 97). Un `/history`
  lento ya no aborta la ejecución.
- Salida con audio nativo real: AAC 24 kHz estéreo, 3,88 s, media −27,3 dB.
  Evidencia en `diagnostics/out/ltx2_audio_20261110/`.
- Motor **expuesto** en Animate (`ltx2_audio`) con los parámetros medidos (8
  pasos, CFG 1,0, 25 fps, fotogramas 8n+1); `compose_prompt` arma el párrafo con
  sonidos. Probe de cableado **14/14**.
- **Matriz arreglada** (`diagnostics/ltx2_matrix_20261110.py`): sin
  `SameFileError`, `matrix.json` conserva filas anteriores y el timeout por
  defecto es 3600 s (la carga fría medida hoy es 828,7 s; la corrida anterior
  tardó 1790 s).
- Pendiente: validación visual con retrato real (cara, boca, diálogo); el
  trabajo `97f_540p_retrato` necesita `--portrait` y no hay esa foto en el
  proyecto. Decidir el último fotograma a 49f.


## 2026-11-10 — v5.1328: Mejoras Image Editor + Face Swap

### Face Swap — Tracking Continuity (v5.1328)
- Archivo: `roop/ProcessMgr.py` línea 5803
- Cambio: Cuando tracking se pierde (det_score < 0.05 y velocity > 100),
  reutiliza `_last_swap_bbox` en lugar de hacer fallback a la cara más grande
  del frame. Previene cambios de cara inesperados en video y mantiene
  consistencia de identidad entre fotogramas.
- Estados: Implementado, compilación verificada con `py_compile`.
- Pendiente: Validación visual en video real con movimiento de cámara.

### Image Editor — Zero Signal Detector (v5.1328)
- Archivo: `roop/img_editor/img_editor_manager.py` líneas 729-743
- Cambio: Cuando `mag ≤ 0.55` Y todas las medidas de concepto son ~0
  (body_exposure_full, body_exposure_local, pose, camera_view, structural),
  fuerza `mag = 0.90` y `forced_strong_regime = True`. Evita routing
  inconsistente cuando el prompt no contiene conceptos reconocidos (ej: "the
  boy is bald" daba mag=0.50 + target=background absurdo).
- Estados: Implementado, compilación verificada con `py_compile`.
- Pendiente: Validación visual con prompts fuera de vocabulario de conceptos.

### Image Editor — Umbrales Unificados (v5.1328)
- Archivo: `roop/img_editor/img_editor_manager.py` líneas 885-892
- Cambio: Usar `full_exposure_min_score` de `config/editor_selection.json`
  (0.3) en lugar de hardcode `0.80`. Elimina discrepancias de comportamiento.
- Estados: Implementado, requiere bloque try/except (líneas 885-889).
- Pendiente: Validar que `self._body_scope_policy()` existe y funciona.

### Image Editor — Modo Calidad UI Mejorada (v5.1328)
- Archivo: `ui/tabs/img_editor/events.py` líneas 536-567
- Cambio: `on_quality_estimate` muestra texto formateado con iconos,
  resolución máxima prevista, fuente, factor de escalado y advertencias.
- Estados: Implementado, infraestructura de wiring ya existía (líneas 629-638).
- Pendiente: Validar visualmente en UI relanzada.

### Face Swap — Halo/Edge Removal (v5.1328)
- Archivo: `roop/quality_enhancements.py` líneas 480-544
- Cambio: `apply_quality_enhancements` aplica `_apply_edge_aware_mask` que
  detecta bordes con Sobel en luminancia y reduce el peso del blend solo en
  bandas de borde, protegiendo contornos y reduciendo halo. Umbral Sobel:
  28.0, fuerza de borde: 0.85.
- Estados: Implementado, parámetros configurables, compilación verificada con `py_compile`.
- Pendiente: Validación visual en video con caras de perfil y movimiento.
  Los parámetros ahora son configurables vía options dict para adaptarse a
  diferentes escenarios de vídeo.

### Deudas pendientes (spec/02-REQUISITOS.md)
- [x] **IE-3**: Mover valores hardcoded `threshold=0.45` y `grow_frac=0.70` en
  `selected_region_edit.py` a `config/editor_selection.json`.
- [x] **AN-2**: Recortar último fotograma en `ltx2_audio.generate()` cuando
  `frames < 97`. Implementado: código en `ltx2_audio.py:315-329` recorta el último
  frame para evitar el defecto de fundido; o elegir 97f para calidad.
- [x] **AN-3**: Exponer 720p (1280×704) en UI Animate cuando VRAM > 8 GB.
  Implementado: `resolve_generation()` en `ltx2_audio.py:191-206` usa 540p (960×544)
  para VRAM ≤ 8 GB y 720p (1280×704) para VRAM > 8 GB.
- [x] **AN-4**: Alinear slider de pasos con los 8 pasos destilados del grafo LTX-2.
  Implementado: `ui/tabs/animate/logic.py:43` - steps=8 fijos para engine=ltx2_audio.
- Validación visual real: retrato con cara/boca/diálogo para Animate.


## 2026-11-10 — v5.1327: obediencia medible y controles de Animate

- Un resultado con **0 px** cambiados ya no se guarda como edición: se informa
  «Edición sin cambios» y se conserva la foto (cubre la máscara vacía de
  CLIPSeg del log 01:01). La etiqueta de éxito publica el cambio medido.
- `measure_change` mide dentro/fuera de la máscara y `verify(before=)` usa el
  recorte cambiado; probes CPU 6/6 y 8/8 respectivamente.
- Animate expone fotogramas/pasos/sonido; defaults intactos (37/12/24).
- Pendiente: validación visual real de una animación con este código; no se
  declara PASS visual. La ruta de cámara con «plano» sigue sin medir («haz un
  plano del salon» → «Make a plan of the living room»).
- **Validación real de cámara**: enruta y cambia en los dos casos. Retrato
  neutro (astronauta, 512×512, 59 s): PASS visual acotado, 89,9 % cambiado con
  cara/identidad coherentes. Composición de dos cuerpos en contacto: FAIL visual
  (anatomía fundida) con la misma receta Q2/4 pasos. Falta decidir si se toca
  la receta para composiciones difíciles (más pasos o Kontext con denoise
  medio); próximas pruebas con fotos neutras, sin repetir la referencia
  explícita.

## 2026-11-10 — v5.1326: búferes de swap y cámara sin tildes

- `_prime_warp_edges` y `_soothe_swap_ring` mezclan in-place y por canal:
  pico 33,6 → 11,4 MB y 27,7 → 13,4 MB a 720×720; salida idéntica bit a bit
  (`probe_warp_edges_buffers_20261110.py`, 12 casos, 0 FAIL).
- Cámara: `_tokenize` pliega tildes, el ámbito corto de cámara se acepta con un
  token y las palabras sueltas en MAYÚSCULAS se bajan antes de Marian
  (`probe_editor_camera_caps_20261110.py`, 12/12).
- Pendiente: repetir un render real con este código (el lote de 14 vídeos
  quedó sin reanudar) y validar visualmente que no hay halo ni cambio de borde.

## 2026-11-10 — v5.1325: comprobación de destrucción corporal

- `_body_destruction_metrics` / `_body_destruction_reason` en
  `selected_region_edit.py`, conectadas a `_vision_zone_pass` y
  `_kontext_zone_pass`: rechazan la pasada (se conserva la foto) si cambia
  >0,2 % de píxeles fuera de la zona escrita o si aplana piel texturada en
  ≥15 % de la piel de la zona con bloque contiguo ≥5 %.
- Probe nuevo `diagnostics/probe_editor_body_destruction_20261110.py` 0 FAIL:
  tono legítimo 0 %, relleno plano 90 %/78 %, bloques 20 %/8 %, fuera de zona
  0,6 %, sin cambios 0 %.
- Pendiente: repetir «la mujer debe ir descalza» en la UI reabierta y
  confirmar que la salida ya no se entrega con las piernas tapadas; el probe
  es CPU y no demuestra cumplimiento visual.

## 2026-11-10 — v5.1324: auditoría de uncover y banda inferior

- `_vision_zone_pass` acepta `write_mask`: los reintentos de residual ven la
  zona ampliada pero escriben solo lo medido al principio (antes 81 345 px de
  banda reescrita en el caso «descalza»).
- `_covered_surface_view` descarta cubiertas >35 % de la persona o >15 % de la
  foto, con log, en vez de pintarlas con el color de alrededor.
- Rutas revisadas y acotadas sin cambios: `_measured_target_zone` (≥400 px,
  ≤50 % de la persona), `_measured_scene_zone` (`max_ratio` 0,5),
  `_measured_clothing_zone` (segmentador + dilate 14), `_kontext_zone_pass`
  (recorte dilatado, pegado limitado a `write_limit & original_zone`).
- Deudas declaradas: la banda inferior sigue siendo geométrica cuando no hay
  ropa ni objetivo medido; el residual fallido conserva el intento del modelo
  (decisión 2026-11-06); la banda sin objetivo no verifica cumplimiento.
- Verificación: probes de uncover 14/15 y 13 PASS, py_compile/ruff y probe
  sintético de cubierta verdes. Pendiente PASS visual con render real.

## 2026-11-10 — v5.1323: quitar sin objeto y casing mixto

- `_requested_removal_target` ya no devuelve los placeholders «objeto» /
  «objeto_removido»: sin complemento gramatical no hay objetivo local, así que
  se evita el prompt «Remove the objeto», la cubierta beige de
  `_covered_surface_view` y los tres reintentos de residual que dejaron
  `edit_1794266098.png` / `edit_1794266288.png` sin piernas.
- `_normalize_mixed_case_words` en `prompt_translator.py`: `cALVO` → `calvo`
  antes de traducir; `iPhone` y las MAYÚSCULAS completas no cambian.
- Verificación: `probe_editor_local_uncover_20261029.py` 14/15 (FAIL sólo por
  fixture borrado), `probe_editor_compound_uncover_20261030.py` exit 0,
  py_compile y ruff verdes.
- Pendiente: repetir en la UI reabierta «la mujer debe ir descalza» y «el
  hombre debe ser calvo» sobre la misma foto y firmar el PASS visual; sin ese
  render no se declara cumplido.

## 2026-11-10 — v5.1322: malla 468, restauración única y calibración real

- Máscara de caras por malla de 468 puntos (`_face_mesh_mask`) con caída a la
  elipse+protector si MediaPipe no encuentra cara; restauración facial única en
  Modo Calidad (TOP al final, contra la imagen editada).
- Calibración medida en `diagnostics/out/quality_calibration_20261110/`
  (`report.md`, `report.json`, `run.log`), 6 fotos reales, 0 FAIL: bandas
  limpias ≤0,231, q8 ≥0,691, desposterizar gana ≥0,351 por foto; pasos suaves
  legítimos suben hasta 0,046 → `band_rise_tolerance` 0,06 (antes 0,03).
  A/B: ratio mín. 5,192 y estructura mín. 0,817 frente a ruido ≤0,007; se
  mantienen `ai_gain_min_ratio` 1,08 y `ai_gain_min_corr` 0,45.
- Validación 1080p/4K preparada: `probe_quality_visual_1080_4k_20261110.py` y
  protocolo en `plans/VALIDACION_modo_calidad_1080_4k_20261110.md`.
- Pendiente: PASS visual del usuario y render real 1080p/4K (el script mide,
  no firma PASS); probar la malla en perfiles/oclusiones y con caras pequeñas;
  confirmar que el A/B rechaza una foto sobreexpandida real (hoy el control
  negativo es ruido).

## 2026-11-10 — v5.1321: Modo Calidad (bandas, identidad y coste)

- `probe_quality_gate_banding_20261110.py` 10/10 PASS y
  `probe_quality_checkbox_finish.py` 19/19 PASS; py_compile y ruff crítico
  verdes en los módulos tocados.
- Nuevo: métrica de bandas en la puerta de desposterizar; identidad facial
  verificada también por encima de 2048 px; máscara de caras con protector de
  bordes; A/B de upscale en 3 zonas con correlación de estructura; tope de
  afilado por ruido medido; `upscale=False` en las rutas que conservan el
  exterior; cancelación en upscale; medida final contra la entrada;
  estimación de resolución en la UI; `use_generative=False` por defecto.
- Pendiente: PASS visual del usuario y render real 1080p/4K; calibrar la
  métrica de bandas con fotos reales (no sólo el fixture) y comprobar que el
  A/B de estructura rechaza en una foto sobreexpandida real.

## 2026-11-09 — v5.1320: corrección en desvestir/sin ropa y sintagma de sujeto

- `MEMORY.md` actualizado con el diagnóstico y cierre de los incidentes de la jornada:
  1. `la mujer debe ir sin ropa` en Auto: `edit_described_subject` autoriza explícitamente LongCat y Kontext con `permit_automatic_model` y asume `vision_zone` para `persona medida menos la cabeza` y `banda inferior` en `uncover`.
  2. `el hombre tumbado debe ser calvo`: `_leading_visual_subject` extrae el sintagma nominal (`lying man`, `man on the right`), evitando que CLIPSeg reciba participios o adjetivos sueltos sin sustantivo.
  3. ComfyUI v0.19.0: flag `--disable-dynamic-vram` para ≤8GB; pendiente vigilar el manejo de descarga entre modelos (`_handle_clone` en `comfy_kitchen`).
  4. Modo Calidad: `min_absolute: 0.60` en `top_tier_finish.py` y caché `blake2b` en `image_quality_pipeline.py`.
- Suites CPU pertinentes: versatilidad 23/23 PASS, contratos de selección 15/15 PASS, régimen corporal 13/13 PASS. Sintaxis `py_compile` y `ruff` (E9, F63, F7, F82) verdes.
- Pendientes vigentes: validación visual del usuario, mapa de versatilidad (zapatos/ropa/postura), FaceSwap perfiles/vídeo representativo y build Android instalado en terminal.

## 2026-11-06 — continuación v5.1312: técnico verificado, calidad abierta

- `plans/WORKLOG_state_logs_fix_20261106.md` actualizado con pruebas nuevas en
  `diagnostics/out/continuation_20261106/`. Latido, residual configurable,
  compositor y geometría temporal: reviewer APPROVE técnico acotado y tester
  PASS. Corregido el último fallo reproducido: propuesta simple sin clasificar
  perdía `measured_request=True`. 84 comprobaciones CPU, Ruff y compilación
  de 49 módulos PASS; build del launcher exit0. No aprobación global del repo.
- **FAIL visual actual del Editor**: neutral de taza azul respeta el soporte
  (101.043 px cambiados; 0 fuera), pero cambia geometría/posición y conserva
  restos de la taza original. `real_single/result.png`; confirmado por tester.
  Falta un resultado natural y aceptación del usuario; no convertir la máscara
  medida en garantía de cobertura completa ni ampliar a ciegas por oscuridad.
- Seis solicitudes en último log UI10:29; dos conservan la misma imagen por
  zona no medida: pendientes de cumplimiento. Las cuatro inferencias siguientes
  completan y tienen salidas presentes; la nota de tercer intento cortado se
  supera. No hay validación visual de esas seis solicitudes en este turno.
- FaceSwap foto frontal real CPU: A/B idéntico y tres modelos ejecutados.
  El probe desactiva suavizado temporal: no valida el fix en un vídeo real.
  Siguen pendientes perfiles/oclusión/halo/parecido y vídeo representativo.
  Los seis outputs del lote histórico no están disponibles para inspección.
- Modo Calidad Auto puede aplicar acabado clásico global solicitado. Sus
  cambios de color/luz/detalle no prueban fuga generativa; selección explícita
  conserva su contrato distinto de exterior exacto. 1080p/4K siguen abiertos.
- Persisten versatilidad visual, POV/primera persona, Android instalado,
  Chroma CFG/negativo y discrepancia histórica39:12; no reinstalar ni cambiar
  motores para cerrar estas deudas sin diagnóstico/autorización pertinente.

## 2026-11-06 — contraste ejecutado del estado actual (sin cambios de procesamiento)

Captura histórica de las 08:52; el estado técnico vigente está en la entrada
anterior y en el WORKLOG. Conservar estos datos como procedencia, no repetirlos
como fallos actuales tras las correcciones verificadas.

- Informe: `plans/REVISION_estado_20261106.md`; evidencia nueva en
  `diagnostics/out/state_review_20261106_085256/`.
- **FAIL de progreso en aquella captura (superado):** 42/43 tests CPU pasan; falla
  `test_heartbeat_keeps_real_step_without_claiming_health`. La consola impone
  `max(heartbeat, 30.0)` aunque la configuración solicite menos. Los callbacks
  tienen su cadencia separada; no demuestra fallo de entrega Gradio ni cuelgue.
- Los hashes actuales de `comfy_progress.py` y `selected_region_edit.py` no
  coinciden con el tester/render v5.1310: su APPROVE/PASS no certifica estos
  cambios posteriores. El probe nuevo de pegado/residual pasa 11/11, pero no
  valida su calidad visual ni sustituye una revisión completa del alcance.
- Ruff Editor/Generate: 94 incidencias (88 de código y 6 E902 de respaldos con
  bytes nulos). Sintaxis y lint crítico de 52 archivos sin esos respaldos PASS.
  No declarar lint global ni build actual verdes; no se recompiló el launcher.
- Parámetros nuevos de residual/expansión siguen literales: `threshold=0.4`
  y `grow_frac=0.35` en `selected_region_edit.py`; revisar conforme a NO HARDCODE.
- Corregida solo documentación: predeterminado Auto en spec/00; estados
  históricos del umbral corporal y build Android aclarados más abajo.
- Permanecen las validaciones visuales/funcionales abiertas de Editor, FaceSwap,
  Android instalado, Chroma CFG/negativo y Modo Calidad 1080p/4K.

## 2026-11-06 — v5.1311: cierre documental y alineación de pendientes

- Petición: «revisa pendientes y bitácora, y requisitos, memory y demás» y
  después «haz lo que debas y las pendientes, sube versión cuando acabes».
  Solo documentación y versión: sin cambios de motor, prompts, recetas ni píxeles.
- Corregido el desfase: `spec/03` afirmaba v5.1309 mientras `AGENTS.md`,
  `run.py` y `roop/core.py` ya eran v5.1310. Vigente **v5.1311 (2026-11-06)**.
- Cerrada la deuda del 29/10 sobre exposición corporal: la ruta lee
  `instruction_scope.body_state_min_intensity` (0,3) de
  `config/editor_selection.json`, valida el rango y ya no usa el literal 0,35.
- Cerradas las 4 tareas del 26/10 en `.autohand/agents/tasks/todos.json`:
  LongCat en cuerpo sin selección es la ruta configurada desde v5.1306, no un
  defecto; solo la apariencia pura medida se redirige a Kontext.
- Verificado hoy: `probe_editor_versatility_integrity_20261105.py` 23/23 PASS y
  retirada regional con `body_transform=False` (la frase de piel desnuda ya no
  se envía para objetos). Falta la muestra real repetida de la cuchara.
- Sigue abierto: FAIL visual de conservación v5.1310; categorías de
  versatilidad sin PASS visual; 39:12; FaceSwap perfiles/oclusión/vídeo;
  Android instalado; Chroma CFG/negativo; Ruff global ~90; render 1080p/4K.
- Sin commit/push: HEAD `4d7ce3c` v5.1280; 65 modificados + 115 sin rastrear.
  UI abierta (PID 15720, :7861) intacta; reabrir carga v5.1311.

## 2026-11-05 — H-EDIT-WAIT-FINISH: límites v5.1310

- FAIL visual de conservación: taza azul sí, pero cambia plato/encuadre/café/
  cuchara y añade vapor. No acredita calidad general ni caso zapatos.
- Latidos cooperativos: HTTP/callback bloqueados pueden retrasarlos. Prueba con
  callback JSONL no acredita entrega Gradio. No se explica el39:12 histórico
  ni se atribuye aceleración del modelo al parche. Fuentes primarias sobre
  offload en fix/research.md sin cambios de flags. No se instaló ni reinició,
  no se hizo commit/push; reabrir app carga cambios.
- Toda evidencia: diagnostics/out/editor_stall_20261105/. Reviewer/tester
  independientes; detalles visuales en fix/tester/real_visual_and_provenance.json.

## 2026-11-05 — auditoría de versatilidad: defectos actuales, v5.1308

Informe y reproducciones: `plans/AUDIT_editor_versatility_20261105.md` y
`diagnostics/out/editor_versatility_audit_20261105/`. Es una revisión del alcance
existente, sin nueva implementación ni cambio de versión. No declarar versatilidad
cumplida por las muestras de apariencia ni por las suites CPU verdes.

- Estado tras la revisión de 2026-11-05 y comparación con el código actual:
  - **P1 `_build_object_removal_mask`**: el código ya no convierte los ceros de la máscara en zona editable. Prueba: `probe_editor_versatility_integrity_20261105.py::RemovalContracts::test_generic_removal_helper_does_not_invent_a_body_surface` / `::test_auto_regional_object_removal_prompt_reconstructs_neutral_scene` → **PASS**.
  - **P1 `_closed_structural_zone`**: ya distingue objetivo medido de personas protegidas. Prueba: `...::test_person_closure_keeps_target_and_excludes_adjacent_neighbor` → **PASS**.
  - **P2 `_compound_clause_zone`**: transmite `structural_remove` y mantiene zona simple/compuesta. Prueba: `...::test_small_target_same_mask_in_simple_and_compound` → **PASS**.
  - **P2 cobertura de probes add/remove**: los controles constantes/búsqueda de texto se reemplazaron por contratos con píxeles, objetivo y ruta real (clases `RemovalContracts`, `AdditionContracts`, `CameraContracts` del mismo archivo). Prueba: 23/23 **PASS** en `python -m pytest diagnostics/probe_editor_versatility_integrity_20261105.py -q`.

- Riesgos de clasificación confirmados (sin FAIL visual generado): abrigo
  azul puede medir recolor, ojos grandes puede medir cuerpo, silla
  desplazada puede medir pose. Requieren muestras específicas antes de
  cambios de rutas.

- No se implementó aún:
  - validación visual del usuario por categoría;
  - pruebas de cámara/POV sobre una imagen adecuada;
  - muestras específicas de ropa/ojos/movimiento;
  - prueba en teléfono Android instalado; el build/lint previo ya pasó
    según spec/05, sin acreditar todavía la ejecución en el dispositivo;
  - vídeo representativo de FaceSwap;
  - actualización de la versión de aplicación (el código ya estaba en v5.1310;
    ver entrada 2026-11-06).

## 2026-11-05 — recuperación entregada y límites visuales, v5.1308

- Auto recupera apariencia focal Kontext y la plantilla regional conserva
  atributos no solicitados. Prueba nativa astronaut sin overrides con exterior0
  y calvicie visible; Grace controlada mejora frente al FAIL conservado.
- La preservación de todos los atributos no es absoluta: Grace retira gorra
  y conserva vello corto. No certificar todas las fotos/categorías ni identidad
  exacta. Aceptación visual del usuario y mapa de versatilidad siguen abiertos.
- Carga fría lenta y variable (283s modelo/371,297s callback en la prueba
  productiva final; otras corridas fueron mayores). No declarar rapidez resuelta.
- Error de máscara local que producía array0d corregido y engine_status28/28.
  Diez suites pertinentes de la plantilla pasan; deudas de lint ajenas siguen
  separadas. No restaurar el stub ni la dilatación genérica posterior de máscara.

## 2026-11-04 — H-EDIT-REFINE-RECIPE, v5.1305

- Comparación aislada posterior: HEAD v5.1280 no carga el Editor por imports
  de funciones/excepción ausentes en sus propias fuentes. Worktree limpio y
  hashes verificados; `head_v51280_import.log`/`head_v51280_provenance.json`.
  No usar ese commit como referencia visual validada ni confundirlo con la
  carpeta de trabajo que pudiera existir el día del commit. Sin parchearlo,
  no hay render comparable. Producción permanece en v5.1305.

- Restaurado el contrato de latente foto y receta configurada del refino,
  manteniendo pegado completo y compuertas faciales. No equivale a resolver
  calidad visual ni remanentes; pendiente aceptación visual del usuario.
- La afirmación de MEMORY v5.1304 «solución final» sin referencia espacial
  queda rectificada en la entrada v5.1305; no usarla para repetir ese override.
- Ruff completo de manager/run/core y diff --check de run/core tienen deuda
  anterior al arreglo; lint crítico, probe y build se verifican por separado.
  No ampliar esta corrección para limpiar el resto de cambios sin commit.
- Prueba neutral «la mujer debe tener el pelo rojo» (astronaut, semilla
  424242): FAIL visual, pelo rojo añadido alrededor del original. La ruta
  region_edit amplió máscara al 87,2 % y omitió refino por área. Evidencia
  `diagnostics/out/refine_recipe_20261104/neutral_real/compare_real.png` y
  `real.log`. La corrección de receta focal no resuelve esta medición/pegado
  anterior; no declarar versatilidad ni calidad general resueltas.
- La dilatación genérica posterior y el hueco de todas las cajas faciales
  se han retirado tras reproducción CPU de fuga a otra persona y A/B real.
  El refino vuelve a ejecutarse con su receta de foto; desaparece el bloque
  frontal protegido, pero la prueba neutral todavía conserva restos claros
  y orla. **FAIL visual de edición completa**; pendiente completar la región
  sin restaurar la expansión arbitraria ni el override sin referencia.
  Evidencia `real_without_expansion.log` y
  `neutral_without_expansion/compare_real.png` en la carpeta de esta historia.

## Continuación verificada — 2026-10-29

- Revisión solicitada: «revisa pendientes y sigue», con relevo antes de agotar
  contexto. Se está contrastando documentación histórica con código actual.
- Bloqueo general reproducido y corregido: `_appearance_work_region` lanzaba `NameError`
  por `body_direction` no definido, también con una máscara local no explícita
  y sin persona. Retirado solo el bloque suelto dentro de H-EDIT-003/H-AUDIT-FIX;
  la inclusión geométrica de la persona ya existe más abajo. Probe nuevo
  `probe_editor_focal_window_20261029.py`: rojo (6 errores) → verde (5 métodos),
  con ventana local, bordes, persona completa, fuente intacta y máscara vacía.
  Reviewer APPROVE y tester independiente PASS acotados; Ruff del manager/probe,
  compilación y build del launcher verdes. Foto neutral real: crop exacto,
  fuente intacta, máscara/persona sin pérdida (geometría explícita, sin render).
- Baseline: compilación Editor/Generate verde; Ruff global del área falla.
  Después del fix quedan 90 incidencias previas (antes 91), ninguna en manager.
  Evidencia y ejecuciones en `diagnostics/out/pending_audit_20261029/145422/`.
  No equivale a aprobación del workspace ni a validación visual general.
  Relevo: `plans/HANDOFF_pendientes_20261029.md`.
- Deuda histórica del lote previo (superada por el contraste del 06/11 en la
  cabecera; se conserva como registro): `measured_full_body_exposure` contiene una
  lista manual que evita los conceptos medidos; `_full_body_exposure_clauses`
  incorpora un umbral literal 0,35. Reviewer CHANGES_REQUESTED por las reglas
  de enrutamiento/configuración. No modificado en esta corrección genérica;
  los contratos actuales de alcance (10/10) no acreditan cumplimiento de esa
  regla ni cubren todo el vocabulario añadido.
- Color guard opcional: contratos 11/11 e identidad 14/14 siguen verdes;
  continúa pendiente aceptación visual del usuario. Riesgo de estiramiento
  con proporciones incompatibles ya medido el 26/10: abierto, sin causa
  atribuida a una edición real. No repetir A/B fallidos sin hipótesis nueva.
- FaceSwap: perfiles/oclusiones/vídeos completos y validación visual abiertos.
  Android: prueba instalada pendiente; el build/lint previo no la sustituye.

## Histórico: revisión para publicación y prueba del usuario — 2026-10-24

Los fallos por contar cláusulas, el F841 de calidad y la comparación de identidad
descritos abajo se corrigieron el 25/10 (H-AUDIT-FIX en spec/02 y MEMORY.md).
Se conservan como historial, no como deuda actual. Los fallbacks nuevos por
vocabulario/umbral del working tree son una deuda posterior, declarada arriba.
Las afirmaciones de motor de este bloque también preceden a la autorización
del 25–28/10; consultar spec/00 y spec/01 actuales.

- Usuario autoriza actualizar documentación, commit y push; hará la prueba
  visual. No se ejecutan nuevos renders ni se reinicia su UI en este turno.
  Mantener Ollama activo. Publicar no equivale a cerrar calidad.
- **CHANGES_REQUESTED del reviewer para el lote de código completo:**
  `measured_full_body_exposure` considera exposición completa cualquier texto
  con dos cláusulas separadas por coma, sin exigir señal de remoción. Repro CPU
  real: «The walls must be gray, the sofa must be blue» devuelve True.
  `_body_state_clause('The sofa must be gray, the woman must go barefoot',
  'uncover', 0.8)` devuelve «The sofa is fully nude…»: alcance ampliado y sujeto
  equivocado. Corregir y probar falsos positivos antes de pedir APPROVE; la
  prueba visual por sí sola no sustituye estos contratos. No se modifica código
  en esta actualización documental.
- Los contratos CPU ejecutados por reviewer pasan (apariencia10/10,
  corporal10/10, cleanup2/2), pero la suite corporal no cubre esos casos.
  Las aprobaciones v5.1249/v5.1250 no se extienden al fallback corporal posterior.
- Chequeo actual del lote: compilación del editor y UI PASS; Ruff FALLA F841
  por `scores` sin usar en `image_quality_pipeline.py:1520`. No corregido en
  este commit documental. No afirmar lint/build/calidad globales verdes.
- Configuración pendiente: los umbrales y el conteo del fallback corporal son
  literales en código; la afirmación previa «config, no hardcode» se rectifica.
- Calidad/identidad: el máximo de comparaciones entre caras en
  `top_tier_finish.py` puede aceptar una restauración porque una cara se conserva
  aunque otra empeore. No certifica identidad individual en fotos con varias
  personas. Pendiente de revisión específica, sin cambiar el diseño aquí.

## Histórico: referente débil y global — v5.1250, 2026-10-24

Actualización 29/10: la repetición de Grace indicada aquí ya se ejecutó el
25/10. PASS acotado de color azul, sin certificar el corte exacto (añade bob y
flequillo); véanse MEMORY.md y spec/05. No repetirla como ejecución ausente.

Estado al pedir «anótalo todo»: relevo vigente
`plans/HANDOFF_editor_global_20261024.md`. Global astronaut PASS; Grace
INTERRUMPIDA, sin veredicto visual. Cierre nativo de proyector PASS en dos
ciclos,1224MB liberados cada uno, sin acumulación. **Conservar Ollama activo por
instrucción explícita del usuario.** Falta repetir solo Grace desde proceso
limpio y con presupuesto de VRAM compatible, sin detener aplicaciones ajenas.
El estado de entonces es el de este resumen; los intentos anteriores quedan en el relevo.

- weak_target lejos/cerca corregido: competir entre todas las detecciones antes de exigir elegibilidad del ganador, sin regresar a una única detección total; fallback sin región sigue permitido y protege vecinos. CPU20/20 y reviewer APPROVE. La deuda15/16 de v5.1249 ya está corregida en código; no marcarla pendiente sin contrastar esta actualización.
- Global: baseline real produjo pelo blanco al pedir calvicie y omitió un retrato por35%. Plantilla regional sin edad inferida, recorte configurado de una sola cara creíble y armonización compartida; CPU20/20 + apariencia10/10 y reviewer APPROVE del cambio. Global corregido1792858682799844700 (astronaut) PASS visual acotado. Grace1792858908810863000 INTERRUMPIDA intencionalmente, sin imagen final validada; no atribuir a OOM ni afirmar dos fotos aprobadas. Log `logs/appearance_global_real_20261024.log`.
- Cierre explícito de Llama+ExitStack corregido:2/2 CPU y reviewer APPROVE. Medición nativa terminada: dos ciclos liberan1224MB cada uno, sin acumulación (`logs/verifier_cleanup_native_20261024.log`). No demuestra rapidez de render con Ollama activo. Próxima prueba visual a cargo del usuario; Grace y calidad general siguen abiertas.

## Apariencia focal — v5.1249, 2026-10-23/24

- Despacho/resolución y cobertura de propuestas corregidos. GPU inicial bloqueada por PermissionError; repetición1792854592938590500 FAIL por pelo restaurado,1792855222215129000 FAIL por óvalo luminoso. Armonización NORMAL_CLONE configurada con guarda y exterior restaurado;10/10 CPU y reviewer APPROVE. Tercer render1792855675823830800 TERMINADO y PASS acotado; fallos anteriores conservados en el relevo.
- Actualización final:1792855675823830800 tester independiente PASS ACOTADO,30656px cambiados y exterior0,una pasada20steps/96s. Código de apariencia reviewer APPROVE;10/10 CPU,Ruff/compileall verdes. Global astronaut también validado; siguen pendientes otra foto y validación del usuario. Para probar propuestas nuevas, el usuario debe reabrir UI y volver a analizar; este turno no reinicia su sesión.
- Deuda `weak_target` de v5.1249 RESUELTA en v5.1250 con20/20 CPU y reviewer APPROVE: se conserva fallback frente a falsos positivos y se impide elegir otra cara cuando el referente medido es débil. No restaurar la guarda de una sola detección total.

## Modo Calidad — puerta de aceptación y topes (v5.1245), 2026-10-23

- Hecho y medido: `diagnostics/probe_quality_benchmark_20261023.py` (tres fixtures + contrato de la puerta y de los topes) y `config/quality_mode.json`. La puerta revierte un paso sólo si empeora una medida: nitidez −6 %, posterización +0,04, recorte de tonos +0,8 %; la restauración de caras se compara antes/después con ArcFace hasta 2048 px. Tope de resolución: 8K apagado por defecto, 4K exige ≥700 px de origen y 12 MP de salida como presupuesto. Upscale: A/B local ESRGAN/Lanczos sobre la zona con más textura antes de gastar el motor IA.
- No hecho: ControlNet Tile. `flux-union-pro2-fp8` no trae `controlnet_mode_embedder`/`task_embedding`, así que no sirve para tile; haría falta un tile dedicado (SDXL ~2,5 GB) y su descarga es decisión del usuario.
- Pendiente de validación real: un render 1080p y otro 4K sobre foto del usuario con este lote (el banco es sintético y no juzga belleza); medir VRAM y tiempo si se activa `allow_8k`; y una foto con bandas reales para comprobar que la métrica de posterización no basta para juzgar «desposterizar» (en el fixture 0,837→0,836).
- Coste medido: la puerta añade ~4 análisis de posterización por pasada (blanda 240×160: 11,9→18,3 s). Se apaga con `acceptance_gate=false` en `config/quality_mode.json`. El A/B de upscale no ha rechazado todavía ningún motor en los fixtures (recorte sin textura medible): falta verlo en una foto real con detalle.

## Image Editor — quitar ropa coordinada con «y» (v5.1248), 2026-10-23

- Medido en el log real (12:06 y 12:13): con v5.1247 cargado, «debe ir descalza y sin pantalones» seguía dejando la zona en la ropa medida (19 850 / 21 568 px) y el ensanche no se imprimía. `_has_unmeasured_clause` partía solo por coma; la frase coordinada es una cláusula. Resultado en GPU: el modelo repintó el top (top de lunares, luego camiseta + póster con caras inventadas) y el pantalón quedó; el visor mostró ediciones grotescas, no el cambio pedido.
- Cambiado: cláusulas con `_split_compound_instruction` y solo cuenta la que no mide ningún régimen y el analizador lee como `uncover` con todo a cero. «without pants» ensancha; «the woman»/«her» (change) y «remove the lamp» (structural_remove) no.
- Verificado: caso `garment_fragment_widens_to_the_person` con las dos formas, fail→pass en ambas (0,39 sin ensanche, >0,85 con él); body_regime 10/10; global_body_prompt 10/10 + dos personas + las dos formas 8/8; selection_contract 15/15; subject_head_restore OK; ruff y py_compile limpios.
- No seguro: sin render de GPU con este lote. La zona de persona entera da más libertad al motor: puede repintar otra prenda, inventar objetos y cambiar la cara. Falta mirar un render real. Límites del criterio: «remove the lamp and her pants» no ensancha; instrucciones partidas con punto tampoco.

## Image Editor — quitar ropa en un fragmento medido (v5.1247), 2026-10-23

- Medido en la foto real: `object-1` = mujer entera (66 243 px), CLIPSeg ve el pantalón (58 998 px de ropa en la foto) pero la zona quedó en 21 568 px (33 % de la persona) porque «without pants»/«without blouse» puntúan 0. Resultado (edit_1792748877.png): chaleco→camiseta, pantalón intacto, diploma inventado dentro de la zona.
- Cambiado: si hay cláusula con señal cero y la ropa medida cubre menos de la mitad de la persona, la zona es «persona medida menos la cabeza» en las dos rutas. No afecta a «descalza» sola, «sin camiseta» (exposición local) ni frases con quitar objeto. Probes: body_regime 10/10 con caso nuevo que falla sin el ensanche (ratio 0,39) y pasa con él (>0,85); global_body_prompt 10/10 + dos personas + descalza/sin pantalón; selection_contract 15/15; subject_head_restore OK; ruff y py_compile limpios.
- No seguro: sin render de GPU con este lote. Queda por ver si el modelo, con la zona de la persona entera, cumple el prompt o repinta otra prenda; y si la costura cabeza/cuello queda limpia. El ensanche no añade pasadas ni mediciones de ropa extra.

## Image Editor — prompt sin selección que quita ropa por partes (v5.1246), 2026-10-23

- Medido y corregido: «la mujer debe ir descalza, sin pantalon, sin blusa» sin marcar devolvía la foto idéntica (edit_1792748232.png y edit_1792748448.png, log 11:33). Causa: el veto por cláusula de `edit_described_subject` exigía body en cada trozo de la frase partida por comas, y «without pants»/«without blouse» no lo miden. Ahora una cláusula neutra no veta; la que pide postura, escena, región, apariencia o quitar un objeto sigue vetando. Probe `probe_editor_global_body_prompt_20261021.py` con caso nuevo fail→pass (7/7) más los dos anteriores; body_regime 9/9, selection_contract 15/15 y subject_head_restore OK.
- No seguro: el probe simula CLIPSeg y los segmentadores sobre un fixture de rectángulos; falta el render de GPU con esa frase en la foto real. La zona se mide como ropa ∩ persona y NO resta la cabeza (sólo queda fuera si el segmentador no la marca como ropa); si la ropa medida no alcanza el pantalón, el cambio no llega ahí. La guarda v5.1244 sigue conservando la foto cuando no logra colocar a una sola persona.

## Image Editor — prompt sin selección (v5.1244), anotado 2026-10-23

- Seguro en código: sin marcar, un uncover que no pide postura ni cámara no reinterpreta el marco. Si no coloca a una persona, se conserva la foto. La frase «la mujer de la izquierda debe ir desnuda» no puede repetir edit_1792709573.png (00:52, identidad 0/2, mask=global) mientras el proceso cargue este código.
- No seguro: no hay render de GPU sobre la foto del usuario. El probe CPU pasó con un segmentador doble. Falta ver si la ropa desaparece. La ventana de las 00:44 no tiene el cambio; hace falta cerrar AutoAuto y abrirlo. Log esperado: «persona medida menos la cabeza». Log viejo: mask=global o «el marco se reinterpreta».

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

## 2026-10-23 — el prompt va en la zona medida (v5.1243)

- El marco reinterpretado (v5.1230) cumplía el texto en otra foto
  (`edit_1792703256.png`, `FastEdit_00836_.png`). El usuario: lo importante,
  más que conservar la cara, es cumplir el prompt y ponerlo donde toca.
- El cambio corporal de selección, haya o no otras personas, escribe con el
  motor con visión en la zona de ropa medida. Fuera de la selección la foto
  queda igual. La cara no se restaura.
- Límite ya medido (v5.1229): dentro del agujero el modelo no siempre llega
  a piel total (ropa de playa, piel 0,43 frente a 0,53 sin agujero). No se
  vuelve al marco entero para tapar ese límite.

## 2026-10-23 — pegado de cabeza revertido (v5.1242)

- v5.1241 devolvía la cabeza medida de la foto original sobre el marco.
  Inspección visual del 23/10: `FastEdit_00836_.png` editaba el cuerpo y
  `edit_1792706837.png` solo difería en un óvalo del pecho (x 302-475,
  y 396-702 de 777×1600). El usuario pidió volver a la entrega que editaba.
- v5.1242: sin otras personas, la entrega es el marco generado. No se pega
  la cabeza original. Probe
  `probe_editor_subject_head_restore_20261022.py` reescrito a ese contrato.
- La cara del marco puede no ser la de la foto. Eso ya estaba medido en
  `edit_1792703256.png`. No reabrir el pegado de cabeza: colocaba la cara
  original en el torso.

## 2026-10-22 — cara de la persona en el cambio corporal (v5.1241) y prueba visual

- REVERTIDO el 2026-10-23. Ver la entrada de arriba. El texto siguiente
  describe el intento que se retiró.
- v5.1241 (implementado): la ruta corporal de seleccion SIN otras personas
  reinterpreta el marco con el motor con vision, y antes entregaba una cara
  inventada (medido: `edit_1792703256.png`, "otra imagen"). Ahora la cabeza
  MEDIDA de la seleccion (cara y pelo) se devuelve de la foto original, con
  realineacion acotada por correlacion local. Probe CPU
  `probe_editor_subject_head_restore_20261022.py` en verde (fail->pass).
- Pendiente de entonces: inspeccion visual del usuario. Hecha el 23/10 y
  fallida (cara original en el pecho). No cerrar ese pegado.
- `probe_editor_selection_contract_20261019.py` estaba en rojo con
  `early_route_precedence` (14/15): la ruta viva de seleccion (v5.1238/v5.1240)
  enruta "overhead camera angle" y "taller and change the posture" al motor de
  vision (LongCat) en vez de Kontext. RESUELTO 2026-10-23: el probe se
  actualizo al contrato vivo (nueva asertiva `assert_vision_frame`: motor de
  visión, foto completa sin mascara, caras ajenas protegidas) y queda 15/15
  PASS sin tocar la ruta en caliente.

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

## Relevo en curso del Editor — 2026-10-28

- Reescritura sin commitear (`working tree`): `image_quality_pipeline.py`
  (~3244 líneas), `selected_region_edit.py` (+450), `top_tier_finish.py`
  (+370), `element_selection.py`, `run.py`, `config/editor_selection.json`,
  `config/generation_prompt.json`. HEAD (`1dacf32`) es el código desplegado.
- Compatibilidad: decidir si el lote se commita (posteando pruebas) o se
  revierte/desfasado, antes de cerrar calidad. Especificar alcance y si afecta
  fuentes fuera de Image Editor (FaceSwap/Generate); no tocar Servidor ComfyUI
  8188 ni un render activo sin comprobar estado.
- Calidad de la inpaint de ropa con visión (H-EDIT-005, nuevo): pendiente de
  pruebas y validación visual del usuario.
- H-EDIT-SELECT / H-EDIT-SELECT-BODY / H-EDIT-SELECT-V ya APPROVE y PASS en
  código; validación visual del usuario sigue pendiente.


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

## 2026-10-28 — Segundo pase residual (LongCat Turbo inpaint sobre tiras)
- Ejecutado: diagnostics/probe_editor_residual_second_pass_20261028.py (arranco ComfyUI propio, cerro con close_if_ours)
- Base: img_editor/edit_1793205552.png (primera edicion LongCat Turbo)
- Mascara residual: 26067 px (CLIPSeg th 0.3/0.4/0.5, sujeto The woman)
- Metricas: dentro=71.91%, fuera=0.00%, seed 424242, denoise 0.95, steps 12
- Tiempo: 111 s
- Resultado: diagnostics/out/residual_second_pass_20261028/20261028_180929/
- Veredicto: segundo pase elimina mayoria de tiras; borde intacto; solucion practica probada.
