# Relevo vigente — Image Editor y prueba SAM2

Fecha: 2026-10-19, 23:20 Europe/Madrid. El usuario pide
«ANOTALO TODO PASRA EL SIGUIENTE». Leer primero este archivo y las cabeceras
de MEMORY.md; el relevo del18/10 es histórico, no el estado actual.

## Objetivo y permisos que persisten

- Continuar el selector visual aprobado: analizar foto, ver regiones medidas,
  elegir una o varias y editar con Kontext, protegiendo el exterior.
- Usuario «PUES ARREGLALO»: corregir máscaras parciales y obediencia visual.
- Usuario «SI» el19/10 autoriza probar **SAM2 ya instalado, sin descargar nada,
  e integrarlo SOLO si demuestra máscaras completas y seguras**. No volver a
  pedir permiso para esa evaluación. Esta autorización no cubre GrabCut,
  instalaciones, modelos nuevos, cambios de motor, APK ni pintado libre.
- Sin catálogos de objetos/prendas/géneros/ediciones escritos a mano para enrutar.
  Semillas, cajas y máscaras salen de mediciones de imagen; no coordenadas VLM.
- Leer C:/Users/INdaHouse/.agents/AGENTS.md y AGENTS.md del proyecto. SDD vigente
  en spec/00…04, plan aprobado plans/editor_object_selection_20261018.md.
- No commits, resets ni limpieza de cambios ajenos. Árbol muy sucio, con cambios
  anteriores en Generate/cámara/FaceSwap. No revertirlos ni atribuirlos a SAM2.
- No declarar HECHO sin reviewer APPROVE y tester PASS pertinentes, con evidencia
  visual. No instalar paquetes/modelos/skills sin aprobación explícita.

## Código de producción ya integrado y comprobado

- element_selection.py: huella RGB completa+tamaño, detector de cabezas real
  con bbox[:4]+landmarks, inventario Moondream→Qwen3B→CLIPSeg locales, máscaras
  medidas, validación estricta, selección múltiple, protección de otras cabezas.
- selected_region_edit.py + generate_intelligent: selección explícita antes de
  NLP/cámara/preflight/calidad; una pasada Kontext, máscara de inpaint=None,
  referencia limpia y pegado restringido incluso tras acabado. Rechazo conserva
  fuente, sin fallback global. Sin selección, mantiene la ruta previa.
- UI/state/selection_events: clic/lista múltiple, invalidación al cambiar/reusar
  foto, sesiones aisladas y descarte de análisis tardíos, lock de generación.
  API legacy13/9/8 preservada; endpoint de selección separado. PNG evita WebP
  al reutilizar. Sin reiniciar la UI activa ni reconstruir Android.
- Corrección geométrica: reference_canvas.py escala isotrópicamente, añade
  padding de borde a múltiplos64 y restaura el recorte antes del pegado.
  Cliente flux_edit_comfy_client.py la usa SOLO con reference_canvas explícito;
  legacy conserva preparación. Budgets inválidos rechazados antes del upload.
- Corrección de contexto: config/editor_selection.json edit.object_context_fraction
  =2.0, por procedencia kind=object; cabezas solas conservan base0.35.
  Se amplía la referencia, nunca el soporte de escritura. Sin palabras de routing.
- Reviewer APPROVE de producción/contratos finales; tester PASS técnico acotado.
  Geometría7/7, backend15/15, contexto independiente4/4, UI14/14 y API13/9/8+PNG
  verdes. Ruff/compileall/diff-check verdes. SAM2 aún NO está en producción.

## Evidencia visual previa: avance real y límites

Raíz: diagnostics/out/editor_selection_20261019/.

- Café fuente skimage/data/coffee.png600×400. Análisis real1792420385485144000:
  coffee cup omite cuerpo/asa; saucer marca cazoleta. Spoon omite mango trasero.
- Baseline cuchara1792421243545046400 y canvas/direct1792422658379575700:
  seguía plateada. Guía5 y T5FP8 local tampoco corrigieron el color. No promover
  esos ajustes ni repetir igual matriz esperando otro resultado.
- Contexto2.0 comparado con0.35 (misma receta directa/seed424242):
  1792441271736905200,106s, cazoleta dorada; raw también dora mango, pero final
  lo conserva plateado porque máscara lo omite. Informe tester_visual_review.md.
- Wrapper normal/seed424243 solicitada por CLI:1792441449254040300,79s,
  cazoleta dorada y exterior0. Esa traza vieja no serializaba semilla: no inventar
  metadatos retrospectivos ni llamarlo A/B solo de semilla frente al texto directo.
- Última muestra producción1792441719739668800,80s, seed424242 YA serializada,
  prompt «Make the selected spoon polished gold. Preserve its shape.», wrapper
  normal, contexto2.0, cazoleta dorada/mango plateado, exterior0. Tester confirmó
  semilla/config/una generación/máscara idéntica; informe junto al resultado.
- Cabeza1792441547925018000,88s: rojo, laterales más cortos, otra persona
  intacta, exterior0; expresión exacta/parecido no certificados.
- H-EDIT-SELECT sigue abierta. Protección de píxeles no certifica cumplir prompt
  ni cubrir objeto completo; la máscara puede estar mal etiquetada.

## SAM2 evaluado de verdad, aún aislado

- Modelo models/sam2/sam2.1-hiera-tiny/model.safetensors155908064bytes,
  SHA25648c14467e5cf9e51870511feb72c89688e82dd74523142c0538b663e193ac2a7.
  Config dice sam2_video, pero se usan Sam2Model y Sam2Processor de imagen
  explícitos, local_files_only=True. Loading_info sin keys faltantes, inesperadas,
  incompatibles ni errores. No descargas ni instalaciones.
- diagnostics/probe_editor_sam2_selection_20261019.py reutiliza embedding CPU,
  prueba single/distributed/seed_box/context_box, tres máscaras por prompt y
  grilla uniforme con dedup geométrico. No GT ni GrabCut en inferencia. Parámetros
  en diagnostics/editor_sam2_20261019.json y editor_sam2_dense_20261019.json.
- Primer intento1792444061870590200 falló serializando sets loading_info;
  corregido, fallo preservado. Corrida completa café1792444083551212000:
  68 máscaras guardadas,12.7s desde carga hasta guardado (no import inicial).
- People1792444143402673500:85 máscaras. Corbata púrpura single13 precisa;
  contexto puede invadir cabeza y traje; men in suits puede quedar en una persona.
- Café denso32×32=1792444274917406400:80 máscaras únicas guardadas. Ninguna
  nueva grilla une cuerpo+asa+borde de taza. Hay propuestas de plato prometedoras,
  pero incluyen fragmentos de taza. Mayor densidad no resuelve la semántica.
- Moto741×500, fuente skimage/data/motorcycle_left.png: inventario actual
  1792444096198289500; SAM2:1792444293037854100 con98 máscaras. Single1 score0.933
  toma solo depósito; distributed3/seed_box6 son más amplias. No basta maxscore.
  Detector también propuso una falsa cabeza sobre logo de motor; no confundirla
  con persona real ni ampliar por ello el alcance de FaceSwap.
- Contraejemplo decisivo: coffee saucer0038 selecciona cuchara y tiene
  IoU predicha0.953/stability0.953/seed_recall0.997. Es casi idéntico a spoon0014
  (IoU entre máscaras0.9955). Refina una semilla equivocada con confianza alta.
- 0014 OMITE mango trasero (bbox326,215–406,326). Mi primera impresión visual
  fue corregida al ampliar y medir: no conservar esa impresión como éxito.
- 0012 sí cubre cazoleta y mango trasero visibles; pequeña incertidumbre de bordes
  y14px del núcleo cup_body. Score0.654/stability0.701: no bajar filtros para
  admitirlo. Tester/reviewer permiten SOLO ensayo manual diagnóstico, no selector
  automático ni integración a producción. No unir todas las alternativas.
- Reviewer CHANGES_REQUESTED para admisión automática por filtros geométricos.
  TESTER_REVIEW.md y tester_sparse_core_metrics.json junto a las corridas;
  overlays de todos los candidatos y láminas VISUAL_REVIEW.md guardados.

## Oráculo y semántica: no falsear conclusiones

- GT sparse de agente, NO densa ni humana certificada:
  diagnostics/out/editor_selection_gt_20261019/coffee_v1/annotations.json.
  Núcleos positivos conservadores, exterior UNKNOWN. Nunca input del segmentador.
- ERRATA_spoon_neck.md: núcleo spoon_neck puede ser reflejo de taza, es
  DISPUTED/UNKNOWN y no decide admisión. Descubierto POST HOC, no evaluación
  ciega. V1/hashes/máscaras/métricas anteriores intactos. No redibujar GT para
  favorecer0012 ni tratar bajo recall del núcleo disputado como falta de cuello.
- diagnostics/evaluate_editor_sam2_cores_20261019.py evalúa todas las propuestas,
  marca parte disputada; respeta checksum canónico original UTF8/LF pese CRLF
  de escritura Windows y añade hash real de bytes para auditoría.
- CLIP local ViT-B16: diagnostics/probe_editor_sam2_semantics_20261019.py,
  settings editor_sam2_semantics_20261019.json,68 máscaras/136 vistas CPU47.8s.
  Scores en diagnostics/out/editor_sam2_semantics_20261019/1792444316088337000/.
  0038→silver spoon detecta error; PERO plato0066→coffee cup y asa0067→saucer
  en ambas vistas. Acuerdo/margen tampoco demuestra correspondencia correcta.
- Multiscale/alias CLIPSeg anterior188 variantes/3fotos también falló. No repetir
  ni promover alias inseguros, lowering thresholds, bbox inventado o convex hull.

## Estado final de los procesos y último render

Comprobar de nuevo antes de tocar nada; estado actualizado al23:20.

1. Render MANUAL diagnóstico de0012 TERMINADO, session48969 exit0, log
   logs/selection_sam2_spoon_manual_20261019.log. Adaptador
   diagnostics/probe_editor_sam2_edit_candidate_20261019.py, analysis real adaptado
   diagnostics/out/editor_sam2_manual_analysis_20261019/1792444518118707500/.
   Guarda máscara medida SAM2 y procedencia/hashes con elección manual post hoc.
   Misma petición breve/seed424242 de muestra1792441719739668800. El probe real
   propaga sam2_provenance y registra seed/config. NO es política automática.
   Resultado diagnostics/out/editor_selection_20261019/1792444518153626400/:
   root abrió result.png, cazoleta Y mango trasero dorados, exterior0. Una pasada,
   seed424242,20pasos,cfg3.5,256s. Referencia497×400→canvas768×640, por lo que
   también cambia el recorte respecto a baseline de máscara CLIPSeg; es comparación
   del flujo completo, no A/B de máscara aislada. Elección MANUAL post hoc, todavía
   NO una política automática. Prompt_id f9d3634b-4873-45c7-816a-20c9685e03b5.
   Tester independiente: PASS acotado bowl+mango dorados; cup_bodycore13px cambiados (14 en máscara), exterior0.
   Informe `diagnostics/out/editor_selection_20261019/1792444518153626400/tester_visual_review.md`; sin admisión automática.
   Se estancó paso2/20 mientras caption activo y luego terminó, sin reiniciar.
2. VLM Moondream diagnóstico INTERRUMPIDO, session50409 exit1, PID26488 ausente
   (padre20156), log
   logs/probe_editor_sam2_moondream_20261019.log, salida
   diagnostics/out/editor_sam2_moondream_20261019/1792444514154364300/.
   Primera crop0012, ninguna respuesta ni results.json; PNG/JPEG exacto
   enviado guardados. Script/config probe_editor_sam2_moondream_20261019.py y
   editor_sam2_moondream_20261019.json. Ctrl+C SOLO a ese probe auxiliar al
   preparar relevo; no se iniciaron otras5inferencias. INTERRUPTED.json conserva
   estado y hashes. No se tocó ComfyUI ni MUSICIA.
3. **CPU total de Moondream NO estaba garantizada.** n_gpu_layers=0 afecta
   texto, pero MoondreamChatHandler hereda Llava15ChatHandler sin opción CPU
   para proyector; llama_chat_format.py2709 llama clip_model_load(path,0), donde
   0 es verbosity. nvidia-smi muestra PID26488 en GPU. No repetir la afirmación
   previa CPU total. Posible competencia con render: VRAM7883/8192MiB, GPU100%,
   RAM física libre~14GB. Es hipótesis de ralentización, no causalidad probada.
4. ComfyUI8188 PID22944, padre25964, cola running0/pending0 verificada al finalizar.
   NO reiniciar ni matar. UI7861 no estaba
   escuchando y no fue relanzada. Proceso11228 GPU es ajeno: no tocarlo;
   MUSICIA/otros servicios del usuario quedan fuera del trabajo.

## Primeros pasos para el siguiente

1. Leer últimos tails de ambos logs y consultar /queue y /history del prompt_id;
   ambos diagnósticos de este relevo ya acabaron (uno completado, uno cancelado).
   Verificar estado actual, sin asumir que otros trabajos posteriores sean propios.
2. Abrir fuente/raw/result/máscara del render manual; revisar informe tester si existe.
   Comprobar mango+cazoleta dorados, taza/plato/resto, exterior0 y procedencia
   manual. No llamar a eso selección automática resuelta.
3. Moondream no produjo respuesta: intento incompleto documentado. Resolver
   aislamiento CPU real del proyector antes de reintentarlo junto a GPU.
4. Actualizar MEMORY.md y este relevo con resultados finales. Ruff/compilación
   de probes nuevos y diff-check ya verdes; no hay producción SAM2 que certificar.
5. Para integrar falta correspondencia semántica/completitud/seguridad verificadas
   en varias fotos, una política independiente de GT y APPROVE/PASS. Si recursos
   instalados no alcanzan, presentar evidencia y consultar NUEVO diseño/descarga.
   No revocar restricciones ni instalar automáticamente. Investigación SAM3 solo
   lectura iniciada, nada instalado ni autorizado; no convertirlo en elección.

## Fuentes primarias consultadas

- https://huggingface.co/docs/transformers/v5.5.2/en/model_doc/sam2
- https://github.com/facebookresearch/sam2/blob/main/sam2/automatic_mask_generator.py
- https://github.com/timojl/clipseg
- https://github.com/facebookresearch/sam3 (solo investigación, no instalación).
