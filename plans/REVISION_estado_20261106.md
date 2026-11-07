# Revisión de memoria, bitácora, requisitos y código — 2026-11-06

Alcance: revisión solicitada, contraste local y contratos CPU. Sin cambios de
procesamiento, versión, dependencias ni servicios. No se ejecutaron renders,
instalaciones, commits ni build nuevo. Evidencias:
`diagnostics/out/state_review_20261106_085256/summary.json` y logs contiguos.

## Estado comprobado

- Disco v5.1311; rama `experiment/klein-9b`, HEAD `4d7ce3c` (v5.1280).
  Git consultado antes de las correcciones documentales: 64 archivos rastreados
  modificados y 115 entradas sin rastrear; una entrada puede ser un directorio.
  El último commit no representa el estado actual de la carpeta.
- `logs/active_server.json` registra v5.1310, PID15720, puerto7861. El PID
  sigue presente; el fichero de registro por sí solo no verifica la versión
  cargada ni la entrega UI. ComfyUI también tiene proceso activo; sin reinicios.
- Auto es el predeterminado. Su ruta general configura LongCat y apariencia
  pura medida Kontext. La elección explícita de motor conserva su contrato.
- `.autohand/memory/events/LOG.jsonl` contiene tres eventos de capacidades,
  sin validaciones de producto. La bitácora efectiva es MEMORY y los WORKLOG.
  `.autohand/agents/tasks/todos.json` contiene cuatro tareas administrativas
  completadas; no representa todos los pendientes de spec/03 y spec/05.

## Resultados nuevos y límites

| Comprobación | Resultado |
|---|---|
| unittest progreso + acabado + versatilidad + controles Generate | 42/43 PASS; 1 FAIL |
| Progreso | 5/6; falla el contrato de latidos de consola |
| Acabado antes de TOP | 8/8 PASS CPU |
| Integridad de versatilidad | 23/23 PASS CPU |
| Controles Generate | 6/6 PASS CPU |
| Probe pegado/residual del 06/11 | 11/11 PASS sintético |
| Ruff Editor/Generate completo | FAIL, 94 incidencias: 88 + 6 respaldos ilegibles |
| Lint crítico sin respaldos | PASS, E9/F821/F822/F823 en 52 archivos |
| Sintaxis sin escribir pyc | PASS 52 archivos; 6 respaldos adicionales con bytes nulos |
| Build y calidad visual actual | No repetidos en esta revisión |

Los contratos sustituyen modelos/segmentación con dobles declarados. Generate
consulta disponibilidad de ComfyUI por HTTP local; no envía una inferencia real.
El runner inicial tuvo un error de salida de consola cp1252 después de guardar
el log CPU completo; se continuaron los checks restantes sin repetir la suite.

El FAIL corresponde a
`diagnostics/probe_comfy_wait_liveness_20261105.py:106`: espera al menos tres
latidos del mismo paso y recibe uno. `comfy_progress.py:510` impone
`max(heartbeat, 30.0)`, distinto del intervalo configurable. Los callbacks se
publican con su propia cadencia. No atribuir a esta prueba una pérdida de
callbacks, un cuelgue real ni la demora histórica 39:12.

## Coherencia y trazabilidad

- `spec/00` describía Kontext como predeterminado; se alinea con Auto sin tocar
  configuración. Se aclaran las notas históricas del umbral corporal y build
  Android en spec/03, conservando el historial.
- Los hashes actuales de `comfy_progress.py` y `selected_region_edit.py` difieren
  de `editor_stall_20261105/fix/tester/cpu_verdict.json` y el manifiesto real.
  Los hashes nuevos quedan en summary.json. Las aprobaciones v5.1310 son
  históricas y acotadas; no acreditan todas las fuentes actuales.
- Existe código posterior de pegado completo/residual y probe del 06/11 que
  no queda explicado por una cabecera de memoria de cambio solo documental.
  Esta observación no atribuye esos cambios al turno documental ni demuestra
  por sí sola que sean defectuosos. El probe actual pasa, falta su trazabilidad
  completa y verificación visual. `threshold=0.4` y `grow_frac=0.35` son
  parámetros literales que deben revisarse según la regla de configuración.
- H-EDIT-VERSATILITY-INTEGRITY sigue con casillas abiertas aunque algunos
  contratos CPU ya pasan; falta evidencia integral/revisión y aceptación visual.
  No marcar la historia terminada por los 23 contratos.
- Se confirman `body_transform=False` para retirada regional de objetos y
  umbral corporal leído/validado desde `instruction_scope` en configuración.

## Requisitos y pendientes efectivos

- Editor: conservar lo no solicitado; calidad por categoría, instrucciones
  compuestas, retirada completa y POV de primera persona siguen sin validación
  general. El neutral v5.1310 hizo la taza azul y cambió elementos ajenos:
  FAIL visual conservado, aunque el seguimiento temporal fue PASS técnico.
- Rendimiento: el caso histórico 39:12 sigue sin causa demostrada; la prueba
  neutral de unos 79 s no lo reproduce ni prueba aceleración general.
- FaceSwap: perfiles, oclusiones, halo, nitidez/identidad y vídeos representativos
  completos. La ruta histórica del vídeo falta; fotos/contratos no la sustituyen.
- Android: build/lint previos documentados; pendiente app instalada y prueba
  real contra Editor/Generate. No se comprobó ADB de nuevo en este turno.
- Generate/Chroma: muestra neutral previa acotada; faltan CFG/negativo,
  anatomía y variedad. Los seis contratos actuales no validan su imagen final.
- Modo Calidad: con texto se edita y después se aplican los pasos marcados;
  sin texto corre acabado. Realismo añade detalle/nitidez de forma clásica,
  sin otra generación en el flujo de UI. No garantiza reparar anatomía,
  textura artificial o una edición fallida; restaurar caras puede alterar rasgos.
  Falta validación visual amplia y render real 1080p/4K.

Revisión independiente inicial: CHANGES_REQUESTED documental por motor
predeterminado desactualizado, estados históricos y hashes distintos. Se corrigen
las inconsistencias documentales señaladas y se declaran los límites actuales;
no se solicita ni se afirma aprobación global del código.

Re-review independiente: **APPROVE documental acotado** del informe, cabecera
de MEMORY y correcciones de spec/00 y spec/03. Confirma correspondencia con
42/43 contratos CPU, residual11/11, Ruff94 y sintaxis/lint crítico52. Acredita
precisión de esta auditoría; no aprueba globalmente el código ni cierra historias.
