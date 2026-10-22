# Bitácora incremental — corrección del selector visual

Petición vigente: «CADA EDIT LO DOCUMENTAS ASÍ PUEDEN RELEVARTE, SIGUE».
Este archivo registra cada edición de código/configuración con finalidad,
verificación y estado pendiente; MEMORY.md y el HANDOFF apuntan al último paso.
No equivale a declarar la historia terminada.

## E00 — 2026-10-19 23:34, retomar con evidencia

- Leídos AGENTS global/proyecto, skill fix, MEMORY, HANDOFF y código actual.
- Producción mantiene corrección de escala/contexto; SAM2 sigue diagnóstico,
  sin admisión automática aprobada. No instalaciones ni cambios de motor/APK.
- Tester ya cerró el último render manual1792444518153626400: PASS visual
  acotado a cazoleta y mango dorados, exterior0,13 píxeles cambiados dentro del
  núcleo de taza invadido por la propia máscara. No afirmar taza intacta total.
- Hipótesis siguiente: contrastar semántica de propuestas con VLM local aislado
  del GPU, porque CLIP confundió plato/taza y el intento VLM anterior no terminó.
- Comprobación actual: ComfyUI cola0/0; GPU100% con1988MiB libres pese cola
  vacía. No usar proyector GPU ni tocar procesos externos. Probar entorno CUDA
  oculto solo en subprocess diagnóstico, verificar que termina y observar backend.
- Criterio: descripciones abiertas reales de máscaras, sin labels/GT en prompt,
  evaluar errores visuales; si falla, no usarlo como admisión ni integrar SAM2.

## E01 — 2026-10-19, trazabilidad del diagnóstico VLM

- Editado diagnostics/probe_editor_sam2_moondream_20261019.py: guardar
  results.json antes de cargar/inferir y tras carga; registrar visibilidad CUDA,
  estado running/completed/interrupted/failed y error. Corregir mensaje que
  decía CPU total cuando solo se exigía n_gpu_layers=0 para texto.
- Motivo: el intento previo se interrumpió sin results.json y el proyector
  podía usar GPU. La visibilidad CUDA se registra como condición de ejecución,
  sin afirmar backend CPU observado antes de comprobarlo.
- Verificación realizada: Ruff y py_compile verdes; diff-check verde.
  Pendiente inmediatamente después de editar:
  ejecución real del mismo probe con CUDA_VISIBLE_DEVICES=-1 en su subprocess.
- Producción: ningún cambio; no se ha promovido ninguna máscara ni filtro.

## E02 — 2026-10-19, observar el backend del proyector

- Editados probe_editor_sam2_moondream_20261019.py y su config diagnóstica:
  projector_load_verbose=true durante construcción; desactivar verbose en esa
  instancia antes de captions para evitar imprimir el data URL.
- Motivo: comprobar fallback de DLL legacy al ocultar CUDA; n_gpu_layers=0
  por sí solo no controla el proyector. Reviewer comprobó que esta DLL no
  exporta setter CPU; el entorno del subprocess es la prueba no invasiva.
- Ejecución siguiente con CUDA_VISIBLE_DEVICES=-1, solo este proceso; guardar
  log/JSON nuevos y comprobar backend y descripciones antes de cualquier admisión.
- Sin cambios de producción, dependencias ni otros procesos.

## T01 — 2026-10-19, veredicto independiente del último ensayo manual

- Añadidas dos líneas junto al resultado manual en MEMORY.md y otras dos en
  plans/HANDOFF_editor_selection_20261019.md; no se reescribieron otras secciones.
- Motivo: dejar cerrado y localizable el veredicto tester del render
  1792444518153626400: PASS visual acotado de cazoleta y mango trasero dorados,
  exterior0; 13 píxeles del núcleo de cuerpo de taza cambiados, de 14 incluidos
  en la máscara. No afirmar preservación total de taza ni admisión automática.
- Evidencia independiente ya guardada:
  diagnostics/out/editor_selection_20261019/1792444518153626400/tester_visual_review.md.
- Verificación de esta edición: lectura de ambas inserciones y contraste con
  el informe; solo documentación. Sin nuevas inferencias ni cambios de producción.
- Pendiente: resolver selección semántica completa y segura en varias fotos;
  el éxito manual post hoc no constituye una política automática aprobada.

## E03 — 2026-10-19, 23:4x, aislamiento real del proyector (CUDA oculta)

- Relevo verificado: el intento de las 23:36 NO falló por el modelo; murió
  cortado por el límite de 30 s del shell (directorio de evidencia vacío, 0
  ficheros, sin `results.json`). El script ya guardaba `results.json` antes de
  cargar, pero el proceso murió antes de escribirlo. Relanzado en segundo plano.
- Corrida real: `diagnostics/out/editor_sam2_moondream_20261019/1792446081024742800/`,
  log `logs/probe_editor_sam2_moondream_cpu_isolated_20261019.log` (+ `.err`).
  Comando: `CUDA_VISIBLE_DEVICES=-1` + `PYTHONPATH` en el venv del proyecto.
  El primer lanzamiento falló con `ModuleNotFoundError: No module named
  'roop'` (cwd/PYTHONPATH); corregido fijando `WorkingDirectory` y `PYTHONPATH`.
- **Aislamiento CUDA PROBADO en runtime**, no supuesto: `ggml_cuda_init: failed
  to initialize CUDA: no CUDA-capable device is detected` →
  `CLIP using CUDA backend` → `CLIP using CPU backend` →
  `params backend buffer size = 867.61 MB (457 tensors)`. El proyector LLaVA
  carga en CPU sin selector expuesto, que era la duda de E02.
  `results.json` con `cuda_visible_devices: "-1"`, `status: running`.
- Sin cambios de producción, sin instalaciones, sin tocar ComfyUI (8188 vivo,
  cola 0/0 verificada) ni procesos ajenos (MUSICIA).
- Pendiente: leer las 6 descripciones de los candidatos medidos, valorarlas
  como semántica (no como admisión automática) y cerrar E02/E03 en el relevo.

## E04 — 2026-10-20, GPU libre: pruebas reales + guard de producción

- La GPU estaba ocupada (613 MiB usados al inicio de esta tanda; Codex la liberó
  a 613/8192 MiB usados, ~7579 MiB libres) y la cola de ComfyUI estaba 0/0.
  Con esa margen se relanzaron las pruebas sin tocar procesos ajenos.
- **Probe de 6 candidatos, GPU real**: `diagnostics/out/editor_sam2_moondream_
  20261019/1792447384656908100/`, log `logs/probe_editor_sam2_moondream_
  20261019_gpu_free.log`. 6/6 descripciones (0012, 0014, 0038, 0062, 0066, 0067)
  entre 8,43 s y 12,43 s, `status: completed`, sin crash. VRAM observada en
  1972 MiB (antes 7759 MiB con la GPU ocupada). En el log:
  `clip_model_load: CLIP using CUDA backend` y
  `params backend buffer size = 867.61 MB (457 tensors)`.
- **Detector de backend validado en runtime**, con salida explícita y proceso
  limpio: `diagnostics/probe_moondream_gpu_backend_20261020.py` y
  `diagnostics/out/moondream_gpu_backend_20261020/`.
  `llama_supports_gpu_offload()` = True con CUDA visible, False con
  `CUDA_VISIBLE_DEVICES=-1` (`ggml_cuda_init: failed to initialize CUDA`).
  El orden de importación NO importa (`order_hazard`: True antes y después de
  añadir las DLL y tras `llama_backend_init`), porque `ggml-cuda.dll` resuelve
  junto a `llama.dll` en `venv/Lib/site-packages/llama_cpp/lib`.
- **Evidencia del crash en CPU ya cerrada** (previo):
  `moondream_cpu_control_20261019/1792446285109944900` deja la fase `vision`
  en `status: running` con `CUDA_VISIBLE_DEVICES=-1` (muerte nativa, no
  capturable), mientras `1792446473658902100` con CUDA visible deja
  `vision: completed` con descripción real. El texto solo en CPU es seguro.
- **Ruta de producción completa con GPU libre**: `analyze_elements` sobre
  `coffee.png` → 5 regiones medidas (coffee cup 17,70 %, silver spoon 2,85 %,
  wooden table 45,66 %, saucer 1,44 %, gray background 3,18 %) en 132,73 s,
  `diagnostics/out/editor_selection_20261019/1792448112336042600/`. Sin crash.
- **Guard de producción implementado**: `roop/img_editor/vision_guard.py`
  (nuevo) con `check_vision_preflight()`; integra en
  `roop/img_editor/element_selection.py::_moondream_description` y en
  `roop/img_editor/instruction_verifier.py::_load` (misma ruta de visión).
  Reglas: el backend ggml debe exponer descarga a GPU y la VRAM libre debe
  cubrir el proyector; si no, se aborta con mensaje claro ANTES de cargar
  modelos, nunca con la ruta CPU-vision que aborta el proceso.
  Parámetros en `config/editor_selection.json` → `vision_guard`
  (`enabled`, `vram_margin_mb`), sin valores mágicos en el código.
  Excepción propia `VisionUnavailableError` para que el rechazo no se
  degrade a «análisis parcial» en `analyze_elements`.
- **Verificación del guard, 4 escenarios reales**
  (`diagnostics/probe_vision_guard_20261020.py`,
  `diagnostics/out/vision_guard_20261020/results.json`): `allow` OK con
  descripción real; `no_cuda` rechazado por backend sin GPU; `low_vram`
  rechazado por VRAM insuficiente; `disabled` (guard apagado) permite la
  visión. En `allow`/`disabled`: 7849 MiB libres, 1891 MiB requeridos
  (proyector 867 MiB + 1024 MiB de margen).
- **HALLAZGO QUE CAMBIÓ EL DISEÑO**: `torch.cuda.mem_get_info` NO ve la VRAM
  de otros procesos. Con una reserva externa de 6400 MiB, `mem_get_info` solo
  bajó 3738 MiB mientras `nvidia-smi` subiría 6400 MiB. Medido en
  `diagnostics/out/vision_guard_busy_gpu_20261020/vram_probe_6.0.json` con
  `diagnostics/probe_vram_visibility_20261020.py`. Con `mem_get_info` el guard
  habría aprobado una ejecución que no cabe. **El guard usa `nvidia-smi`
  (usada/total) como fuente de VRAM**, con `mem_get_info` solo como respaldo
  si `nvidia-smi` no está disponible.
- **Escenario decisivo: GPU ocupada por otro proceso**
  (`diagnostics/probe_vision_guard_busy_gpu_20261020.py`): con 6656 MiB
  reservados por un proceso ajeno, el guard rechaza con
  `VRAM insuficiente: 800 MiB libres y se necesitan 1891 MiB para el
  proyector` y el proceso **termina con exit 0, sin crash nativo**. Con
  5120 MiB reservados (2612 MiB libres) permite y la descripción real se
  genera sin error.
- **Verifier protegido**: `instruction_verifier.py` usa el mismo proyector.
  Con el guard activo: `[InstructionVerifier] Moondream: proyector en GPU
  listo` → carga el modelo → veredicto `Fail` correcto (la taza sigue en la
  imagen). Evidencia `diagnostics/out/vision_guard_20261020/
  instruction_verifier.json`.
- **Regresión de producción**: `analyze_elements` con el guard activo →
  5 regiones idénticas (coffee cup, silver spoon, wooden table, saucer, gray
  background), 83,58 s, `diagnostics/out/editor_selection_20261019/
  1792449159773368700/`. Sin regresión respecto a la corrida sin guard.
- **El mensaje de rechazo SÍ llega a la UI**: el callback real
  `selection_events.on_analyze` devuelve
  `Error analizando: Moondream no se puede ejecutar: VRAM insuficiente: 1056
  MiB libres y se necesitan 1891 MiB para el proyector`, sin excepción y con
  la sesión intacta (`diagnostics/probe_vision_guard_ui_busy_20261020.py`,
  `diagnostics/out/vision_guard_20261020/ui_busy_gpu.json`, RETURNCODE 0).
  La UI no se relanzó, pero el callback de producción que la UI invoca sí
  está verificado.
- Ruff (E9,F63,F7,F82) y py_compile verdes en los tres ficheros tocados y en
  los seis probes nuevos. Sin descargas, sin instalaciones, sin tocar FaceSwap,
  Generate, Android ni ComfyUI.
- Pendiente: decidir si el margen de 1024 MiB es el adecuado cuando ComfyUI
  está cargando un modelo en paralelo (no medido).

