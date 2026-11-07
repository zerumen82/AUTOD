# spec/01 — Arquitectura

## Procesos

| Proceso | Puerto/ruta | Quién lo usa |
|---|---|---|
| AutoAuto UI (Gradio) | `ui/main.py` | Pestañas: 🚀 Generate, 🎨 Image Editor, Face Swap, Animate, Settings |
| ComfyUI (local) | 8188, fijo (`ui/tabs/comfy_launcher.py:18-21`) | Generate, Image Editor, Animate — **compartido** |
| Android (APK nativa Kotlin) | HTTP contra la API Gradio | `MainActivity.kt` + `GradioClient.kt` en `android_app/autoapp/` |

## Módulos clave

Selector visual aprobado 2026-10-18: `element_selection.py` propone regiones
medidas mediante detectores/segmentación locales, con parámetros en
`config/editor_selection.json`. Cada análisis contiene huella RGB, tamaño y
regiones (id, etiqueta, máscara, caja, procedencia). `gr.State` de sesión mantiene
análisis/selección; la imagen marcada es solo visualización. Un callback separado
envía al manager la imagen limpia y la selección validada. Los endpoints antiguos
conservan su firma y su flujo sin selección explícita.

La selección múltiple se une en una sola máscara y una pasada Kontext.
`selected_region_edit.py` se invoca antes de preflight/NLP/cámara/calidad en
`generate_intelligent`: valida huella RGB completa, tamaño, IDs y máscaras;
un rechazo conserva la foto, sin pasar a edición global. Recorta la fuente
limpia y limita el pegado al soporte medido, incluso tras acabado clásico.
La referencia de objetos recibe contexto configurable más amplio por su
procedencia medida; las cabezas conservan contexto local. `reference_canvas.py`
escala sin deformar y añade padding a bloques latentes, que el cliente retira
antes de restituir el tamaño del recorte. Esto no amplía la máscara de escritura
ni cambia la preparación legacy sin selección explícita.
`selection_events.py` mantiene revisión de análisis en `SelectionSession`
deepcopyable y descarta resultados tardíos tras cambiar/reutilizar la foto.
El backbone CLIP local se inyecta por parámetro; no se modifica `clip.load`.

### 🚀 Generate (`ui/tabs/gen/` + `roop/img_editor/flux_gen_comfy_client.py`)
- `ui.py` → formulario (prompt, estilo/luz/encuadre/piel, orientación, motor, ESRGAN, rewriter).
- `events.py` → `on_generate_image` → `get_flux_gen_client().generate_ai(...)`.
- `flux_gen_comfy_client.py` → traducción ES→EN, rewriter LLM, ensamblado de
  prompt (`gen_prompt_modifiers.py`) y workflows para SDXL, FLUX, Chroma y
  Klein; HART se delega a su cliente. `generate()` distingue T2I de imagen de
  referencia y admite de una a cuatro muestras. Sin seed explícita usa tiempo.
- `on_generate_image` conserva los diez primeros argumentos para la APK;
  el evento de escritorio añade controles, referencia y elección de ruta.
- Salida: `generate/gen_<ts>.png` y muestras adicionales con sufijo numérico
  (`roop/output_paths.py` y `ui/tabs/gen/events.py`).

### 🎨 Image Editor (`ui/tabs/img_editor/` + `roop/img_editor/img_editor_manager.py`)

- H-EDIT-REFERENCE (24/25-10): `_instruction_clauses` mantiene relativa y
  continuación del predicado juntas antes de extraer alcance/sujeto/hechos.
  En grounding focal de cabeza Kontext, CLIPSeg usa una plantilla fotográfica
  configurada que contiene el sujeto íntegro una sola vez; head/hair y prompts
  generativos siguen sus contratos. Bboxes y cobertura de todos los candidatos
  quedan en log; configuración inválida se rechaza sin elegir otra cabeza.
- H-EDIT-ENGINE-STATUS (24/10): el callback captura un informe por contexto de
  ejecución. Kontext elegido fija el archivo configurado como modelo requerido;
  la ruta medida con visión puede declarar expresamente otro modelo mediante
  `permit_vision_model` / `allow_vision_engine=True` (autorización posterior del
  25–28/10, registrada en MEMORY.md). Los clientes comprueban la política antes de sus efectos
  externos y registran solamente ejecuciones que devuelven imagen. El informe
  alimenta el estado visible; no altera prompts, máscaras ni muestreo. Un rechazo
  permanece durante la petición para impedir reintentos/fallbacks encubiertos.
- `generate_intelligent()` es la puerta de entrada: NLP
  (`nlp/semantic_analyzer.py`), elección de motor (`_route_edit_engine`),
  medición de máscaras, ventana local (`_appearance_work_region`), integración
  (`_integrate_local_generation`) y verificación VLM (`instruction_verifier`).
- Clientes: `flux_edit_comfy_client.py` (Kontext/Klein y compatibilidad de
  alias antiguos), `qwen_camera_comfy_client.py` (POV), `qwen_edit`, `boogu`,
  `hart`, `icedit`, `omnigen2`. La presencia de un cliente no autoriza usarlo
  cuando el contexto exige el modelo Kontext configurado sin excepción explícita.
- El cliente de edición puede conservar el modelo residente para reutilizarlo;
  no hay descarga obligatoria tras cada render. La liberación explícita usa
  `POST /autoauto/unload`; Generate aplica su propia descarga al finalizar.
  El nodo está en `ui/tob/ComfyUI/custom_nodes/autoauto_unload.py`.

### Contratos compartidos

- H-EDIT-WAIT-FINISH (05/11): `comfy_progress.py` mide espera/fases/pasos con
  reloj monotónico. El progreso WS disponible precede al sondeo HTTP; los
  latidos conservan pasos reales y muestran edad del evento/latencias, sin
  acreditar salud del proceso. El historial aporta duración del servidor
  por separado. Cadencias en `generation_prompt.json/comfy_progress`.
  En la rama corporal sin selección de `edit_described_subject`, el candidato
  se valida antes de TOP; el caller conserva rechazo/reintento y validación
  final tras el acabado. No cambia la receta ni la selección de motor.
- `roop/img_editor/prompt_translator.py` (ES→EN), `prompt_rewriter.py`, `comfy_progress.py`, `job_cancel.py` (scopes `SCOPE_GENERATION` / `SCOPE_IMG_EDITOR`).
- `roop/comfy_workflows.get_comfyui_url()` — compatibilidad de URL ComfyUI local.
- Calidad: `image_quality_pipeline.py` (ESRGAN/tiers), `hyperreal_polish.py`, `top_tier_finish.py`.

### Config
- `config/model_configs.json` → modelo, steps/cfg/sampler por motor, leído por
  los clientes de generación y edición.
- `kontext.model_file/steps/cfg` es la fuente de modelo y receta del Editor.
  Los identificadores LongCat legacy se normalizan antes del caché/loaders.
  Para apariencia focal, la región visual selecciona una cabeza, Kontext edita
  su recorte sin inpaint y el pegado/restauración protegen el exterior y todas
  las otras cabezas detectadas, incluso las de confianza insuficiente para elegir.
- `config/generation_prompt.json` → prefijos, `lora_catalog`.
- `config/generation_prompt.json/editor_ui` → default del Editor independiente
  de Generate y de los defaults compatibles de la API antigua.
- `config/camera_pose.json` → señal de medición y discriminación de candidatos
  de cámara; las 96 frases entrenadas siguen siendo el contrato del LoRA.
- Chroma usa T5 único con loader `type=chroma`, opciones del tokenizador y CFG
  real de `model_configs.json`; FLUX mantiene su workflow dual y FluxGuidance.

## Reglas de integración Generate ↔ Editor

1. El puente Gen→Edit ya existe: con imagen de referencia, `_route_is_editor`
   en `ui/tabs/gen/events.py` respeta `editor`/`img2img`; en `auto` consulta
   `route_prefers_editor`. La rama de edición importa el manager de forma
   diferida. Sin imagen no se usa el puente; `img2img` es el default actual.
2. Cualquier parámetro nuevo de Generate usa el mismo patrón que el editor: aceptado por `generate(...)`, con default tomado de `config/`.
3. El servidor 8188 no se reinicia entre pestañas; toda prueba real debe comprobar antes si hay un render activo.
