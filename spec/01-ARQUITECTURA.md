# spec/01 — Arquitectura

## Procesos

| Proceso | Puerto/ruta | Quién lo usa |
|---|---|---|
| AutoAuto UI (Gradio) | `ui/main.py` | Pestañas: 🚀 Generate, 🎨 Image Editor, Face Swap, Animate, Settings |
| ComfyUI (local) | 8188, fijo (`ui/tabs/comfy_launcher.py:18-21`) | Generate, Image Editor, Animate — **compartido** |
| Android (APK) | HTTP contra la UI Gradio | `android_app/autoapp/` |

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
- `flux_gen_comfy_client.py` → traducción ES→EN, rewriter LLM, ensamblado de prompt (`gen_prompt_modifiers.py`), workflow inline:
  - rama SDXL `:674-697`, rama FLUX/LongCat/Klein `:698-772`, HART delegado `:655-672`.
  - `denoise=1.0`, `batch=1`, seed por tiempo si no se pasa (`:691`, `:766`).
- Salida: `generate/gen_<ts>.png` (`roop/output_paths.py`).

### 🎨 Image Editor (`ui/tabs/img_editor/` + `roop/img_editor/img_editor_manager.py`)
- `generate_intelligent()` (`:3975`) es la puerta de entrada: NLP (`nlp/semantic_analyzer.py`), routing de motor (`_route_edit_engine :467`), máscaras (objeto `:3719`, persona `:3741/3821`, fondo `:2186`, ropa `:5540`), ventana local (`_appearance_work_region :1616`), integración (`_integrate_local_generation`), verificación VLM (`instruction_verifier`).
- Clientes: `flux_edit_comfy_client.py` (LongCat/Kontext/Klein), `qwen_camera_comfy_client.py` (POV), `qwen_edit`, `boogu`, `hart`, `icedit`, `omnigen2`.
- Libera VRAM tras generar (`POST /autoauto/unload`, `custom_nodes/autoauto_unload.py:48`).

### Contratos compartidos
- `roop/img_editor/prompt_translator.py` (ES→EN), `prompt_rewriter.py`, `comfy_progress.py`, `job_cancel.py` (scopes `SCOPE_GENERATION` / `SCOPE_IMG_EDITOR`).
- `roop/comfy_workflows.get_comfyui_url()` — único símbolo usado de ese módulo legacy.
- Calidad: `image_quality_pipeline.py` (ESRGAN/tiers), `hyperreal_polish.py`, `top_tier_finish.py`.

### Config
- `config/model_configs.json` → steps/cfg/sampler por motor (leído en `flux_gen_comfy_client.py:224-240`).
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

1. Generate **no** importa `img_editor_manager` hoy; la Fase 2 lo añade como ruta opcional (solo cuando hay imagen de entrada y la instrucción es de edición).
2. Cualquier parámetro nuevo de Generate usa el mismo patrón que el editor: aceptado por `generate(...)`, con default tomado de `config/`.
3. El servidor 8188 no se reinicia entre pestañas; toda prueba real debe comprobar antes si hay un render activo.
