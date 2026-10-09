# spec/01 — Arquitectura

## Procesos

| Proceso | Puerto/ruta | Quién lo usa |
|---|---|---|
| AutoAuto UI (Gradio) | `ui/main.py` | Pestañas: 🚀 Generate, 🎨 Image Editor, Face Swap, Animate, Settings |
| ComfyUI (local) | 8188, fijo (`ui/tabs/comfy_launcher.py:18-21`) | Generate, Image Editor, Animate — **compartido** |
| Android (APK) | HTTP contra la UI Gradio | `android_app/autoapp/` |

## Módulos clave

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
- `config/generation_prompt.json` → prefijos, `lora_catalog`.

## Reglas de integración Generate ↔ Editor

1. Generate **no** importa `img_editor_manager` hoy; la Fase 2 lo añade como ruta opcional (solo cuando hay imagen de entrada y la instrucción es de edición).
2. Cualquier parámetro nuevo de Generate usa el mismo patrón que el editor: aceptado por `generate(...)`, con default tomado de `config/`.
3. El servidor 8188 no se reinicia entre pestañas; toda prueba real debe comprobar antes si hay un render activo.
