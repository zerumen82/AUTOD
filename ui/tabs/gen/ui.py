import gradio as gr
from roop.img_editor.flux_gen_comfy_client import (
    get_default_generation_engine,
    get_installed_generation_engines,
)
from roop.img_editor.gen_prompt_modifiers import get_dropdown_choices, get_compatible_dropdown_choices, preview_modifiers
from roop.img_editor.gen_prompt_config import get_generate_ui_config, get_lora_catalog


def _lora_dropdown_choices():
    """Dropdown de LoRA: Auto, Sin LoRA y el catálogo completo visible.

    El usuario ve y elige sus LoRAs (v5.1210); la elección manual se aplica
    sin gates con la fuerza del catálogo.
    """
    choices = [
        ("🤖 Auto (según escena)", "Auto"),
        ("🚫 Sin LoRA", "None"),
    ]
    for entry in get_lora_catalog():
        tags = []
        if entry.get("nsfw"):
            tags.append("NSFW")
        if entry.get("sdxl_only"):
            tags.append("SDXL")
        if entry.get("pony_only"):
            tags.append("Pony")
        label = str(entry.get("label") or entry.get("file") or "")
        if tags:
            label = f"{label} ({', '.join(tags)})"
        choices.append((label, str(entry.get("file") or "")))
    return choices


CSS_STYLE = """
    .gen-tab-container {
        background: #020617;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #1e293b;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .gen-tab-header { text-align: center; margin-bottom: 25px; }
    .gen-tab-header h2 {
        background: linear-gradient(90deg, #a855f7, #22d3ee);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 800;
    }
    .prompt-box-gen {
        background: #0f172a !important;
        border: 2px solid #a855f7 !important;
        border-radius: 12px !important;
        font-size: 20px !important;
    }
    .prompt-box-gen textarea {
        font-size: 20px !important;
        line-height: 1.6 !important;
    }
"""


def build_gen_ui():
    gr.HTML(f"<style>{CSS_STYLE}</style>")

    with gr.Column(elem_classes=["gen-tab-container"]):
        with gr.Group(elem_classes=["gen-tab-header"]):
            gr.Markdown("## 🚀 GENERAR")
            gr.Markdown(
                "_Escribe en español. Tu prompt va primero; estilo solo si lo eliges en los desplegables._"
            )

        prompt = gr.Textbox(
            label="Prompt",
            placeholder="Describe la escena que quieres generar...",
            lines=4,
            elem_classes=["prompt-box-gen"],
        )

        with gr.Accordion(
            "🖼️ Imagen de referencia (opcional — img2img)",
            open=False,
        ):
            gr.Markdown(
                "_Sube una foto y el generador arrancará de ella (fuerza desde "
                "`config/model_configs.json` → `img2img_denoise`). Con LongCat la usa "
                "como referencia de edición._"
            )
            # Ruta por defecto en config/generation_prompt.json →
            # generate_ui.default_route. Decisión del usuario 2026-10-16:
            # Generate y el Editor son herramientas paralelas e
            # independientes; Generate NO debe enrutar sola una petición al
            # Editor. "editor" sigue disponible como elección manual.
            _route_value = str(
                get_generate_ui_config().get("default_route", "img2img")
            ).strip().lower()
            if _route_value not in ("auto", "editor", "img2img"):
                _route_value = "img2img"
            route = gr.Dropdown(
                choices=[
                    ("Auto (detecta si es una edición)", "auto"),
                    ("Image Editor (cambios guiados)", "editor"),
                    ("img2img (variar la foto)", "img2img"),
                ],
                value=_route_value,
                label="Ruta con imagen de referencia",
            )
            ref_image = gr.Image(label="Foto de partida", type="pil", height=300)

        with gr.Accordion("🎨 Estilo (opcional — los defaults ya son lo más realista)", open=False):
            with gr.Row():
                image_style = gr.Dropdown(
                    choices=get_dropdown_choices("image_type"),
                    value="photoreal",
                    label="Tipo de imagen",
                )
                lighting = gr.Dropdown(
                    choices=get_dropdown_choices("lighting"),
                    value="natural",
                    label="Iluminación",
                )
                shot_framing = gr.Dropdown(
                    choices=get_dropdown_choices("framing"),
                    value="auto",
                    label="Plano / encuadre",
                )
                skin_finish = gr.Dropdown(
                    choices=get_dropdown_choices("skin_finish"),
                    value="detailed",
                    label="Piel",
                )
                color_grade = gr.Dropdown(
                    choices=get_dropdown_choices("color_grade"),
                    value="auto",
                    label="Color / gradación",
                )

        modifier_preview = gr.HTML("")

        with gr.Row():
            orientation = gr.Radio(
                choices=["Horizontal (Paisaje)", "Vertical (Retrato)"],
                value="Horizontal (Paisaje)",
                label="Forma",
            )
            _engines = get_installed_generation_engines()
            _default = get_default_generation_engine()
            # v5.1131f — asegurar que las opciones que envía la APK móvil
            # ('flux_dev_q4', 'flux_dev_abliterated', 'schnell', 'klein_base') SIEMPRE están en
            # choices: Gradio rechaza valores fuera de la lista en preprocess
            # (mismo fallo que 'imagine' en Edit). 'klein_base' no sale de
            # get_installed_generation_engines (es GGUF diffusion, no checkpoint)
            # pero el gen client lo soporta nativo (flux_alias_map).
            _apk_choices = [
                ("FLUX Dev Q4", "flux_dev_q4"),
                ("FLUX Abliterated (ultra realista)", "flux_dev_abliterated"),
                ("FLUX Schnell (rápido ~8 pasos)", "schnell"),
                ("FLUX.2 Klein 4B base", "klein_base"),
            ]
            _existing = {v for _, v in (_engines or [])}
            for _c in _apk_choices:
                if _c[1] not in _existing:
                    _engines.append(_c)
            engine_model = gr.Dropdown(
                choices=_engines or [("FLUX Abliterated (ultra realista)", "flux_dev_abliterated")],
                value=_default,
                label="Modelo",
            )

        with gr.Row():
            enhance_quality = gr.Checkbox(
                label="✨ Mejorar nitidez (ESRGAN, más lento)",
                value=False,
            )
            use_rewriter = gr.Checkbox(
                label="🧠 Rewriter LLM (mejor prompt, +5-15s)",
                value=False,
            )
            nsfw_content = gr.Checkbox(
                label="🔞 Contenido NSFW",
                value=bool(get_generate_ui_config().get("nsfw_content_default", False)),
                info="Permiso para el sesgo adulto: solo se aplica si el prompt lo pide (el prompt manda).",
            )

        lora_choice = gr.Dropdown(
            choices=_lora_dropdown_choices(),
            value="Auto",
            label="LoRA",
            info="Auto la elige según la escena; lo que tú elijas se aplica tal cual.",
        )

        gr.Markdown("_👆 Prompt + GENERAR ya funciona: los defaults (photoreal, natural, piel detallada) son los más realistas. Los extras son opcionales._")

        with gr.Row():
            width = gr.Slider(
                minimum=512, maximum=1536, step=64, value=1152,
                label="Ancho (px)",
            )
            height = gr.Slider(
                minimum=512, maximum=1536, step=64, value=768,
                label="Alto (px)",
            )

        with gr.Accordion("⚙️ Avanzado (vacío/0 = automático)", open=False):
            with gr.Row():
                seed = gr.Number(
                    value=0, precision=0,
                    label="Semilla (0 = aleatoria)",
                )
                steps = gr.Number(
                    value=0, precision=0,
                    label="Pasos (0 = del modelo)",
                )
                cfg_scale = gr.Number(
                    value=0.0,
                    label="CFG (0 = del modelo)",
                )
            negative_prompt = gr.Textbox(
                label="Negative prompt (vacío = el del modelo)",
                lines=2,
            )
            with gr.Row():
                enhance_tier = gr.Radio(
                    choices=[("HD (1920px)", "hd"), ("4K (3840px)", "4k")],
                    value="hd",
                    label="Calidad de salida",
                    info="Lado mayor del acabado; ampliar no inventa detalle.",
                )
                samples = gr.Radio(
                    choices=[("1", "1"), ("2", "2"), ("3", "3"), ("4", "4")],
                    value="1",
                    label="Muestras por corrida",
                    info="Solo txt2img; con imagen de referencia sale 1.",
                )

        with gr.Row():
            gen_btn = gr.Button("GENERAR", variant="primary", size="lg", scale=3)
            btn_cancel = gr.Button("⏹ CANCELAR", variant="stop", interactive=False, scale=1)

        status_html = gr.HTML("<div style='text-align:center;color:#64748b;padding:10px;'>Listo</div>")
        output_img = gr.Image(label="Resultado", height=600)

        with gr.Row():
            bt_to_editor = gr.Button("→ ✨ Editar este resultado en Image Editor", size="sm")
            bt_open_folder = gr.Button("📂 Abrir carpeta de salidas", size="sm")

        # Contrato legacy de la APK (GradioClient.callGenerate envía 10 valores
        # posicionales): se registra aparte para no romper ese orden.
        api_legacy_btn = gr.Button(visible=False)

    return {
        "prompt": prompt,
        "ref_image": ref_image,
        "route": route,
        "image_style": image_style,
        "lighting": lighting,
        "shot_framing": shot_framing,
        "skin_finish": skin_finish,
        "color_grade": color_grade,
        "orientation": orientation,
        "engine_model": engine_model,
        "enhance_quality": enhance_quality,
        "use_rewriter": use_rewriter,
        "nsfw_content": nsfw_content,
        "lora_choice": lora_choice,
        "modifier_preview": modifier_preview,
        "width": width,
        "height": height,
        "seed": seed,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "negative_prompt": negative_prompt,
        "enhance_tier": enhance_tier,
        "samples": samples,
        "gen_btn": gen_btn,
        "btn_cancel": btn_cancel,
        "status_html": status_html,
        "output_img": output_img,
        "bt_to_editor": bt_to_editor,
        "bt_open_folder": bt_open_folder,
        "api_legacy_btn": api_legacy_btn,
    }
