import gradio as gr

from roop.img_editor.gen_prompt_config import get_editor_ui_config


CSS_STYLE = """
    .img-editor-container {
        background: #020617;
        padding: 30px;
        border-radius: 20px;
        border: 1px solid #1e293b;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    .img-editor-header { text-align: center; margin-bottom: 25px; }
    .img-editor-header h2 {
        background: linear-gradient(90deg, #22d3ee, #a855f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 32px;
        font-weight: 800;
    }
    .btn-transform-main {
        background: linear-gradient(135deg, #06b6d4 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: 900 !important;
        height: 64px !important;
        border-radius: 14px !important;
        font-size: 18px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: none !important;
    }
    .prompt-box-img {
        background: #0f172a !important;
        border: 2px solid #22d3ee !important;
        border-radius: 12px !important;
    }
"""


def build_img_editor_ui():
    gr.HTML(f"<style>{CSS_STYLE}</style>")

    with gr.Column(elem_classes=["img-editor-container"]):
        with gr.Group(elem_classes=["img-editor-header"]):
            gr.Markdown("## ✨ IMAGE EDITOR AI")
            gr.Markdown(
                "_Edita cualquier cosa con lenguaje natural: color, pose, fondo, "
                "añadir/quitar objetos, ropa, calidad… Sin censura._"
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(
                    label="Imagen de Entrada", type="pil", format="png", height=480,
                )

                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder=(
                        "Ej: colorea esta foto B/N · hazla de pie · quita la persona del fondo · "
                        "añade un sombrero · vestido rojo · cambia el fondo por una playa · "
                        "más saturado y nítido…"
                    ),
                    lines=3,
                    elem_classes=["prompt-box-img"]
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ TRANSFORMAR", variant="primary", elem_classes=["btn-transform-main"], scale=3)

                with gr.Row():
                    quality_mode = gr.Checkbox(
                        label="🎨 Modo Calidad",
                        value=False,
                        info=(
                            "Elige los pasos. Sin texto solo corre el acabado; "
                            "con texto edita primero y luego aplica los pasos marcados. "
                            "Ampliar no inventa detalle."
                        ),
                    )

                with gr.Group(visible=False) as quality_options:
                    quality_deposter = gr.Checkbox(
                        label="Desposterizar",
                        value=True,
                        info="Suaviza bloques y bandas. En una foto ya nítida la fuerza es baja.",
                    )
                    quality_upscale = gr.Checkbox(
                        label="Ampliar",
                        value=True,
                        info="Sube a HD, 4K u 8K si la foto es más pequeña. Desmarcado, conserva el tamaño.",
                    )
                    enhance_tier = gr.Radio(
                        choices=[("HD (1920px)", "hd"), ("4K (3840px)", "4k"), ("8K (7680px)", "8k")],
                        value="hd",
                        label="Tamaño si Ampliar está marcado",
                        info="Lado mayor: HD 1920, 4K 3840 o 8K 7680 px. Ampliar no recupera detalle ausente.",
                    )
                    quality_color = gr.Checkbox(
                        label="Color y luz",
                        value=True,
                        info="Ajusta contraste, color y luz sobre la misma foto.",
                    )
                    quality_realism = gr.Checkbox(
                        label="Realismo",
                        value=True,
                        info="Añade detalle local y un poco de nitidez. No lanza otra generación.",
                    )
                    quality_preserve_faces = gr.Checkbox(
                        label="Restaurar caras (puede alterar rasgos)",
                        value=True,
                        info="Desactívalo si las caras originales ya se ven bien o si cambia la identidad.",
                    )
                legacy_api_btn = gr.Button(visible=False)
                legacy_options_btn = gr.Button(visible=False)

                with gr.Accordion("⚙️ Opciones", open=False):
                    engine_choices = [
                        ("🎨 Auto: motor según la instrucción (POV→Qwen, cuerpo fuerte→Kontext)", "kontext"),
                        ("📷 Qwen Cámara (POV/ángulos, ~1152px en 8 GB)", "qwen_camera"),
                        ("Compatibilidad app antigua (Kontext)", "imagine"),
                        ("✨ FLUX.2 Klein 4B (img2img rápido, 4 pasos, entra en 8GB)", "klein"),
                        ("⚠️ FLUX.2 Klein 9B (NO entra en 8GB, usar solo con 12GB+)", "klein_9b"),
                        ("🎯 FLUX.1 Kontext Q4_K_M (12B, ~6.5GB)", "kontext"),
                    ]
                    default_engine = get_editor_ui_config().get("default_engine")
                    if default_engine not in {value for _label, value in engine_choices}:
                        raise ValueError(
                            "editor_ui.default_engine debe nombrar un motor del Editor "
                            "en config/generation_prompt.json"
                        )
                    engine = gr.Dropdown(
                        choices=engine_choices,
                        value=default_engine, label="Motor"
                    )
                    with gr.Row():
                        denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0,
                                            label="Fuerza de Edición (0 = Auto)",
                                            info="0.2: Sutil, 0.6: Medio, 0.9: Radical")

                with gr.Accordion("🔄 Generación Autoregresiva", open=False):
                    gr.Markdown("""
                        La **generación autoregresiva** encadena iteraciones:
                        el resultado de cada paso se realimenta como entrada.
                        Útil para cambios complejos: "hazlo más oscuro" → "añade un sombrero" → "cambia el fondo".
                    """)
                    autoregressive = gr.Checkbox(
                        label="🔁 Modo Autoregresivo",
                        value=False,
                        info="Cada iteración usa el resultado anterior como nueva entrada"
                    )
                    auto_iterations = gr.Slider(
                        minimum=1, maximum=20, step=1, value=3,
                        label="Iteraciones",
                        info="Número de veces que el prompt se aplica autoregresivamente"
                    )

                status = gr.HTML("<div style='text-align:center; color:#64748b; padding:10px;'>Listo</div>")

            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🖼️ RESULTADO"):
                        output_img = gr.Image(label="Resultado", height=500)
                    with gr.TabItem("🎭 MÁSCARA"):
                        mask_preview = gr.Image(label="Máscara generada", height=400)

                with gr.Row():
                    bt_use_as_input = gr.Button("🔄 USAR COMO ENTRADA", variant="secondary", scale=2)
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA", scale=1)

    return {
        "input_img": input_img,
        "prompt": prompt,
        "gen_btn": gen_btn,
        "output_img": output_img,
        "status": status,
        "mask_preview": mask_preview,
        "engine": engine,
        "denoise": denoise,
        "autoregressive": autoregressive,
        "auto_iterations": auto_iterations,
        "quality_mode": quality_mode,
        "quality_options": quality_options,
        "quality_deposter": quality_deposter,
        "quality_upscale": quality_upscale,
        "quality_color": quality_color,
        "quality_realism": quality_realism,
        "enhance_tier": enhance_tier,
        "quality_preserve_faces": quality_preserve_faces,
        "legacy_api_btn": legacy_api_btn,
        "legacy_options_btn": legacy_options_btn,
        "bt_use_as_input": bt_use_as_input,
        "bt_open_folder": bt_open_folder,
    }
