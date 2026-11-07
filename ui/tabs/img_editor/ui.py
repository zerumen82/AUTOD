import gradio as gr

from roop.img_editor.gen_prompt_config import get_editor_ui_config
from .state import SelectionSession


if hasattr(gr, "ImageEditor"):
    ImageEditorWidget = gr.ImageEditor
else:
    class ImageEditorWidget(gr.Image):
        """Compatibilidad con Gradio 3.x: la máscara manual se degrada a un Image normal si no existe el widget avanzado."""

        def __init__(self, *args, **kwargs):
            kwargs.pop("image_mode", None)
            kwargs.pop("layers", None)
            super().__init__(*args, **kwargs)

    gr.ImageEditor = ImageEditorWidget


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
        font-size: 36px;
        font-weight: 800;
    }
    .btn-transform-main {
        background: linear-gradient(135deg, #06b6d4 0%, #a855f7 100%) !important;
        color: white !important;
        font-weight: 900 !important;
        height: 64px !important;
        border-radius: 14px !important;
        font-size: 20px !important;
        letter-spacing: 1px;
        text-transform: uppercase;
        border: none !important;
    }
    .prompt-box-img {
        background: #0f172a !important;
        border: 2px solid #22d3ee !important;
        border-radius: 12px !important;
        font-size: 23px !important;
        font-weight: 700 !important;
    }
    /* Gradio pinta el texto dentro de un textarea, que no hereda el tamaño
       ni el peso del contenedor: sin esta regla el prompt seguía viéndose
       pequeño y fino. */
    .prompt-box-img textarea, .prompt-box-img input {
        font-size: 23px !important;
        font-weight: 700 !important;
        line-height: 1.45 !important;
    }
"""


def _compat_image(**kwargs):
    fmt = kwargs.pop("format", None)
    component = gr.Image(**kwargs)
    try:
        component.format = fmt
    except Exception:
        pass
    return component


def _compat_image_editor(**kwargs):
    fmt = kwargs.pop("format", None)
    component = ImageEditorWidget(**kwargs)
    try:
        component.format = fmt
    except Exception:
        pass
    return component


def build_img_editor_ui():
    gr.HTML(f"<style>{CSS_STYLE}</style>")

    # Estado de sesión del selector visual: análisis medido + regiones marcadas.
    # Mutable por sesión para que un análisis en curso vea una invalidación.
    selection_state = gr.State(SelectionSession())

    with gr.Column(elem_classes=["img-editor-container"]):
        with gr.Group(elem_classes=["img-editor-header"]):
            gr.Markdown("## ✨ IMAGE EDITOR AI")
            gr.Markdown(
                "_Edita cualquier cosa con lenguaje natural: color, pose, fondo, "
                "añadir/quitar objetos, ropa, calidad… Sin censura._"
            )

        with gr.Row():
            with gr.Column(scale=1):
                input_img = _compat_image(
                    label="Imagen de Entrada", type="pil", format="png", height=480,
                )

                with gr.Accordion("🖌️ Delimitar zona a mano (opcional)", open=False):
                    gr.Markdown(
                        "_Pinta toda la zona que quieres cambiar y un pequeño margen "
                        "alrededor. El prompt indica qué hacer; el trazo solo delimita "
                        "dónde. Si no pintas nada, el prompt solo también funciona igual._"
                    )
                    manual_mask_editor = _compat_image_editor(
                        label="Pinta la zona que quieres editar", type="pil",
                        image_mode="RGBA", format="png", layers=True, height=380,
                    )

                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder=(
                        "Ej: colorea esta foto B/N · hazla de pie · quita la persona del fondo · "
                        "añade un sombrero · vestido rojo · cambia el fondo por una playa · "
                        "más saturado y nítido…"
                    ),
                    lines=4,
                    elem_classes=["prompt-box-img"]
                )

                # Oculto por decisión del usuario (2026-11-03): la zona se delimita
                # pintando. Los componentes siguen construidos para que el contrato
                # de API (edit_manual_region, 15 entradas) no cambie.
                with gr.Accordion("🎯 Seleccionar zonas (opcional)", open=False, visible=False):
                    gr.Markdown(
                        "_Analiza la foto, marca una o varias zonas detectadas y edita solo "
                        "esas. También puedes pulsar una zona de la vista previa. "
                        "Las propuestas pueden ser parciales o incorrectas: comprueba el resaltado, "
                        "que delimita la zona que se editará. La edición genera una sola pasada. "
                        "El exterior conserva sus píxeles y el tamaño de entrada, también con Modo Calidad._"
                    )
                    analyze_btn = gr.Button("🔍 ANALIZAR ZONAS", variant="secondary")
                    overlay_img = _compat_image(
                        label="Zonas detectadas: pulsa para marcar o desmarcar", height=380,
                        type="pil", interactive=False,
                    )
                    region_choices = gr.CheckboxGroup(
                        choices=[], label="Elementos detectados (marca uno o varios)",
                        visible=False,
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
                legacy_select_btn = gr.Button(visible=False)
                legacy_region_btn = gr.Button(visible=False)

                with gr.Accordion("⚙️ Opciones", open=False):
                    engine_choices = [
                        ("♻️ Auto — LongCat; POV medido → Qwen Cámara", "auto"),
                        ("LongCat Turbo (8 GB)", "longcat"),
                        ("🎯 FLUX.1 Kontext Q4_K_M", "kontext"),
                        ("📷 Qwen Cámara (POV/ángulos, ~1152px en 8 GB)", "qwen_camera"),
                        ("Compatibilidad app antigua (Kontext)", "imagine"),
                        ("✨ FLUX.2 Klein 4B (img2img rápido, 4 pasos, entra en 8GB)", "klein"),
                        ("⚠️ FLUX.2 Klein 9B (NO entra en 8GB, usar solo con 12GB+)", "klein_9b"),
                    ]
                    default_engine = get_editor_ui_config().get("default_engine")
                    if default_engine not in {value for _label, value in engine_choices}:
                        raise ValueError(
                            "editor_ui.default_engine debe nombrar un motor del Editor "
                            "en config/generation_prompt.json"
                        )
                    engine = gr.Dropdown(
                        choices=engine_choices,
                        value=default_engine, label="Motor",
                        info=(
                            "Auto usa el motor visual general y las rutas especializadas "
                            "que ya miden la solicitud. Puedes forzar un motor; el resultado "
                            "muestra el motor ejecutado, el alcance y la verificación."
                        ),
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
                        output_img = _compat_image(label="Resultado", height=500, type="pil", format="png")
                    with gr.TabItem("🎭 MÁSCARA"):
                        mask_preview = gr.Image(label="Máscara generada", height=400)

                with gr.Row():
                    bt_use_as_input = gr.Button("🔄 USAR COMO ENTRADA", variant="secondary", scale=2)
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA", scale=1)

    return {
        "input_img": input_img,
        "manual_mask_editor": manual_mask_editor,
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
        "legacy_select_btn": legacy_select_btn,
        "legacy_region_btn": legacy_region_btn,
        "bt_use_as_input": bt_use_as_input,
        "bt_open_folder": bt_open_folder,
        "selection_state": selection_state,
        "analyze_btn": analyze_btn,
        "overlay_img": overlay_img,
        "region_choices": region_choices,
    }
