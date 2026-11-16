"""UI de la pestaña «✨ Mejorar fotos» estilo local profesional.

Diseño reestructurado a 3 tabs por petición de usuario:
- Tab 1: Importar fotos
- Tab 2: Ajustar (TODO integrado: sliders + previsualización al instante, estilo local)
- Tab 3: Settings (configuración de salida, presets, modo automático)

SIN duplicado de pestaña: el código se registra UNA sola vez en ui/main.py.
"""

import gradio as gr

from roop.output_paths import get_enhance_output_dir

from .logic import load_enhance_config
from .state import EnhanceSessionState


ENHANCE_CSS = """
.img-enhance-container {
    background: #0b1021;
    padding: 28px;
    border-radius: 24px;
    border: 1px solid #1f2a44;
    box-shadow: 0 14px 40px rgba(0,0,0,0.6);
}
.img-enhance-header { text-align: center; margin-bottom: 26px; }
.img-enhance-header h2 {
    background: linear-gradient(90deg, #22d3ee 0%, #a855f7 55%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 36px;
    font-weight: 900;
    letter-spacing: -0.5px;
    margin: 0 0 8px 0;
}
.img-enhance-header p {
    color: #94a3b8;
    font-size: 15px;
    margin: 0;
}
.enhance-card {
    background: #111827;
    border: 1px solid #1f2a44;
    border-radius: 16px;
    padding: 12px 14px;
    margin-bottom: 10px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.35);
}
.enhance-card h3 {
    color: #f8fafc;
    font-size: 15px;
    font-weight: 800;
    margin: 0 0 4px 0;
    display: flex;
    align-items: center;
    gap: 6px;
}
.enhance-card p.sub {
    color: #94a3b8;
    font-size: 11px;
    margin: 0 0 8px 0;
}
.enhance-slider-wrap {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 6px;
}
.enhance-slider-label {
    min-width: 130px;
    color: #e2e8f0;
    font-size: 12px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 4px;
}
.enhance-slider-value {
    min-width: 32px;
    text-align: right;
    color: #22d3ee;
    font-variant-numeric: tabular-nums;
    font-size: 11px;
    font-weight: 700;
}
.btn-batch {
    background: linear-gradient(135deg, #22d3ee 0%, #a855f7 100%) !important;
    color: #000 !important;
    font-weight: 900 !important;
    height: 44px !important;
    border-radius: 10px !important;
    font-size: 14px !important;
    letter-spacing: 0.3px;
    text-transform: uppercase;
    border: none !important;
}
.btn-secondary {
    background: linear-gradient(135deg, #334155 0%, #475569 100%) !important;
    color: #f8fafc !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: 1px solid #475569 !important;
    padding: 2px 8px !important;
    font-size: 11px !important;
}
.btn-apply {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: white !important;
    font-weight: 800 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 2px 8px !important;
    font-size: 11px !important;
}
.btn-save {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 2px 8px !important;
    font-size: 11px !important;
}
.enhance-tabs .tab-nav button {
    font-weight: 700 !important;
    color: #94a3b8 !important;
    padding: 8px 12px !important;
}
.enhance-tabs .tab-nav button.selected {
    color: #22d3ee !important;
    background: #0b1225 !important;
    border-radius: 6px !important;
}
.enhance-preview {
    background: #1a1d29;
    border: 1px solid #1f2a44;
    border-radius: 10px;
    padding: 10px;
    margin-top: 8px;
}
.enhance-status {
    text-align: center;
    color: #94a3b8;
    padding: 8px;
    border-radius: 6px;
    background: #0f172a;
    border: 1px solid #1f2a44;
    font-size: 11px;
    margin-top: 6px;
}
.enhance-dataframe {
    font-size: 11px;
}
"""
def _slider(label, emoji, color, minimum, maximum, step, value, info, slider_key):
    """Slider con etiqueta HTML coloreada (emoji + texto) del diseño Enhance.

    Devuelve el gr.Slider; el readout del valor es el nativo de Gradio
    (sin valores falsos). slider_key solo etiqueta la clase CSS.
    """
    gr.HTML(
        f'<div class="enhance-slider-wrap">'
        f'<div class="enhance-slider-label" style="color:{color}">{emoji} {label}</div>'
        f"</div>"
    )
    return gr.Slider(
        minimum=minimum,
        maximum=maximum,
        step=step,
        value=value,
        info=info,
        show_label=False,
        elem_classes=[f"enhance-slider-{slider_key}"],
    )


def build_enhance_ui():
    gr.HTML(f"<style>{ENHANCE_CSS}</style>")

    session_state = gr.State(EnhanceSessionState())

    config = load_enhance_config()
    defaults = config.get("default_slider_values", {})

    with gr.Column(elem_classes=["img-enhance-container"]):
        # ---------- Cabecera ----------
        with gr.Group(elem_classes=["img-enhance-header"]):
            gr.Markdown("## \u2728 MEJORAR FOTOS")
            gr.Markdown(
                "_Estilo local: importa fotos, ajusta cada mejora por separado, \n                ve el resultado al instante y exporta el lote con IA 100 % en tu equipo._"
            )

        # ---------- 1️⃣ Importar ----------
        with gr.Tab("1\ufffd\ufffd\u20e3 Importar"):
            with gr.Row():
                input_files = gr.Files(
                    label="\U0001F4C1 Seleccionar fotos (varias)",
                    file_types=["image"],
                )
                input_folder = gr.File(
                    label="\U0001F4C2 O carpeta entera",
                    file_count="directory",
                    file_types=["image"],
                )
            with gr.Row():
                btn_load_sample = gr.Button("📦 Cargar ejemplo", variant="secondary", size="sm")
            queue_table = gr.Dataframe(
                headers=["Archivo", "Estado", "Tama\u00f1o"],
                interactive=False,
                label="Cola de im\u00e1genes a procesar",
                elem_classes=["enhance-dataframe"],
            )
# ---------- 2️⃣ Ajustar (TODO integrado, estilo local) ----------
        with gr.Tab("2️⃣ Ajustar"):
            gr.Markdown("Mueve los sliders y la previsualización se actualiza al instante. Todas las mejoras son locales y 100 % en tu equipo.")

            with gr.Column(scale=1):
                with gr.Group(elem_classes=["enhance-card"]):
                    gr.Markdown("### 🧹 Denoise y Upscale", elem_classes=["enhance-card-h"])
                    gr.Markdown("Elimina ruido y amplía resolución.", elem_classes=["enhance-card-p"])
                    denoise_slider = _slider("Denoise (fuerza)", "🧹", "#22d3ee", minimum=0, maximum=100, step=1, value=float(defaults.get("denoise_strength", 65)), info="0 = sin cambio · 100 = reducción agresiva.", slider_key="denoise")
                    upscale_factor = gr.Dropdown(choices=[1, 2, 4, 6], value=int(defaults.get("upscale_factor", 2)), label="⬆️ Factor", show_label=True)

                with gr.Group(elem_classes=["enhance-card"]):
                    gr.Markdown("### 🔎 Nitidez y Caras", elem_classes=["enhance-card-h"])
                    gr.Markdown("Recupera detalle y mejora rostros.", elem_classes=["enhance-card-p"])
                    sharpen_slider = _slider("Sharpen (nitidez)", "🔎", "#a855f7", minimum=0, maximum=100, step=1, value=float(defaults.get("sharpen_strength", 40)), info="0 = sin cambio · 100 = máximo realce.", slider_key="sharpen")
                    recover_slider = _slider("Recover Faces (GFPGAN)", "🧑", "#f472b6", minimum=0, maximum=100, step=1, value=float(defaults.get("recover_faces_strength", 0.8) * 100), info="0 = sin tocar caras · 100 = restauración total.", slider_key="recover")

                with gr.Group(elem_classes=["enhance-card"]):
                    gr.Markdown("### 💡 Luz y Color", elem_classes=["enhance-card-h"])
                    gr.Markdown("Exposición, sombras, luces y temperatura.", elem_classes=["enhance-card-p"])
                    exposure_slider = _slider("Exposición", "💡", "#fbbf24", minimum=0.5, maximum=2.0, step=0.1, value=float(defaults.get("exposure", 1.0)), info="1.0 = sin cambio · mayor = más claro.", slider_key="exposure")
                    shadow_slider = _slider("Sombras", "🌑", "#94a3b8", minimum=0.5, maximum=2.0, step=0.1, value=float(defaults.get("shadow", 1.0)), info="1.0 = sin cambio · sube para recuperar detalle en zonas oscuras.", slider_key="shadow")
                    highlight_slider = _slider("Luces", "✨", "#fbbf24", minimum=0.0, maximum=2.0, step=0.05, value=float(defaults.get("highlight", 1.0)), info="1.0 = sin cambio · sube para intensificar las luces.", slider_key="highlight")
                    temperature_slider = _slider("Temperatura", "🌡️", "#38bdf8", minimum=0, maximum=100, step=1, value=float(defaults.get("temperature", 50)), info="0 = neutra · + = más cálida.", slider_key="temperature")
# Grano y panel derecho
                with gr.Group(elem_classes=["enhance-card"]):
                    gr.Markdown("### 🎞️ Grano Fotográfico", elem_classes=["enhance-card-h"])
                    gr.Markdown("Textura de película analógica o cinematográfica.", elem_classes=["enhance-card-p"])
                    grain_strength_slider = _slider("Grano (fuerza)", "🎞️", "#818cf8", minimum=0, maximum=100, step=1, value=float(defaults.get("grain_strength", 20)), info="0 = sin grano · 100 = grano intenso.", slider_key="grain_strength")
                    grain_size_slider = _slider("Grano (tamaño)", "🎞️", "#818cf8", minimum=1, maximum=20, step=1, value=float(defaults.get("grain_size", 10)), info="1 = fina · 20 = grueso.", slider_key="grain_size")

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### 👁️ Previsualización en tiempo real", elem_classes=["enhance-preview"])
                preview_slider = gr.ImageSlider(label="Arrastra: izquierda = original · derecha = mejorada", type="pil", height=360)
                preview_status = gr.Markdown("_Carga una foto y mueve los sliders_")

            with gr.Row():
                status_msg = gr.HTML("<div class='enhance-status'>Listo para importar y ajustar.</div>")
                btn_preview = gr.Button("🔍 Generar vista previa", variant="secondary", scale=1)

        with gr.Tab("3️⃣ Settings"):
            gr.Markdown("### Configuración de salida y presets")
            with gr.Row():
                output_path = gr.Textbox(label="Carpeta de destino", value=get_enhance_output_dir(), interactive=False)
                format_dropdown = gr.Dropdown(choices=["png", "jpg", "jpeg"], value=config.get("export", {}).get("default_format", "png"), label="Formato de salida")
            with gr.Row():
                auto_mode = gr.Checkbox(label="🤖 Modo automático (mide cada foto y propone la pila)", value=False)
                apply_filters_to_filename = gr.Checkbox(label="Añadir resumen de filtros al nombre del archivo", value=bool(config.get("export", {}).get("add_applied_filters_to_filename", True)))
            with gr.Row():
                preset_dropdown = gr.Dropdown(choices=["Retrato", "Paisaje", "Nocturna", "Producto", "Archivo"], value="Retrato", label="Preset de fábrica")
                preset_apply_btn = gr.Button("✅ Aplicar preset", variant="primary", elem_classes=["btn-apply"])
                preset_save_name = gr.Textbox(label="Guardar como…", placeholder="Mi preset")
                preset_save_btn = gr.Button("💾 Guardar", variant="secondary", elem_classes=["btn-save"])
            preset_status = gr.Markdown("")

            gr.Markdown("---")
            gr.Markdown("### Resultado del lote")
            result_table = gr.Dataframe(headers=["Entrada", "Salida", "Notas", "OK"], interactive=False, label="Resultados", elem_classes=["enhance-dataframe"])

            with gr.Row():
                batch_btn = gr.Button("🚀 PROCESAR LOTE", variant="primary", elem_classes=["btn-batch"], scale=3)
                cancel_btn = gr.Button("⏹ Cancelar", variant="stop", scale=1)

    return {
        "session_state": session_state,
        "input_files": input_files,
        "input_folder": input_folder,
        "queue_table": queue_table,
        "preview_slider": preview_slider,
        "btn_preview": btn_preview,
        "preview_status": preview_status,
        "denoise_slider": denoise_slider,
        "sharpen_slider": sharpen_slider,
        "recover_slider": recover_slider,
        "upscale_factor": upscale_factor,
        "exposure_slider": exposure_slider,
        "shadow_slider": shadow_slider,
        "highlight_slider": highlight_slider,
        "temperature_slider": temperature_slider,
        "grain_strength_slider": grain_strength_slider,
        "grain_size_slider": grain_size_slider,
        "preset_dropdown": preset_dropdown,
        "preset_apply_btn": preset_apply_btn,
        "preset_save_name": preset_save_name,
        "preset_save_btn": preset_save_btn,
        "preset_status": preset_status,
        "auto_mode": auto_mode,
        "output_path": output_path,
        "format_dropdown": format_dropdown,
        "apply_filters_to_filename": apply_filters_to_filename,
        "batch_btn": batch_btn,
        "cancel_btn": cancel_btn,
        "status_msg": status_msg,
        "result_table": result_table,
    }