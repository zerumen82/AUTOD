"""UI de la pestaña «✨ Mejorar fotos» estilo Topaz, compacta y continua.

Layout (docs/PLAN_PESTAÑA_TOPAZ_2026-11-15.md §4.2): vista de trabajo tipo Topaz:
  Foto protagonista (antes/después) a la izquierda, columna continua de
  controles (importar + mejoras) a la derecha, banda de presets/exportación abajo.
Sin cajas por categoría: solo separadores finos. Paleta Topaz plana + azul.

El código se registra UNA sola vez en ui/main.py (sin duplicado de pestaña).
Todas las claves del dict de retorno son el contrato con events.py; no
cambiarlas sin actualizar events.py.
"""

import gradio as gr

from roop.output_paths import get_enhance_output_dir

from .logic import load_enhance_config
from .state import EnhanceSessionState


ENHANCE_CSS = """
.img-enhance-container {
    background: #131313;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid #262626;
}
.img-enhance-header { text-align: center; margin-bottom: 8px; }
.img-enhance-header h2 {
    background: linear-gradient(90deg, #4da3ff 0%, #7dd3fc 60%, #a5f3fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 26px;
    font-weight: 900;
    letter-spacing: -0.3px;
    margin: 0 0 2px 0;
}
.img-enhance-header p { color: #9a9a9a; font-size: 11px; margin: 0; }
.enhance-section {
    color: #e8e8e8 !important;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 1.1px;
    text-transform: uppercase;
    margin: 8px 0 4px 0;
    border-bottom: 1px solid #2a2a2a;
    padding-bottom: 3px;
}
.enhance-section::before {
    content: "";
    display: inline-block;
    width: 8px;
    height: 8px;
    border-radius: 2px;
    background: #0f82ff;
    margin-right: 6px;
    vertical-align: middle;
}
.enhance-hero {
    background: #0d0d0d;
    border: 1px solid #1f4d8f;
    box-shadow: 0 0 22px rgba(15, 130, 255, 0.18);
    border-radius: 10px;
    padding: 10px;
}
.enhance-panel {
    background: #181818;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 10px;
}
.enhance-slider-wrap {
    display: flex;
    align-items: center;
    gap: 6px;
    margin: 0;
}
.enhance-slider-label {
    color: #d9d9d9;
    font-size: 11.5px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 5px;
    border-left: 3px solid currentColor;
    padding-left: 7px;
}
.enhance-slider-value {
    color: #0f82ff;
    font-size: 10px;
    font-weight: 700;
    font-variant-numeric: tabular-nums;
}
/* Densidad: compacta bloques internos de Gradio solo dentro de la pestaña */
.img-enhance-container .info-text { font-size: 10px !important; margin: 0 !important; }
.img-enhance-container .block { padding: 2px 0 !important; }
.img-enhance-container .gap { gap: 4px !important; }
.img-enhance-container .wrap { gap: 4px !important; }
.btn-batch {
    background: linear-gradient(90deg, #0f82ff 0%, #22b8ff 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    height: 42px !important;
    border-radius: 9px !important;
    font-size: 13px !important;
    letter-spacing: 0.6px;
    text-transform: uppercase;
    border: none !important;
    box-shadow: 0 0 16px rgba(15, 130, 255, 0.35);
}
.btn-secondary {
    background: #242424 !important;
    color: #e8e8e8 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
    padding: 2px 8px !important;
    font-size: 10px !important;
}
.btn-apply {
    background: #0f82ff !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 2px 8px !important;
    font-size: 10px !important;
}
.btn-save {
    background: #242424 !important;
    color: #e8e8e8 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
    padding: 2px 8px !important;
    font-size: 10px !important;
}
.enhance-status {
    text-align: center;
    color: #9a9a9a;
    padding: 5px;
    border-radius: 6px;
    background: #161616;
    border: 1px solid #2a2a2a;
    font-size: 10px;
}
.enhance-dataframe { font-size: 10px; }
"""


def _slider(label, emoji, color, minimum, maximum, step, value, info, slider_key):
    """Slider con etiqueta HTML (emoji + texto). Readout nativo de Gradio.

    slider_key solo etiqueta la clase CSS.
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
            gr.Markdown("## ✨ MEJORAR FOTOS")
            gr.Markdown("_Ajusta a la derecha, mira el resultado al instante, exporta el lote. 100 % local._")

        # ---------- Vista de trabajo: foto protagonista + panel de control ----------
        with gr.Row():
            with gr.Column(scale=5, elem_classes=["enhance-hero"]):
                preview_slider = gr.ImageSlider(label="Antes / después — arrastra el comparador", type="pil", height=460)
                with gr.Row():
                    btn_preview = gr.Button("🔍 Vista previa", variant="secondary", scale=1, elem_classes=["btn-secondary"])
                    preview_status = gr.Markdown("_Carga una foto y mueve los sliders_")
                status_msg = gr.HTML("<div class='enhance-status'>Listo para importar y ajustar.</div>")

            with gr.Column(scale=4, elem_classes=["enhance-panel"]):
                gr.Markdown("📥 Importar", elem_classes=["enhance-section"])
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
                    btn_load_sample = gr.Button("📦 Ejemplo", variant="secondary", size="sm", elem_classes=["btn-secondary"])
                queue_table = gr.Dataframe(
                    headers=["Archivo", "Estado", "Tamaño"],
                    interactive=False,
                    label="Cola",
                    elem_classes=["enhance-dataframe"],
                )

                gr.Markdown("🧩 Mejoras", elem_classes=["enhance-section"])
                with gr.Row():
                    denoise_slider = _slider("Denoise", "🧹", "#22d3ee", minimum=0, maximum=100, step=1, value=float(defaults.get("denoise_strength", 65)), info="0 = sin cambio · 100 = agresivo.", slider_key="denoise")
                    sharpen_slider = _slider("Sharpen", "🔎", "#a855f7", minimum=0, maximum=100, step=1, value=float(defaults.get("sharpen_strength", 40)), info="0 = sin cambio · 100 = máximo.", slider_key="sharpen")
                with gr.Row():
                    recover_slider = _slider("Recover Faces", "🧑", "#f472b6", minimum=0, maximum=100, step=1, value=float(defaults.get("recover_faces_strength", 0.8) * 100), info="0 = sin tocar caras · 100 = total.", slider_key="recover")
                    upscale_factor = gr.Dropdown(choices=[1, 2, 4, 6], value=int(defaults.get("upscale_factor", 2)), label="⬆️ Factor upscale", show_label=True)
                with gr.Row():
                    exposure_slider = _slider("Exposición", "💡", "#fbbf24", minimum=0.5, maximum=2.0, step=0.1, value=float(defaults.get("exposure", 1.0)), info="1.0 = sin cambio.", slider_key="exposure")
                    shadow_slider = _slider("Sombras", "🌑", "#94a3b8", minimum=0.5, maximum=2.0, step=0.1, value=float(defaults.get("shadow", 1.0)), info="1.0 = sin cambio.", slider_key="shadow")
                with gr.Row():
                    highlight_slider = _slider("Luces", "✨", "#facc15", minimum=0.0, maximum=2.0, step=0.05, value=float(defaults.get("highlight", 1.0)), info="1.0 = sin cambio.", slider_key="highlight")
                    temperature_slider = _slider("Temperatura", "🌡️", "#38bdf8", minimum=0, maximum=100, step=1, value=float(defaults.get("temperature", 50)), info="0 = neutra · + = cálida.", slider_key="temperature")
                with gr.Row():
                    grain_strength_slider = _slider("Grano (fuerza)", "🎞️", "#818cf8", minimum=0, maximum=100, step=1, value=float(defaults.get("grain_strength", 20)), info="0 = sin grano.", slider_key="grain_strength")
                    grain_size_slider = _slider("Grano (tamaño)", "🎞️", "#818cf8", minimum=1, maximum=20, step=1, value=float(defaults.get("grain_size", 10)), info="1 = fina · 20 = gruesa.", slider_key="grain_size")

        # ---------- Banda de presets y exportación ----------
        gr.Markdown("🎛️ Presets · Exportación", elem_classes=["enhance-section"])
        with gr.Row():
            preset_dropdown = gr.Dropdown(choices=["Retrato", "Paisaje", "Nocturna", "Producto", "Archivo"], value="Retrato", label="Preset de fábrica")
            preset_apply_btn = gr.Button("✅ Aplicar preset", variant="primary", elem_classes=["btn-apply"])
            preset_save_name = gr.Textbox(label="Guardar como…", placeholder="Mi preset")
            preset_save_btn = gr.Button("💾 Guardar", variant="secondary", elem_classes=["btn-save"])
        preset_status = gr.Markdown("")
        with gr.Row():
            auto_mode = gr.Checkbox(label="🤖 Modo automático (medido por foto)", value=False)
            apply_filters_to_filename = gr.Checkbox(label="Filtros en el nombre del archivo", value=bool(config.get("export", {}).get("add_applied_filters_to_filename", True)))
            output_path = gr.Textbox(label="Carpeta de destino", value=get_enhance_output_dir(), interactive=False, scale=2)
            format_dropdown = gr.Dropdown(choices=["png", "jpg", "jpeg"], value=config.get("export", {}).get("default_format", "png"), label="Formato", scale=1)
        result_table = gr.Dataframe(headers=["Entrada", "Salida", "Notas", "OK"], interactive=False, label="Resultado del lote", elem_classes=["enhance-dataframe"])
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
