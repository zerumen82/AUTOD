"""UI de la pestaña «Mejorar fotos» estilo Topaz, compacta y continua.

Vista de trabajo: comparador ANTES | DESPUÉS arriba, importación y presets
debajo, ajustes manuales plegados y exportación al final.

El código se registra UNA sola vez en ui/main.py (sin duplicado de pestaña).
Todas las claves del dict de retorno son el contrato con events.py; no
cambiarlas sin actualizar events.py.
"""

import gradio as gr

from roop.output_paths import get_enhance_output_dir

from .logic import load_enhance_config
from .presets import icon_for_preset, load_presets
from .state import EnhanceSessionState


ENHANCE_CSS = """
.img-enhance-container {
    background: #131313;
    padding: 8px;
    border-radius: 12px;
    border: 1px solid #262626;
    --color-accent: #0f82ff;
    --color-accent-soft: rgba(15, 130, 255, 0.25);
    --slider-color: #0f82ff;
}
.img-enhance-header {
    text-align: center;
    margin-bottom: 10px;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}
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
    margin: 10px 0 4px 0;
    border-bottom: 1px solid #2a2a2a;
    border-left: 3px solid #0f82ff;
    padding: 0 0 3px 8px;
}
.img-enhance-container .enhance-section p { margin: 0 !important; }
.enhance-hero {
    background: #0d0d0d;
    border: 1px solid #1f4d8f;
    box-shadow: 0 0 18px rgba(15, 130, 255, 0.14);
    border-radius: 10px;
    padding: 10px;
    position: sticky;
    top: 0;
    align-self: flex-start;
}
.enhance-panel {
    background: #181818;
    border: 1px solid #262626;
    border-radius: 10px;
    padding: 8px;
}
/* ANTES | DESPUÉS lado a lado, misma altura y sin deformación.
   `flex-wrap: nowrap` + `min-width: 0` evita que la imagen fuerce el ancho
   mínimo de la columna y la tire debajo (causa raíz del apilado). */
.enhance-compare-row { gap: 10px !important; align-items: stretch !important; flex-wrap: nowrap !important; }
.enhance-compare-row > .enhance-view { min-width: 0 !important; width: 0 !important; flex: 1 1 0 !important; }
.enhance-view {
    background: #0a0a0a;
    border: 1px solid #1f1f1f;
    border-radius: 10px;
    padding: 2px;
    height: 100%;
    min-width: 0;
}
.enhance-view-title {
    color: #9a9a9a;
    font-size: 10.5px;
    font-weight: 800;
    letter-spacing: 1.2px;
    text-align: center;
    padding: 0 0 4px 0;
}
.enhance-view-title-after { color: #4da3ff; }
.enhance-img { background: #0a0a0a; border-radius: 8px; overflow: hidden; min-width: 0; max-width: 100%; height: auto !important; }
.enhance-img > div { height: auto !important; }
.enhance-img img { max-width: 100% !important; width: 100% !important; height: auto !important; object-fit: contain !important; }
/* Cada slider vive en su propia columna: etiqueta arriba, control abajo */
.enhance-grid { gap: 8px !important; align-items: stretch !important; }
.enhance-slider-col {
    background: #141414;
    border: 1px solid #232323;
    border-radius: 8px;
    padding: 6px 9px 2px 9px;
}
.enhance-slider-label {
    color: #d9d9d9;
    font-size: 11px;
    font-weight: 700;
    display: flex;
    align-items: center;
    gap: 5px;
    border-left: 3px solid #444;
    padding-left: 7px;
    min-height: 17px;
}
.enhance-summary {
    color: #cfcfcf;
    font-size: 11.5px;
    line-height: 1.4;
    padding: 2px 0 0 0;
    min-height: 18px;
}
/* Densidad: compacta bloques internos de Gradio solo dentro de la pestaña */
.img-enhance-container .info-text { font-size: 10px !important; margin: 0 !important; }
.img-enhance-container .block { padding: 2px 0 !important; }
.img-enhance-container .gap { gap: 4px !important; }
.img-enhance-container .wrap { gap: 4px !important; }
/* Los paneles conservan su padding aunque Gradio los marque como .block */
.img-enhance-container .enhance-hero.block { padding: 10px !important; }
.img-enhance-container .enhance-panel.block { padding: 10px !important; }
.img-enhance-container .enhance-view.block { padding: 8px !important; }
.img-enhance-container .enhance-slider-col.block { padding: 6px 9px 2px 9px !important; }
.img-enhance-container .enhance-img { width: 100% !important; }
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
    box-shadow: 0 0 12px rgba(15, 130, 255, 0.22);
}
.btn-secondary {
    background: #242424 !important;
    color: #e8e8e8 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
    padding: 5px 12px !important;
    font-size: 12px !important;
    min-height: 32px !important;
}
.img-enhance-container .btn-apply {
    background: #0f82ff !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: none !important;
    padding: 6px 14px !important;
    font-size: 12px !important;
    min-height: 32px !important;
}
.img-enhance-container .btn-apply button,
.img-enhance-container .btn-apply > button {
    background: #0f82ff !important;
    color: #ffffff !important;
    border: none !important;
}
.img-enhance-container .btn-apply:hover button {
    background: #2b93ff !important;
}
.btn-save {
    background: #242424 !important;
    color: #e8e8e8 !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
    border: 1px solid #3a3a3a !important;
    padding: 5px 12px !important;
    font-size: 12px !important;
    min-height: 32px !important;
}
.btn-open {
    background: #1d3a5c !important;
    color: #dbeafe !important;
    font-weight: 700 !important;
    border-radius: 8px !important;
    border: 1px solid #2c5a8f !important;
    padding: 8px 16px !important;
    font-size: 12px !important;
    min-height: 36px !important;
}
.enhance-status {
    text-align: center;
    color: #b9b9b9;
    padding: 7px 8px;
    border-radius: 6px;
    background: #161616;
    border: 1px solid #2a2a2a;
    font-size: 11px;
}
.enhance-dataframe { font-size: 10px; }
.enhance-preset-grid { gap: 6px !important; }
.img-enhance-container .enhance-preset-button {
    background: #192232 !important;
    color: #e9f3ff !important;
    border: 1px solid #31445e !important;
    border-radius: 9px !important;
    min-height: 38px !important;
    font-size: 11px !important;
    font-weight: 650 !important;
}
.img-enhance-container .enhance-preset-button:hover {
    background: #243a59 !important;
    border-color: #5da8f4 !important;
}
.img-enhance-container .enhance-preset-button img { width: 19px !important; height: 19px !important; }
.img-enhance-container .enhance-settings-accordion {
    margin-top: 8px;
    border-color: #31445e !important;
    min-width: 0 !important;
    overflow-x: hidden !important;
}
.img-enhance-container .enhance-settings-accordion > * { min-width: 0 !important; }
/* Lotes grandes: la lista sigue visible pero el navegador no maqueta las
   filas fuera del area visible (content-visibility, soportado por WebView2). */
.img-enhance-container .enhance-import-input .file-preview tr {
    content-visibility: auto;
    contain-intrinsic-size: 28px;
}
"""


def _slider(label, color, minimum, maximum, step, value, info, slider_key):
    """Slider con etiqueta encima, dentro de su propia columna.

    La etiqueta y el control son hermanos de la MISMA columna (no de la fila):
    así cada fila de dos mejoras queda en dos columnas alineadas.
    """
    with gr.Column(scale=1, min_width=170, elem_classes=["enhance-slider-col"]):
        gr.HTML(
            f'<div class="enhance-slider-label" style="border-left-color:{color}">'
            f"{label}</div>"
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


def _dropdown(label, color, choices, value, key):
    """Dropdown con la misma estructura de etiqueta que un slider."""
    with gr.Column(scale=1, min_width=170, elem_classes=["enhance-slider-col"]):
        gr.HTML(
            f'<div class="enhance-slider-label" style="border-left-color:{color}">'
            f"{label}</div>"
        )
        return gr.Dropdown(
            choices=choices,
            value=value,
            show_label=False,
            elem_classes=[f"enhance-dropdown-{key}"],
        )


def build_enhance_ui():
    gr.HTML(f"<style>{ENHANCE_CSS}</style>")

    session_state = gr.State(EnhanceSessionState())

    config = load_enhance_config()
    defaults = config.get("default_slider_values", {})
    limits = config["control_limits"]
    presets = load_presets()

    with gr.Column(elem_classes=["img-enhance-container"]):
        # ---------- Cabecera ----------
        with gr.Column(elem_classes=["img-enhance-header"]):
            gr.Markdown("## MEJORAR FOTOS")
            gr.Markdown("_Importa, aplica un preset y compara el resultado. Ajusta y exporta el lote en local._")

        # ---------- Vista de trabajo: ANTES | DESPUÉS (ancho completo) ----------
        with gr.Row(elem_classes=["enhance-compare-row"], equal_height=True):
            with gr.Column(scale=1, elem_classes=["enhance-view"]):
                gr.HTML('<div class="enhance-view-title">ANTES</div>')
                preview_before = gr.Image(
                    label="Antes",
                    type="pil",
                    interactive=False,
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["enhance-img"],
                    height=420,
                )
            with gr.Column(scale=1, elem_classes=["enhance-view"]):
                gr.HTML('<div class="enhance-view-title enhance-view-title-after">DESPUÉS</div>')
                preview_after = gr.Image(
                    label="Después",
                    type="pil",
                    interactive=False,
                    show_label=False,
                    buttons=["fullscreen"],
                    elem_classes=["enhance-img"],
                    height=420,
                )
        with gr.Row():
            btn_prev_photo = gr.Button("Anterior", variant="secondary", scale=1, min_width=90, interactive=False, elem_classes=["btn-secondary"])
            preview_photo = gr.Dropdown(
                choices=[],
                value=None,
                label="Foto en vista previa",
                interactive=True,
                scale=4,
                # Con lotes grandes el desplegable se limita; la navegación
                # Anterior/Siguiente debe poder seleccionar rutas fuera de él.
                allow_custom_value=True,
            )
            btn_next_photo = gr.Button("Siguiente", variant="secondary", scale=1, min_width=90, interactive=False, elem_classes=["btn-secondary"])
            btn_preview = gr.Button("Actualizar vista", variant="secondary", scale=1, elem_classes=["btn-secondary"])
        preview_counter = gr.Markdown("_0 / 0 fotos_", elem_classes=["enhance-summary"])
        preview_status = gr.Markdown("_Importa fotos para ver el antes/después._")

        # ---------- Panel de control (ancho completo, sliders en horizontal) ----------
        with gr.Column(elem_classes=["enhance-panel"]):
            gr.Markdown("Importar", elem_classes=["enhance-section"])
            with gr.Tabs():
                with gr.Tab("Archivos"):
                    input_files = gr.Files(
                        label="Seleccionar fotos (varias)",
                        file_count="multiple",
                        height=200,
                        elem_classes=["enhance-import-input"],
                    )
                with gr.Tab("Carpeta"):
                    input_folder = gr.File(
                        label="Carpeta entera",
                        file_count="directory",
                        file_types=["image"],
                        height=200,
                        elem_classes=["enhance-import-input"],
                    )
            select_all = gr.Checkbox(
                label="Seleccionar todas (mismo preset/ajustes para todas las fotos)",
                value=False,
                info="Si está activo, el preset o ajuste actual se aplica igual a todas las fotos del lote.",
            )
            with gr.Row():
                with gr.Column(scale=4, min_width=200):
                    queue_summary = gr.Markdown("_Sin fotos en cola._", elem_classes=["enhance-summary"])
                btn_load_sample = gr.Button("Cargar ejemplo", variant="secondary", size="sm", scale=1, min_width=120, elem_classes=["btn-secondary"])

            gr.Markdown("Presets", elem_classes=["enhance-section"])
            preset_buttons = {}
            factory_presets = [(name, data) for name, data in presets.items() if data["_factory"]]
            for offset in range(0, len(factory_presets), 5):
                with gr.Row(elem_classes=["enhance-preset-grid"]):
                    for name, data in factory_presets[offset:offset + 5]:
                        preset_buttons[name] = gr.Button(
                            name, icon=str(icon_for_preset(data)), size="sm",
                            scale=1, min_width=125,
                            elem_classes=["enhance-preset-button"],
                        )
            with gr.Row():
                custom_presets = [name for name, data in presets.items() if not data["_factory"]]
                preset_dropdown = gr.Dropdown(choices=custom_presets, value=None, allow_custom_value=True, label="Mis presets", scale=2)
                preset_apply_btn = gr.Button("Aplicar mi preset", variant="primary", elem_classes=["btn-apply"], scale=1)
                preset_save_name = gr.Textbox(label="Guardar como…", placeholder="Mi preset", scale=2)
                preset_save_btn = gr.Button("Guardar", variant="secondary", elem_classes=["btn-save"], scale=1)
            preset_status = gr.Markdown("")
            preset_stack = gr.Markdown("_Pila activa: (ninguna)_", elem_classes=["enhance-summary"])
            with gr.Row():
                btn_clear_stack = gr.Button("Limpiar pila de presets", variant="secondary", size="sm", elem_classes=["btn-secondary"], scale=0)

            with gr.Accordion("Ajustes manuales", open=False, elem_classes=["enhance-settings-accordion"]):
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    denoise_slider = _slider("Denoise", "#22d3ee", value=float(defaults.get("denoise_strength", 65)), info="0 = sin cambio · 100 = agresivo.", slider_key="denoise", **limits["denoise_strength"])
                    sharpen_slider = _slider("Sharpen", "#a855f7", value=float(defaults.get("sharpen_strength", 40)), info="0 = sin cambio · 100 = máximo.", slider_key="sharpen", **limits["sharpen_strength"])
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    recover_slider = _slider("Recover Faces", "#f472b6", value=float(defaults.get("recover_faces_strength", 0.8) * 100), info="0 = sin tocar caras · 100 = total.", slider_key="recover", **limits["recover_faces_strength"])
                    upscale_factor = _dropdown("Factor upscale", "#38bdf8", choices=limits["upscale_factor"]["choices"], value=int(defaults.get("upscale_factor", 2)), key="upscale")
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    exposure_slider = _slider("Exposición", "#fbbf24", value=float(defaults.get("exposure", 1.0)), info="1.0 = sin cambio.", slider_key="exposure", **limits["exposure"])
                    shadow_slider = _slider("Sombras", "#94a3b8", value=float(defaults.get("shadow", 1.0)), info="1.0 = sin cambio.", slider_key="shadow", **limits["shadow"])
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    highlight_slider = _slider("Luces", "#facc15", value=float(defaults.get("highlight", 1.0)), info="1.0 = sin cambio.", slider_key="highlight", **limits["highlight"])
                    temperature_slider = _slider("Temperatura", "#38bdf8", value=float(defaults.get("temperature", 50)), info="50 = neutra · 0 = fría · 100 = cálida.", slider_key="temperature", **limits["temperature"])
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    grain_strength_slider = _slider("Grano (fuerza)", "#818cf8", value=float(defaults.get("grain_strength", 20)), info="0 = sin grano.", slider_key="grain_strength", **limits["grain_strength"])
                    grain_size_slider = _slider("Grano (tamaño)", "#818cf8", value=float(defaults.get("grain_size", 10)), info="1 = fina · 20 = gruesa.", slider_key="grain_size", **limits["grain_size"])
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    deband_slider = _slider("Deband", "#34d399", value=float(defaults.get("deband_strength", 0)), info="0 = sin cambio · 100 = máximo.", slider_key="deband", **limits["deband_strength"])
                    vibrance_slider = _slider("Color (vibrance)", "#fb7185", value=float(defaults.get("vibrance", 100)), info="100 = sin cambio · + = más saturación.", slider_key="vibrance", **limits["vibrance"])
                with gr.Row(elem_classes=["enhance-grid"], equal_height=True):
                    face_model = _dropdown("Motor de caras", "#f472b6", choices=limits["face_model"]["choices"], value=str(defaults.get("face_model", "GFPGAN")), key="face_model")
                    gr.Markdown("GFPGAN = rápido · CodeFormer/RestoreFormer++ = más detalle · GPEN = suaviza.", elem_classes=["enhance-summary"])
                with gr.Row():
                    btn_reset_sliders = gr.Button("↺ Reset sliders", variant="secondary", size="sm", elem_classes=["btn-secondary"])

            gr.Markdown("Destino", elem_classes=["enhance-section"])
            with gr.Row():
                output_path = gr.Textbox(label="Carpeta de destino", value=get_enhance_output_dir(), interactive=True, scale=5)
                btn_browse_folder = gr.Button("Examinar…", variant="secondary", elem_classes=["btn-secondary"], scale=1)
                btn_open_folder = gr.Button("Abrir carpeta", variant="secondary", elem_classes=["btn-open"], scale=1)
                format_dropdown = gr.Dropdown(choices=["png", "jpg", "jpeg"], value=config.get("export", {}).get("default_format", "png"), label="Formato", scale=1)
            folder_status = gr.Markdown("", elem_classes=["enhance-summary"])

        # ---------- Exportación ----------
        gr.Markdown("Exportación", elem_classes=["enhance-section"])
        with gr.Row():
            auto_mode = gr.Checkbox(label="Modo automático (medido por foto)", value=False)
            apply_filters_to_filename = gr.Checkbox(label="Filtros en el nombre del archivo", value=bool(config.get("export", {}).get("add_applied_filters_to_filename", True)))
        result_summary = gr.Markdown("", elem_classes=["enhance-summary"])
        status_msg = gr.HTML("<div class='enhance-status'>Listo para importar y ajustar.</div>")
        with gr.Row():
            batch_btn = gr.Button("PROCESAR LOTE", variant="primary", elem_classes=["btn-batch"], scale=3)
            cancel_btn = gr.Button("Cancelar", variant="stop", scale=1)

    return {
        "session_state": session_state,
        "input_files": input_files,
        "input_folder": input_folder,
        "select_all": select_all,
        "btn_load_sample": btn_load_sample,
        "queue_summary": queue_summary,
        "preview_before": preview_before,
        "preview_after": preview_after,
        "preview_photo": preview_photo,
        "btn_prev_photo": btn_prev_photo,
        "btn_next_photo": btn_next_photo,
        "preview_counter": preview_counter,
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
        "deband_slider": deband_slider,
        "vibrance_slider": vibrance_slider,
        "face_model": face_model,
        "btn_reset_sliders": btn_reset_sliders,
        "preset_dropdown": preset_dropdown,
        "preset_apply_btn": preset_apply_btn,
        "preset_save_name": preset_save_name,
        "preset_save_btn": preset_save_btn,
        "preset_status": preset_status,
        "preset_stack": preset_stack,
        "btn_clear_stack": btn_clear_stack,
        "preset_buttons": preset_buttons,
        "auto_mode": auto_mode,
        "output_path": output_path,
        "btn_browse_folder": btn_browse_folder,
        "btn_open_folder": btn_open_folder,
        "folder_status": folder_status,
        "format_dropdown": format_dropdown,
        "apply_filters_to_filename": apply_filters_to_filename,
        "batch_btn": batch_btn,
        "cancel_btn": cancel_btn,
        "status_msg": status_msg,
        "result_summary": result_summary,
    }
