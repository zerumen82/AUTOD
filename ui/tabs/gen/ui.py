import gradio as gr
from roop.img_editor.flux_gen_comfy_client import (
    get_default_generation_engine,
    get_installed_generation_engines,
)
from roop.img_editor.gen_prompt_modifiers import get_dropdown_choices, get_compatible_dropdown_choices, preview_modifiers


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

        modifier_preview = gr.HTML("")

        with gr.Row():
            orientation = gr.Radio(
                choices=["Horizontal (Paisaje)", "Vertical (Retrato)"],
                value="Horizontal (Paisaje)",
                label="Forma",
            )
            _engines = get_installed_generation_engines()
            _default = get_default_generation_engine()
            engine_model = gr.Dropdown(
                choices=_engines or [("FLUX Abliterated (ultra realista)", "flux_dev_abliterated")],
                value=_default,
                label="Modelo",
            )

        with gr.Row():
            enhance_quality = gr.Checkbox(
                label="Mejorar nitidez (ESRGAN, más lento)",
                value=False,
            )
            use_rewriter = gr.Checkbox(
                label="Rewriter LLM (mejor prompt, +5-15s)",
                value=False,
            )

        width = gr.Slider(minimum=512, maximum=1536, step=64, value=1152, visible=False)
        height = gr.Slider(minimum=512, maximum=1536, step=64, value=768, visible=False)

        with gr.Row():
            gen_btn = gr.Button("GENERAR", variant="primary", size="lg", scale=3)
            btn_cancel = gr.Button("⏹ CANCELAR", variant="stop", interactive=False, scale=1)

        status_html = gr.HTML("<div style='text-align:center;color:#64748b;padding:10px;'>Listo</div>")
        output_img = gr.Image(label="Resultado", height=600)

        bt_open_folder = gr.Button("📂 Abrir carpeta de salidas", size="sm")

    return {
        "prompt": prompt,
        "image_style": image_style,
        "lighting": lighting,
        "shot_framing": shot_framing,
        "skin_finish": skin_finish,
        "orientation": orientation,
        "engine_model": engine_model,
        "enhance_quality": enhance_quality,
        "use_rewriter": use_rewriter,
        "modifier_preview": modifier_preview,
        "width": width,
        "height": height,
        "gen_btn": gen_btn,
        "btn_cancel": btn_cancel,
        "status_html": status_html,
        "output_img": output_img,
        "bt_open_folder": bt_open_folder,
    }
