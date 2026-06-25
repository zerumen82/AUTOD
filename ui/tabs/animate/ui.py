import gradio as gr
from roop.output_paths import get_animate_output_dir


def open_animations_folder():
    path = get_animate_output_dir()
    try:
        import os
        os.startfile(path)
    except Exception as e:
        print(f"[UI] No se pudo abrir la carpeta: {e}")


def build_animate_ui():
    gr.HTML("""
        <style>
            .animate-container { background: #020617; padding: 30px; border-radius: 20px; border: 1px solid #1e293b; box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
            .animate-header { text-align: center; margin-bottom: 25px; }
            .animate-header h2 { background: linear-gradient(90deg, #a78bfa, #f472b6); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 32px; font-weight: 800; }
            .btn-animate-main { background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important; color: white !important; font-weight: 900 !important; height: 64px !important; border-radius: 14px !important; font-size: 18px !important; letter-spacing: 1px; text-transform: uppercase; border: none !important; }
            .prompt-box-anim { background: #0f172a !important; border: 2px solid #a78bfa !important; border-radius: 12px !important; }
        </style>
    """)

    with gr.Column(elem_classes=["animate-container"]):
        with gr.Group(elem_classes=["animate-header"]):
            gr.Markdown("## 🎬 ANIMATE IMAGE")
            gr.Markdown("_Sube foto + escribe qué debe pasar (~6s). Audio ambiente y voz en español automáticos._")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)

                prompt = gr.Textbox(
                    label="¿Qué debe pasar?",
                    placeholder="Ej: deben ponerse a bailar, viento en el pelo, que diga hola a cámara...",
                    lines=3,
                    elem_classes=["prompt-box-anim"]
                )

                with gr.Row():
                    btn_animate = gr.Button(
                        "🎬 ANIMAR",
                        variant="primary",
                        elem_classes=["btn-animate-main"],
                        scale=3,
                    )
                    btn_cancel = gr.Button("⏹ CANCELAR", variant="stop", interactive=False, scale=1)

                with gr.Accordion("⚙️ Avanzado (opcional)", open=False):
                    gr.Markdown("*Defaults automáticos — no hace falta abrir esto.*")
                    from animate_photo import AnimatePhoto
                    animator = AnimatePhoto()
                    lora_list = ["None"] + animator.get_available_loras()
                    lora_name = gr.Dropdown(
                        label="LoRA WanVideo",
                        choices=lora_list,
                        value="None",
                    )
                    lora_strength = gr.Slider(
                        label="Intensidad LoRA",
                        minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                    )

                progress_html = gr.HTML(
                    "<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>Listo</div>"
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="Resultado (~6 segundos)", height=520)
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_animations_folder)

    return {
        "input_img": input_img,
        "prompt": prompt,
        "btn_animate": btn_animate,
        "btn_cancel": btn_cancel,
        "video_output": video_output,
        "progress_html": progress_html,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
    }