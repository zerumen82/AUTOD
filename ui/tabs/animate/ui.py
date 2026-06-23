import os
import gradio as gr
from animate_photo import AnimatePhoto


def open_animations_folder():
    path = os.path.abspath("output/animations")
    os.makedirs(path, exist_ok=True)
    try:
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
            gr.Markdown("## 🎬 ANIMATE IMAGE AI")
            gr.Markdown("_Da vida a tus fotos con IA._")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)

                prompt = gr.Textbox(
                    label="¿Qué movimiento quieres?",
                    placeholder="Ej: el viento sopla su cabello, mira a la cámara y sonríe...",
                    lines=3,
                    elem_classes=["prompt-box-anim"]
                )

                with gr.Accordion("🎨 Ajustes de Estilo (LoRA)", open=False):
                    animator = AnimatePhoto()
                    lora_list = ["None"] + animator.get_available_loras()
                    
                    lora_name = gr.Dropdown(
                        label="Seleccionar LoRA",
                        choices=lora_list,
                        value="None",
                        info="Aplica un estilo específico (motor WanVideo)"
                    )
                    
                    lora_strength = gr.Slider(
                        label="Intensidad LoRA",
                        minimum=0.0,
                        maximum=2.0,
                        step=0.05,
                        value=1.0
                    )

                with gr.Row():
                    btn_suggest = gr.Button("🪄 SUGERIR", size="sm", variant="secondary", scale=1)
                    btn_animate = gr.Button(
                        "🎬 GENERAR ANIMACIÓN", variant="primary",
                        elem_classes=["btn-animate-main"],
                        scale=3
                    )

                stabilize = gr.Checkbox(
                    label="💠 Estabilizar rostro (post-proceso)",
                    value=False,
                    info="Aplica restauración facial frame a frame al final (alarga el proceso)"
                )

                add_mmaudio = gr.Checkbox(
                    label="🔊 Añadir audio (MMAudio)",
                    value=False,
                    info="Genera sonido ambiente sincronizado al vídeo tras la animación (requiere ComfyUI-MMAudio)"
                )

                audio_prompt = gr.Textbox(
                    label="Descripción del sonido (opcional)",
                    placeholder="Ej: viento suave, pasos, ambiente de fiesta, olas del mar...",
                    lines=2,
                    visible=False,
                )

                def _toggle_audio_prompt(enabled):
                    return gr.update(visible=bool(enabled))

                add_mmaudio.change(
                    fn=_toggle_audio_prompt,
                    inputs=[add_mmaudio],
                    outputs=[audio_prompt],
                )

                progress_html = gr.HTML(
                    "<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>Listo</div>"
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="", height=520)
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_animations_folder)

    return {
        "input_img": input_img,
        "prompt": prompt,
        "btn_animate": btn_animate,
        "btn_suggest": btn_suggest,
        "video_output": video_output,
        "progress_html": progress_html,
        "stabilize": stabilize,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
        "add_mmaudio": add_mmaudio,
        "audio_prompt": audio_prompt,
    }
