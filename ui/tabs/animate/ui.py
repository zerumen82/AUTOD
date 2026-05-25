import os
import gradio as gr


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

                btn_animate = gr.Button(
                    "🎬 GENERAR ANIMACIÓN", variant="primary",
                    elem_classes=["btn-animate-main"]
                )

                with gr.Accordion("⚙️ Calidad / Velocidad", open=False):
                    quality = gr.Radio(
                        choices=["Rápido (20 pasos, 1 bloque)", "Normal AR (25 pasos, 2 bloques)", "Calidad AR (30 pasos, 3 bloques)"],
                        value="Normal AR (25 pasos, 2 bloques)", label="Modo"
                    )
                    stabilize = gr.Checkbox(
                        label="💠 Estabilizar rostro (post-proceso)",
                        value=False,
                        info="Aplica restauración facial frame a frame al final (alarga el proceso)"
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
        "video_output": video_output,
        "progress_html": progress_html,
        "quality": quality,
        "stabilize": stabilize,
    }
