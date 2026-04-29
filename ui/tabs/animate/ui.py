import os
import gradio as gr
import ui.tabs.animate.state as state


def build_animate_ui():
    gr.HTML("""
        <style>
            .animate-container { background: #020617; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
            .prompt-box-v { background: #0f172a !important; border: 2px solid #8b5cf6 !important; border-radius: 12px !important; font-size: 16px !important; }
            .btn-animate-main { background: linear-gradient(90deg, #8b5cf6, #d946ef) !important; color: white !important; font-weight: bold !important; height: 56px !important; border-radius: 10px !important; font-size: 16px !important; }
        </style>
    """)

    with gr.Column(elem_classes=["animate-container"]):
        gr.Markdown("### 🎬 ANIMAR IMAGEN (estilo Grok)")
        gr.Markdown("_Sube una foto, describe el movimiento, y el AI genera el video._")

        with gr.Row():
            with gr.Column(scale=1):
                from ui.tabs.img_editor_tab import create_maskable_image_input
                input_img = create_maskable_image_input()

                prompt = gr.Textbox(
                    label="",
                    placeholder="Ej: el viento sopla su cabello, parpadea naturalmente, sonríe lentamente...",
                    lines=2, elem_classes=["prompt-box-v"]
                )

                btn_animate = gr.Button("🎬 GENERAR VIDEO", variant="primary", elem_classes=["btn-animate-main"])

                with gr.Accordion("⚙️ Opciones (Automático)", open=False):
                    model_choice = gr.Dropdown(
                        choices=[("Wan 2.2 (Calidad)", "wan_video")],
                        value="wan_video", label="Motor"
                    )
                    face_stabilize = gr.Checkbox(label="💎 Anti-Melting (Cara)", value=True)
                    motion_bucket = gr.Slider(1, 255, 127, step=1, label="Movimiento", visible=False)
                    num_frames = gr.Slider(16, 128, 81, step=1, label="Frames", visible=False)
                    fps = gr.Slider(8, 30, 16, step=1, label="FPS", visible=False)

                progress_html = gr.HTML("<div style='text-align:center; color:#8b5cf6; padding:10px;'>Listo</div>")

            with gr.Column(scale=1):
                video_output = gr.Video(label="RESULTADO", height=500)
                with gr.Row():
                    gr.Button("📂 CARPETA").click(lambda: os.startfile(os.path.abspath("output/animations")))

    return {
        "input_img": input_img, "prompt": prompt,
        "model_choice": model_choice, "face_stabilize": face_stabilize,
        "motion_bucket": motion_bucket, "num_frames": num_frames, "fps": fps,
        "btn_animate": btn_animate, "video_output": video_output,
        "progress_html": progress_html
    }