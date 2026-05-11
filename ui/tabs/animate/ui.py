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
            .animate-container {
                background: #020617;
                padding: 30px;
                border-radius: 20px;
                border: 1px solid #1e293b;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .animate-header { text-align: center; margin-bottom: 25px; }
            .animate-header h2 {
                background: linear-gradient(90deg, #a78bfa, #f472b6);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 800;
            }
            .btn-animate-main {
                background: linear-gradient(135deg, #8b5cf6 0%, #ec4899 100%) !important;
                color: white !important;
                font-weight: 900 !important;
                height: 64px !important;
                border-radius: 14px !important;
                font-size: 18px !important;
                letter-spacing: 1px;
                text-transform: uppercase;
                border: none !important;
            }
            .expression-btn {
                background: #1e293b !important;
                border: 1px solid #334155 !important;
                border-radius: 10px !important;
                color: #94a3b8 !important;
            }
            .expression-btn:hover {
                background: #334155 !important;
                color: white !important;
                border-color: #8b5cf6 !important;
            }
            .audio-panel {
                background: rgba(15, 23, 42, 0.5);
                border: 1px dashed #334155;
                padding: 15px;
                border-radius: 12px;
                margin-top: 15px;
            }
            .prompt-box-v {
                background: #0f172a !important;
                border: 2px solid #8b5cf6 !important;
                border-radius: 12px !important;
            }
        </style>
    """)

    with gr.Column(elem_classes=["animate-container"]):
        with gr.Group(elem_classes=["animate-header"]):
            gr.Markdown("## 🎬 ANIMATE PHOTO AI")
            gr.Markdown("_Da vida a tus fotos con IA: habla, sonríe y muévete con realismo cinematográfico._")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)

                with gr.Tabs():
                    with gr.TabItem("💬 TEXTO A VIDEO"):
                        prompt = gr.Textbox(
                            label="Prompt de movimiento",
                            placeholder="Ej: el viento sopla su cabello, mira a la cámara y sonríe...",
                            lines=3,
                            elem_classes=["prompt-box-v"]
                        )
                    with gr.TabItem("👄 LIP-SYNC"):
                        with gr.Column(elem_classes=["audio-panel"]):
                            audio_text = gr.Textbox(
                                label="Texto que dirá la foto",
                                placeholder="Escribe aquí lo que quieres que diga..."
                            )
                            with gr.Row():
                                use_tts = gr.Checkbox(label="Generar Voz AI", value=True)
                                language = gr.Dropdown(
                                    ["Español", "Inglés", "Portugués", "Francés"],
                                    value="Español",
                                    label="Idioma"
                                )
                            ref_voice = gr.Audio(label="Voz de Referencia (Clonación)", type="filepath")
                            gr.Markdown("ℹ️ _Sube un audio corto (10s) para clonar esa voz específica._")

                gr.Markdown("**✨ Acciones Rápidas:**")
                with gr.Row():
                    btn_smile = gr.Button("😊 Sonreír", elem_classes=["expression-btn"], size="sm")
                    btn_wink = gr.Button("😉 Guiñar", elem_classes=["expression-btn"], size="sm")
                    btn_angry = gr.Button("😠 Serio", elem_classes=["expression-btn"], size="sm")
                    btn_wind = gr.Button("🌬️ Viento", elem_classes=["expression-btn"], size="sm")

                btn_animate = gr.Button(
                    "🎬 GENERAR ANIMACIÓN",
                    variant="primary",
                    elem_classes=["btn-animate-main"]
                )

                with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                    model_choice = gr.Dropdown(
                        choices=[
                            ("Wan 2.2 (obedece prompt)", "wan_video"),
                            ("SVD Turbo (movimiento genérico)", "svd_turbo"),
                            ("LivePortrait (solo LipSync)", "live_portrait")
                        ],
                        value="wan_video",
                        label="Motor de Animación"
                    )
                    face_stabilize = gr.Checkbox(label="💎 Anti-Melting (Preservar Rostro)", value=True)
                    with gr.Row():
                        motion_bucket = gr.Slider(1, 255, 127, step=1, label="Intensidad Movimiento")
                        num_frames = gr.Slider(17, 129, 81, step=4, label="Duración (Frames)")

                progress_html = gr.HTML(
                    "<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>Listo</div>"
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="", height=520)
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_animations_folder)
                    gr.Button("🔄 REINTENTAR").click(fn=None, _js="window.location.reload()")

    return {
        "input_img": input_img,
        "prompt": prompt,
        "audio_text": audio_text,
        "use_tts": use_tts,
        "language": language,
        "ref_voice": ref_voice,
        "btn_smile": btn_smile,
        "btn_wink": btn_wink,
        "btn_angry": btn_angry,
        "btn_wind": btn_wind,
        "model_choice": model_choice,
        "face_stabilize": face_stabilize,
        "motion_bucket": motion_bucket,
        "num_frames": num_frames,
        "btn_animate": btn_animate,
        "video_output": video_output,
        "progress_html": progress_html,
    }
