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
            gr.Markdown("_Sube foto + escribe qué debe pasar. Animación autoregresiva automática (~6–9s). Audio ambiente y voz en español._")

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

                with gr.Accordion("⚙️ Configuración y Autoregresión", open=False):
                    from animate_photo import AnimatePhoto
                    animator = AnimatePhoto()
                    lora_list = ["None"] + animator.get_available_loras()
                    engine = gr.Dropdown(
                        label="Motor de animación",
                        choices=["framepack", "wan_video", "ltx2_audio"],
                        value="framepack",
                        info="ltx2_audio: LTX-2 destilado Q4 con audio nativo (960x544, 25 fps, 8 pasos). FramePack/Wan no generan audio nativo.",
                        # "ltx2_audio"  # prueba en el probe del cableado
                    )
                    with gr.Row():
                        autoregressive_chunks = gr.Slider(
                            label="Bloques Autoregresivos (Duración)",
                            minimum=1, maximum=5, step=1, value=2,
                            info="1 bloque = ~4s, 2 bloques = ~8s (concatena manteniendo el último frame)."
                        )
                        add_audio = gr.Checkbox(
                            label="🔊 Generar Audio (MMAudio + Voz Español)",
                            value=True,
                            info="Añade sonido ambiente y voz si hay diálogo. ltx2_audio lo genera nativo: el checkbox no cambia esa ruta."
                        )
                    with gr.Row():
                        frames = gr.Slider(
                            label="Fotogramas por bloque",
                            minimum=25, maximum=97, step=4, value=97,
                            info="**720p: 1280×704** (1280×704, 97 fotogramas, 25 fps). ltx2_audio usa 97 (múltiplo de 8 + 1) para 720p; FramePack 49 y wan_video 33. Más fotogramas = más movimiento y más tiempo.",
                        )
                        steps = gr.Slider(
                            label="Pasos por bloque",
                            minimum=1, maximum=40, step=1, value=8,
                            info="**LTX-2 destilado fija 8 pasos** (CFG 1,0, euler). Este control no afecta al motor ltx2_audio; el valor se ignora y siempre usa 8 pasos. FramePack/Wan sí respetan este slider."
                        )
                    audio_prompt = gr.Textbox(
                        label="Sonido deseado (opcional)",
                        placeholder="Ej: viento suave, pasos en la hierba",
                        lines=1,
                        info="Vacío = MMAudio usa el prompt de movimiento.",
                    )
                    lora_name = gr.Dropdown(
                        label="LoRA WanVideo",
                        choices=lora_list,
                        value="None",
                    )
                    lora_strength = gr.Slider(
                        label="Intensidad LoRA",
                        minimum=0.0, maximum=2.0, step=0.05, value=1.0,
                    )
                    face_stabilize = gr.Checkbox(
                        label="Estabilizar rostro (opcional, ~2–4 min extra)",
                        value=False,
                    )
                    gr.Markdown(
                        "*Déjalo desactivado para velocidad. Actívalo solo si hay personas y la cara se deforma en el vídeo.*"
                    )

                progress_html = gr.HTML(
                    "<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>Listo</div>"
                )

            with gr.Column(scale=1):
                video_output = gr.Video(label="Resultado (duración automática)", height=520)
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_animations_folder)

    return {
        "input_img": input_img,
        "prompt": prompt,
        "engine": engine,
        "autoregressive_chunks": autoregressive_chunks,
        "add_audio": add_audio,
        "frames": frames,
        "steps": steps,
        "audio_prompt": audio_prompt,
        "btn_animate": btn_animate,
        "btn_cancel": btn_cancel,
        "video_output": video_output,
        "progress_html": progress_html,
        "lora_name": lora_name,
        "lora_strength": lora_strength,
        "face_stabilize": face_stabilize,
    }