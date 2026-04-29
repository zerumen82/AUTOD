import os
import gradio as gr
import ui.tabs.animate.state as state

def build_animate_ui():
    """Interfaz Minimalista para Animación (Prompt-First)"""
    
    gr.HTML("""
        <style>
            .animate-container { background: #020617; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
            .prompt-box-v { background: #0f172a !important; border: 2px solid #8b5cf6 !important; border-radius: 12px !important; font-size: 16px !important; }
            .btn-animate-main { background: linear-gradient(90deg, #8b5cf6, #d946ef) !important; color: white !important; font-weight: bold !important; height: 60px !important; border-radius: 10px !important; }
        </style>
    """)

    with gr.Column(elem_classes=["animate-container"]):
        gr.Markdown("<h2 style='text-align:center; color:white;'>🎬 ANIMATE PRO</h2>")
        
        with gr.Row():
            # IZQUIERDA: INPUT
            with gr.Column(scale=1):
                from ui.tabs.img_editor_tab import create_maskable_image_input
                input_img = create_maskable_image_input()
                
                prompt = gr.Textbox(
                    label="✨ ¿QUÉ DEBE PASAR EN EL VÍDEO?", 
                    placeholder="Ej: el viento sopla su cabello, que parpadee, que sonría, que camine...",
                    lines=3, elem_classes=["prompt-box-v"]
                )
                
                with gr.Row():
                    btn_animate = gr.Button("🎬 GENERAR ANIMACIÓN", variant="primary", elem_classes=["btn-animate-main"], scale=2)
                    btn_describe = gr.Button("🌙 DESCRIBIR", scale=1)

                # TODO LO TÉCNICO OCULTO POR DEFECTO
                with gr.Accordion("⚙️ AJUSTES AVANZADOS (AUTOMÁTICOS)", open=False):
                    with gr.Row():
                        model_choice = gr.Dropdown(
                            choices=[("SVD Turbo (Rápido)", "svd_turbo"), ("Wan 2.2 (Ultra Calidad)", "wan_video"), ("LTX-Video (Equilibrado)", "ltx_video")],
                            value="svd_turbo", label="Motor de Vídeo"
                        )
                        face_stabilize = gr.Checkbox(label="💎 Anti-Melt (Cara)", value=True)
                    
                    with gr.Row():
                        mask_mode = gr.Radio(
                            choices=[("Auto", "global"), ("Pintar 🖌️", "manual"), ("Smart IA 🤖", "smart")],
                            value="global", label="Modo de Máscara"
                        )
                        mask_prompt = gr.Textbox(label="Detección Smart", placeholder="ej: pelo, ojos...", visible=False)

                    with gr.Row():
                        motion_bucket = gr.Slider(1, 255, 127, step=1, label="Fuerza de Movimiento")
                        num_frames = gr.Slider(16, 128, 81, step=1, label="Duración (Frames)")
                    
                    fps = gr.Slider(8, 30, 16, step=1, label="FPS")

                progress_html = gr.HTML("<div style='text-align:center; color:#8b5cf6; padding:10px;'>Listo</div>")

            # DERECHA: VISTA PREVIA
            with gr.Column(scale=1):
                video_output = gr.Video(label="VISTA PREVIA", height=500)
                
                with gr.Row():
                    gr.Button("📂 ABRIR CARPETA").click(lambda: os.startfile(os.path.abspath("output/animations")))
                    btn_upscale = gr.Button("✨ RE-ESCALAR 4K")

    # EVENTOS
    mask_mode.change(lambda m: gr.update(visible=(m=="smart")), inputs=[mask_mode], outputs=[mask_prompt])
    
    return {
        "input_img": input_img, "prompt": prompt, "btn_describe": btn_describe,
        "mask_mode": mask_mode, "mask_prompt": mask_prompt,
        "motion_bucket": motion_bucket, "num_frames": num_frames, "fps": fps,
        "model_choice": model_choice, "face_stabilize": face_stabilize,
        "btn_animate": btn_animate, "video_output": video_output,
        "progress_html": progress_html, "btn_upscale": btn_upscale
    }
