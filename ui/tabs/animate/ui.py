import os
import cv2
import numpy as np
from PIL import Image
import gradio as gr
import roop.globals
import ui.tabs.animate.state as state

def build_animate_ui():
    """Estructura visual inspirada en Grok Imagine para Animación"""
    
    gr.HTML("""
        <style>
            .animate-panel { border: 2px solid #8b5cf6 !important; border-radius: 12px; padding: 15px; background: rgba(139, 92, 246, 0.05); }
            .video-preview { background: #000; border-radius: 8px; min-height: 400px; }
            .grok-btn { background: linear-gradient(90deg, #8b5cf6, #d946ef) !important; color: white !important; font-weight: bold !important; }
        </style>
    """)

    with gr.Row(elem_classes=["animate-panel"]):
        # COLUMNA IZQUIERDA: Configuración y Prompts
        with gr.Column(scale=2):
            # CANVAS DE ANIMACIÓN CON MÁSCARAS
            from ui.tabs.img_editor_tab import create_maskable_image_input
            input_img = create_maskable_image_input()
            
            with gr.Row(variant="compact"):
                mask_mode = gr.Radio(
                    choices=[("Todo", "global"), ("Pintar 🖌️", "manual"), ("IA (CLIPSeg) 🤖", "smart")],
                    value="global",
                    label="🎯 ¿Qué animar?"
                )
                mask_prompt = gr.Textbox(
                    label="Objeto a animar", 
                    placeholder="ej: pelo, ojos, agua...",
                    visible=False,
                    scale=2
                )
            
            with gr.Row():
                prompt = gr.Textbox(
                    label="¿Qué debe pasar en el área seleccionada?", 
                    placeholder="Ej: el viento sopla suavemente...",
                    lines=3
                )
                btn_describe = gr.Button("🌙 Ver con Moondream", scale=1)
            
            with gr.Accordion("🎨 Control de Movimiento", open=False):
                motion_bucket = gr.Slider(1, 255, value=127, label="Fuerza del Movimiento")
                num_frames = gr.Slider(16, 128, value=81, step=1, label="Duración (Frames)")
                fps = gr.Slider(8, 30, value=16, step=1, label="FPS")
            
            with gr.Row():
                model_choice = gr.Dropdown(
                    choices=[("Wan 2.2 (Máxima Calidad)", "wan_video"), ("LTX-Video (Velocidad)", "ltx_video")],
                    value="wan_video", label="Motor de Vídeo"
                )
                face_stabilize = gr.Checkbox(label="💎 Estabilidad Facial (Anti-Melt)", value=True)

            btn_animate = gr.Button("🎬 GENERAR VÍDEO (Grok Style)", variant="primary", elem_classes=["grok-btn"])

        # COLUMNA DERECHA: Resultado
        with gr.Column(scale=3):
            video_output = gr.Video(label="Resultado", elem_classes=["video-preview"])
            
            with gr.Row():
                gr.Button("📂 Abrir Carpeta").click(lambda: os.startfile(os.path.abspath("output/animations")))
                btn_upscale = gr.Button("✨ Super-Resolución (4K)", variant="secondary")
            
            progress_html = gr.HTML("<div style='text-align:center; color:#8b5cf6;'>Esperando inicio...</div>")

    return {
        "input_img": input_img, "prompt": prompt, "btn_describe": btn_describe,
        "mask_mode": mask_mode, "mask_prompt": mask_prompt,
        "motion_bucket": motion_bucket, "num_frames": num_frames, "fps": fps,
        "model_choice": model_choice, "face_stabilize": face_stabilize,
        "btn_animate": btn_animate, "video_output": video_output,
        "progress_html": progress_html, "btn_upscale": btn_upscale
    }
