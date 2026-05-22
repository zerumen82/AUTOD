#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import requests
import json
import time
from PIL import Image
import io
from roop.img_editor.flux_gen_comfy_client import get_flux_gen_client

def check_engine_status():
    """Verifica si el motor de generación (ComfyUI) está activo"""
    client = get_flux_gen_client()
    if client.is_available():
        return "🟢 Motor Longshot Listo"
    return "🔴 Motor Detenido"

def on_generate_image(prompt, negative_prompt, steps, cfg_scale, width, height):
    from roop.img_editor.prompt_translator import translate_prompt
    prompt = translate_prompt(prompt)
    client = get_flux_gen_client()
    if not client.is_available():
        return None, "❌ El motor de IA no está activo. Inícialo en los controles de arriba."

    try:
        res, msg = client.generate(
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=steps,
            guidance_scale=cfg_scale,
            width=width,
            height=height
        )
        
        if res:
            # Guardar automáticamente
            output_dir = os.path.abspath("output/generation")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"gen_{int(time.time())}.png")
            res.image.save(out_path)
            return res.image, f"✅ Generación completada ({res.time_taken:.1f}s) - Guardado en output/generation"
        else:
            return None, f"❌ Error: {msg}"
    except Exception as e:
        return None, f"❌ Error crítico: {str(e)}"

def open_generation_folder():
    import subprocess
    import sys
    path = os.path.abspath("output/generation")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["explorer", path])
    except:
        pass
    return None

def generation_tab():
    gr.HTML("""
        <style>
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
            .btn-generate-main {
                background: linear-gradient(135deg, #a855f7 0%, #06b6d4 100%) !important;
                color: white !important;
                font-weight: 900 !important;
                height: 64px !important;
                border-radius: 14px !important;
                font-size: 18px !important;
                letter-spacing: 1px;
                text-transform: uppercase;
                border: none !important;
            }
            .prompt-box-gen {
                background: #0f172a !important;
                border: 2px solid #a855f7 !important;
                border-radius: 12px !important;
            }
        </style>
    """)
    
    with gr.Column(elem_classes=["gen-tab-container"]):
        with gr.Group(elem_classes=["gen-tab-header"]):
            gr.Markdown("## 🚀 AI GENERATOR (LONGSHOT)")
            gr.Markdown("_Crea imágenes realistas desde cero usando el motor de alta fidelidad._")

        with gr.Row():
            # COLUMNA DE ENTRADA
            with gr.Column(scale=1):
                status_box = gr.Textbox(value=check_engine_status(), label="Estado del Motor", interactive=False)
                
                prompt = gr.Textbox(
                    label="¿Qué quieres crear?",
                    placeholder="Ej: A realistic portrait of a viking warrior, snowy mountains background...",
                    lines=4,
                    elem_classes=["prompt-box-gen"]
                )
                
                neg_prompt = gr.Textbox(
                    label="Evitar (Prompt Negativo)",
                    value="blurry, distorted, low quality, bad anatomy, text, watermark",
                    lines=2
                )

                with gr.Row():
                    gen_btn = gr.Button("🚀 GENERAR IMAGEN", variant="primary", elem_classes=["btn-generate-main"], scale=3)
                    refresh_btn = gr.Button("↻", size="sm", scale=1)

                with gr.Accordion("⚙️ Ajustes Avanzados", open=False):
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=50, step=1, value=8, label="Pasos (Longshot Turbo: 8 es ideal)")
                        cfg = gr.Slider(minimum=1, maximum=10, step=0.5, value=1.0, label="CFG Scale (Turbo: 1.0)")
                    
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=1024, step=64, value=512, label="Ancho")
                        height = gr.Slider(minimum=256, maximum=1024, step=64, value=768, label="Alto")

                status_html = gr.HTML("<div style='text-align:center; color:#64748b; padding:10px;'>Listo</div>")

            # COLUMNA DE SALIDA
            with gr.Column(scale=1):
                output_img = gr.Image(label="Imagen Generada", height=600)
                
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_generation_folder)

    gen_btn.click(
        on_generate_image,
        [prompt, neg_prompt, steps, cfg, width, height],
        [output_img, status_html],
        concurrency_limit=None
    )
    
    refresh_btn.click(fn=check_engine_status, outputs=[status_box])

    return {
        "prompt": prompt, "gen_btn": gen_btn, "output_img": output_img, "status": status_html
    }
