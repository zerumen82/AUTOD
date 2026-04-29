#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager

def create_maskable_image_input():
    common_kwargs = {"label": "📷 IMAGEN", "height": 420, "type": "pil"}
    try:
        if hasattr(gr, "ImageEditor"): return gr.ImageEditor(**common_kwargs)
    except: pass
    return gr.Image(**common_kwargs)

def on_generate(img_data, p_text, engine_val, f_preserve):
    if not p_text:
        yield None, "Escribe un prompt"
        return
    if img_data is None:
        yield None, "Sube una imagen"
        return

    if isinstance(img_data, dict):
        img = img_data.get("background")
    else:
        img = img_data

    if img is None:
        yield None, "Imagen inválida"
        return

    manager = get_img_editor_manager()
    yield None, f"⏳ Analizando prompt y preparando edición..."

    res_img, msg = manager.generate_intelligent(
        image=img, prompt=p_text,
        face_preserve=f_preserve, use_rewriter=True,
        engine=engine_val
    )

    if res_img:
        yield res_img, f"✅ {msg}"
    else:
        yield None, f"❌ {msg}"

def create_img_editor_tab():
    gr.HTML("""
        <style>
            .grok-container { background: #020617; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
            .grok-prompt { background: #0f172a !important; border: 2px solid #3b82f6 !important; border-radius: 12px !important; font-size: 16px !important; }
            .grok-btn { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; color: white !important; font-weight: bold !important; height: 56px !important; border-radius: 10px !important; font-size: 16px !important; }
            .grok-result { border-radius: 12px !important; overflow: hidden !important; }
        </style>
    """)

    with gr.Column(elem_classes=["grok-container"]):
        gr.Markdown("### ✨ Editor Inteligente (estilo Grok Imagine)")
        gr.Markdown("_Sube una imagen, escribe qué quieres cambiar, y el AI lo hace solo._")

        with gr.Row():
            with gr.Column(scale=1):
                input_img = create_maskable_image_input()

                prompt = gr.Textbox(
                    label="",
                    placeholder="Ej: ponle gafas de sol, cambia el fondo a una playa, que sonría...",
                    lines=2, elem_classes=["grok-prompt"]
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ APLICAR CAMBIO", variant="primary", elem_classes=["grok-btn"], scale=2)
                    btn_analyze = gr.Button("🔍 ANALIZAR", scale=1)

                with gr.Accordion("⚙️ Opciones (Automático)", open=False):
                    engine = gr.Dropdown(
                        choices=[("FLUX.2 Klein (rápido)", "flux_klein"), ("FLUX.1 Schnell (más rápido)", "flux_schnell")],
                        value="flux_klein", label="Motor"
                    )
                    f_preserve = gr.Checkbox(label="Preservar rostros", value=True)

                status = gr.Textbox(label="", interactive=False)

            with gr.Column(scale=1):
                output_img = gr.Image(label="RESULTADO", height=500, elem_classes=["grok-result"])
                with gr.Row():
                    gr.Button("📂 CARPETA").click(lambda: os.startfile(os.path.abspath("output/img_editor")))

    gen_btn.click(
        on_generate,
        [input_img, prompt, engine, f_preserve],
        [output_img, status]
    )

    def analyze_click(img):
        if not img: return "Sube una imagen primero"
        if isinstance(img, dict): img = img.get("background")
        try:
            from moondream_analyzer import analyze_image_with_moondream
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                res = analyze_image_with_moondream(tmp.name)
            return res.get('positive', 'No se pudo analizar')
        except:
            return "Análisis no disponible (instala moondream_analyzer)"
    btn_analyze.click(analyze_click, [input_img], [prompt])

    return {
        "input_img": input_img, "prompt": prompt, "gen_btn": gen_btn,
        "output_img": output_img, "status": status
    }