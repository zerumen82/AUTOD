#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Editor - Ultra Minimalist (Prompt-First)
Elimina distracciones y automatiza la lógica de máscaras.
"""

import gradio as gr
import os
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager

def create_maskable_image_input():
    common_kwargs = {"label": "📤 IMAGEN ORIGEN", "height": 420, "type": "pil"}
    try:
        if hasattr(gr, "ImageEditor"): return gr.ImageEditor(**common_kwargs)
    except: pass
    return gr.Image(**common_kwargs)

def get_metrics_html(percent, status):
    return f"""
    <div style="background: #0f172a; padding: 10px; border-radius: 8px; border: 1px solid #1e293b;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="color: #64748b; font-size: 11px;">PROGRESO: <b style="color: #3b82f6;">{percent:.0f}%</b></span>
            <span style="color: #64748b; font-size: 11px;">ESTADO: <b style="color: #8b5cf6;">{status}</b></span>
        </div>
        <div style="background: #1e293b; border-radius: 4px; height: 4px; overflow: hidden;">
            <div style="width: {percent}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #8b5cf6);"></div>
        </div>
    </div>
    """

def on_generate(img_data, p_text, creativity, preserve, steps, num_var, f_preserve, a_enhance, res_label, engine_val, m_mode, m_prompt):
    img = None
    manual_mask = None
    if not p_text: yield [], "Escribe un prompt", get_metrics_html(0, "Error"); return
    if img_data is None: yield [], "Sube una imagen", get_metrics_html(0, "Error"); return
    
    if isinstance(img_data, dict):
        img = img_data.get("background")
        if m_mode == "manual" and "layers" in img_data and img_data["layers"]: 
            manual_mask = img_data["layers"][0]
    else: 
        img = img_data
        
    if img is None: yield [], "Imagen inválida", get_metrics_html(0, "Error"); return

    manager = get_img_editor_manager()
    ref_meta = {"resolution_label": res_label, "creativity": creativity, "preserve": preserve}
    results = []
    
    for op_idx in range(int(num_var)):
        pct = (op_idx / int(num_var)) * 100
        yield results, f"Procesando...", get_metrics_html(pct, f"Motor: {engine_val}")
        
        res_img, msg = manager.generate_intelligent(
            image=img, prompt=p_text, num_inference_steps=steps,
            face_preserve=f_preserve, use_rewriter=a_enhance, ref_metadata=ref_meta, 
            engine=engine_val, mask_image=manual_mask, mask_mode=m_mode, mask_prompt=m_prompt
        )
        if res_img: results.append(res_img)
        yield results, msg, get_metrics_html(((op_idx+1)/int(num_var))*100, "Listo")

def create_img_editor_tab():
    gr.HTML("""
        <style>
            .minimal-container { background: #020617; padding: 20px; border-radius: 15px; border: 1px solid #1e293b; }
            .prompt-box { background: #0f172a !important; border: 2px solid #3b82f6 !important; border-radius: 12px !important; font-size: 16px !important; }
            .main-btn { background: linear-gradient(90deg, #3b82f6, #8b5cf6) !important; color: white !important; font-weight: bold !important; height: 60px !important; border-radius: 10px !important; }
        </style>
    """)

    with gr.Column(elem_classes=["minimal-container"]):
        with gr.Row():
            # IZQUIERDA: INPUT
            with gr.Column(scale=1):
                input_img = create_maskable_image_input()
                
                prompt = gr.Textbox(
                    label="✨ ¿QUÉ QUIERES CAMBIAR?", 
                    placeholder="Ej: ponle una armadura dorada, cámbiale el fondo a una playa, que sonría...",
                    lines=3, elem_classes=["prompt-box"]
                )
                
                with gr.Row():
                    gen_btn = gr.Button("🎨 APLICAR CAMBIOS", variant="primary", elem_classes=["main-btn"], scale=2)
                    btn_analyze = gr.Button("🔍 IA SUGGEST", scale=1)

                # TODO LO TÉCNICO OCULTO POR DEFECTO
                with gr.Accordion("🛠️ AJUSTES TÉCNICOS (AUTOMÁTICOS)", open=False):
                    with gr.Row():
                        engine = gr.Dropdown(
                            choices=[("FLUX.2 Klein", "flux_klein"), ("FLUX.1 Schnell", "flux_schnell"), ("OmniGen2", "omnigen2")],
                            value="flux_klein", label="Motor IA"
                        )
                        res = gr.Radio(["720p", "1024p"], value="1024p", label="Resolución")
                    
                    with gr.Row():
                        mask_mode = gr.Radio(
                            choices=[("Auto (Recomendado)", "global"), ("Pintar 🖌️", "manual"), ("Smart IA 🤖", "smart")],
                            value="global", label="Área de Trabajo"
                        )
                        mask_prompt = gr.Textbox(label="Detección Smart", placeholder="ej: ropa, ojos...", visible=False)

                    with gr.Row():
                        creativity = gr.Slider(0, 1, 0.6, label="Creatividad (Denoise)")
                        preserve = gr.Slider(0, 1, 0.4, label="Fidelidad (Fuerza)")
                    
                    with gr.Row():
                        steps = gr.Slider(4, 30, 12, step=1, label="Pasos")
                        num_var = gr.Slider(1, 5, 1, step=1, label="Variaciones")
                    
                    with gr.Row():
                        f_preserve = gr.Checkbox(label="Preservar Rostro", value=True)
                        a_enhance = gr.Checkbox(label="Auto-Prompt Boost", value=True)

                metrics = gr.HTML(get_metrics_html(0, "Listo"))
                status = gr.Textbox(label="Estado", interactive=False)

            # DERECHA: RESULTADOS
            with gr.Column(scale=1):
                gallery = gr.Gallery(label="RESULTADOS", columns=1, height=500, preview=True)
                
                with gr.Row():
                    gr.Button("📂 CARPETA").click(lambda: os.startfile(os.path.abspath("output/img_editor")))
                    reload_btn = gr.Button("🔄 USAR COMO INPUT")

                with gr.Accordion("🌓 COMPARAR ANTES/DESPUÉS", open=True):
                    comp_slider = gr.Slider(0, 100, 50, label="Desliza")
                    preview_comp = gr.Image(label="", interactive=False)

    # EVENTOS
    mask_mode.change(lambda m: gr.update(visible=(m=="smart")), inputs=[mask_mode], outputs=[mask_prompt])
    
    gen_btn.click(
        on_generate,
        [input_img, prompt, creativity, preserve, steps, num_var, f_preserve, a_enhance, res, engine, mask_mode, mask_prompt],
        [gallery, status, metrics]
    )

    def use_as_input(g_data):
        if g_data and len(g_data) > 0:
            item = g_data[0]
            return item['name'] if isinstance(item, dict) else (item[0] if isinstance(item, (list, tuple)) else None)
        return None
    reload_btn.click(use_as_input, [gallery], [input_img])

    def analyze_click(img):
        if not img: return "Sube una imagen"
        if isinstance(img, dict): img = img.get("background")
        try:
            from moondream_analyzer import analyze_image_with_moondream
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                res = analyze_image_with_moondream(tmp.name)
            return res['positive']
        except: return "No se pudo analizar"
    btn_analyze.click(analyze_click, [input_img], [prompt])

    def update_comp(orig_data, results, pct):
        if not orig_data or not results: return None
        orig = orig_data if not isinstance(orig_data, dict) else orig_data.get("background")
        if not orig: return None
        gen = results[0].resize(orig.size, Image.LANCZOS)
        w, h = orig.size
        split = int(w * (pct / 100))
        comp = Image.new("RGB", (w, h))
        comp.paste(orig.crop((0, 0, split, h)), (0, 0))
        comp.paste(gen.crop((split, 0, w, h)), (split, 0))
        return comp
    comp_slider.change(update_comp, [input_img, gallery, comp_slider], [preview_comp])

    return {
        "input_img": input_img, "prompt": prompt, "gen_btn": gen_btn,
        "gallery": gallery, "status": status, "metrics": metrics
    }
