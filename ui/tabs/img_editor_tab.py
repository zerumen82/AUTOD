#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager

def create_maskable_image_input(label="Imagen de Entrada", height=480):
    return gr.Image(label=label, type="pil", height=height)

_is_generating = False

def on_generate(img_data, p_text, engine_val, f_preserve):
    global _is_generating
    
    # Prevenir ciclos múltiples
    if _is_generating:
        print("[ImgEditor] Generación ya en progreso, ignorando...")
        return None, "⚠️ Ya hay una transformación en proceso", None
    
    p_text = (p_text or "").strip()
    if not p_text:
        return None, "Escribe un prompt", None
    if img_data is None:
        return None, "Sube una imagen", None

    if isinstance(img_data, dict):
        img = img_data.get("background")
    else:
        img = img_data

    if img is None:
        return None, "Imagen inválida", None

    # Verificar si el prompt parece una descripción automática (evitar ciclos)
    auto_descriptions = [
        "the image features", "there is a", "this is a photo of",
        "a group of people", "a person sitting", "a photo showing"
    ]
    p_text_lower = p_text.lower()
    if any(p_text_lower.startswith(desc) for desc in auto_descriptions):
        print(f"[ImgEditor] Warn: Prompt parece descripción automática: {p_text[:50]}...")
        # Permitir si el usuario explicitly confirmó
    
    try:
        _is_generating = True
        manager = get_img_editor_manager()
        
        res_img, msg, mask_img = manager.generate_intelligent(
            image=img, prompt=p_text,
            face_preserve=f_preserve, use_rewriter=True,
            engine=engine_val
        )

        if res_img:
            return res_img, f"✅ {msg}", mask_img
        else:
            return None, f"❌ {msg}", mask_img
    finally:
        _is_generating = False

def analyze_click(img):
    if not img: return "Sube una imagen primero"
    if isinstance(img, dict): img = img.get("background")
    try:
        from scripts.moondream_analyzer import analyze_image_with_moondream
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            res = analyze_image_with_moondream(tmp.name)
        return res.get('positive', 'No se pudo analizar')
    except:
        return "Análisis no disponible"

def open_output_folder():
    path = os.path.abspath("output/img_editor")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    try:
        os.startfile(path)
    except Exception as e:
        print(f"[UI] No se pudo abrir la carpeta: {e}")

def create_img_editor_tab():
    with gr.Column():
        gr.Markdown("## ✨ IMAGE EDITOR AI")
        gr.Markdown("_Transforma tus imágenes con lenguaje natural._")

        with gr.Row():
            # COLUMNA DE ENTRADA
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)
                
                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder="Ej: Ponle un traje de payaso...",
                    lines=3
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ TRANSFORMAR", variant="primary", scale=3)
                    btn_analyze = gr.Button("🔍 ANALIZAR", scale=1)

                with gr.Accordion("⚙️ Opciones", open=False):
                    engine = gr.Dropdown(
                        choices=[
                            ("FLUX.1 Dev", "flux_dev_abliterated"), 
                            ("FLUX.2 Klein", "klein_base"), 
                            ("OmniGen", "omnigen2")
                        ],
                        value="flux_dev_abliterated", label="Motor"
                    )
                    f_preserve = gr.Checkbox(label="Preservar Rostro", value=True)

                status = gr.Textbox(label="ESTADO", interactive=False)

            # COLUMNA DE SALIDA
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🖼️ RESULTADO"):
                        output_img = gr.Image(label="Resultado", height=500)
                    with gr.TabItem("🎭 MÁSCARA"):
                        mask_preview = gr.Image(label="Máscara generada", height=400)
                
                with gr.Row():
                    bt_open_folder = gr.Button("📂 CARPETA")
                    bt_open_folder.click(fn=open_output_folder)

    gen_btn.click(
        on_generate,
        [input_img, prompt, engine, f_preserve],
        [output_img, status, mask_preview]
    )

    btn_analyze.click(analyze_click, [input_img], [prompt])

    return {
        "input_img": input_img, "prompt": prompt, "gen_btn": gen_btn,
        "output_img": output_img, "status": status, "mask_preview": mask_preview
    }
