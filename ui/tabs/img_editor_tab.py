#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager


def open_output_folder():
    path = os.path.abspath("output/img_editor")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    try:
        os.startfile(path)
    except Exception as e:
        print(f"[UI] No se pudo abrir la carpeta: {e}")


_is_generating = False


def on_generate(img_data, p_text, engine_val, f_preserve):
    global _is_generating
    
    if _is_generating:
        return None, "⚠️ Ya hay una transformación en proceso", None
    
    p_text = (p_text or "").strip()
    if not p_text:
        return None, "Escribe un prompt", None
    if img_data is None:
        return None, "Sube una imagen", None

    if isinstance(img_data, dict):
        img = img_data.get("background")
    elif isinstance(img_data, str):
        # Es una ruta de archivo
        from PIL import Image
        img = Image.open(img_data).convert("RGB")
    else:
        img = img_data

    if img is None:
        return None, "Imagen inválida", None
    
    # Asegurar que es PIL Image
    from PIL import Image as PILImage
    if not isinstance(img, PILImage.Image):
        return None, "Formato de imagen no soportado", None

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


def analyze_click(img, user_prompt):
    if not img: return "<div style='color:#f87171;'>Sube una imagen primero</div>", ""
    if isinstance(img, dict): 
        img = img.get("background")
    elif isinstance(img, str):
        from PIL import Image
        img = Image.open(img).convert("RGB")
    try:
        from scripts.moondream_analyzer import analyze_image_with_moondream
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            res = analyze_image_with_moondream(tmp.name)
        desc = res.get('positive', 'No se pudo analizar')
        
        # Combinar descripción con prompt del usuario
        combined = f"{desc[:150]}... {user_prompt}" if user_prompt else desc[:200]
        
        status_html = f"<div style='color:#22d3ee; font-size:12px;'><b>Imagen:</b> {desc[:150]}...</div>"
        return status_html, combined
    except:
        return "<div style='color:#f87171;'>Análisis no disponible</div>", user_prompt


def create_img_editor_tab():
    gr.HTML("""
        <style>
            .img-editor-container {
                background: #020617;
                padding: 30px;
                border-radius: 20px;
                border: 1px solid #1e293b;
                box-shadow: 0 10px 30px rgba(0,0,0,0.5);
            }
            .img-editor-header { text-align: center; margin-bottom: 25px; }
            .img-editor-header h2 {
                background: linear-gradient(90deg, #22d3ee, #a855f7);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 32px;
                font-weight: 800;
            }
            .btn-transform-main {
                background: linear-gradient(135deg, #06b6d4 0%, #a855f7 100%) !important;
                color: white !important;
                font-weight: 900 !important;
                height: 64px !important;
                border-radius: 14px !important;
                font-size: 18px !important;
                letter-spacing: 1px;
                text-transform: uppercase;
                border: none !important;
            }
            .prompt-box-img {
                background: #0f172a !important;
                border: 2px solid #22d3ee !important;
                border-radius: 12px !important;
            }
        </style>
    """)
    
    with gr.Column(elem_classes=["img-editor-container"]):
        with gr.Group(elem_classes=["img-editor-header"]):
            gr.Markdown("## ✨ IMAGE EDITOR AI")
            gr.Markdown("_Transforma tus imágenes con lenguaje natural._")

        with gr.Row():
            # COLUMNA DE ENTRADA
            with gr.Column(scale=1):
                input_img = gr.Image(label="Imagen de Entrada", type="pil", height=480)
                
                prompt = gr.Textbox(
                    label="¿Qué quieres cambiar?",
                    placeholder="Ej: Ponle un traje de payaso, cambia el fondo por una playa...",
                    lines=3,
                    elem_classes=["prompt-box-img"]
                )

                with gr.Row():
                    gen_btn = gr.Button("✨ TRANSFORMAR", variant="primary", elem_classes=["btn-transform-main"], scale=3)
                    btn_analyze = gr.Button("🔍 ANALIZAR", size="sm", scale=1)

                with gr.Accordion("⚙️ Opciones Avanzadas", open=False):
                    engine = gr.Dropdown(
                        choices=[
                            ("FLUX.1 Dev Abliterated", "flux_dev_abliterated"), 
                            ("FLUX.2 Klein", "klein_base"), 
                            ("OmniGen 2", "omnigen2")
                        ],
                        value="flux_dev_abliterated", label="Motor de Generación"
                    )
                    f_preserve = gr.Checkbox(label="💎 Preservar Rostro", value=True)

                status = gr.HTML("<div style='text-align:center; color:#64748b; padding:10px;'>Listo</div>")

            # COLUMNA DE SALIDA
            with gr.Column(scale=1):
                with gr.Tabs():
                    with gr.TabItem("🖼️ RESULTADO"):
                        output_img = gr.Image(label="Resultado", height=500)
                    with gr.TabItem("🎭 MÁSCARA"):
                        mask_preview = gr.Image(label="Máscara generada", height=400)
                
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_output_folder)

    gen_btn.click(
        on_generate,
        [input_img, prompt, engine, f_preserve],
        [output_img, status, mask_preview]
    )

    btn_analyze.click(analyze_click, [input_img, prompt], [status, prompt])

    return {
        "input_img": input_img, "prompt": prompt, "gen_btn": gen_btn,
        "output_img": output_img, "status": status, "mask_preview": mask_preview
    }