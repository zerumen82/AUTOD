#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import gradio as gr
import os
import sys
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager


def open_output_folder():
    import subprocess
    path = os.path.abspath("output/img_editor")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    # Abrir carpeta en explorador
    try:
        if sys.platform == "win32":
            os.startfile(path)
        else:
            subprocess.Popen(["explorer", path])
    except:
        pass
    return None


_is_generating = False


def on_generate(img_data, p_text, engine_val, f_preserve, use_ai_val, enhance_val, lora_name, lora_strength, denoise_val):
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
            face_preserve=f_preserve, use_rewriter=use_ai_val,
            engine=engine_val,
            enhance_faces=enhance_val,
            lora_name=lora_name,
            lora_strength=lora_strength,
            denoise=denoise_val if denoise_val > 0 else None
        )

        if res_img:
            # GUARDAR AUTOMÁTICAMENTE en output/img_editor
            output_dir = os.path.abspath("output/img_editor")
            os.makedirs(output_dir, exist_ok=True)
            import time
            ts = int(time.time())
            out_path = os.path.join(output_dir, f"edit_{ts}.png")
            res_img.save(out_path)
            print(f"[ImgEditor] Imagen guardada: {out_path}")
            return res_img, f"✅ {msg} - Guardado en output/img_editor", mask_img
        else:
            return None, f"❌ {msg}", mask_img
    finally:
        _is_generating = False


def analyze_click(img, user_prompt):
    if not img: return "<div style='color:#f87171;'>Sube una imagen primero</div>", ""
    
    if isinstance(img, dict): 
        img_pil = img.get("background")
    else:
        img_pil = img

    if img_pil is None:
        return "<div style='color:#f87171;'>Imagen inválida</div>", user_prompt

    try:
        from scripts.image_analyzer_for_prompt import ImageAnalyzer
        analyzer = ImageAnalyzer()
        
        # Guardar temporalmente para analizar
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            img_pil.save(tmp.name)
            tmp_path = tmp.name
        
        analysis = analyzer.analyze(tmp_path)
        os.unlink(tmp_path)
        
        desc = analysis.get('suggested_prompt', "No se pudo generar descripción")
        
        # Combinar con el prompt del usuario si existe
        if user_prompt and user_prompt.strip():
            combined = f"{user_prompt}, {desc}"
        else:
            combined = desc
            
        status_html = f"<div style='color:#22d3ee; font-size:12px;'><b>Análisis IA:</b> {desc}</div>"
        return status_html, combined
    except Exception as e:
        print(f"[ImgEditor] Error en análisis: {e}")
        return f"<div style='color:#f87171;'>Error en análisis: {str(e)}</div>", user_prompt


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
                    use_ai = gr.Checkbox(label="🧠 Análisis inteligente", value=True, info="Por defecto activado. Análisis local ligero automático (sin cargar nada pesado, sin internet). El usuario solo sube foto y escribe la instrucción.")
                    
                    with gr.Row():
                        denoise = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.0, label="Fuerza de Edición (0 = Auto)", info="0.2: Sutil, 0.6: Medio, 0.9: Radical")

                    engine = gr.Dropdown(
                        choices=[
                            ("✨ Grok Imagine (default - edición fiel, sin censura)", "imagine"),
                            ("LongCat Image Edit Turbo", "longcat"),
                            ("LongCat Image Edit (Full, CFG=4.5)", "longcat_full"),
                            ("HART (Autoregressive)", "hart"),
                            ("FLUX.2 Klein", "klein_base"),
                            ("FLUX.1 Dev Q2", "flux_q2"),
                            ("OmniGen 2", "omnigen2"),
                            ("FLUX.1 Dev Abliterated", "flux_dev_abliterated"),
                            ("Qwen Image Edit", "qwen_edit")
                        ],
                        value="imagine", label="Motor de Generación"
                    )
                    f_preserve = gr.Checkbox(label="💎 Preservar Rostro", value=True)
                    enhance_faces = gr.Checkbox(label="🌟 Mejorar Rostro (CodeFormer)", value=False, info="Post-procesa los rostros con CodeFormer para más realismo (usa VRAM)")

                    with gr.Row():
                        from ui.tabs.generation_tab import get_available_loras
                        lora_dropdown = gr.Dropdown(
                            choices=get_available_loras(),
                            value="None",
                            label="LoRA (Estilo/Personaje)"
                        )
                        lora_strength = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Fuerza LoRA")
                    
                    bt_refresh_loras = gr.Button("🔄 Refrescar LoRAs", size="sm")
                    bt_refresh_loras.click(fn=lambda: gr.update(choices=get_available_loras()), outputs=[lora_dropdown])

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
                    bt_use_as_input = gr.Button("🔄 USAR COMO ENTRADA")
                    
                    bt_open_folder.click(fn=open_output_folder)
                    bt_use_as_input.click(fn=lambda x: x, inputs=[output_img], outputs=[input_img])

    gen_btn.click(
        on_generate,
        [input_img, prompt, engine, f_preserve, use_ai, enhance_faces, lora_dropdown, lora_strength, denoise],
        [output_img, status, mask_preview],
        concurrency_limit=None
    )


    btn_analyze.click(analyze_click, [input_img, prompt], [status, prompt])

    return {
        "input_img": input_img, "prompt": prompt, "gen_btn": gen_btn,
        "output_img": output_img, "status": status, "mask_preview": mask_preview,
        "enhance_faces": enhance_faces
    }