#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Editor - Editor de Imágenes con IA (Versión Completa Restaurada)
"""

import gradio as gr
import os
import sys
import logging
import time
import threading
from PIL import Image
import requests
from roop.img_editor.img_editor_manager import get_img_editor_manager

# Directorio de salida
COMFYUI_OUTPUT_DIR = os.path.abspath(os.path.join(os.getcwd(), "output", "img_editor"))
os.makedirs(COMFYUI_OUTPUT_DIR, exist_ok=True)

def get_metrics_html(percent, variation, time_remaining, status):
    """Genera HTML de métricas profesional"""
    progress_color = "#3b82f6" if status not in ["Error", "Completado"] else ("#10b981" if status == "Completado" else "#ef4444")
    bar_color = "linear-gradient(90deg, #3b82f6, #10b981)" if status != "Error" else "linear-gradient(90deg, #ef4444, #f59e0b)"
    safe_percent = max(0, min(100, percent))  # Asegurar que está entre 0-100
    return f"""
    <div style="background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%); padding: 15px; border-radius: 10px; margin: 10px 0; box-shadow: 0 4px 6px rgba(0,0,0,0.3); border: 1px solid #334155;">
        <h3 style="color: #3b82f6; margin-top: 0; font-size: 16px; border-bottom: 1px solid #334155; padding-bottom: 5px;">📊 Progreso en Tiempo Real</h3>
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px;">
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">PROGRESO</div><div style="color: {progress_color}; font-size: 22px; font-weight: bold;">{safe_percent:.1f}%</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">RESTANTE</div><div style="color: #10b981; font-size: 22px; font-weight: bold;">{time_remaining}</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">VARIACIÓN</div><div style="color: #f59e0b; font-size: 22px; font-weight: bold;">{variation}</div></div>
            <div style="text-align: center;"><div style="color: #94a3b8; font-size: 11px;">ESTADO</div><div style="color: #8b5cf6; font-size: 14px; font-weight: bold;">{status}</div></div>
        </div>
        <div style="margin-top: 15px; background: rgba(255,255,255,0.05); border-radius: 5px; height: 10px; overflow: hidden;">
            <div style="width: {safe_percent}%; height: 100%; background: {bar_color}; transition: width 0.4s ease-out;"></div>
        </div>
    </div>
    """

def apply_text_overlay(image: Image.Image, text_cfg: dict) -> Image.Image:
    try:
        from PIL import ImageDraw, ImageFont
        res = image.copy(); draw = ImageDraw.Draw(res)
        text, pos_key, style = text_cfg.get("text", ""), text_cfg.get("position", "bottom-center"), text_cfg.get("style", "modern")
        if not text: return image
        f_size = max(20, min(image.width, image.height) // 18)
        try: font = ImageFont.truetype("arial.ttf", f_size)
        except: font = ImageFont.load_default()
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]; p = 30
        pos_map = {
            "top-left": (p, p), "top-center": ((image.width-w)//2, p), "top-right": (image.width-w-p, p),
            "bottom-left": (p, image.height-h-p), "bottom-center": ((image.width-w)//2, image.height-h-p), "bottom-right": (image.width-w-p, image.height-h-p),
            "center": ((image.width-w)//2, (image.height-h)//2)
        }
        x, y = pos_map.get(pos_key, pos_map["bottom-center"])
        draw.text((x+2, y+2), text, font=font, fill="black") # Sombra
        draw.text((x, y), text, font=font, fill="white") # Texto
        return res
    except: return image

def on_enhance_prompt(prompt_text):
    if not prompt_text or not prompt_text.strip(): 
        return "⚠️ Escribe un prompt primero", ""
    try:
        manager = get_img_editor_manager()
        enhanced = manager.rewrite_prompt(prompt_text)
        return "✨ Prompt mejorado con éxito", enhanced
    except Exception as e: 
        return f"❌ Error: {str(e)}", ""

def on_generate(img, p_text, num_var, quality, f_preserve, a_enhance, res_val, c_ref, r_list, use_c_ref, add_t, t_in, t_p, t_s, b_mode, b_files, engine_val, p_enh, qwen_version="q3", zimage_version="q4"):
    logging.info(f"[OnGenerate] Motor: {engine_val}, Qwen: {qwen_version}, Z-Image: {zimage_version}")

    # Validaciones de imagen
    if b_mode:
        if not b_files: yield [], "❌ Sube imágenes para Batch", get_metrics_html(0, "0/0", "--:--", "Error"); return
        work_images = [Image.open(f.name) for f in b_files]
    else:
        if not img: yield [], "❌ Sube una imagen", get_metrics_html(0, "0/0", "--:--", "Error"); return
        work_images = [img]

    manager = get_img_editor_manager()
    yield [], "Preparando motor de IA...", get_metrics_html(0, "0/0", "--:--", "⏳ Inicializando...")

    # Preparación de Metadatos
    final_p = p_enh if p_enh else p_text
    ref_meta = {
        "character_ref": c_ref if use_c_ref else None,
        "multi_refs": r_list if r_list else [],
        "text_overlay": {"text": t_in, "position": t_p, "style": t_s} if add_t else None
    }

    # Configuración de calidad
    # FLUX con CPU offload necesita pocos pasos para ser usable
    q_cfg = {"fast": 8, "balanced": 12, "high": 20}.get(quality, 12)

    results = []
    total_imgs = len(work_images)
    total_vars = int(num_var)
    total_ops = total_imgs * total_vars
    op_idx = 0
    start_t = time.time()

    # Bucle de Generación (Batch + Variaciones)
    for img_idx, current_img in enumerate(work_images):
        for var_idx in range(total_vars):
            pct = (op_idx / total_ops) * 100
            elapsed = time.time() - start_t
            avg_time = elapsed / (op_idx + 1) if op_idx > 0 else 0
            remaining_ops = total_ops - op_idx
            time_remaining = f"{int(avg_time * remaining_ops)}s" if avg_time > 0 else "Calculando..."
            status_msg = f"Imagen {img_idx+1}/{total_imgs} | Var {var_idx+1}/{total_vars}"
            yield results, f"Procesando {status_msg}...", get_metrics_html(pct, f"{op_idx+1}/{total_ops}", time_remaining, f"Motor {engine_val.upper()}")

            res_img, msg = manager.generate_intelligent(
                image=current_img, prompt=final_p, num_inference_steps=q_cfg,
                face_preserve=f_preserve, auto_enhance=a_enhance, ref_metadata=ref_meta, 
                engine=engine_val, qwen_version=qwen_version, zimage_version=zimage_version
            )

            if res_img:
                if ref_meta["text_overlay"]: res_img = apply_text_overlay(res_img, ref_meta["text_overlay"])
                results.append(res_img)

            op_idx += 1
            elapsed_total = time.time() - start_t
            yield results, f"Completada {status_msg}", get_metrics_html((op_idx/total_ops)*100, f"{op_idx}/{total_ops}", f"{int(elapsed_total)}s", "OK")

    elapsed_final = time.time() - start_t
    yield results, f"✅ {len(results)} imágenes generadas con éxito", get_metrics_html(100, f"{len(results)}/{total_ops}", f"{int(elapsed_final)}s", "Finalizado")

def create_img_editor_tab():
    gr.Markdown("## 🎨 Image Editor")
    
    # Estados
    current_img_state = gr.State(); char_ref_state = gr.State(); ref_list_state = gr.State(value=[])

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("📦 Procesamiento por Lote (Batch)", open=False):
                batch_mode = gr.Checkbox(label="Activar Modo Batch", value=False)
                batch_files = gr.File(label="Sube múltiples imágenes", file_count="multiple", visible=False)
            
            input_img = gr.Image(label="Imagen Original", type="pil", height=250)
            prompt = gr.Textbox(label="Instrucciones", placeholder="Ej: cámbiale la pose a bailando...", lines=3)
            enhance_btn = gr.Button("✨ Auto-Mejorar Prompt", variant="secondary")
            prompt_enh = gr.Textbox(label="Prompt Optimizado (Sugerencia)", interactive=False)

            with gr.Accordion("👤 Referencias de Personaje/Estilo", open=False):
                char_ref_img = gr.Image(label="Character Reference", type="pil", height=150)
                use_char_ref = gr.Checkbox(label="Usar Character Reference", value=False)
                gr.Markdown("---")
                gr.Markdown("**📚 Multi-Reference (Estilo/Entorno)**")
                with gr.Row():
                    r1 = gr.Image(label="R1", type="pil", height=80); r2 = gr.Image(label="R2", type="pil", height=80); r3 = gr.Image(label="R3", type="pil", height=80)

            with gr.Accordion("📝 Texto y Logos", open=False):
                add_text = gr.Checkbox(label="Añadir texto sobre la imagen")
                t_in = gr.Textbox(label="Contenido del texto", visible=False)
                t_p = gr.Radio(choices=[("Arriba", "top-center"), ("Abajo", "bottom-center"), ("Centro", "center")], value="bottom-center", label="Posición", visible=False)
                t_s = gr.Radio(choices=[("Moderno", "modern"), ("Clásico", "classic")], value="modern", label="Estilo", visible=False)

            with gr.Accordion("⚙️ Configuración del Motor", open=True):
                gr.Markdown("**🎯 Recomendado: OmniGen2 (nuevo AR) > Z-Image > Qwen > HART**")
                with gr.Row():
                    engine = gr.Radio(
                        choices=[
                            ("🐷 OmniGen2 GGUF (AR)", "omnigen2"),
                            ("⚡ Z-Image Turbo (híbrido)", "zimage"),
                            ("🤖 Qwen 2509 (AR)", "qwen2509"),
                            ("🤖 Qwen 2512 (AR)", "qwen2512"),
                            ("🔬 HART (AR)", "hart"),
                            ("🔷 FLUX (DiT)", "flux")
                        ],
                        value="omnigen2",
                        label="Motor"
                    )
                    with gr.Column(scale=1):
                        with gr.Row():
                            qwen_version = gr.Radio(
                                choices=[
                                    ("Q3", "q3"),
                                    ("Q2", "q2")
                                ],
                                value="q3",
                                visible=True,
                                label="Qwen"
                            )
                            zimage_version = gr.Radio(
                                choices=[
                                    ("Q4 (6GB)", "q4"),
                                    ("Q5 (8GB)", "q5")
                                ],
                                value="q4",
                                visible=True,
                                label="Z-Image"
                            )
                num_var = gr.Slider(1, 8, 1, step=1, label="Variaciones")
                quality = gr.Radio(
                    ["fast", "balanced", "high"], 
                    value="fast", 
                    label="Velocidad",
                    info="Fast=8 pasos | Balanced=12 | High=20"
                )
                res = gr.Radio(["480p", "720p", "1024p"], value="720p", label="Resolución")
                with gr.Row():
                    f_preserve = gr.Checkbox(label="Preservar Cara", value=True)
                    a_enhance = gr.Checkbox(label="Auto-Mejorar", value=True)

            with gr.Row():
                gen_btn = gr.Button("🎨 Generar Imágenes", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Limpiar Todo", variant="stop")
            
            status = gr.Textbox(label="Estado del Sistema", interactive=False)
            metrics = gr.HTML(value=get_metrics_html(0, "0/0", "--:--", "Listo para empezar"))

        with gr.Column(scale=2):
            gallery = gr.Gallery(label="Galería de Resultados", columns=2, height=750, preview=True)
            with gr.Row():
                btn_open = gr.Button("📂 Abrir Carpeta de Salida"); btn_input = gr.Button("🔄 Usar como Nuevo Input")

    # --- Lógica de Callbacks ---
    
    # Batch Toggle
    batch_mode.change(lambda c: {input_img: gr.update(visible=not c), batch_files: gr.update(visible=c)}, [batch_mode], [input_img, batch_files])
    
    # Texto Toggle
    add_text.change(lambda c: {t_in: gr.update(visible=c), t_p: gr.update(visible=c), t_s: gr.update(visible=c)}, [add_text], [t_in, t_p, t_s])
    
    # Enhance Prompt
    enhance_btn.click(on_enhance_prompt, [prompt], [status, prompt_enh])
    
    # Multi-Ref Handler
    def on_ref_change(i1, i2, i3):
        refs = [x for x in [i1, i2, i3] if x is not None]
        return refs, f"📚 {len(refs)} referencias cargadas"
    r1.upload(on_ref_change, [r1, r2, r3], [ref_list_state, status])
    r2.upload(on_ref_change, [r1, r2, r3], [ref_list_state, status])
    r3.upload(on_ref_change, [r1, r2, r3], [ref_list_state, status])

    # GENERACIÓN PRINCIPAL
    gen_btn.click(
        on_generate,
        [input_img, prompt, num_var, quality, f_preserve, a_enhance, res, char_ref_img, ref_list_state, use_char_ref, add_text, t_in, t_p, t_s, batch_mode, batch_files, engine, prompt_enh, qwen_version, zimage_version],
        [gallery, status, metrics]
    )
    
    # Utilidades
    btn_open.click(lambda: os.startfile(COMFYUI_OUTPUT_DIR) if os.name=="nt" else None)
    btn_input.click(lambda imgs: imgs[0] if imgs else None, [gallery], [input_img])
    clear_btn.click(lambda: (None, "", None, None, None, None, None, ""), outputs=[input_img, prompt, char_ref_img, r1, r2, r3, gallery, status])

def img_editor_tab():
    create_img_editor_tab()
