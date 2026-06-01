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
from roop.comfy_client import ComfyClient

def get_available_loras():
    """Obtiene la lista de LoRAs desde ComfyUI"""
    try:
        client = ComfyClient()
        loras = client.get_loras()
        if not loras:
            return ["None"]
        return ["None"] + sorted(loras)
    except:
        return ["None"]

def check_engine_status():
    """Verifica si el motor de generación (ComfyUI) está activo"""
    client = get_flux_gen_client()
    if client.is_available():
        return "🟢 Motor Longshot Listo"
    return "🔴 Motor Detenido"

def set_orientation(orientation):
    if orientation == "Vertical (Retrato)":
        return 768, 1024
    elif orientation == "Horizontal (Paisaje)":
        return 1024, 768
    elif orientation == "Cuadrado":
        return 768, 768
    return 512, 768

def update_model_params(engine_val):
    """Actualiza los sliders según el modelo seleccionado cargando desde JSON"""
    try:
        from roop.img_editor.flux_gen_comfy_client import get_project_root
        path = os.path.join(get_project_root(), "config", "model_configs.json")
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                conf = json.load(f)
                m_conf = conf.get(engine_val, conf.get("default", {}))
                return m_conf.get("steps", 20), m_conf.get("cfg", 3.5)
    except:
        pass
    
    # Fallback por si falla el JSON
    if "sdxl" in engine_val or "miamodel" in engine_val or "nova" in engine_val or "lazy" in engine_val:
        return 25, 7.0
    return 20, 3.5

def on_generate_image(prompt, negative_prompt, steps, cfg_scale, width, height, use_ai, engine_val, lora_name, lora_strength, manual_prompt, use_override):
    client = get_flux_gen_client()
    if not client.is_available():
        return None, "❌ El motor de IA no está activo. Inícialo en los controles de arriba."

    try:
        # Cargar el modelo seleccionado
        success, load_msg = client.load(engine_val)
        if not success:
            return None, f"❌ Error cargando modelo: {load_msg}"

        # Si hay override, forzamos el prompt y saltamos reescritura
        skip_rewrite = False
        final_p = prompt
        if use_override and manual_prompt:
            final_p = manual_prompt
            skip_rewrite = True

        # Si usa IA, los sliders se ignoran (se pasan como None para que el cliente decida)
        final_steps = steps if not use_ai else None
        final_cfg = cfg_scale if not use_ai else None

        if use_ai:
            res, msg = client.generate_ai(
                prompt=final_p,
                negative_prompt=negative_prompt,
                steps=final_steps,
                guidance_scale=final_cfg,
                width=width,
                height=height,
                use_ai=use_ai,
                lora_name=lora_name,
                lora_strength=lora_strength,
                _skip_rewrite=skip_rewrite
            )
        else:
            res, msg = client.generate(
                prompt=final_p,
                negative_prompt=negative_prompt,
                steps=final_steps,
                guidance_scale=final_cfg,
                width=width,
                height=height,
                use_ai=use_ai,
                lora_name=lora_name,
                lora_strength=lora_strength,
                _skip_rewrite=skip_rewrite
            )
        
        if res:
            output_dir = os.path.abspath("output/generation")
            os.makedirs(output_dir, exist_ok=True)
            out_path = os.path.join(output_dir, f"gen_{int(time.time())}.png")
            res.image.save(out_path)
            
            status_msg = f"✅ Generación completada ({res.time_taken:.1f}s)<br><br>"
            status_msg += f"<div style='font-size:11px; color:#94a3b8; background:#1e293b; padding:8px; border-radius:6px; border:1px solid #334155;'>"
            status_msg += f"<b>Prompt Final IA:</b><br>{res.final_prompt}</div>"
            
            return res.image, status_msg
        else:
            return None, f"❌ Error: {msg}"
    except Exception as e:
        import traceback
        traceback.print_exc()
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
            .info-btn {
                display: inline-flex; align-items: center; justify-content: center;
                width: 22px; height: 22px; border-radius: 50%;
                background: #a855f7; color: white; cursor: pointer;
                font-size: 14px; font-weight: bold; margin-left: 8px;
                border: none; line-height: 1;
            }
            .info-btn:hover { background: #9333ea; }
            .info-popup {
                display: none; position: absolute; z-index: 999;
                background: #1e293b; border: 1px solid #a855f7;
                border-radius: 12px; padding: 16px; max-width: 400px;
                font-size: 13px; color: #e2e8f0; line-height: 1.5;
                box-shadow: 0 8px 32px rgba(0,0,0,0.5); margin-top: 4px;
            }
            .info-popup.show { display: block; }
            .info-popup strong { color: #a855f7; }
            .info-wrap { position: relative; display: inline-block; }
        </style>
    """)
    
    with gr.Column(elem_classes=["gen-tab-container"]):
        with gr.Group(elem_classes=["gen-tab-header"]):
            gr.Markdown("## 🚀 AI GENERATOR")
            gr.Markdown("_Crea imágenes realistas desde cero usando el motor de alta fidelidad._")

        with gr.Row():
            # COLUMNA DE ENTRADA
            with gr.Column(scale=1):
                
                gr.HTML("""
                <div style="display:flex; align-items:center; gap:4px; margin-bottom:4px;">
                    <label style="color: #e2e8f0; font-size: 14px; font-weight: 600;">¿Qué quieres crear?</label>
                    <div class="info-wrap">
                        <button class="info-btn" onclick="var p=this.nextElementSibling; var isOpen=p.classList.contains('show'); document.querySelectorAll('.info-popup').forEach(function(x){x.classList.remove('show')}); if(!isOpen) { p.classList.add('show'); setTimeout(function(){ p.classList.remove('show'); }, 15000); } event.stopPropagation();">i</button>
                        <div class="info-popup">
                            <strong>Estructura recomendada:</strong><br>
                            <em>[persona], [descripción física], [acción], [entorno], [iluminación/ambiente]</em><br><br>
                            <strong>Ejemplo:</strong><br>
                            mujer de 40 años, delgada, pelo negro liso, piel muy clara, arrodillada en un baño, chupando polla, expresión de sumisión, iluminación natural<br><br>
                            <strong>Consejos:</strong><br>
                            • Sé específico (posición, entorno, expresión)<br>
                            • Describe iluminación (natural, oscuro, cine)<br>
                            • Los tags de calidad se añaden automáticamente<br>
                            • El modelo LongCat Full no tiene filtros
                        </div>
                    </div>
                </div>
                <script>
                document.addEventListener('click', function() {
                    document.querySelectorAll('.info-popup').forEach(function(p) { p.classList.remove('show'); });
                });
                </script>
                """)

                prompt = gr.Textbox(
                    label="",
                    placeholder="Ej: mujer de 40 años, delgada, pelo negro liso, arrodillada en un baño, chupando polla, iluminación natural",
                    lines=4,
                    elem_classes=["prompt-box-gen"]
                )

                with gr.Accordion("👁️ Edición del Prompt Final (Control Total)", open=True):
                    final_prompt_preview = gr.Textbox(
                        label="Prompt que recibirá el motor (Puedes editarlo aquí directamente)",
                        lines=3,
                        interactive=True
                    )
                    with gr.Row():
                        use_manual_override = gr.Checkbox(label="Usar este prompt editado manualmente (Ignorar el original)", value=False)
                        auto_update_preview = gr.Checkbox(label="Actualizar automáticamente mientras escribo", value=True)

                neg_prompt = gr.Textbox(
                    label="Evitar (Prompt Negativo)",
                    value="blurry, distorted, low quality, bad anatomy, text, watermark",
                    lines=2
                )

                with gr.Row():
                    gen_btn = gr.Button("🚀 GENERAR IMAGEN", variant="primary", elem_classes=["btn-generate-main"], scale=1)

                with gr.Accordion("⚙️ Ajustes Avanzados", open=False):
                    engine_model = gr.Dropdown(
                        choices=[
                            ("Miamodel SDXL (NSFW Realista - Rápido)", "miamodel_nsfw"),
                            ("FLUX.1 Dev Abliterated (PRO Uncensored)", "flux_dev_abliterated"),
                            ("LongCat Full (Realista + Sin Filtros)", "longcat_full"),
                            ("Nova Illustrous (NSFW Artístico - Rápido)", "nova_nsfw"),
                            ("Realistic Lazy Mix (NSFW Balanceado)", "lazy_nsfw"),
                            ("FLUX.2 Klein (Ultra Realista - Restrictivo)", "klein_base"),
                            ("LongCat Turbo (Rápido)", "longcat"),
                            ("FLUX.1 Dev Q2 (Bajo VRAM)", "flux_q2")
                        ],
                        value="miamodel_nsfw",
                        label="Motor de Generación"
                    )
                    gr.Markdown("<small>🔥 **Uncensored Pro:** Usa 'Abliterated' para máxima libertad en Flux, o los modelos 'SDXL' para velocidad y NSFW real en 8GB.</small>")
                    
                    orientation = gr.Radio(
                        choices=["Vertical (Retrato)", "Horizontal (Paisaje)", "Cuadrado", "Personalizado"],
                        value="Vertical (Retrato)",
                        label="Orientación"
                    )
                    
                    with gr.Row():
                        width = gr.Slider(minimum=256, maximum=1280, step=64, value=768, label="Ancho")
                        height = gr.Slider(minimum=256, maximum=1280, step=64, value=1024, label="Alto")
                    
                    with gr.Row():
                        steps = gr.Slider(minimum=1, maximum=50, step=1, value=25, label="Pasos")
                        cfg = gr.Slider(minimum=1, maximum=15, step=0.5, value=7.0, label="CFG Scale")
                    
                    with gr.Row():
                        lora_dropdown = gr.Dropdown(
                            choices=get_available_loras(),
                            value="None",
                            label="LoRA (Mejora de Estilo/Personaje)",
                            scale=2
                        )
                        lora_strength = gr.Slider(minimum=-2.0, maximum=2.0, step=0.05, value=1.0, label="Fuerza LoRA", scale=1)
                    
                    bt_refresh_loras = gr.Button("🔄 Refrescar LoRAs", size="sm")
                    bt_refresh_loras.click(fn=lambda: gr.update(choices=get_available_loras()), outputs=[lora_dropdown])

                    # Conectar cambio de modelo a actualización de sliders
                    engine_model.change(
                        fn=update_model_params,
                        inputs=[engine_model],
                        outputs=[steps, cfg]
                    )
                    
                    use_ai_logic = gr.Checkbox(
                        label="🧠 Análisis Inteligente (IA)",
                        value=True,
                        info="Detecta automáticamente los parámetros ideales y mejora el realismo según tu prompt."
                    )

                # Conectar la vista previa después de definir engine_model y use_ai_logic
                def update_prompt_preview(p, ai_enabled, model_val, auto_upd, manual_on, current_val):
                    # Si el auto-update está apagado o el manual override está encendido, no tocamos el cuadro
                    if not auto_upd or manual_on:
                        return current_val
                    
                    if not p or len(p) < 3: return current_val
                    
                    from roop.img_editor.flux_gen_comfy_client import get_flux_gen_client
                    client = get_flux_gen_client()
                    # Simular la preparación del prompt
                    if ai_enabled:
                        final, _ = client._prepare_prompt_intelligent(p)
                    else:
                        # Si no hay IA, necesitamos el alias del modelo para saber si es Longcat etc.
                        client.load(model_val)
                        final = client._prepare_prompt(p)
                    return final

                prompt.change(
                    fn=update_prompt_preview,
                    inputs=[prompt, use_ai_logic, engine_model, auto_update_preview, use_manual_override, final_prompt_preview],
                    outputs=[final_prompt_preview],
                    show_progress="hidden"
                )
                use_ai_logic.change(fn=update_prompt_preview, inputs=[prompt, use_ai_logic, engine_model, auto_update_preview, use_manual_override, final_prompt_preview], outputs=[final_prompt_preview])
                engine_model.change(fn=update_prompt_preview, inputs=[prompt, use_ai_logic, engine_model, auto_update_preview, use_manual_override, final_prompt_preview], outputs=[final_prompt_preview])

                status_html = gr.HTML("<div style='text-align:center; color:#64748b; padding:10px;'>Listo</div>")

            # COLUMNA DE SALIDA
            with gr.Column(scale=1):
                output_img = gr.Image(label="Imagen Generada", height=600)
                
                with gr.Row():
                    bt_open_folder = gr.Button("📂 ABRIR SALIDA")
                    bt_open_folder.click(fn=open_generation_folder)

    orientation.change(
        fn=set_orientation,
        inputs=[orientation],
        outputs=[width, height]
    )

    gen_btn.click(
        fn=lambda: (None, gr.update(value="<div style='text-align:center; color:#a855f7; padding:10px; font-weight:bold;'>🚀 Generando imagen... (esto puede tardar 2-3 min)</div>")),
        outputs=[output_img, status_html]
    ).then(
        on_generate_image,
        [prompt, neg_prompt, steps, cfg, width, height, use_ai_logic, engine_model, lora_dropdown, lora_strength, final_prompt_preview, use_manual_override],
        [output_img, status_html],
        concurrency_limit=None
    )

    return {
        "prompt": prompt, "gen_btn": gen_btn, "output_img": output_img, "status": status_html
    }
