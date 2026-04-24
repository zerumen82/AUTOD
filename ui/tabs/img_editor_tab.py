#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Editor - Versión Simplificada y Limpia (2026)
"""

import gradio as gr
import os
import sys
import logging
import time
import tempfile
from PIL import Image
from roop.img_editor.img_editor_manager import get_img_editor_manager

# Añadir scripts al path para importar analizador
SCRIPTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'scripts')
sys.path.insert(0, os.path.abspath(SCRIPTS_DIR))
try:
    from image_analyzer_for_prompt import analyze_image_for_prompt
    HAS_ANALYZER = True
except ImportError:
    HAS_ANALYZER = False
    print("[WARN] No se pudo importar image_analyzer_for_prompt")


def create_maskable_image_input():
    """
    Compatibilidad simple - Si falla el modo dibujo, devuelve imagen normal
    """
    common_kwargs = {
        "label": "Imagen Original",
        "height": 450,
        "type": "pil"
    }
    
    # Intentar con ImageEditor (Gradio 5+) o Image con brush (Gradio 4.x)
    try:
        if hasattr(gr, "ImageEditor"):
            return gr.ImageEditor(**common_kwargs)
    except:
        pass
    
    # Fallback a Image normal
    return gr.Image(**common_kwargs)

def get_metrics_html(percent, variation, time_remaining, status):
    progress_color = "#3b82f6" if status not in ["Error", "Completado"] else ("#10b981" if status == "Completado" else "#ef4444")
    bar_color = "linear-gradient(90deg, #3b82f6, #10b981)" if status != "Error" else "linear-gradient(90deg, #ef4444, #f59e0b)"
    return f"""
    <div style="background: #111827; padding: 12px; border-radius: 8px; border: 1px solid #374151;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
            <span style="color: #9ca3af; font-size: 12px;">PROGRESO: <b style="color: {progress_color}">{percent:.1f}%</b></span>
            <span style="color: #9ca3af; font-size: 12px;">VARIACIÓN: <b>{variation}</b></span>
            <span style="color: #9ca3af; font-size: 12px;">ESTADO: <b style="color: #8b5cf6;">{status}</b></span>
        </div>
        <div style="background: #374151; border-radius: 4px; height: 6px; overflow: hidden;">
            <div style="width: {percent}%; height: 100%; background: {bar_color};"></div>
        </div>
    </div>
    """

def on_generate(img_data, p_text, creativity, preserve, steps, num_var, f_preserve, a_enhance, res_label, c_ref, use_c_ref, b_mode, b_files, engine_val, m_mode, m_prompt):
    # m_mode: "Global", "Manual 🖌️", "IA Inteligente 🤖"
    # img_data en Gradio 4.x/5.x con tool='sketch' es un dict
    
    img = None
    manual_mask = None
    
    if img_data is None:
        yield [], "Error: No hay imagen", get_metrics_html(0, "0/0", "0s", "Error")
        return

    if isinstance(img_data, dict):
        # Gradio 4.x/5.x format
        img = img_data.get("background")
        if m_mode == "manual" and "layers" in img_data and img_data["layers"]:
            # La máscara suele estar en la primera capa
            manual_mask = img_data["layers"][0]
    else:
        img = img_data

    # Fallback si por algún motivo img sigue siendo None tras el dict
    if img is None:
        yield [], "Error: Formato de imagen inválido", get_metrics_html(0, "0/0", "0s", "Error")
        return

    if b_mode:
        if not b_files: yield [], "Error: Sube imagenes", get_metrics_html(0, "0/0", "0s", "Error"); return
        work_images = [Image.open(f.name) for f in b_files]
    else:
        work_images = [img]

    manager = get_img_editor_manager()
    yield [], "Preparando...", get_metrics_html(0, "0/0", "0s", "Inicializando")
    
    guidance = 1.5 + (creativity * 2.0)
    denoise = 0.95 - (preserve * 0.65)
    denoise = max(0.2, min(0.95, denoise))
    
    ref_meta = {
        "character_ref": c_ref if use_c_ref else None,
        "resolution_label": res_label,
        "guidance_scale": guidance,
        "denoise": denoise
    }
    
    results = []
    total_ops = len(work_images) * int(num_var)
    op_idx = 0

    for img_idx, current_img in enumerate(work_images):
        for var_idx in range(int(num_var)):
            pct = (op_idx / total_ops) * 100
            yield results, f"Procesando...", get_metrics_html(pct, f"{op_idx+1}/{total_ops}", "...", f"Engine: {engine_val}")

            res_img, msg = manager.generate_intelligent(
                image=current_img, prompt=p_text, num_inference_steps=steps,
                face_preserve=f_preserve, auto_enhance=a_enhance, ref_metadata=ref_meta, 
                engine=engine_val,
                mask_image=manual_mask,
                mask_mode=m_mode,
                mask_prompt=m_prompt
            )

            if res_img: 
                results.append(res_img)
                op_idx += 1
                yield results, "OK", get_metrics_html((op_idx/total_ops)*100, f"{op_idx}/{total_ops}", "...", "Listo")
            else:
                op_idx += 1
                yield results, f"Fallo: {msg}", get_metrics_html((op_idx/total_ops)*100, f"{op_idx}/{total_ops}", "...", "Error")

    yield results, f"Completado", get_metrics_html(100, f"{len(results)}/{total_ops}", "...", "Completado")

def create_img_editor_tab():
    with gr.Row():
        with gr.Column(scale=1.1):
            # CANVAS DE INPAINTING PRO
            input_img = create_maskable_image_input()
            
            with gr.Row(variant="compact"):
                mask_mode = gr.Radio(
                    choices=[("Todo", "global"), ("Pintar 🖌️", "manual"), ("IA (CLIPSeg) 🤖", "smart")],
                    value="global",
                    label="🎯 ¿Qué modificar?"
                )
                btn_preview_mask = gr.Button("👁️ Ver Máscara", visible=False, scale=1)
                mask_prompt = gr.Textbox(
                    label="IA: ¿Qué objeto?", 
                    placeholder="ej: camisa, gafas, fondo...",
                    visible=False,
                    scale=2
                )
            
            # Prompt row con botón de análisis
            with gr.Row():
                prompt = gr.Textbox(label="Instrucción de Edición", placeholder="Ej: cámbiale la ropa por un vestido rojo...", lines=3, scale=4)
                btn_analyze = gr.Button("🔍 Sugerir Prompt", scale=1)
            
            # Negative prompt (opcional, se puede expandir)
            negative_prompt = gr.Textbox(
                label="Negative Prompt (opcional)", 
                placeholder="Dejar vacío para默认...",
                lines=2,
                value="clothed, dressed, safe, censored, blurry, low quality, bad anatomy, deformed, ugly, watermark, text, logo, children, minor, underage, worst quality"
            )
            
            with gr.Row():
                engine = gr.Dropdown(
                    choices=[
                        ("FLUX.2-klein (Editor, 4B)", "flux_klein"),
                        ("FLUX.1-schnell (Editor, 4 pasos)", "flux_schnell"),
                        ("HART (Generador puro, 512px)", "hart"),
                        ("OmniGen2 (Editor, 2B, 6GB)", "omnigen2")
                    ],
                    value="flux_klein", label="Motor IA"
                )
                res = gr.Radio(["512p", "720p", "1024p"], value="1024p", label="Resolución")
            
            params_info = gr.HTML(value="<small style='color:#6b7280;'>Editores: usa Preservar/Creatividad | Generador: usa solo Guidance</small>")
            with gr.Row():
                # Slider creativo: 0 = conservador, 1 = muy creativo
                creativity = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="Creatividad",
                    info="0=Fiel a original, 1=Mas cambios"
                )
                # Slider denoise: cuánto respeta la imagen original
                preserve = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.7, step=0.05,
                    label="Preservar original",
                    info="0=Todo nuevo, 1=Igual al original"
                )

            with gr.Row():
                steps = gr.Slider(
                    minimum=4, maximum=20, value=8, step=1,
                    label="Pasos de calidad",
                    info="4-8 rapido, 20 mas calidad (FLUX)"
                )
                num_var = gr.Number(value=1, label="Variaciones", precision=0)

            with gr.Row():
                f_preserve = gr.Checkbox(label="Restaurar Cara", value=True)
                a_enhance = gr.Checkbox(label="Mejorar Prompt", value=True)

            # Guía completa en accordion (plegada por defecto para no saturar)
            with gr.Accordion("📖 Guía de uso (sliders, ejemplos, consejos)", open=False):
                gr.Markdown("""
### 🎯 Sliders (ajustes libres)
| Control | Rango | Qué hace |
|---|---|---|
| **Preservar original** | 0.0 - 1.0 | Cuánto mantiene la imagen base. **Alto** (0.7-1.0) = fondo/pose intactos. **Bajo** (0.1-0.3) = regenera casi todo. |
| **Creatividad** | 0.0 - 1.0 | Libertad del modelo. Alto (0.7-1.0) = más cambios, Bajo (0-0.3) = fiel al prompt. |
| **Pasos** | 4 - 20 | Calidad vs velocidad. FLUX: 8-12 óptimo en 8GB. Más pasos → más lento, poca mejora. |

### 📝 Ejemplos prácticos
- **Cambiar ropa/color** (mantener fondo):  
  `Preservar original`=0.7-0.9 | `Creatividad`=0.2-0.4 | Pasos=8 | Prompt: *"camisa azul, manteniendo el fondo"*
- **Cambiar cara/peinado** (entorno similar):  
  `Preservar`=0.5-0.7 | `Creatividad`=0.4-0.6 | Pasos=10-12 | Prompt: *"pelo más oscuro, expresión sonriendo"*
- **🔞 Desnudar 1 persona**:  
  `Preservar`=0.15-0.2 | `Creatividad`=0.8-0.9 | Pasos=10-12 | Prompt: *"completely naked, nude, no clothing, explicit nsfw"*
- **👥 Desnudar GRUPAL (4+ personas)**:  
  `Preservar`=0.1-0.15 | `Creatividad`=0.9 | Pasos=12 | Prompt: *"4 naked people, 1 man and 3 women, explicit nsfw, full body"*
- **Cambiar pose** (sentarse, girar):  
  `Preservar`=0.1-0.3 | `Creatividad`=0.7-0.8 | Pasos=12-16 | Prompt: *"sentado en una silla"*, *"girando 90°"*
- **Escena nueva** (reemplazar todo):  
  `Preservar`=0.0-0.1 | `Creatividad`=0.9-1.0 | Pasos=16-20 | Prompt: *"astronauta en el espacio"*

### ⚙️ Motores
- **FLUX.2-klein (4B) + LoRA NSFW**: ✅ Edita sin censura. Calidad alta. 8-12 pasos → 10-20 min. **Recomendado para NSFW**.
- **HART (Generador puro)**: ⚠️ NO edita. Generación autoregresiva, no edición. Genera imagen NUEVA inspirada en el prompt (512px).
- **OmniGen2**: Rápido (~1-2 min), calidad moderada. Tiene safe filters.

### 🔄 Restaurar Cara (Face Preserve)
- **Activado**: Restaura **TODAS las caras** detectadas en la imagen original → identidad 100% preservada para cada persona.
  - Usar con: **NSFW individual** y **Grupal (4+ personas)**.
  - Requiere cara detectable en imagen original para cada persona.
- **Desactivado**: FLUX genera caras nuevas (pueden ser otras personas).

### 💡 Consejos para FOTOS GRUPALES (4+ personas)
1. **Preservar MUY BAJO** (0.1-0.15): así FLUX regenera completamente los cuerpos/ropa.
2. **Creatividad ALTA** (0.85-0.9): para que siga el prompt "naked people" sin bloquear.
3. **Prompt explícito**: menciona número y género: *"4 naked people, 1 man and 3 women, explicit nsfw"*
4. **Restaurar Cara ACTIVADO**: restaurará automáticamente las 4 caras originales (o tantas como detecte).
5. Si alguna cara **no se restaura**: verifica que esté bien detectada en el original (buena resolución, frontal).
6. **Pasos 12**: suficiente calidad sin ser excesivamente lento.

### 🛠️ Solución de problemas
- **FLUX genera ropa aún con LoRA**: 
  1. Asegúrate de que `NSFW_MASTER_FLUX.safetensors` está en `models/loras/`
  2. Usa `Preservar` ≤ 0.2
  3. Prompt con palabras clave: `nude, naked, no clothing, explicit`
- **Solo restaura 1 cara (no 4)**: versión antigua — actualiza a la última versión del código (multi-face ya está implementado).
- **HART falla (OOM)**: HART necesita 10-12GB para 1024px. En 8GB usa solo FLUX.
""", elem_classes=["help-text"])

            with gr.Accordion("[TOOLS] Opciones Avanzadas", open=False):
                with gr.Tab("[BATCH] Batch"):
                    batch_mode = gr.Checkbox(label="Activar Procesamiento por Lotes", value=False)
                    batch_files = gr.File(label="Imágenes", file_count="multiple")
                with gr.Tab("[CHAR] Personaje"):
                    char_ref_img = gr.Image(label="Referencia de Rostro", type="pil")
                    use_char_ref = gr.Checkbox(label="Usar esta cara", value=False)

            gen_btn = gr.Button("[IMG] GENERAR CAMBIOS", variant="primary", size="lg")
            metrics = gr.HTML(value=get_metrics_html(0, "0/0", "0s", "Listo"))
            status = gr.Textbox(label="Estado", interactive=False)

        with gr.Column(scale=1.5):
            gallery = gr.Gallery(label="Resultados", columns=2, height=600, preview=True)
            
            # COMPARADOR VISUAL (Cortina)
            with gr.Accordion("🌓 Comparar Antes/Después", open=False):
                with gr.Row():
                    compare_slider = gr.Slider(minimum=0, maximum=100, value=50, label="Desliza para comparar", info="Izquierda: Original | Derecha: Editada")
                preview_compare = gr.Image(label="Vista de Comparación", interactive=False)

            with gr.Row():
                gr.Button("[OPEN] Abrir Carpeta").click(lambda: os.startfile(os.path.abspath("output/img_editor")))
                
                def use_as_input(gallery_data):
                    if gallery_data and len(gallery_data) > 0:
                        item = gallery_data[0]
                        if isinstance(item, dict): return item['name']
                        if isinstance(item, (list, tuple)): return item[0]
                    return None

                gr.Button("[RELOAD] Usar como Input").click(
                    fn=use_as_input,
                    inputs=[gallery],
                    outputs=[input_img]
                )

    # --- EVENTOS DE MÁSCARA ---
    def on_mask_mode_change(mode):
        is_smart = (mode == "smart")
        return gr.update(visible=is_smart), gr.update(visible=is_smart)

    mask_mode.change(
        fn=on_mask_mode_change, 
        inputs=[mask_mode], 
        outputs=[mask_prompt, btn_preview_mask]
    )

    def preview_mask_click(img_data, m_prompt, m_mode):
        img = None
        manual_mask = None
        
        if isinstance(img_data, dict):
            img = img_data.get("background")
            manual_mask = img_data.get("layers")[0] if img_data.get("layers") else None
        else:
            img = img_data

        if not img: return None, "Sube una imagen primero"
        
        manager = get_img_editor_manager()
        res_mask, msg = manager.preview_smart_mask(
            image=img, 
            mask_prompt=m_prompt, 
            mask_image=manual_mask, 
            mask_mode=m_mode
        )
        return res_mask, msg

    btn_preview_mask.click(
        fn=preview_mask_click,
        inputs=[input_img, mask_prompt, mask_mode],
        outputs=[preview_compare, status]
    )

    # --- GENERACIÓN ---
    gen_btn.click(
        on_generate,
        [input_img, prompt, creativity, preserve, steps, num_var, f_preserve, a_enhance, res, char_ref_img, use_char_ref, batch_mode, batch_files, engine, mask_mode, mask_prompt],
        [gallery, status, metrics]
    )

    # --- LÓGICA DE COMPARACIÓN ---
    def update_comparison(original, results, split_pct):
        if not original or not results or len(results) == 0:
            return None
        
        # Obtener imágenes PIL
        orig = original if not isinstance(original, dict) else original.get("background")
        if not orig: return None
        
        gen = results[0] # Comparamos con el primer resultado
        
        # Asegurar mismo tamaño
        if orig.size != gen.size:
            gen = gen.resize(orig.size, Image.LANCZOS)
        
        # Crear imagen compuesta (cortina)
        w, h = orig.size
        split_x = int(w * (split_pct / 100))
        
        comp = Image.new("RGB", (w, h))
        comp.paste(orig.crop((0, 0, split_x, h)), (0, 0))
        comp.paste(gen.crop((split_x, 0, w, h)), (split_x, 0))
        
        # Añadir línea separadora
        import PIL.ImageDraw as ImageDraw
        draw = PIL.ImageDraw.Draw(comp)
        draw.line([(split_x, 0), (split_x, h)], fill="white", width=4)
        
        return comp

    compare_slider.change(
        fn=update_comparison,
        inputs=[input_img, gallery, compare_slider],
        outputs=[preview_compare]
    )

    def on_engine_change(engine_val):
        if engine_val == "hart":
            return gr.update(visible=False), gr.update(value=0.5, label="Guidance", info="0=Consistente, 1=Más creativo")
        else:
            return gr.update(visible=True), gr.update(value=0.7, label="Preservar original", info="0=Todo nuevo, 1=Igual al original")

    engine.change(fn=on_engine_change, inputs=[engine], outputs=[preserve, creativity])

# === FUNCIÓN DE ANÁLISIS DE IMAGEN PARA PROMPT ===
    def analyze_image_click(img):
        if img is None:
            return "ERROR: No hay imagen cargada. Sube una imagen primero."
        
        if not HAS_ANALYZER:
            return "ERROR: Módulo de análisis no disponible."
        
        # Handle Gradio 4.x/5.x ImageEditor dict format
        if isinstance(img, dict):
            img = img.get("background")
        
        if img is None:
            return "ERROR: No se pudo extraer la imagen."
        
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
                img.save(tmp.name)
                tmp_path = tmp.name
            
            from image_analyzer_for_prompt import ImageAnalyzer
            analyzer = ImageAnalyzer()
            result = analyzer.generate_full_prompt(tmp_path, nsfw_level='explicit')
            
            try:
                os.unlink(tmp_path)
            except:
                pass
            
            generated_prompt = result['positive']
            
            num_people = result['analysis'].get('num_people', 0)
            if num_people > 0:
                print(f"[ANALIZADOR] Detectadas {num_people} personas")
                for i, face in enumerate(result['analysis']['faces']):
                    g = face.get('gender', 'desconocido')
                    a = face.get('age', '?')
                    print(f"  - Persona {i+1}: {g}, {a} años")
            
            return generated_prompt
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return f"ERROR: {str(e)}"
    
    btn_analyze.click(
        fn=analyze_image_click,
        inputs=[input_img],
        outputs=[prompt]
    )
