#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ImgEditor Tab - Editor de imagenes con Stable Diffusion via ComfyUI
"""

import gradio as gr
import os
import subprocess
import sys
from PIL import Image
import requests
import socket
import logging
from roop.img_editor.img_editor_manager import get_img_editor_manager

# Directorio de salida de ComfyUI
COMFYUI_OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tob", "ComfyUI", "output"))

# Puerto de ComfyUI - intentara detectar automaticamente
DETECTED_COMFYUI_PORT = None


def detect_comfyui_port():
    """Detecta automaticamente el puerto de ComfyUI"""
    possible_ports = [8188, 8189, 8190, 8888, 8000]
    
    for port in possible_ports:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/system_stats", timeout=1)
            if response.status_code == 200:
                return port
        except:
            continue
    return None


def get_comfyui_url():
    """Obtiene la URL de ComfyUI, detectandola si es necesario"""
    global DETECTED_COMFYUI_PORT
    
    if DETECTED_COMFYUI_PORT:
        return f"http://127.0.0.1:{DETECTED_COMFYUI_PORT}"
    
    detected = detect_comfyui_port()
    if detected:
        DETECTED_COMFYUI_PORT = detected
        return f"http://127.0.0.1:{detected}"
    
    default_port = os.environ.get('COMFYUI_PORT', '8188')
    return f"http://127.0.0.1:{default_port}"


def open_output_folder():
    """Abre la carpeta de salida de ComfyUI"""
    try:
        if os.name == 'nt':
            os.startfile(COMFYUI_OUTPUT_DIR)
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', COMFYUI_OUTPUT_DIR])
        else:
            subprocess.Popen(['xdg-open', COMFYUI_OUTPUT_DIR])
        return f"Abriendo: {COMFYUI_OUTPUT_DIR}"
    except Exception as e:
        return f"Error: {str(e)}"


def check_comfy_status():
    """Verifica si ComfyUI esta corriendo y tiene modelos"""
    try:
        comfy_url = get_comfyui_url()
        response = requests.get(f"{comfy_url}/system_stats", timeout=2)
        if response.status_code != 200:
            return "No conectado"
        
        response2 = requests.get(f"{comfy_url}/object_info", timeout=5)
        if response2.status_code == 200:
            data = response2.json()
            if "CheckpointLoaderSimple" in data:
                node = data["CheckpointLoaderSimple"]
                if "input" in node and "required" in node["input"]:
                    ckpt_list = node["input"]["required"].get("ckpt_name")
                    if ckpt_list and len(ckpt_list) > 0 and len(ckpt_list[0]) > 0:
                        return f"Listo ({len(ckpt_list[0])} modelos)"
        return "Conectado (cargando...)"
    except Exception as e:
        return "No conectado"


def check_models_status():
    """Verifica qué modelos están disponibles (FLUX, ComfyUI, ControlNet, IP-Adapter, CLIPSeg)"""
    try:
        from roop.img_editor.img_editor_manager import get_img_editor_manager
        manager = get_img_editor_manager()
        models = manager._check_models_available()
        
        status_parts = []
        if models.get("FLUX"):
            status_parts.append("✅ FLUX")
        if models.get("ComfyUI"):
            status_parts.append("✅ ComfyUI")
        if models.get("ControlNet"):
            status_parts.append("✅ ControlNet")
        if models.get("IP-Adapter"):
            status_parts.append("✅ IP-Adapter")
        if models.get("CLIPSeg"):
            status_parts.append("✅ CLIPSeg")
        
        if status_parts:
            return " | ".join(status_parts)
        else:
            return "❌ Ningún modelo disponible"
    except Exception as e:
        return f"Error: {str(e)}"


def create_img_editor_tab():
    """Crea la interfaz del tab ImgEditor"""
    
    manager = get_img_editor_manager()
    
    gr.Markdown("**Editor de Imagenes - Describe los cambios que quieres**")
    
    with gr.Row():
        comfy_status = gr.Textbox(
            value="No conectado",
            interactive=False,
            label="Estado ComfyUI",
            scale=3
        )
        models_status = gr.Textbox(
            value="No verificado",
            interactive=False,
            label="Modelos",
            scale=3
        )
        btn_refresh_status = gr.Button("Refrescar", size="sm", scale=1)
    
    current_image = gr.State(value=None)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Imagen")
            
            input_image = gr.Image(
                label="Sube una imagen",
                sources=["upload", "clipboard"],
                type="pil",
                height=300,
            )
            
            gr.Markdown("### Prompt")
            
            prompt = gr.Textbox(
                label="Describe los cambios",
                placeholder="Ej: 'mujer desnuda con cuerpo realista, high quality, detailed'",
                lines=3,
            )
            
            negative_prompt = gr.Textbox(
                label="Negative Prompt",
                placeholder="Lo que NO quieres...",
                lines=2,
                value="low quality, blurry, distorted, bad anatomy, ugly, deformed, child, underage, minor, censored",
            )
            
            with gr.Row():
                advanced_mode = gr.Checkbox(
                    label="⚙️ Modo Avanzado",
                    value=False,
                    elem_id="checkbox_advanced_mode",
                    info="Desactiva para usar presets optimizados (recomendado)"
                )
            
            with gr.Column(visible=True) as basic_panel:
                gr.Markdown("**⚡ Modo Rápido - Presets Optimizados**")
                
                edit_mode = gr.Dropdown(
                    label="Tipo de Edición",
                    choices=[
                        "Cambiar Ropa/Cuerpo",
                        "Cambiar Entorno/Fondo",
                        "Retoques Sutiles"
                    ],
                    value="Cambiar Ropa/Cuerpo",
                    elem_id="dropdown_edit_mode",
                    info="Selecciona qué tipo de edición quieres hacer"
                )
                
                quality_preset = gr.Dropdown(
                    label="Calidad",
                    choices=["fast", "balanced", "high_quality", "max_quality"],
                    value="balanced",
                    elem_id="dropdown_quality_preset",
                    info="fast: rápido, balanced: equilibrado, high_quality: alta calidad, max_quality: máxima"
                )
                
                with gr.Row():
                    face_preserve = gr.Checkbox(
                        label="Preservar Cara",
                        value=True,
                        elem_id="checkbox_face_preserve",
                        info="Restaura la cara original después de generar"
                    )
                    
                    use_flux = gr.Checkbox(
                        label="Usar FLUX (más rápido, mejor calidad)",
                        value=True,
                        elem_id="use_flux",
                        info="Activa para usar FLUX Fill Pipeline (si está disponible)"
                    )
                
                with gr.Row():
                    auto_detect_clothing = gr.Checkbox(
                        label="🎯 Detección Automática de Ropa (CLIPSeg)",
                        value=True,
                        elem_id="auto_detect_clothing",
                        info="Detecta ropa automáticamente y aplica inpaint SOLO en esas áreas"
                    )
                
                with gr.Column(visible=True) as mask_controls:
                    with gr.Row():
                        mask_threshold = gr.Slider(
                            label="Sensibilidad de Detección",
                            minimum=0.2,
                            maximum=0.8,
                            value=0.5,
                            step=0.05,
                            elem_id="slider_mask_threshold",
                            info="0.2: detecta más (puede incluir piel), 0.5: balance, 0.8: solo ropa obvia"
                        )
                        
                        mask_dilation = gr.Slider(
                            label="Dilatar Máscara",
                            minimum=0,
                            maximum=30,
                            value=6,
                            step=1,
                            elem_id="slider_mask_dilation",
                            info="Píxeles a expandir la máscara (0: exacto, 15: cubre más)"
                        )
                    
                    with gr.Row():
                        inpaint_denoise = gr.Slider(
                            label="Fuerza Inpaint",
                            minimum=0.7,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            elem_id="slider_inpaint_denoise",
                            info="0.7: conserva más, 0.85: balance, 1.0: regenera completamente"
                        )
                        
                        exclude_skin = gr.Checkbox(
                            label="Excluir Piel",
                            value=True,
                            elem_id="exclude_skin",
                            info="No modificar áreas de piel detectadas"
                        )
                    
                    with gr.Row():
                        preview_mask_btn = gr.Button("👁️ Ver Máscara", size="sm")
                
                mask_preview = gr.Image(
                    label="Vista Previa de Máscara (rojo = área a modificar)",
                    type="pil",
                    height=200,
                    visible=False
                )
            
            with gr.Column(visible=False) as advanced_panel:
                gr.Markdown("**⚙️ Modo Avanzado - Control Total**")
                
                with gr.Row():
                    adv_steps = gr.Slider(
                        label="Pasos",
                        minimum=15,
                        maximum=50,
                        value=25,
                        step=1,
                        elem_id="slider_adv_steps",
                        info="15-25: normal, 30-50: alta calidad"
                    )
                    
                    adv_guidance = gr.Slider(
                        label="Guidance",
                        minimum=5,
                        maximum=15,
                        value=8.5,
                        step=0.5,
                        elem_id="slider_adv_guidance",
                        info="5-8: suave, 8-12: balance, 12-15: fuerte"
                    )
                    
                    adv_strength = gr.Slider(
                        label="Fuerza (Denoise)",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.65,
                        step=0.05,
                        elem_id="slider_adv_strength",
                        info="0.1-0.4: conserva estructura, 0.5-0.8: balance, 0.9-1.0: fuerte"
                    )
                
                with gr.Row():
                    controlnet_strength = gr.Slider(
                        label="ControlNet",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.35,
                        step=0.05,
                        elem_id="slider_controlnet",
                        info="0.0: desactivado, 0.3-0.5: balance, 0.6-1.0: fijo"
                    )
                    
                    ipadapter_strength = gr.Slider(
                        label="IP-Adapter",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.7,
                        step=0.05,
                        elem_id="slider_ipadapter",
                        info="0.0: desactivado, 0.5-0.8: balance, 0.9-1.0: fijo"
                    )
                
                seed = gr.Slider(
                    label="Seed",
                    minimum=0,
                    maximum=2147483647,
                    value=42,
                    step=1,
                    elem_id="slider_seed",
                    info="0 = aleatorio"
                )
                
                gr.Markdown("""
                **💡 Guía Rápida:**
                - **Cambiar Ropa/Cuerpo**: ControlNet 0.35, IP-Adapter 0.6, strength 0.6-0.8
                - **Cambiar Entorno/Fondo**: ControlNet 0.4, sin IP-Adapter, strength 0.55-0.75
                - **Retoques Sutiles**: Ambos activos, ControlNet 0.5, IP-Adapter 0.8, strength 0.4-0.5
                """)
            
            with gr.Row():
                generate_btn = gr.Button("🎨 Generar", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Limpiar", variant="stop")
            
            status = gr.Textbox(label="Estado", lines=2, interactive=False)
        
        with gr.Column():
            gr.Markdown("### Resultado")
            
            result = gr.Image(
                label="Resultado",
                type="pil",
                height=300,
            )
            
            with gr.Row():
                download = gr.Button("Descargar")
                use_input = gr.Button("Usar como input")
                open_folder = gr.Button("Carpeta", size="sm")
    
    btn_refresh_status.click(
        fn=lambda: (check_comfy_status(), check_models_status()),
        outputs=[comfy_status, models_status]
    )
    open_folder.click(fn=open_output_folder, outputs=[status])
    
    def toggle_panels(mode):
        is_advanced = mode
        return (
            gr.update(visible=not is_advanced),
            gr.update(visible=is_advanced)
        )
    
    advanced_mode.change(
        fn=toggle_panels,
        inputs=[advanced_mode],
        outputs=[basic_panel, advanced_panel]
    )
    
    def on_upload(img):
        if img:
            return img, f"Imagen: {img.size[0]}x{img.size[1]}px"
        return None, ""
    
    input_image.upload(on_upload, [input_image], [current_image, status])
    
    def on_generate(img, p, np, face_preserve, edit_mode,
                    advanced_mode, quality_preset,
                    adv_steps, adv_guidance, adv_strength,
                    controlnet_strength, ipadapter_strength,
                    seed, use_flux, auto_detect_clothing, mask_threshold,
                    mask_dilation, inpaint_denoise, exclude_skin):
        if not img:
            return None, "❌ Sube una imagen primero"
        if not p or not p.strip():
            return None, "❌ Escribe un prompt"
        
        # Detectar si es contenido adulto
        is_adult_content = any(keyword in p.lower() for keyword in 
            ["nude", "desnuda", "naked", "adult", "explicit", "topless", "nsfw", "desnudo", "sin ropa"])
        
        if advanced_mode:
            final_steps = adv_steps
            final_guidance = adv_guidance
            final_strength = adv_strength
            final_controlnet_strength = controlnet_strength
            final_ipadapter_strength = ipadapter_strength
            use_controlnet = controlnet_strength > 0.05
            use_ipadapter = ipadapter_strength > 0.05
            
            print(f"[ImgEditorTab] ⚙️ MODO AVANZADO: steps={final_steps}, guidance={final_guidance}, strength={final_strength}, controlnet={final_controlnet_strength}, ipadapter={final_ipadapter_strength}, seed={seed}")
            
        else:
            preset_config = {
                "fast": {"steps": 12, "guidance": 6.0, "denoise": 0.6},
                "balanced": {"steps": 20, "guidance": 7.5, "denoise": 0.65},
                "high_quality": {"steps": 30, "guidance": 9.0, "denoise": 0.7},
                "max_quality": {"steps": 40, "guidance": 11.0, "denoise": 0.75}
            }
            
            preset = preset_config.get(quality_preset, preset_config["balanced"])
            final_steps = preset["steps"]
            final_guidance = preset["guidance"]
            final_strength = preset["denoise"]
            
            if edit_mode == "Cambiar Ropa/Cuerpo":
                use_controlnet = True
                use_ipadapter = True
                final_controlnet_strength = 0.35
                final_ipadapter_strength = 0.6
                final_strength = max(final_strength, 0.6)
                print(f"[ImgEditorTab] 👔 MODO ROPA/CUERPO (preset={quality_preset}): steps={final_steps}, guidance={final_guidance}, strength={final_strength}, ControlNet={final_controlnet_strength}, IP-Adapter={final_ipadapter_strength}")
                
            elif edit_mode == "Cambiar Entorno/Fondo":
                use_controlnet = True
                use_ipadapter = False
                final_controlnet_strength = 0.4
                final_ipadapter_strength = 0.0
                final_strength = max(final_strength, 0.55)
                print(f"[ImgEditorTab] 🏔️ MODO ENTORNO (preset={quality_preset}): steps={final_steps}, guidance={final_guidance}, strength={final_strength}, ControlNet={final_controlnet_strength}")
                
            elif edit_mode == "Retoques Sutiles":
                use_controlnet = True
                use_ipadapter = True
                final_controlnet_strength = 0.5
                final_ipadapter_strength = 0.8
                final_strength = min(final_strength, 0.5)
                if quality_preset in ["fast"]:
                    final_steps = max(final_steps, 20)
                print(f"[ImgEditorTab] ✨ MODO SUTIL (preset={quality_preset}): steps={final_steps}, guidance={final_guidance}, strength={final_strength}, ControlNet={final_controlnet_strength}, IP-Adapter={final_ipadapter_strength}")
            
            else:
                use_controlnet = False
                use_ipadapter = False
                final_controlnet_strength = 0.0
                final_ipadapter_strength = 0.0
                print(f"[ImgEditorTab] 📋 MODO DEFAULT (preset={quality_preset}): steps={final_steps}, guidance={final_guidance}, strength={final_strength}")
        
        print(f"[ImgEditorTab] ✅ VALORES FINALES: steps={final_steps}, guidance={final_guidance}, strength={final_strength}, controlnet={use_controlnet}({final_controlnet_strength:.2f}), ipadapter={use_ipadapter}({final_ipadapter_strength:.2f}), seed={seed}")
        print(f"[ImgEditorTab] 🎯 Inpaint Selectivo: {auto_detect_clothing}, threshold={mask_threshold}")
        
        from roop.comfy_client import get_comfyui_url
        comfy_url = get_comfyui_url()
        try:
            response = requests.get(f"{comfy_url}/system_stats", timeout=2)
            if response.status_code != 200:
                return None, "❌ ComfyUI no conectado"
        except Exception as e:
            return None, f"❌ ComfyUI no conectado: {str(e)}"
        
        try:
            manager = get_img_editor_manager()
            
            # USAR INPAINT SELECTIVO si está activado y es contenido adulto
            if auto_detect_clothing and is_adult_content:
                print(f"[ImgEditorTab] 🎯 Usando INPAINT SELECTIVO para contenido adulto")
                print(f"[ImgEditorTab] 📊 Parámetros: threshold={mask_threshold}, dilation={mask_dilation}, denoise={inpaint_denoise}, exclude_skin={exclude_skin}")
                res, msg = manager.generate_selective(
                    image=img,
                    prompt=p,
                    negative_prompt=np or "",
                    num_inference_steps=int(final_steps),
                    guidance_scale=float(final_guidance),
                    strength=float(inpaint_denoise),  # Usar el denoise del slider
                    seed=int(seed) if seed is not None else None,
                    face_preserve=face_preserve,
                    auto_detect_clothing=True,
                    mask_threshold=float(mask_threshold),
                    mask_dilation=int(mask_dilation),
                    exclude_skin=exclude_skin,
                    use_flux=use_flux
                )
            else:
                # Usar el método normal
                res, msg = manager.generate(
                    image=img,
                    prompt=p,
                    negative_prompt=np or "",
                    num_inference_steps=int(final_steps),
                    guidance_scale=float(final_guidance),
                    strength=float(final_strength),
                    seed=int(seed) if seed is not None else None,
                    face_preserve=face_preserve,
                    use_ipadapter=use_ipadapter,
                    use_controlnet=use_controlnet,
                    controlnet_strength=float(final_controlnet_strength),
                    ipadapter_strength=float(final_ipadapter_strength),
                    use_flux=use_flux
                )
            
            if res is not None:
                mode_label = " [Avanzado]" if advanced_mode else f" [{edit_mode}]"
                if auto_detect_clothing and is_adult_content:
                    mode_label = " [Inpaint Selectivo]"
                if face_preserve:
                    mode_label += " [Cara]"
                return res, f"✅ {msg}{mode_label}"
            else:
                return None, f"❌ {msg}"
                
        except Exception as e:
            logging.error(f"Error en generación: {e}")
            return None, f"❌ Error: {str(e)}"
    
    generate_btn.click(
        on_generate,
        [
            current_image,
            prompt,
            negative_prompt,
            face_preserve,
            edit_mode,
            advanced_mode,
            quality_preset,
            adv_steps, adv_guidance, adv_strength,
            controlnet_strength,
            ipadapter_strength,
            seed,
            use_flux,
            auto_detect_clothing,
            mask_threshold,
            mask_dilation,
            inpaint_denoise,
            exclude_skin
        ],
        [result, status]
    )
    
    # Función para preview de máscara
    def on_preview_mask(img, threshold):
        if not img:
            return None, "❌ Sube una imagen primero"
        
        try:
            manager = get_img_editor_manager()
            preview, msg = manager.preview_clothing_mask(
                image=img,
                threshold=threshold
            )
            if preview is not None:
                return preview, f"✅ {msg}"
            else:
                return None, f"❌ {msg}"
        except Exception as e:
            return None, f"❌ Error: {str(e)}"
    
    preview_mask_btn.click(
        on_preview_mask,
        [current_image, mask_threshold],
        [mask_preview, status]
    )
    
    # Mostrar/ocultar preview de máscara
    def toggle_mask_preview(img):
        if img:
            return gr.update(visible=True)
        return gr.update(visible=False)
    
    input_image.change(
        toggle_mask_preview,
        [input_image],
        [mask_preview]
    )
    
    clear_btn.click(
        lambda: (
            None, "", "", None, None, "", True, "Cambiar Ropa/Cuerpo", False, "balanced",
            0.0, 0.7, 25, 8.5, 0.65, 42, True, True, 0.5, 6, 0.9, True, None
        ),
        outputs=[
            input_image, prompt, negative_prompt, current_image, result, status,
            face_preserve, edit_mode, advanced_mode, quality_preset,
            controlnet_strength, ipadapter_strength, adv_steps, adv_guidance, adv_strength, seed,
            use_flux, auto_detect_clothing, mask_threshold, mask_dilation, inpaint_denoise, exclude_skin, mask_preview
        ]
    )
    
    def use_as(img):
        if img:
            return img, img
        return current_image.value, current_image.value
    
    use_input.click(
        use_as,
        [result],
        [input_image, current_image]
    )


def img_editor_tab():
    """Wrapper function"""
    create_img_editor_tab()
