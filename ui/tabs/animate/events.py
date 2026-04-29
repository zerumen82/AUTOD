import gradio as gr
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state

def wire_animate_events(ui):
    """Conecta los componentes de animación con la lógica Grok"""
    
    # Manejar cambios en el modo de máscara
    def on_mask_mode_change(mode):
        return gr.update(visible=mode == "smart")
    
    ui["mask_mode"].change(fn=on_mask_mode_change, inputs=[ui["mask_mode"]], outputs=[ui["mask_prompt"]])

    def on_describe(img_data):
        if img_data is None: return "Sube una imagen primero"
        
        # Extraer imagen del componente ImageEditor
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        if img is None: return "Error: No se pudo extraer la imagen"
        
        from moondream_analyzer import analyze_image_with_moondream
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            img.save(tmp.name)
            res = analyze_image_with_moondream(tmp.name)
            
        return res['positive']

    ui["btn_describe"].click(
        fn=on_describe,
        inputs=[ui["input_img"]],
        outputs=[ui["prompt"]]
    )

    def on_animate_click(img_data, p, m_mode, m_prompt, motion, frames, fps, model, stabilize):
        if img_data is None: yield None, "Falta imagen"; return
        
        # Extraer imagen y máscara manual del componente ImageEditor
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        manual_mask = None
        if isinstance(img_data, dict) and m_mode == "manual":
            manual_mask = img_data.get("layers")[0] if img_data.get("layers") else None
            
        if img is None: yield None, "Error: Imagen no válida"; return
        
        state.is_animating = True
        yield None, "Iniciando Animación Inteligente..."
        
        video, msg = logic.generate_grok_animation(
            img, p, motion, frames, fps, model, stabilize,
            mask_mode=m_mode, mask_prompt=m_prompt, mask_image=manual_mask
        )
        
        state.is_animating = False
        yield video, msg

    ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[
            ui["input_img"], ui["prompt"], ui["mask_mode"], ui["mask_prompt"],
            ui["motion_bucket"], ui["num_frames"], ui["fps"], ui["model_choice"], ui["face_stabilize"]
        ],
        outputs=[ui["video_output"], ui["progress_html"]]
    )

    def on_upscale(video_path):
        if not video_path: return None, "No hay vídeo generado"
        from roop.animate.animate_manager import get_animate_manager
        manager = get_animate_manager()
        res_path, msg = manager.upscale_video(video_path)
        return res_path, msg

    ui["btn_upscale"].click(
        fn=on_upscale,
        inputs=[ui["video_output"]],
        outputs=[ui["video_output"], ui["progress_html"]]
    )
