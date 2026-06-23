import gradio as gr
import queue
import threading
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state
from animate_photo import AnimatePhoto


def _status(message):
    return f"<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>{message}</div>"


def wire_animate_events(ui):
    def on_animate_click(img_data, p, stabilize_on, l_name, l_strength, add_audio_on, audio_p):
        if state.is_animating:
            yield None, _status("Ya hay una animacion en proceso")
            return

        p = (p or "").strip()
        if img_data is None:
            yield None, _status("Sube una imagen primero")
            return

        if not AnimatePhoto().check_comfyui_status():
            yield None, _status("ComfyUI no esta activo. Inicia ComfyUI primero.")
            return
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        if img is None:
            yield None, _status("Imagen no valida")
            return

        state.is_animating = True
        updates = queue.Queue()
        result = {"video": None, "msg": "Error desconocido"}

        def progress_callback(message):
            updates.put(message)

        def worker():
            try:
                video, msg = logic.generate_grok_animation(
                    img, p,
                    stabilize=stabilize_on,
                    progress_callback=progress_callback,
                    lora_name=l_name,
                    lora_strength=l_strength,
                    add_mmaudio=add_audio_on,
                    audio_prompt=audio_p or "",
                )
                result["video"] = video
                result["msg"] = msg
            except Exception as e:
                result["msg"] = f"Excepcion: {str(e)}"
            finally:
                state.is_animating = False
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        yield None, _status("Generando animacion...")
        while True:
            message = updates.get()
            if message is None:
                break
            yield None, _status(message)

        yield result["video"], _status(result["msg"])

    # Evento para sugerir prompt
    def on_suggest_click(img):
        if img is None:
            return gr.update(value="", placeholder="Sube una imagen primero...")
        
        suggestion = logic.suggest_motion_prompt(img)
        return gr.update(value=suggestion)

    ui["btn_suggest"].click(
        fn=on_suggest_click,
        inputs=[ui["input_img"]],
        outputs=[ui["prompt"]]
    )

    ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[
            ui["input_img"], 
            ui["prompt"], 
            ui["stabilize"],
            ui["lora_name"],
            ui["lora_strength"],
            ui["add_mmaudio"],
            ui["audio_prompt"],
        ],
        outputs=[ui["video_output"], ui["progress_html"]]
    )
