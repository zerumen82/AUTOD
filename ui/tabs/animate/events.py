import gradio as gr
import queue
import threading
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state
from animate_photo import AnimatePhoto
from ui.job_cancel import (
    SCOPE_ANIMATE,
    btn_idle,
    btn_running,
    cancel_status_html,
    clear as clear_cancel,
    is_cancelled,
    request as request_cancel,
)


def _status(message):
    return f"<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>{message}</div>"


def on_cancel_animate():
    request_cancel(SCOPE_ANIMATE, interrupt_comfy=True)
    state.is_animating = False
    return _status("Cancelando animación…"), *btn_idle()


def wire_animate_events(ui):
    def on_animate_click(img_data, p, l_name, l_strength):
        clear_cancel(SCOPE_ANIMATE)

        if state.is_animating:
            yield None, _status("Ya hay una animación en proceso"), *btn_idle()
            return

        p = (p or "").strip()
        if img_data is None:
            yield None, _status("Sube una imagen primero"), *btn_idle()
            return

        if not AnimatePhoto().check_comfyui_status():
            yield None, _status("ComfyUI no está activo. Inicia ComfyUI primero."), *btn_idle()
            return
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        if img is None:
            yield None, _status("Imagen no válida"), *btn_idle()
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
                    progress_callback=progress_callback,
                    lora_name=l_name,
                    lora_strength=l_strength,
                    add_mmaudio=True,
                    cancel_check=lambda: is_cancelled(SCOPE_ANIMATE),
                )
                result["video"] = video
                result["msg"] = msg
            except Exception as e:
                result["msg"] = f"Excepción: {str(e)}"
            finally:
                state.is_animating = False
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        try:
            yield None, _status("Generando animación (~6s)..."), *btn_running()
            while True:
                if is_cancelled(SCOPE_ANIMATE):
                    break
                try:
                    message = updates.get(timeout=0.35)
                except queue.Empty:
                    yield gr.skip(), gr.skip(), gr.skip(), gr.skip()
                    continue
                if message is None:
                    break
                yield None, _status(message), gr.skip(), gr.skip()

            if is_cancelled(SCOPE_ANIMATE):
                yield None, cancel_status_html(), *btn_idle()
                return

            yield result["video"], _status(result["msg"]), *btn_idle()
        finally:
            clear_cancel(SCOPE_ANIMATE)

    animate_event = ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[
            ui["input_img"],
            ui["prompt"],
            ui["lora_name"],
            ui["lora_strength"],
        ],
        outputs=[ui["video_output"], ui["progress_html"], ui["btn_animate"], ui["btn_cancel"]],
    )

    ui["btn_cancel"].click(
        fn=on_cancel_animate,
        outputs=[ui["progress_html"], ui["btn_animate"], ui["btn_cancel"]],
        queue=False,
    )
    ui["btn_cancel"].click(fn=None, cancels=[animate_event], queue=False)