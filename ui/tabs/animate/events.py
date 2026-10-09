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
    # is_animating no se limpia aquí: el worker es dueño de la bandera y
    # liberarla con el hilo vivo dejaría la animación corriendo en silencio.
    request_cancel(SCOPE_ANIMATE, interrupt_comfy=True)
    return _status("Cancelando animación…"), *btn_idle()


def wire_animate_events(ui):
    def on_animate_click(img_data, p, engine, l_name, l_strength, stabilize_face, ar_chunks, add_audio):
        # El guard va ANTES de clear_cancel: mientras un worker cancelado
        # todavía frena, limpiar primero reviviría su bandera de cancelación.
        if state.is_animating:
            yield None, _status("Ya hay una animación en proceso"), *btn_idle()
            return

        clear_cancel(SCOPE_ANIMATE)

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
                    engine=engine,
                    stabilize=stabilize_face,
                    progress_callback=progress_callback,
                    lora_name=l_name,
                    lora_strength=l_strength,
                    add_mmaudio=add_audio,
                    autoregressive_chunks=ar_chunks,
                    cancel_check=lambda: is_cancelled(SCOPE_ANIMATE),
                )
                result["video"] = video
                result["msg"] = msg
            except Exception as e:
                result["msg"] = f"Excepción: {str(e)}"
            finally:
                # El worker es dueño de las banderas del ámbito animate.
                clear_cancel(SCOPE_ANIMATE)
                state.is_animating = False
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        # La cancelación la limpia el worker en su propio finally: hacerlo
        # aquí con el hilo vivo reviviría el trabajo cancelado.
        yield None, _status("Generando animación autoregresiva con audio..."), *btn_running()
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

    animate_event = ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[
            ui["input_img"],
            ui["prompt"],
            ui["engine"],
            ui["lora_name"],
            ui["lora_strength"],
            ui["face_stabilize"],
            ui["autoregressive_chunks"],
            ui["add_audio"],
        ],
        outputs=[ui["video_output"], ui["progress_html"], ui["btn_animate"], ui["btn_cancel"]],
    )

    ui["btn_cancel"].click(
        fn=on_cancel_animate,
        outputs=[ui["progress_html"], ui["btn_animate"], ui["btn_cancel"]],
        queue=False,
    )
    ui["btn_cancel"].click(fn=None, cancels=[animate_event], queue=False)