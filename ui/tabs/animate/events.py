import gradio as gr
import queue
import threading
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state


def _status(message):
    return f"<div style='text-align:center; color:#8b5cf6; padding:10px; font-weight:bold;'>{message}</div>"


def wire_animate_events(ui):
    def on_animate_click(img_data, p, stabilize_on):
        p = (p or "").strip()
        if img_data is None:
            yield None, _status("Sube una imagen primero")
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
                    progress_callback=progress_callback
                )
                result["video"] = video
                result["msg"] = msg
            except Exception as e:
                result["msg"] = f"Excepcion: {str(e)}"
            finally:
                updates.put(None)

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

        yield None, _status("Generando animacion...")
        while True:
            message = updates.get()
            if message is None:
                break
            yield None, _status(message)

        state.is_animating = False
        yield result["video"], _status(result["msg"])

    ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[ui["input_img"], ui["prompt"], ui["stabilize"]],
        outputs=[ui["video_output"], ui["progress_html"]]
    )
