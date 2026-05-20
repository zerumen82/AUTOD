import gradio as gr
import ui.tabs.animate.logic as logic
import ui.tabs.animate.state as state


def wire_animate_events(ui):
    def on_animate_click(img_data, p):
        p = (p or "").strip()
        if img_data is None:
            yield None, "Sube una imagen primero"
            return
        img = img_data if not isinstance(img_data, dict) else img_data.get("background")
        if img is None:
            yield None, "Imagen no válida"
            return

        state.is_animating = True
        yield None, "⏳ Generando animación..."

        video, msg = logic.generate_grok_animation(img, p)

        state.is_animating = False
        yield video, msg

    ui["btn_animate"].click(
        fn=on_animate_click,
        inputs=[ui["input_img"], ui["prompt"]],
        outputs=[ui["video_output"], ui["progress_html"]]
    )
